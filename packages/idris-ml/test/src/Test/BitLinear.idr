module Test.BitLinear

import Data.Vect

import Harness
import Device
import Tensor
import Array
import Layer
import Layer.BitLinear
import TestConfig


----------------------------------------------------------------------
-- BitLinear forward oracle (#411 B2 / #424)
----------------------------------------------------------------------
--
-- Mirror of the C-side oracle test
-- `packages/backends/test/common/nn/quantization/test_bitlinear_fwd.c`
-- through the Idris-side ABI: builds the same fixed [3, 4] ternary
-- weight + per-row scale + 4-vec input + 3-vec bias, runs
-- `tBitlinearFwd`, asserts each output matches the PyTorch oracle
-- (`packages/pytorch/torch_ref/models/bitlinear.py`).
--
-- Fixture (o=3, i=4):
--   W_ternary = [[ 1,  0, -1,  1],   row 0
--                [-1,  1,  1,  0],   row 1
--                [ 0, -1,  0,  1]]   row 2
--   w_scale   = [0.5, 0.25, 0.75]
--   x         = [1.0, 2.0, -0.5, 0.25]
--   bias      = [0.1, -0.2, 0.3]
--   expected y = [0.975, -0.075, -1.0125]
--
-- 2-bit packed bytes (one byte per row, slot 0 in low two bits):
--   row 0 → 0x71, row 1 → 0x17, row 2 → 0x4C
-- (See the C-side test for the bit-by-bit derivation.)


-- Build a Tensor [n] from a Vect of Doubles. Goes through an IO
-- bracket so the FFI chain inside `bulkToTensor` fires in
-- deterministic order; without it Idris/Chez can reorder the calls
-- across multiple let-bound tensors and one of them ends up
-- referencing an uninitialised arena slot.
mkVecDt : {0 dt : DType} -> RuntimeDType dt => Compatible TestDevice dt =>
          {n : Nat} -> Vect n Double -> IO (Tensor [n] TestDevice dt WithGrad)
mkVecDt xs = do
  raw <- ioRerun (\_ => bulkToTensor {d=TestDevice} {dt=dt}
                                     (VArray (map SArray xs)))
  pure (tinput1d {n} raw)

mkVec : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestDevice TestDType WithGrad)
mkVec = mkVecDt {dt=TestDType}


-- 1-vec helper: lift NoGrad scale + bias for the BitLinear field
-- types (scale is NoGrad by construction). Goes through an IO step
-- so the underlying FFI chain inside `bulkToTensor` fires when the
-- IO action runs; without an IO bracket the Tensor was lazily
-- evaluated mid-FFI-call and showed rank=0 in the kernel.
mkVecNoGradDt : {0 dt : DType} -> RuntimeDType dt => Compatible TestDevice dt =>
                {n : Nat} -> Vect n Double -> IO (Tensor [n] TestDevice dt NoGrad)
mkVecNoGradDt xs = do
  raw <- ioRerun (\_ => bulkToTensor {d=TestDevice} {dt=dt}
                                     (VArray (map SArray xs)))
  weakenGrad {d=TestDevice} (tinput1d {n} raw)

mkVecNoGrad : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestDevice TestDType NoGrad)
mkVecNoGrad = mkVecNoGradDt {dt=TestDType}


-- Build the 3-byte packed buffer via prim__allocBytes + prim__setByte.
-- Returns the (buffer, byte_count) pair ready to hand to
-- tCreateTernaryPacked2d.
buildFixtureBytes : IO (AnyPtr, Int)
buildFixtureBytes = do
  let buf  = prim__allocBytes 3
      buf' = prim__setByte buf 0 0x71
      buf'' = prim__setByte buf' 1 0x17
      buf''' = prim__setByte buf'' 2 0x4C
  pure (buf''', 3)


-- Read element `k` of a [3] result Tensor as a Double.
readElem3 : Tensor [3] TestDevice TestDType g -> Int -> IO Double
readElem3 t k = do
  s <- telemSelect {d=TestDevice} {n=3} t k
  pure (tensorItem {d=TestDevice} s)

-- Read element `k` of a [3] result Tensor parameterised by dtype.
readElem3Dt : {0 dt : DType} -> {0 g : GradMode} ->
              Tensor [3] TestDevice dt g -> Int -> IO Double
readElem3Dt t k = do
  s <- telemSelect {d=TestDevice} {n=3} t k
  pure (tensorItem {d=TestDevice} s)


-- The oracle assertion. Builds the fixture, runs forward, asserts
-- y[0..2] within 1e-6 of the PyTorch-computed expected values.
bitlinearForwardOracle : IO Bool
bitlinearForwardOracle = do
  (bytesPtr, byteCount) <- buildFixtureBytes
  w <- tCreateTernaryPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr byteCount
  s <- mkVecNoGrad (the (Vect 3 Double) [0.5, 0.25, 0.75])
  x <- mkVec       (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  b <- mkVec       (the (Vect 3 Double) [0.1, -0.2, 0.3])
  y <- tBitlinearFwd {d=TestDevice} {cDt=TestDType} w s x b
  y0 <- readElem3 y 0
  y1 <- readElem3 y 1
  y2 <- readElem3 y 2
  let tol = 1.0e-6
  ok0 <- checkClose "y[0] matches PyTorch oracle"   0.975    y0 tol
  ok1 <- checkClose "y[1] matches PyTorch oracle" (-0.075)   y1 tol
  ok2 <- checkClose "y[2] matches PyTorch oracle" (-1.0125)  y2 tol
  pure (ok0 && ok1 && ok2)


-- BitLinearState round-trips through the public constructor and
-- exposes its forward via the standalone helper. Doesn't go through
-- the LayerLikeMixed interface (which depends on the broader Network
-- machinery); the layer-level test is filed under #424's follow-up
-- once BitLinear gains a paramId-aware constructor.
bitlinearStateRoundtrip : IO Bool
bitlinearStateRoundtrip = do
  (bytesPtr, byteCount) <- buildFixtureBytes
  w  <- tCreateTernaryPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr byteCount
  s <- mkVecNoGrad (the (Vect 3 Double) [0.5, 0.25, 0.75])
  b <- mkVec       (the (Vect 3 Double) [0.1, -0.2, 0.3])
  let st : BitLinearState 4 3 TestDevice Ternary TestDType WithGrad
      st = bitLinearFromTensors w s b
  x <- mkVec (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  y <- tBitlinearFwd {d=TestDevice} st.weightT st.scaleT x st.biasT
  y0 <- readElem3 y 0
  checkClose "BitLinearState forward through stored fields" 0.975 y0 1.0e-6


-- BitLinear slots into NetworkMixed via the LayerLikeMixed instance
-- (B2 follow-up). Builds a single-layer NetworkMixed BitLinear,
-- runs forwardVarMixed, and asserts the output matches the same
-- PyTorch oracle (the network is just the layer wrapped in
-- OutputLayerMixed). This proves the cross-layer plumbing —
-- LayerLikeMixed instance + AnyLayerMixed wrapping + forwardVarMixed
-- chain — works end-to-end for a quantized layer.
bitlinearLayerLikeMixedOracle : IO Bool
bitlinearLayerLikeMixedOracle = do
  (bytesPtr, byteCount) <- buildFixtureBytes
  w <- tCreateTernaryPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr byteCount
  s <- mkVecNoGrad (the (Vect 3 Double) [0.5, 0.25, 0.75])
  b <- mkVec       (the (Vect 3 Double) [0.1, -0.2, 0.3])
  let anyL : AnyLayerMixed 4 3 TestDevice Ternary TestDType WithGrad
      anyL = bitLinearFromTensorsAny w s b
      net : NetworkMixed 4 [] 3 TestDevice Ternary TestDType WithGrad
      net = OutputLayerMixed anyL
  x <- mkVec (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  (_, y) <- forwardVarMixed net x
  y0 <- readElem3 y 0
  y1 <- readElem3 y 1
  y2 <- readElem3 y 2
  let tol = 1.0e-6
  ok0 <- checkClose "NetworkMixed BitLinear y[0]"   0.975    y0 tol
  ok1 <- checkClose "NetworkMixed BitLinear y[1]" (-0.075)   y1 tol
  ok2 <- checkClose "NetworkMixed BitLinear y[2]" (-1.0125)  y2 tol
  pure (ok0 && ok1 && ok2)


-- F32 oracle: same fixture as `bitlinearForwardOracle`, but scale/x/
-- bias materialised in F32. Exercises the F32 path of the kernel —
-- on tape this is the dedicated `tensor_bitlinear_fwd_f32` dispatch
-- (#411 B2 follow-up); on torch/mlx it's the native int8-mul-cast
-- path operating with `at::ScalarType::Float` / `mx::float32`.
-- F32 precision target: 1e-5 (single-precision relative error budget).
bitlinearForwardOracleF32 : IO Bool
bitlinearForwardOracleF32 = do
  (bytesPtr, byteCount) <- buildFixtureBytes
  w <- tCreateTernaryPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr byteCount
  s <- mkVecNoGradDt {dt=F32} (the (Vect 3 Double) [0.5, 0.25, 0.75])
  x <- mkVecDt       {dt=F32} (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  b <- mkVecDt       {dt=F32} (the (Vect 3 Double) [0.1, -0.2, 0.3])
  y <- tBitlinearFwd {d=TestDevice} {cDt=F32} w s x b
  y0 <- readElem3Dt y 0
  y1 <- readElem3Dt y 1
  y2 <- readElem3Dt y 2
  let tol = 1.0e-5
  ok0 <- checkClose "F32 y[0] matches PyTorch oracle"   0.975    y0 tol
  ok1 <- checkClose "F32 y[1] matches PyTorch oracle" (-0.075)   y1 tol
  ok2 <- checkClose "F32 y[2] matches PyTorch oracle" (-1.0125)  y2 tol
  pure (ok0 && ok1 && ok2)


----------------------------------------------------------------------
-- Load-time absmean quant: real-valued weights → ternary
----------------------------------------------------------------------
--
-- Roundtrip the per-row absmean quantization recipe. Starts from a
-- real-valued weight (the kind a HF BitNet checkpoint actually
-- carries on disk), runs `tAbsmeanTernaryQuant2d`, runs the
-- resulting (W_ternary, scale) through `tBitlinearFwd`, and asserts
-- the output matches the dequant-then-bitlinear oracle.
--
-- Fixture weights are pre-snapped to {-0.5, 0, +0.5} so the
-- absmean is unambiguous (0.25 for rows with all three values, etc.)
-- and the resulting (W_ternary, scale) pair is byte-deterministic.
--
-- Fixture (o=3, i=4) — chosen so each row has a different non-zero
-- absmean and the ternary pattern matches `FIXTURE_W_TERNARY` of the
-- oracle test above (so we can reuse the same expected y).
--
--   W_raw = [[ 0.5,  0.0, -0.5,  0.5],   row 0, absmean = 0.375
--            [-1.0,  1.0,  1.0,  0.0],   row 1, absmean = 0.75
--            [ 0.0, -0.5,  0.0,  0.5]]   row 2, absmean = 0.25
--
-- Quantization:
--   scale[j] = mean_k(|W[j, k]|)
--   t[j, k]  = round(W[j, k] / scale[j]).clamp(-1, +1)
-- Yields:
--   ternary = [[ 1,  0, -1,  1],   (row 0 / 0.375 = ±1.33 → ±1, clamp)
--              [-1,  1,  1,  0],
--              [ 0, -1,  0,  1]]
--   scale   = [0.375, 0.75, 0.25]
--
-- Forward with x = [1.0, 2.0, -0.5, 0.25], bias = [0.1, -0.2, 0.3]:
--   y[0] = 0.375 * (1*1.0 + 0*2.0 + (-1)*(-0.5) + 1*0.25) + 0.1
--        = 0.375 * 1.75 + 0.1 = 0.65625 + 0.1 = 0.75625
--   y[1] = 0.75 * ((-1)*1.0 + 1*2.0 + 1*(-0.5) + 0*0.25) - 0.2
--        = 0.75 * 0.5 - 0.2 = 0.375 - 0.2 = 0.175
--   y[2] = 0.25 * (0*1.0 + (-1)*2.0 + 0*(-0.5) + 1*0.25) + 0.3
--        = 0.25 * (-1.75) + 0.3 = -0.4375 + 0.3 = -0.1375

mkMat34NoGrad : Vect 3 (Vect 4 Double) ->
                IO (Tensor [3, 4] TestDevice TestDType NoGrad)
mkMat34NoGrad xs = do
  raw <- ioRerun (\_ => bulkToTensor2d {d=TestDevice} {dt=TestDType}
                                       (map (\row => VArray (map SArray row)) xs))
  weakenGrad {d=TestDevice} (tinput2d {m=3} {n=4} raw)


absmeanQuantRoundtripOracle : IO Bool
absmeanQuantRoundtripOracle = do
  w <- mkMat34NoGrad [
         [ 0.5,  0.0, -0.5,  0.5],
         [-1.0,  1.0,  1.0,  0.0],
         [ 0.0, -0.5,  0.0,  0.5]]
  (ternaryW, scale) <- tAbsmeanTernaryQuant2d w
  x <- mkVec       (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  b <- mkVec       (the (Vect 3 Double) [0.1, -0.2, 0.3])
  y <- tBitlinearFwd {d=TestDevice} {cDt=TestDType} ternaryW scale x b
  y0 <- readElem3 y 0
  y1 <- readElem3 y 1
  y2 <- readElem3 y 2
  let tol = 1.0e-6
  ok0 <- checkClose "absmean-quant y[0] matches hand-computed" 0.75625   y0 tol
  ok1 <- checkClose "absmean-quant y[1] matches hand-computed" 0.175     y1 tol
  ok2 <- checkClose "absmean-quant y[2] matches hand-computed" (-0.1375) y2 tol
  -- Also assert the scale matches the absmean by hand:
  s0 <- readElem3 scale 0
  s1 <- readElem3 scale 1
  s2 <- readElem3 scale 2
  ok3 <- checkClose "scale[0] = 0.375" 0.375 s0 tol
  ok4 <- checkClose "scale[1] = 0.75"  0.75  s1 tol
  ok5 <- checkClose "scale[2] = 0.25"  0.25  s2 tol
  pure (ok0 && ok1 && ok2 && ok3 && ok4 && ok5)


export
tests : List (IO Bool)
tests =
  [ bitlinearForwardOracle
  , bitlinearStateRoundtrip
  , bitlinearLayerLikeMixedOracle
  , bitlinearForwardOracleF32
  , absmeanQuantRoundtripOracle
  ]
