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


export
tests : List (IO Bool)
tests =
  [ bitlinearForwardOracle
  , bitlinearStateRoundtrip
  , bitlinearLayerLikeMixedOracle
  , bitlinearForwardOracleF32
  ]
