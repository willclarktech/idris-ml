module Test.BitLinear

import Data.Vect

import Test.Harness
import Device
import Tensor
import Array
import Layer
import Layer.BitLinear
import Test.Config


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


----------------------------------------------------------------------
-- HF-format ternary load (microsoft/bitnet-b1.58-2B-4T-style)
----------------------------------------------------------------------
--
-- Roundtrip the HF -> ours layout reshuffle + encoding remap. Takes a
-- known HF-packed byte sequence (the SAME logical ternary matrix as
-- `bitlinearForwardOracle` above, but stored in HF's `[(o+3)/4, i]`
-- layout with `{-1, 0, +1} -> {0, 1, 2}` encoding), runs through
-- `tCreateTernaryFromHfPacked2d`, then through `tBitlinearFwd`, and
-- asserts the output matches FIXTURE_EXPECTED_Y.
--
-- Derivation for the [3, 4] fixture (o=3 → hf_row_dim = (3+3)/4 = 1):
--   Original ternary:
--     [[ 1,  0, -1,  1],
--      [-1,  1,  1,  0],
--      [ 0, -1,  0,  1]]
--   +1 (HF encoding):
--     [[ 2,  1,  0,  2],
--      [ 0,  2,  2,  1],
--      [ 1,  0,  1,  2]]
--   HF packed shape: [hf_row_dim, i] = [1, 4]. Each byte holds 4
--   chunks of 2 bits (low to high) corresponding to the 4 output rows
--   at the same column. With o=3, only chunks 0..2 are populated;
--   chunk 3 stays 0.
--     col 0: low=2, then 0, then 1, then 0 → 2 | (0<<2) | (1<<4) | (0<<6) = 0x12
--     col 1: 1 | (2<<2) | (0<<4) | (0<<6) = 0x09
--     col 2: 0 | (2<<2) | (1<<4) | (0<<6) = 0x18
--     col 3: 2 | (1<<2) | (2<<4) | (0<<6) = 0x26
--   So FIXTURE_HF_PACKED = [0x12, 0x09, 0x18, 0x26], total bytes = 4.

buildHfFixtureBytes : IO (AnyPtr, Int)
buildHfFixtureBytes = do
  let buf    = prim__allocBytes 4
      buf'   = prim__setByte buf  0 0x12
      buf''  = prim__setByte buf' 1 0x09
      buf''' = prim__setByte buf'' 2 0x18
      buf4   = prim__setByte buf''' 3 0x26
  pure (buf4, 4)


bitlinearHfPackedRoundtrip : IO Bool
bitlinearHfPackedRoundtrip = do
  (bytesPtr, _) <- buildHfFixtureBytes
  w <- tCreateTernaryFromHfPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr
  s <- mkVecNoGrad (the (Vect 3 Double) [0.5, 0.25, 0.75])
  x <- mkVec       (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  b <- mkVec       (the (Vect 3 Double) [0.1, -0.2, 0.3])
  y <- tBitlinearFwd {d=TestDevice} {cDt=TestDType} w s x b
  y0 <- readElem3 y 0
  y1 <- readElem3 y 1
  y2 <- readElem3 y 2
  let tol = 1.0e-6
  ok0 <- checkClose "HF-pack y[0] matches the our-pack oracle"   0.975    y0 tol
  ok1 <- checkClose "HF-pack y[1] matches the our-pack oracle" (-0.075)   y1 tol
  ok2 <- checkClose "HF-pack y[2] matches the our-pack oracle" (-1.0125)  y2 tol
  pure (ok0 && ok1 && ok2)


----------------------------------------------------------------------
-- Activation quantization (per-token symmetric int8)
----------------------------------------------------------------------
--
-- Mirrors HF transformers' `BitLinear.activation_quant` recipe (from
-- `packages/pytorch/.venv/.../transformers/integrations/bitnet.py`):
--
--   scale  = 127 / max(|x|).clamp(min=1e-5)
--   quant  = round(x * scale).clamp(-128, 127)
--
-- Hand-computed for fixture x = [-1.5, 0.3, 0.8, -0.6]:
--   max(|x|)    = 1.5
--   safe_max    = max(1.5, 1e-5) = 1.5
--   in_scale    = 127 / 1.5 ≈ 84.6666666...
--   x * in_scale = [-127.0, 25.4, 67.7333..., -50.8]
--   round       = [-127, 25, 68, -51]
--   clamp(-128,127) = same (none out of range)

activationQuantInt8Oracle : IO Bool
activationQuantInt8Oracle = do
  x <- mkVec (the (Vect 4 Double) [-1.5, 0.3, 0.8, -0.6])
  (xq, inScale) <- tActivationQuantInt8 x
  -- Read 4 elements of xq
  q0 <- readElemN xq 0
  q1 <- readElemN xq 1
  q2 <- readElemN xq 2
  q3 <- readElemN xq 3
  let tol = 1.0e-6
  ok0 <- checkClose "act-quant x[0] = -127"  (-127.0) q0 tol
  ok1 <- checkClose "act-quant x[1] =   25"    25.0   q1 tol
  ok2 <- checkClose "act-quant x[2] =   68"    68.0   q2 tol
  ok3 <- checkClose "act-quant x[3] =  -51"  (-51.0)  q3 tol
  -- 127 / 1.5 = 84.66666...
  ok4 <- checkClose "input_scale = 127/1.5" (127.0 / 1.5) inScale 1.0e-9
  pure (ok0 && ok1 && ok2 && ok3 && ok4)
  where
    readElemN : Tensor [4] TestDevice TestDType NoGrad -> Int -> IO Double
    readElemN t k = do
      s <- telemSelect {d=TestDevice} {n=4} t k
      pure (tensorItem {d=TestDevice} s)


----------------------------------------------------------------------
-- Fused HF BitLinear (tBitlinearFwdHfQuant) consistency vs composed
----------------------------------------------------------------------
--
-- The fused `tBitlinearFwdHfQuant` and the composed
-- `tActivationQuantInt8` + `tBitlinearFwd` formulation should
-- produce identical results (modulo FP rounding within the
-- intermediate accumulators). Validates both paths agree on a
-- known fixture. The cross-language gate vs HF transformers'
-- actual `BitLinear.forward` lands in a separate slice once
-- HfBitNet.idr ships an end-to-end model load path.
--
-- Fixture (o=3, i=4, no RMSNorm):
--   W_ternary = bitlinearForwardOracle's [[1,0,-1,1], [-1,1,1,0], [0,-1,0,1]]
--   weight_scale = 0.5 (HF's scalar)
--   x   = [1.0, 2.0, -0.5, 0.25]
--   bias = [0.1, -0.2, 0.3]

bitlinearFwdHfQuantConsistency : IO Bool
bitlinearFwdHfQuantConsistency = do
  (bytesPtr, byteCount) <- buildFixtureBytes
  w <- tCreateTernaryPacked2d {d=TestDevice} {o=3} {i=4} bytesPtr byteCount
  x <- mkVec    (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  b <- mkVec    (the (Vect 3 Double) [0.1, -0.2, 0.3])

  -- Composed path: tActivationQuantInt8 + tBitlinearFwd with
  -- per-row scale = 1 / (in_scale * w_scale) broadcast to [o].
  let weightScale = 0.5
  (xq, inScale) <- tActivationQuantInt8 x
  let perRowScale = 1.0 / (inScale * weightScale)
  s <- mkVecNoGrad (the (Vect 3 Double) [perRowScale, perRowScale, perRowScale])
  -- xq is NoGrad; weakenGrad? Actually tBitlinearFwd wants matching grad mode on x and bias.
  -- xq : Tensor [4] d dt NoGrad; bias : WithGrad. Promote bias to NoGrad to match.
  bNoG <- weakenGrad {d=TestDevice} b
  yComposed <- tBitlinearFwd {d=TestDevice} {cDt=TestDType} w s xq bNoG

  -- Fused path: tBitlinearFwdHfQuant (useRmsNorm = False; rmsW is placeholder).
  xNoG <- weakenGrad {d=TestDevice} x
  yFused <- tBitlinearFwdHfQuant {d=TestDevice} {cDt=TestDType} w weightScale x b False xNoG 1.0e-5

  yc0 <- readElem3 yComposed 0
  yc1 <- readElem3 yComposed 1
  yc2 <- readElem3 yComposed 2
  yf0 <- readElem3 yFused 0
  yf1 <- readElem3 yFused 1
  yf2 <- readElem3 yFused 2
  let tol = 1.0e-9
  ok0 <- checkClose "fused y[0] matches composed" yc0 yf0 tol
  ok1 <- checkClose "fused y[1] matches composed" yc1 yf1 tol
  ok2 <- checkClose "fused y[2] matches composed" yc2 yf2 tol
  pure (ok0 && ok1 && ok2)


export
tests : List (IO Bool)
tests =
  [ bitlinearForwardOracle
  , bitlinearStateRoundtrip
  , bitlinearLayerLikeMixedOracle
  , bitlinearForwardOracleF32
  , absmeanQuantRoundtripOracle
  , bitlinearHfPackedRoundtrip
  , activationQuantInt8Oracle
  , bitlinearFwdHfQuantConsistency
  ]
