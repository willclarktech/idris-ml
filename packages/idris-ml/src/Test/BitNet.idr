module Test.BitNet

import Data.Vect

import Array
import Executor
import Layer
import Layer.BitLinear
import Layer.RmsNorm
import Tensor
import Test.Config
import Test.Harness

----------------------------------------------------------------------
-- BitNet MLP-block cross-language oracle (#411 B4.2)
----------------------------------------------------------------------
--
-- Composes three BitLinears + SiLU + elementwise multiply + RMSNorm
-- (ffn_sub_norm) into the BitNet MLP block, mirroring HF's
-- `BitNetMLP(GemmaMLP)`:
--
--     down_proj(ffn_sub_norm(silu(gate_proj(x)) * up_proj(x)))
--
-- Validates that the existing `tBitlinearFwd` kernel composes
-- correctly with the rest of the Layer stack (silu / elementwise-mul
-- / RMSNorm) across a real multi-layer forward path. The single-
-- layer BitLinear tests (#424) verify the kernel in isolation; this
-- bucket gates the composition.
--
-- Fixture (hidden=4, intermediate=6, seed=411) comes from
-- `packages/pytorch/torch_ref/models/bitnet.py --dump-idris`. Re-run
-- the dumper and paste the constants below if the fixture seed or
-- sizes ever change.
--
-- Semantics match the per-row absmean BitLinear (NOT HF's scalar
-- weight_scale + activation-quant variant — that's a separate
-- primitive landing in B4.3). The PyTorch oracle in `bitnet.py`
-- uses the same per-row variant; the cross-language tolerance is
-- 1e-6 (F64 on tape; F32 on torch-mps / mlx-gpu when ExampleDType
-- = F32).

----------------------------------------------------------------------
-- Fixture constants — pasted from `bitnet.py --dump-idris`
----------------------------------------------------------------------

FIXTURE_HIDDEN : Nat
FIXTURE_HIDDEN = 4

FIXTURE_INTERMEDIATE : Nat
FIXTURE_INTERMEDIATE = 6

FIXTURE_X : Vect 4 Double
FIXTURE_X = [-0.16570060924449506, 0.50697898891016568, 0.79047761974204911, -0.46094930205813683]

-- gate_proj weight: shape [6, 4], packed bytes per row = 1.
FIXTURE_GATE_W_BYTES : Vect 6 Int
FIXTURE_GATE_W_BYTES = [28, 223, 19, 92, 49, 17]
FIXTURE_GATE_S : Vect 6 Double
FIXTURE_GATE_S = [1.0458343978844533, 0.80335316314062011, 0.72107903464495759, 0.90019304354758134, 0.70381265384720038, 0.50734301333859877]
FIXTURE_GATE_B : Vect 6 Double
FIXTURE_GATE_B = [-0.040995353570576543, -0.035795525364578293, -0.044564600878273686, 0.030879881296616868, -0.069945560162894146, -0.017090924322408108]

-- up_proj weight: shape [6, 4], packed bytes per row = 1.
FIXTURE_UP_W_BYTES : Vect 6 Int
FIXTURE_UP_W_BYTES = [81, 15, 125, 220, 197, 213]
FIXTURE_UP_S : Vect 6 Double
FIXTURE_UP_S = [1.2468767907674523, 0.32924646460102575, 0.88216728390228916, 0.93283072336308259, 0.26621754151994453, 0.90992481027233185]
FIXTURE_UP_B : Vect 6 Double
FIXTURE_UP_B = [-0.0026141452258246402, -0.010914310791314961, -0.0026670306590134661, -0.054180323131770125, 0.044094906348529514, -0.011729664554701579]

-- down_proj weight: shape [4, 6], packed bytes per row = 2.
FIXTURE_DOWN_W_BYTES : Vect 8 Int
FIXTURE_DOWN_W_BYTES = [93, 12, 207, 12, 220, 13, 213, 4]
FIXTURE_DOWN_S : Vect 4 Double
FIXTURE_DOWN_S = [1.0727296794870445, 0.6287099828117314, 0.91146445502605611, 0.62375227880976059]
FIXTURE_DOWN_B : Vect 4 Double
FIXTURE_DOWN_B = [0.098925397866279879, 0.026898583540059876, 0.093769239001927046, -0.022116587823148945]

FIXTURE_FFN_SUB_NORM : Vect 6 Double
FIXTURE_FFN_SUB_NORM = [0.97311909206546721, 0.97154058008200295, 1.0842038806934902, 0.90390995124749218, 1.0821430005508705, 0.89429305716671603]

FIXTURE_EXPECTED_Y : Vect 4 Double
FIXTURE_EXPECTED_Y = [-3.1641801259368396, -0.29296268251998431, -2.7523220431951523, -1.1193214854352063]

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------
--
-- Duplicated from Test/BitLinear.idr — each test bucket stays self-
-- contained per the project's test-file convention. Going through
-- `ioRerun (\_ => bulkToTensor xs)` is load-bearing here for the
-- same reason as in the BitLinear test: `bulkToTensor`'s pure
-- `AnyPtr` return lets Idris/Chez reorder the FFI side-effects
-- across sibling let-bound tensors, and one of them ends up
-- referencing an uninitialised arena slot. See gotchas.md
-- "Pure-typed FFI helpers reorder across sibling let-bindings".

mkVec : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType WithGrad)
mkVec xs = do
  raw <- ioRerun (\_ => bulkToTensor {ex=TestExecutor} {dt=TestDType}
                                     (VArray (map SArray xs)))
  pure (tinput1d {n} raw)

mkVecNoGrad : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType NoGrad)
mkVecNoGrad xs = do
  raw <- ioRerun (\_ => bulkToTensor {ex=TestExecutor} {dt=TestDType}
                                     (VArray (map SArray xs)))
  weakenGrad {ex=TestExecutor} (tinput1d {n} raw)

-- Write a Vect n Int into a freshly allocated byte buffer; returns
-- the buffer pointer. Threaded through `ioRerun` to keep the
-- prim__allocBytes + prim__setByte chain inside one IO unit and
-- prevent the same pure-FFI reorder that bit `bulkToTensor`.
writeBytesPure : AnyPtr -> Int -> Vect k Int -> AnyPtr
writeBytesPure b _   []        = b
writeBytesPure b off (v :: vs) = writeBytesPure (prim__setByte b off v) (off + 1) vs

buildPackedBytes : {n : Nat} -> Vect n Int -> IO (AnyPtr, Int)
buildPackedBytes {n} xs = do
  raw <- ioRerun (\_ =>
           writeBytesPure (prim__allocBytes (cast n)) 0 xs)
  pure (raw, cast n)

readElemN : {n : Nat} -> {0 g : GradMode} ->
            Tensor [n] TestExecutor TestDType g -> Int -> IO Double
readElemN {n} t k = do
  s <- telemSelect {ex=TestExecutor} {n} t k
  pure (tensorItem {ex=TestExecutor} s)

----------------------------------------------------------------------
-- BitNet MLP block forward
----------------------------------------------------------------------
--
--   gate     = bitlinear(x,   gate_w, gate_s, gate_b)       : [m]
--   up       = bitlinear(x,   up_w,   up_s,   up_b)         : [m]
--   gated    = silu(gate) * up                              : [m]
--   normed   = rmsnorm(gated, ffn_sub_norm_weight, eps=1e-5): [m]
--   out      = bitlinear(normed, down_w, down_s, down_b)    : [h]
--
-- Each tBitlinearFwd output carries the input's grad mode (here
-- `WithGrad`, since x and biases are constructed via mkVec). The
-- RmsNorm weight is built WithGrad so the kernel's grad-mode slot
-- matches the input grad mode.

bitnetMlpBlockOracle : IO Bool
bitnetMlpBlockOracle = do
  -- Pack the three ternary weights into freshly-allocated byte buffers.
  (gateBytes, gateByteCount) <- buildPackedBytes FIXTURE_GATE_W_BYTES
  gateW <- tCreateTernaryPacked2d {ex=TestExecutor}
             {o=FIXTURE_INTERMEDIATE} {i=FIXTURE_HIDDEN}
             gateBytes gateByteCount
  (upBytes, upByteCount) <- buildPackedBytes FIXTURE_UP_W_BYTES
  upW <- tCreateTernaryPacked2d {ex=TestExecutor}
           {o=FIXTURE_INTERMEDIATE} {i=FIXTURE_HIDDEN}
           upBytes upByteCount
  (downBytes, downByteCount) <- buildPackedBytes FIXTURE_DOWN_W_BYTES
  downW <- tCreateTernaryPacked2d {ex=TestExecutor}
             {o=FIXTURE_HIDDEN} {i=FIXTURE_INTERMEDIATE}
             downBytes downByteCount

  -- Scale + bias tensors (per-row scale: NoGrad by kernel signature).
  gateS <- mkVecNoGrad FIXTURE_GATE_S
  upS   <- mkVecNoGrad FIXTURE_UP_S
  downS <- mkVecNoGrad FIXTURE_DOWN_S
  gateB <- mkVec FIXTURE_GATE_B
  upB   <- mkVec FIXTURE_UP_B
  downB <- mkVec FIXTURE_DOWN_B

  -- FFN sub-norm weight + input.
  subNormW <- mkVec FIXTURE_FFN_SUB_NORM
  x        <- mkVec FIXTURE_X

  -- Compose the block forward.
  gate   <- tBitlinearFwd {ex=TestExecutor} {cDt=TestDType} gateW gateS x gateB
  up     <- tBitlinearFwd {ex=TestExecutor} {cDt=TestDType} upW   upS   x upB
  siluG  <- tsilu gate
  gated  <- tmul siluG up
  let subNormState = MkRmsNorm subNormW
  (_, normed) <- applyRmsNormEps {ex=TestExecutor} 1.0e-5 subNormState gated
  y      <- tBitlinearFwd {ex=TestExecutor} {cDt=TestDType} downW downS normed downB

  -- Assert each output element matches the PyTorch oracle within 1e-6.
  y0 <- readElemN y 0
  y1 <- readElemN y 1
  y2 <- readElemN y 2
  y3 <- readElemN y 3
  let tol              = 1.0e-6
  let [e0, e1, e2, e3] = FIXTURE_EXPECTED_Y
  ok0 <- checkClose "block y[0] matches PyTorch oracle" e0 y0 tol
  ok1 <- checkClose "block y[1] matches PyTorch oracle" e1 y1 tol
  ok2 <- checkClose "block y[2] matches PyTorch oracle" e2 y2 tol
  ok3 <- checkClose "block y[3] matches PyTorch oracle" e3 y3 tol
  pure (ok0 && ok1 && ok2 && ok3)

export
tests : List (IO Bool)
tests =
  [ bitnetMlpBlockOracle
  ]
