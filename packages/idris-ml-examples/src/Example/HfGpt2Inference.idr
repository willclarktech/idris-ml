||| HfGpt2Inference — load `hf-internal-testing/tiny-random-gpt2` and
||| exercise the typed GPT-2 forward pass against the HF Python oracle.
|||
||| `hf-internal-testing/tiny-random-gpt2` is HF's random-init GPT-2
||| test fixture: vocab=1000, hidden=32, n_layer=5, n_head=4, head_dim=8,
||| intermediate=37, max_pos=512. Random weights (not pretrained), but
||| architecturally complete and — critically — ships safetensors on
||| disk (sshleifer/tiny-gpt2 has only pytorch_model.bin). Exercises
||| every GPT-2 architectural piece (fused QKV via c_attn, Conv1D
||| transpose storage, learned positional embeddings, causal mask,
||| tied LM head). Useful for validating architectural plumbing; not
||| useful for generation quality (weights are random).
|||
||| The binary dumps the final-position hidden state (after `ln_f`) to
||| stdout, one float per line. `scripts/compare_inference.py` reads
||| that and asserts elementwise agreement with the Python oracle
||| produced by `scripts/save_oracle_gpt2.py`.
|||
||| Future: when the Phase-2 tokenizer subprocess lands, this binary
||| gains a user-facing string-in / string-out generation mode (mirror
||| of the `--dump-pooled` vs default split in HfBertInference). For
||| v1 the binary is gate-only.
|||
||| Pre-requisites (CI handles these via the make targets):
|||   - `packages/idris-transformers/models/hf-internal-testing/tiny-random-gpt2/model.safetensors`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh
|||                   hf-internal-testing/tiny-random-gpt2`
module Example.HfGpt2Inference

import Data.Vect
import System
import System.File

import Array
import BuildConfig
import Checkpoint
import Device
import HfGpt2
import Tensor


----------------------------------------------------------------------
-- Config (tinyGpt2Config dims pinned at the type level)
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 1000

Hidden : Nat
Hidden = 32

NumLayers : Nat
NumLayers = 5

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 8

Intermediate : Nat
Intermediate = 128

MaxPos : Nat
MaxPos = 512


modelDir : String
modelDir = "packages/idris-transformers/models/hf-internal-testing/tiny-random-gpt2"

hfWeightsPath : String
hfWeightsPath = modelDir ++ "/model.safetensors"


----------------------------------------------------------------------
-- Build small input-ID + position tensors
----------------------------------------------------------------------

mkIds : {n : Nat} -> Vect n Double
     -> Tensor [n] ExampleDevice ExampleDType WithGrad
mkIds xs =
  let raw = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

arangeVect : (n : Nat) -> Vect n Double
arangeVect n = go n 0.0
  where
    go : (k : Nat) -> Double -> Vect k Double
    go Z     _ = []
    go (S k) v = v :: go k (v + 1.0)


----------------------------------------------------------------------
-- Dump a [hidden]-shape tensor to stdout, one float per line
----------------------------------------------------------------------

printRow : Int -> Int -> AnyPtr -> IO ()
printRow end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {d=ExampleDevice} p i
      putStrLn (show v)
      printRow end (i + 1) p


----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

main : IO ()
main = do
  -- Build a tinyGpt2Config model. Each param registers under the
  -- literal HF name (`transformer.wte.weight`, etc.).
  model <- hfGpt2Model {d=ExampleDevice} {dt=ExampleDType}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       ""
  -- Load the HF checkpoint. loadModelAllowCast handles BF16/F32→target
  -- dtype widening at the loader; this checkpoint is F32 on disk so
  -- it's a straight load on F64 backends (tape) and a copy on F32
  -- backends (mlx-gpu / torch-mps).
  ok <- loadModelAllowCast {d=ExampleDevice} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()

  -- Fixed input matching save_oracle_gpt2.py: token IDs [42, 137] in the
  -- 1000-token random vocab. Two tokens: minimum that exercises the
  -- learned positional embedding AND the causal mask (position 1
  -- attends to position 0; position 0 cannot attend to position 1).
  let inputIds = mkIds (the (Vect 2 Double) [42.0, 137.0])
      posIds   = mkIds (arangeVect 2)
  out <- hfGpt2Forward {d=ExampleDevice} {dt=ExampleDType}
                       {seqLen       = 2}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       model inputIds posIds
  -- Pull row 1 (last position; carries info from position 0 via causal
  -- attention) — matches save_oracle_gpt2.py's last_hidden[0, -1, :].
  lastRow <- trowSelect out 1
  printRow (cast {to=Int} Hidden) 0 lastRow.tensorPtr
