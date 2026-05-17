||| HfBertInference — load `google/bert_uncased_L-2_H-128_A-2` weights
||| via the HF-aligned `HfBert` module and run a forward pass on
||| `[CLS] hello [SEP]`. Output is the 128-dim pooled `[CLS]` vector,
||| one value per line on stdout.
|||
||| The companion `packages/idris-transformers/scripts/save_oracle.py`
||| produces a `bert-tiny-oracle.safetensors` containing the same
||| forward output computed by HF transformers' Python. The
||| `test-hf-bert-roundtrip` Makefile target wires the two together:
||| run this binary, capture stdout, hand off to `compare_inference.py`
||| which loads the oracle and asserts element-wise agreement within
||| F32 tolerance.
|||
||| Pre-requisites (CI handles these automatically):
|||   - `packages/idris-transformers/fixtures/google/bert_uncased_L-2_H-128_A-2/model.safetensors`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh`
|||   - Live registry — `hfBertModel "bert"` must register exactly the
|||     param names HF's state_dict() exposes (locked down by
|||     HfBert.bertParamNames + the package's unit test bucket 2).
module Example.HfBertInference

import Data.Vect
import System

import Array
import BuildConfig
import Checkpoint
import Device
import HfBert
import Tensor


-- BERT-tiny config dims pinned at the type level so the
-- divisibility proof (128 = 2 * 64) resolves cleanly.
VocabSize : Nat
VocabSize = 30522

Hidden : Nat
Hidden = 128

NumLayers : Nat
NumLayers = 2

NumHeads : Nat
NumHeads = 2

HeadDim : Nat
HeadDim = 64

Intermediate : Nat
Intermediate = 512

MaxPos : Nat
MaxPos = 512

TypeVocab : Nat
TypeVocab = 2

SeqLen : Nat
SeqLen = 3


-- Path to the HF safetensors fixture. The path is relative to the
-- repo root so the binary works under `make example-hf-bert-inference`
-- (which is invoked from the repo root).
hfWeightsPath : String
hfWeightsPath =
  "packages/idris-transformers/fixtures/" ++
  "google/bert_uncased_L-2_H-128_A-2/model.safetensors"


-- Print 128 doubles, one per line, by walking primItem1d. The
-- comparator script reads stdout line-by-line.
printOutput : Int -> Int -> AnyPtr -> IO ()
printOutput end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {d=ExampleDevice} p i
      putStrLn (show v)
      printOutput end (i + 1) p


main : IO ()
main = do
  -- Build the HF-aligned BERT model. Registers 39 params under
  -- HF-native names like `bert.embeddings.word_embeddings.weight`.
  model <- hfBertModel {d=ExampleDevice} {dt=ExampleDType}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       {typeVocab    = TypeVocab}
                       "bert"
  -- Load the HF safetensors. allow_cast=True because the on-disk
  -- dtype is F32 and the active tape build is F64 — the safetensors
  -- loader widens F32 → F64 via the lingua-franca double pivot.
  ok <- loadModelAllowCast {d=ExampleDevice} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()

  -- Build the three input ID tensors. [CLS] hello [SEP] = [101, 7592,
  -- 102]; position IDs are arange(0, seqLen); token-type IDs are all
  -- zeros (single-sentence input). Same values save_oracle.py uses.
  let inputIdsRaw = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                                 (VArray [SArray 101.0, SArray 7592.0, SArray 102.0])
      posIdsRaw   = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                                 (VArray [SArray 0.0, SArray 1.0, SArray 2.0])
      typeIdsRaw  = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                                 (VArray [SArray 0.0, SArray 0.0, SArray 0.0])
  let inputIds : Tensor [SeqLen] ExampleDevice ExampleDType WithGrad
      inputIds = tinput1d {n=SeqLen} inputIdsRaw
      posIds : Tensor [SeqLen] ExampleDevice ExampleDType WithGrad
      posIds = tinput1d {n=SeqLen} posIdsRaw
      typeIds : Tensor [SeqLen] ExampleDevice ExampleDType WithGrad
      typeIds = tinput1d {n=SeqLen} typeIdsRaw

  -- Forward. Auto-proof resolves Hidden = NumHeads * HeadDim
  -- (128 = 2 * 64).
  out <- hfBertForward {d=ExampleDevice} {dt=ExampleDType}
                       {seqLen       = SeqLen}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       {typeVocab    = TypeVocab}
                       model inputIds posIds typeIds

  -- Dump all 128 pooled-output values to stdout.
  printOutput (cast {to=Int} Hidden) 0 out.tensorPtr
