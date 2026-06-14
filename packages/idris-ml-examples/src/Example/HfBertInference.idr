||| HfBertInference — fill-in-the-mask with `google/bert_uncased_L-2_H-128_A-2`.
|||
||| By default the binary loads the HF checkpoint, runs three short
||| sentences with a `[MASK]` token through `BertForMaskedLM`, and
||| prints the top-5 predicted fill-ins per sentence — the canonical
||| BERT demo, in idris-ml's typed-tensor surface.
|||
||| The CI gate (`make test-hf-bert-roundtrip`) invokes this same
||| binary with `--dump-pooled`, which switches output to the legacy
||| 128-dim pooled `[CLS]` vector on the input `[101, 7592, 102]`
||| (one float per line). `scripts/compare_inference.py` reads that and
||| asserts element-wise agreement with the Python oracle.
|||
||| One binary, two modes — keeps the cross-language correctness gate
||| and the user-facing demo on the same code path.
|||
||| Pre-requisites (CI handles these automatically via the make targets):
|||   - `packages/idris-transformers/models/google/bert_uncased_L-2_H-128_A-2/model.safetensors`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh`
|||   - Python `transformers` package available via packages/pytorch's uv
|||     venv — `Transformers.Tokenizer.idr` shells out to `scripts/hf_tokenize.py`
|||     for encode + decode, replacing the pre-2026-05-26 hardcoded ID
|||     lists + vocab.txt lookup table.
module Example.HfBertInference

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import System.File

import Array
import BuildConfig
import Checkpoint
import Example.Common.HfInferenceHelper
import Executor
import Tensor
import Transformers.Bert
import Transformers.Tokenizer
import Util

----------------------------------------------------------------------
-- Config (bert-tiny dims pinned at the type level)
----------------------------------------------------------------------

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

modelDir : String
modelDir =
  "models/google/bert_uncased_L-2_H-128_A-2"

hfWeightsPath : String
hfWeightsPath = modelDir ++ "/model.safetensors"

----------------------------------------------------------------------
-- Build small input-ID tensors (mkIds lives in HfInferenceHelper)
----------------------------------------------------------------------

arangeVect : (n : Nat) -> Vect n Double
arangeVect n = go n 0.0
  where
    go : (k : Nat) -> Double -> Vect k Double
    go Z     _ = []
    go (S k) v = v :: go k (v + 1.0)

zerosVect : (n : Nat) -> Vect n Double
zerosVect n = Vect.replicate n 0.0

----------------------------------------------------------------------
-- Top-K over the vocab logits
----------------------------------------------------------------------

-- Drain `vocab` floats out of a [vocab]-shape Tensor's raw pointer
-- into a (index, logit) pair list. Done host-side because we want
-- index ordering for the eventual top-5 print, and primItem1d is
-- cheap — even 30522 calls take well under a second on tape.
readLogits : (vocab : Nat) -> AnyPtr -> IO (List (Nat, Double))
readLogits vocab p = go (cast {to=Int} vocab) 0 []
  where
    go : Int -> Int -> List (Nat, Double) -> IO (List (Nat, Double))
    go end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=ExampleExecutor} p i
             in go end (i + 1) ((cast {to=Nat} i, v) :: acc)

-- O(n log n) sort by descending logit + take 5. With vocab=30522 and
-- k=5 a selection algorithm would shave a small constant; the sort is
-- the obvious code and runs in tens of ms.
topK : Nat -> List (Nat, Double) -> List (Nat, Double)
topK k xs = take k (sortBy (\(_, a), (_, b) => compare b a) xs)

----------------------------------------------------------------------
-- One fill-in-the-mask demo
----------------------------------------------------------------------

-- BERT's WordPiece vocab assigns id 103 to the [MASK] token. The Idris
-- side searches for this token in the tokenized input to find the mask
-- position; tokenizing "paris is the capital of [MASK] ." returns
-- [101, 3000, …, 103, 1012, 102] (verified upstream by the tokenizer
-- subprocess + pinned in Test/Transformers.Tokenizer.idr's BERT encode test).
bertMaskTokenId : Nat
bertMaskTokenId = 103

-- The detokenize call returns the top-5 predicted tokens
-- space-separated (BERT WordPiece decode joins tokens with single
-- spaces), so we split on whitespace via `words` to recover
-- individual words. `joinBy` for the formatted output comes from
-- Data.String.

runMaskDemo : Tokenizer VocabSize
           -> (model : BertForMaskedLmState VocabSize Hidden NumLayers
                                            Intermediate MaxPos TypeVocab
                                            ExampleExecutor ExampleDType NoGrad)
           -> (sentence : String)
           -> IO ()
runMaskDemo tok evalModel sentence = do
  Right (seqLen ** tokens) <- tokenize tok sentence
    | Left err => putStrLn ("  ERR: tokenize: " ++ show err)
  case findIndex (\f => finToNat f == bertMaskTokenId) tokens of
    Nothing => putStrLn ("  ERR: input has no [MASK] token: " ++ show sentence)
    Just maskFin => do
      -- Each Fin VocabSize → Double via finToNat → cast.
      let idDoubles = map (cast {to=Double} . finToNat) tokens
          inputIds  = retypeGrad (mkIds idDoubles)
          posIds    = retypeGrad (mkIds (arangeVect seqLen))
          typeIds   = retypeGrad (mkIds (zerosVect seqLen))
      logits <- hfBertMlmForward {ex=ExampleExecutor} {dt=ExampleDType}
                                 {seqLen}
                                 {vocab        = VocabSize}
                                 {hidden       = Hidden}
                                 {numLayers    = NumLayers}
                                 {numHeads     = NumHeads}
                                 {headDim      = HeadDim}
                                 {intermediate = Intermediate}
                                 {maxPos       = MaxPos}
                                 {typeVocab    = TypeVocab}
                                 evalModel inputIds posIds typeIds Nothing
      maskRow <- trowSelect logits (cast {to=Int} (finToNat maskFin))
      pairs   <- readLogits VocabSize maskRow.tensorPtr
      let top5     = topK 5 pairs
          topIds   = map fst top5
          topLogits = map snd top5
      -- Lift each Nat → Fin VocabSize. mapMaybe drops any that fail;
      -- our IDs come from readLogits over the 30522-wide logits row
      -- so all of them are < VocabSize and Nothings should be impossible.
      let lifted : List (Fin VocabSize)
          lifted = mapMaybe (\n => natToFin n VocabSize) topIds
      -- Detokenize all 5 IDs in one subprocess call; BERT WordPiece's
      -- decode joins space-separated → split back on whitespace.
      Right decoded <- detokenize tok (fromList lifted)
        | Left err => putStrLn ("  ERR: detokenize: " ++ show err)
      let topWords = words decoded
          formatted = zipWith fmt topWords topLogits
      putStrLn ("Input:  " ++ sentence)
      putStrLn ("Top-5:  " ++ joinBy ", " formatted)
      putStrLn ""
  where
    -- Two-decimal-place logit formatter — same shape as the original.
    fmt : String -> Double -> String
    fmt tok x =
      let scaled : Int = cast (x * 100.0 + (if x < 0.0 then -0.5 else 0.5))
          whole = scaled `div` 100
          frac  = abs (scaled `mod` 100)
          sign  = if x < 0.0 then "-" else "+"
          fracStr = if frac < 10 then "0" ++ show frac else show frac
      in tok ++ " (" ++ sign ++ show (abs whole) ++ "." ++ fracStr ++ ")"

----------------------------------------------------------------------
-- --dump-pooled: legacy 128-float output for the CI comparator
----------------------------------------------------------------------

printPooled : Int -> Int -> AnyPtr -> IO ()
printPooled end i p =
  if i >= end
    then pure ()
    else do
      let v = primItem1d {ex=ExampleExecutor} p i
      putStrLn (show v)
      printPooled end (i + 1) p

runPooledDump : (model : BertForMaskedLmState VocabSize Hidden NumLayers
                                              Intermediate MaxPos TypeVocab
                                              ExampleExecutor ExampleDType NoGrad)
             -> IO ()
runPooledDump model = do
  -- Same fixed input save_oracle.py uses: [CLS] hello [SEP].
  -- `retypeGrad` lifts the WithGrad ids `mkIds` builds to the model's
  -- NoGrad (inference) gradmode — the single-g forward needs them to match.
  let inputIds = retypeGrad (mkIds (the (Vect 3 Double) [101.0, 7592.0, 102.0]))
      posIds   = retypeGrad (mkIds (the (Vect 3 Double) [0.0, 1.0, 2.0]))
      typeIds  = retypeGrad (mkIds (the (Vect 3 Double) [0.0, 0.0, 0.0]))
  out <- hfBertForward {ex=ExampleExecutor} {dt=ExampleDType}
                       {seqLen       = 3}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       {typeVocab    = TypeVocab}
                       model.base inputIds posIds typeIds Nothing
  printPooled (cast {to=Int} Hidden) 0 out.tensorPtr

----------------------------------------------------------------------
-- main (stageStamp lives in HfInferenceHelper)
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let dumpPooled = elem "--dump-pooled" args
  t0 <- clockTime Monotonic

  -- Build the full BertForMaskedLM (encoder + pooler + MLM head, 44 params).
  -- Construct the inference model directly in NoGrad: every param is born
  -- with requires_grad=0 (no post-construction `eval` flip), so the forward
  -- below is genuinely tape-free and this model can't be fed to an optimizer.
  model <- hfBertForMaskedLm {ex=ExampleExecutor} {dt=ExampleDType} {g=NoGrad}
                             {vocab        = VocabSize}
                             {hidden       = Hidden}
                             {numLayers    = NumLayers}
                             {numHeads     = NumHeads}
                             {intermediate = Intermediate}
                             {maxPos       = MaxPos}
                             {typeVocab    = TypeVocab}
                             "bert"
  stageStamp "hfBertForMaskedLm ok" t0
  ok <- loadModelAllowCast {ex=ExampleExecutor} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()
  stageStamp "loadModelAllowCast ok" t0

  if dumpPooled
    then runPooledDump model
    else do
      -- Construct the BERT tokenizer once; each demo reuses it for both
      -- the input-string encode AND the top-5 token-id decode. The
      -- subprocess startup cost (~1s) amortises across the three demos.
      tokR <- mkTokenizer "google/bert_uncased_L-2_H-128_A-2" VocabSize
      case tokR of
        Left err => do
          putStrLn ("ERR: mkTokenizer: " ++ show err)
          exitFailure
        Right tok => do
          putStrLn ""
          putStrLn "BERT fill-in-the-mask — google/bert_uncased_L-2_H-128_A-2"
          putStrLn "=========================================================="
          putStrLn ""
          -- Three sentences fed through the tokenizer. The "[MASK]"
          -- literal in each string is tokenized to BERT's mask token
          -- id 103 (by AutoTokenizer); runMaskDemo searches for it to
          -- locate the position to score.
          benchT0 <- clockTime Monotonic
          runMaskDemo tok model "paris is the capital of [MASK] ."
          runMaskDemo tok model "i went to the [MASK] to buy bread ."
          runMaskDemo tok model "the man worked as a [MASK] ."
          benchT1 <- clockTime Monotonic
          -- Axis D perf marker: token count = 25 (wordpiece-aware:
          -- 8 + 8 + 9 across the three sentences including [CLS]/[SEP]
          -- as tokenized by bert-tiny's WordPiece). Wall is the sum
          -- across all three demos including tokenize + forward +
          -- decode subprocess hops; that's the user-observable inference
          -- cost so it's the right number to report.
          let benchMs =
                let s  = cast {to=Double} (seconds benchT1 - seconds benchT0)
                    ns = cast {to=Double} (nanoseconds benchT1 - nanoseconds benchT0)
                in s * 1000.0 + ns / 1000000.0
          putStrLn ""
          putStrLn ("PERF_GENERATE_TOKENS=25")
          putStrLn ("PERF_GENERATE_WALL_MS=" ++ show benchMs)
