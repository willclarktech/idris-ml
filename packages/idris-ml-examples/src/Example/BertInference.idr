||| BertInference — fill-in-the-mask with `google/bert_uncased_L-2_H-128_A-2`.
|||
||| By default the binary loads the HF checkpoint, runs three short
||| sentences with a `[MASK]` token through `BertForMaskedLM`, and
||| prints the top-5 predicted fill-ins per sentence — the canonical
||| BERT demo, in idris-ml's typed-tensor surface.
|||
||| The model is loaded with `Transformers.Bert.fromPretrained`: it reads
||| `<dir>/config.json` for the dims, builds a tape-free `NoGrad` model at
||| those dims, and fills params from `<dir>/model.safetensors`. Nothing
||| about bert-tiny is hardcoded here — the returned `(cfg ** model)` ties
||| the model's type to the file's dims, and the per-head split is
||| recovered from the config (a runtime divisibility check supplies the
||| `hidden = numHeads * headDim` proof the forward needs).
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
|||   - `models/google/bert_uncased_L-2_H-128_A-2/{config.json,model.safetensors}`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh`
|||   - Python `transformers` package available via packages/pytorch's uv
|||     venv — `Transformers.Tokenizer.idr` shells out to `scripts/hf_tokenize.py`
|||     for encode + decode, replacing the pre-2026-05-26 hardcoded ID
|||     lists + vocab.txt lookup table.
module Example.BertInference

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import Decidable.Equality
import System
import System.Clock
import System.File

import Ml.Array
import Ml.Checkpoint
import Ml.Executor
import Ml.Tensor
import Ml.Util
import Transformers.Bert
import Transformers.Tokenizer

import BuildConfig
import Example.Common.InferenceHelper

----------------------------------------------------------------------
-- Model location (dims come from the file, not from here)
----------------------------------------------------------------------

modelDir : String
modelDir = "models/google/bert_uncased_L-2_H-128_A-2"

----------------------------------------------------------------------
-- Build small input-ID tensors (mkIds lives in InferenceHelper)
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

runMaskDemo : (cfg : BertConfig)
           -> (nHeads, hDim : Nat)
           -> (prf : hidden cfg = nHeads * hDim)
           -> Tokenizer (vocabSize cfg)
           -> BertForMaskedLm cfg ExampleExecutor ExampleDType NoGrad
           -> (sentence : String)
           -> IO ()
runMaskDemo cfg nHeads hDim prf tok model sentence = do
  Right (seqLen ** tokens) <- tokenize tok sentence
    | Left err => putStrLn ("  ERR: tokenize: " ++ show err)
  case findIndex (\f => finToNat f == bertMaskTokenId) tokens of
    Nothing      => putStrLn ("  ERR: input has no [MASK] token: " ++ show sentence)
    Just maskFin => do
      -- Each Fin (vocabSize cfg) → Double via finToNat → cast.
      let idDoubles = map (cast {to=Double} . finToNat) tokens
          inputIds = retypeGrad (mkIds idDoubles)
          posIds   = retypeGrad (mkIds (arangeVect seqLen))
          typeIds  = retypeGrad (mkIds (zerosVect seqLen))
      logits <- hfBertMlmForward {ex=ExampleExecutor} {dt=ExampleDType}
                                 {seqLen}
                                 {vocab        = vocabSize cfg}
                                 {hidden       = hidden cfg}
                                 {numLayers    = numLayers cfg}
                                 {numHeads     = nHeads}
                                 {headDim      = hDim}
                                 {intermediate = intermediate cfg}
                                 {maxPos       = maxPosition cfg}
                                 {typeVocab    = typeVocabSize cfg}
                                 {prf}
                                 model inputIds posIds typeIds Nothing
      maskRow <- trowSelect logits (cast {to=Int} (finToNat maskFin))
      pairs   <- readLogits (vocabSize cfg) maskRow.tensorPtr
      let top5     = topK 5 pairs
          topIds    = map fst top5
          topLogits = map snd top5
      -- Lift each Nat → Fin (vocabSize cfg). mapMaybe drops any that fail;
      -- our IDs come from readLogits over the (vocabSize cfg)-wide logits
      -- row so all of them are in range and Nothings should be impossible.
      let lifted : List (Fin (vocabSize cfg))
          lifted = mapMaybe (\n => natToFin n (vocabSize cfg)) topIds
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
          whole   = scaled `div` 100
          frac    = abs (scaled `mod` 100)
          sign    = if x < 0.0 then "-" else "+"
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

runPooledDump : (cfg : BertConfig)
             -> (nHeads, hDim : Nat)
             -> (prf : hidden cfg = nHeads * hDim)
             -> BertForMaskedLm cfg ExampleExecutor ExampleDType NoGrad
             -> IO ()
runPooledDump cfg nHeads hDim prf model = do
  -- Same fixed input save_oracle.py uses: [CLS] hello [SEP].
  -- `retypeGrad` lifts the WithGrad ids `mkIds` builds to the model's
  -- NoGrad (inference) gradmode — the single-g forward needs them to match.
  let inputIds = retypeGrad (mkIds (the (Vect 3 Double) [101.0, 7592.0, 102.0]))
      posIds  = retypeGrad (mkIds (the (Vect 3 Double) [0.0, 1.0, 2.0]))
      typeIds = retypeGrad (mkIds (the (Vect 3 Double) [0.0, 0.0, 0.0]))
  out <- hfBertForward {ex=ExampleExecutor} {dt=ExampleDType}
                       {seqLen       = 3}
                       {vocab        = vocabSize cfg}
                       {hidden       = hidden cfg}
                       {numLayers    = numLayers cfg}
                       {numHeads     = nHeads}
                       {headDim      = hDim}
                       {intermediate = intermediate cfg}
                       {maxPos       = maxPosition cfg}
                       {typeVocab    = typeVocabSize cfg}
                       {prf}
                       model.base inputIds posIds typeIds Nothing
  printPooled (cast {to=Int} (hidden cfg)) 0 out.tensorPtr

----------------------------------------------------------------------
-- main (stageStamp lives in InferenceHelper)
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let dumpPooled = elem "--dump-pooled" args
  t0 <- clockTime Monotonic

  -- Load the full BertForMaskedLM straight from the HF checkpoint dir:
  -- fromPretrained reads config.json for the dims, builds the model at
  -- those dims (NoGrad → born requires_grad=0, so the forward is
  -- genuinely tape-free), and fills every param from model.safetensors.
  Right (cfg ** model) <- fromPretrained {ex=ExampleExecutor} {dt=ExampleDType} {g=NoGrad} modelDir
    | Left err => do
        putStrLn ("ERR: fromPretrained " ++ modelDir ++ ": " ++ show err)
        exitFailure
  stageStamp "fromPretrained ok" t0

  -- Recover the per-head split from the config. The forward needs a
  -- proof that hidden = numHeads * headDim; for a config read at runtime
  -- that's a divisibility check, not a compile-time literal — decEq
  -- supplies the proof, or we bail if hidden isn't divisible by heads.
  let nHeads = numHeads cfg
      hDim   = hidden cfg `div` nHeads
  case decEq (hidden cfg) (nHeads * hDim) of
    No _ => do
      putStrLn ("ERR: hidden " ++ show (hidden cfg)
                 ++ " not divisible by num heads " ++ show nHeads)
      exitFailure
    Yes prf =>
      if dumpPooled
        then runPooledDump cfg nHeads hDim prf model
        else do
          -- Construct the BERT tokenizer once; each demo reuses it for both
          -- the input-string encode AND the top-5 token-id decode. The
          -- subprocess startup cost (~1s) amortises across the three demos.
          tokR <- mkTokenizer "google/bert_uncased_L-2_H-128_A-2" (vocabSize cfg)
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
              runMaskDemo cfg nHeads hDim prf tok model "paris is the capital of [MASK] ."
              runMaskDemo cfg nHeads hDim prf tok model "i went to the [MASK] to buy bread ."
              runMaskDemo cfg nHeads hDim prf tok model "the man worked as a [MASK] ."
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
