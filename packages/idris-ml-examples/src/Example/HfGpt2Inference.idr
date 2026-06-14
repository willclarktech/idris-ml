||| HfGpt2Inference — load `distilgpt2` and exercise the typed GPT-2
||| forward + greedy generation against the HF Python oracle.
|||
||| `distilgpt2` (~350 MB safetensors, pretrained, 6 layers / hidden 768
||| / 12 heads / head_dim 64 / FFN 3072 / max_pos 1024 / vocab 50257)
||| is real GPT-2 — distilled from gpt2 by HF. Same architecture as
||| gpt2 / gpt2-medium / gpt2-large / gpt2-xl; same on-disk naming
||| conventions, so the Transformers.Gpt2 module covers the whole family.
|||
||| The model is loaded with `Transformers.Gpt2.fromPretrained`: it reads
||| `<dir>/config.json` for the dims (GPT-2 keys: `n_embd` / `n_head` /
||| `n_layer` / `n_positions`; `head_dim = n_embd / n_head`), builds a
||| tape-free `NoGrad` model, and fills params from `model.safetensors`.
||| Nothing about distilgpt2 is hardcoded — the `(cfg ** model)` ties the
||| model's type to the file, and a `decEq` recovers the head split.
|||
||| Two modes — same shape as HfBert:
|||
|||   --dump-final-hidden    CI gate. Fixed input [15496, 995]
|||                          ("Hello world"). Forward once. Print the
|||                          last-position hidden state (post `ln_f`)
|||                          to stdout, one float per line. Diffed
|||                          against `save_oracle_gpt2.py` by
|||                          `scripts/compare_inference.py`.
|||
|||   (default)              User-facing demo. Read a string prompt
|||                          from argv (`--prompt "..."`) or use the
|||                          default ("The quick brown fox"). Tokenize
|||                          via the HF subprocess, greedy-decode N
|||                          tokens, detokenize, print.
|||
||| The user-facing demo runs ~8 forward passes on growing sequences;
||| at distilgpt2's scale (~82M params, hidden=768) each forward is
||| seconds on tape, so the demo takes 30-60s end-to-end. KV-cache
||| optimisation would drop this dramatically (filed for Phase 4 along
||| with HfLlama).
|||
||| Pre-requisites (CI handles these via the make targets):
|||   - `models/distilgpt2/{config.json,model.safetensors}`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh distilgpt2`
|||   - Python `transformers` available via the pytorch venv (for the
|||     Tokenizer subprocess).
module Example.HfGpt2Inference

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import Decidable.Equality
import System
import System.Clock
import System.File

import Array
import BuildConfig
import Checkpoint
import Example.Common.HfInferenceHelper
import Executor
import Tensor
import Transformers.Gpt2
import Transformers.Tokenizer
import Util

----------------------------------------------------------------------
-- Model location (dims come from the file, not from here)
----------------------------------------------------------------------

ModelRepo : String
ModelRepo = "distilgpt2"

modelDir : String
modelDir = "models/" ++ ModelRepo

----------------------------------------------------------------------
-- Build small input-ID + position tensors (mkIds lives in HfInferenceHelper)
----------------------------------------------------------------------

arangeVect : (n : Nat) -> Vect n Double
arangeVect n = go n 0.0
  where
    go : (k : Nat) -> Double -> Vect k Double
    go Z     _ = []
    go (S k) v = v :: go k (v + 1.0)

----------------------------------------------------------------------
-- --dump-final-hidden mode: forward [15496, 995] once, dump last hidden
----------------------------------------------------------------------

runDumpHidden : (cfg : Gpt2Config)
             -> (nHeads, hDim : Nat)
             -> (prf : hidden cfg = nHeads * hDim)
             -> Gpt2Model cfg ExampleExecutor ExampleDType NoGrad
             -> IO ()
runDumpHidden cfg nHeads hDim prf model = do
  -- BPE("Hello world") = [15496, 995]. Same as save_oracle_gpt2.py.
  let inputIds = retypeGrad (mkIds (the (Vect 2 Double) [15496.0, 995.0]))
      posIds   = retypeGrad (mkIds (arangeVect 2))
  out <- hfGpt2Forward {ex=ExampleExecutor} {dt=ExampleDType}
                       {seqLen       = 2}
                       {vocab        = vocabSize cfg}
                       {hidden       = hidden cfg}
                       {numLayers    = numLayers cfg}
                       {numHeads     = nHeads}
                       {headDim      = hDim}
                       {intermediate = intermediate cfg}
                       {maxPos       = maxPosition cfg}
                       {prf}
                       model inputIds posIds
  lastRow <- trowSelect out 1
  printRow (cast {to=Int} (hidden cfg)) 0 lastRow.tensorPtr

----------------------------------------------------------------------
-- Greedy generation (argmaxRow / toExistVect live in HfInferenceHelper)
----------------------------------------------------------------------

-- One generation step: forward the current sequence, pick argmax of
-- the last position's LM-head logits, return the next token ID as a
-- Fin (vocabSize cfg). The accumulator is `List (Fin (vocabSize cfg))`
-- rather than `Vect n` to dodge the `(n + 1) ~ S n` rewrite tax inside
-- the generation loop — every step would otherwise need a
-- `plusCommutative`-style proof. We convert to `Vect` via fromList
-- once at each genOneStep call (the length is whatever the list is).
genOneStep : (cfg : Gpt2Config)
          -> (nHeads, hDim : Nat)
          -> (prf : hidden cfg = nHeads * hDim)
          -> Gpt2Model cfg ExampleExecutor ExampleDType NoGrad
          -> List (Fin (vocabSize cfg))
          -> IO (Maybe (Fin (vocabSize cfg)))
genOneStep cfg nHeads hDim prf model toksList = do
  -- Convert via dependent pair so the runtime length is bound to a
  -- fresh type variable `curLen`. Idris doesn't reduce
  -- `length idsList` against a let-bound name automatically, so the
  -- DPair pattern is the path that actually unifies.
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      let inputIds = retypeGrad (mkIds idDoubles)
          posIds   = retypeGrad (mkIds (arangeVect curLen))
      logits <- hfGpt2ForwardLm {ex=ExampleExecutor} {dt=ExampleDType}
                                {seqLen       = curLen}
                                {vocab        = vocabSize cfg}
                                {hidden       = hidden cfg}
                                {numLayers    = numLayers cfg}
                                {numHeads     = nHeads}
                                {headDim      = hDim}
                                {intermediate = intermediate cfg}
                                {maxPos       = maxPosition cfg}
                                {prf}
                                model inputIds posIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow (vocabSize cfg) lastRow.tensorPtr
      pure (natToFin nextN (vocabSize cfg))

-- Generate `remaining` more tokens, snoc'ing each onto the input.
-- Returns the full token list (prompt + generated tokens).
genLoop : (cfg : Gpt2Config)
       -> (nHeads, hDim : Nat)
       -> (prf : hidden cfg = nHeads * hDim)
       -> Gpt2Model cfg ExampleExecutor ExampleDType NoGrad
       -> List (Fin (vocabSize cfg))
       -> (remaining : Nat)
       -> IO (List (Fin (vocabSize cfg)))
genLoop _   _      _    _   _     tokens Z     = pure tokens
genLoop cfg nHeads hDim prf model tokens (S k) = do
  mNext <- genOneStep cfg nHeads hDim prf model tokens
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure tokens
    Just next =>
      genLoop cfg nHeads hDim prf model (tokens ++ [next]) k

----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : (cfg : Gpt2Config)
           -> (nHeads, hDim : Nat)
           -> (prf : hidden cfg = nHeads * hDim)
           -> Tokenizer (vocabSize cfg)
           -> Gpt2Model cfg ExampleExecutor ExampleDType NoGrad
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate cfg nHeads hDim prf tok model prompt numTokens = do
  Right (promptLen ** promptIds) <- tokenize tok prompt
    | Left err => putStrLn ("ERR: tokenize: " ++ show err)
  putStrLn ""
  putStrLn "GPT-2 greedy generation — distilgpt2"
  putStrLn "===================================="
  putStrLn ""
  putStrLn ("Prompt:    " ++ prompt)
  putStrLn ("Tokens in: " ++ show promptLen ++ ", generating: " ++ show numTokens)
  putStrLn ""
  let promptList = toList promptIds
  finalList <- genLoop cfg nHeads hDim prf model promptList numTokens
  -- detokenize the FULL sequence so the printed text includes the
  -- prompt and the generation continuously. Convert List → Vect via
  -- fromList (length-existential is fine because detokenize accepts
  -- any n).
  Right text <- detokenize tok (fromList finalList)
    | Left err => putStrLn ("ERR: detokenize: " ++ show err)
  putStrLn ("Output:    " ++ text)

----------------------------------------------------------------------
-- main (stageStamp lives in HfInferenceHelper)
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args
  t0 <- clockTime Monotonic

  -- Load distilgpt2 straight from the HF checkpoint dir. fromPretrained
  -- reads config.json for the dims, builds a tape-free NoGrad model, and
  -- fills params from model.safetensors (HF-native `transformer.*` names).
  Right (cfg ** model) <- fromPretrained {ex=ExampleExecutor} {dt=ExampleDType} {g=NoGrad} modelDir
    | Left err => do
        putStrLn ("ERR: fromPretrained " ++ modelDir ++ ": " ++ show err)
        exitFailure
  stageStamp "fromPretrained ok" t0

  -- Recover the per-head split from the config (head_dim = n_embd /
  -- n_head). decEq supplies the `hidden = numHeads * headDim` proof the
  -- forward needs, or we bail if n_embd isn't divisible by n_head.
  let nHeads = numHeads cfg
      hDim   = headDim cfg
  case decEq (hidden cfg) (nHeads * hDim) of
    No _ => do
      putStrLn ("ERR: n_embd " ++ show (hidden cfg)
                 ++ " not divisible by n_head " ++ show nHeads)
      exitFailure
    Yes prf =>
      if dumpHidden
        then runDumpHidden cfg nHeads hDim prf model
        else do
          tokR <- mkTokenizer ModelRepo (vocabSize cfg)
          case tokR of
            Left err  => do
              putStrLn ("ERR: mkTokenizer: " ++ show err)
              exitFailure
            Right tok => do
              let numTokens = extractNumTokens 8 args
              benchT0 <- clockTime Monotonic
              runGenerate cfg nHeads hDim prf tok model
                          (extractPrompt "The quick brown fox" args) numTokens
              benchT1 <- clockTime Monotonic
              let benchMs =
                    let s  = cast {to=Double} (seconds benchT1 - seconds benchT0)
                        ns = cast {to=Double} (nanoseconds benchT1 - nanoseconds benchT0)
                    in s * 1000.0 + ns / 1000000.0
              putStrLn ""
              putStrLn ("PERF_GENERATE_TOKENS=" ++ show numTokens)
              putStrLn ("PERF_GENERATE_WALL_MS=" ++ show benchMs)
