||| HfGpt2Inference — load `distilgpt2` and exercise the typed GPT-2
||| forward + greedy generation against the HF Python oracle.
|||
||| `distilgpt2` (~350 MB safetensors, pretrained, 6 layers / hidden 768
||| / 12 heads / head_dim 64 / FFN 3072 / max_pos 1024 / vocab 50257)
||| is real GPT-2 — distilled from gpt2 by HF. Same architecture as
||| gpt2 / gpt2-medium / gpt2-large / gpt2-xl; same on-disk naming
||| conventions, so the HfGpt2 module covers the whole family.
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
|||   - `packages/idris-transformers/models/distilgpt2/model.safetensors`
|||     — fetch with `bash packages/idris-transformers/scripts/hf-download.sh distilgpt2`
|||   - Python `transformers` available via the pytorch venv (for the
|||     Tokenizer subprocess).
module Example.HfGpt2Inference

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
import Executor
import Example.Common.HfInferenceHelper
import HfGpt2
import Tensor
import Util
import Tokenizer


----------------------------------------------------------------------
-- Config (distilgpt2 dims, pinned at the type level)
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 50257

Hidden : Nat
Hidden = 768

NumLayers : Nat
NumLayers = 6

NumHeads : Nat
NumHeads = 12

HeadDim : Nat
HeadDim = 64

Intermediate : Nat
Intermediate = 3072

MaxPos : Nat
MaxPos = 1024

ModelRepo : String
ModelRepo = "distilgpt2"

modelDir : String
modelDir = "models/" ++ ModelRepo

hfWeightsPath : String
hfWeightsPath = modelDir ++ "/model.safetensors"


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

runDumpHidden : Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                               ExampleExecutor ExampleDType WithGrad
             -> IO ()
runDumpHidden model = do
  -- BPE("Hello world") = [15496, 995]. Same as save_oracle_gpt2.py.
  let inputIds = mkIds (the (Vect 2 Double) [15496.0, 995.0])
      posIds   = mkIds (arangeVect 2)
  out <- hfGpt2Forward {ex=ExampleExecutor} {dt=ExampleDType}
                       {seqLen       = 2}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       model inputIds posIds
  lastRow <- trowSelect out 1
  printRow (cast {to=Int} Hidden) 0 lastRow.tensorPtr


----------------------------------------------------------------------
-- Greedy generation (argmaxRow / toExistVect live in HfInferenceHelper)
----------------------------------------------------------------------

-- One generation step: forward the current sequence, pick argmax of
-- the last position's LM-head logits, return the next token ID as a
-- Fin VocabSize. The accumulator is `List (Fin VocabSize)` rather
-- than `Vect n` to dodge the `(n + 1) ~ S n` rewrite tax inside the
-- generation loop — every step would otherwise need a
-- `plusCommutative`-style proof. We convert to `Vect` via fromList
-- once at each genOneStep call (the length is whatever the list is).
genOneStep : Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                            ExampleExecutor ExampleDType WithGrad
          -> List (Fin VocabSize)
          -> IO (Maybe (Fin VocabSize))
genOneStep model toksList = do
  -- Convert via dependent pair so the runtime length is bound to a
  -- fresh type variable `curLen`. Idris doesn't reduce
  -- `length idsList` against a let-bound name automatically, so the
  -- DPair pattern is the path that actually unifies.
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      let inputIds = mkIds idDoubles
          posIds   = mkIds (arangeVect curLen)
      logits <- hfGpt2ForwardLm {ex=ExampleExecutor} {dt=ExampleDType}
                                {seqLen       = curLen}
                                {vocab        = VocabSize}
                                {hidden       = Hidden}
                                {numLayers    = NumLayers}
                                {numHeads     = NumHeads}
                                {headDim      = HeadDim}
                                {intermediate = Intermediate}
                                {maxPos       = MaxPos}
                                model inputIds posIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow VocabSize lastRow.tensorPtr
      pure (natToFin nextN VocabSize)

-- Generate `remaining` more tokens, snoc'ing each onto the input.
-- Returns the full token list (prompt + generated tokens).
genLoop : Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                         ExampleExecutor ExampleDType WithGrad
       -> List (Fin VocabSize)
       -> (remaining : Nat)
       -> IO (List (Fin VocabSize))
genLoop _     tokens Z     = pure tokens
genLoop model tokens (S k) = do
  mNext <- genOneStep model tokens
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure tokens
    Just next =>
      genLoop model (tokens ++ [next]) k


----------------------------------------------------------------------
-- --prompt argv parsing (helpers in HfInferenceHelper)
----------------------------------------------------------------------


----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : Tokenizer VocabSize
           -> Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                             ExampleExecutor ExampleDType WithGrad
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate tok model prompt numTokens = do
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
  finalList <- genLoop model promptList numTokens
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
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args
  t0 <- clockTime Monotonic

  -- Build a distilgpt2 model. Each param registers under the literal
  -- HF name (`transformer.wte.weight`, etc.).
  model <- hfGpt2Model {ex=ExampleExecutor} {dt=ExampleDType}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       ""
  stageStamp "hfGpt2Model ok" t0
  -- Load the HF checkpoint. loadModelAllowCast handles dtype
  -- widening at the loader; distilgpt2 is F32 on disk so the cost is
  -- a copy on F32 backends (mlx-gpu / torch-mps) and a widen on F64
  -- backends (tape).
  ok <- loadModelAllowCast {ex=ExampleExecutor} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()
  stageStamp "loadModelAllowCast ok" t0

  if dumpHidden
    then runDumpHidden model
    else do
      tokR <- mkTokenizer ModelRepo VocabSize
      case tokR of
        Left err  => do
          putStrLn ("ERR: mkTokenizer: " ++ show err)
          exitFailure
        Right tok => do
          let numTokens = extractNumTokens 8 args
          benchT0 <- clockTime Monotonic
          runGenerate tok model (extractPrompt "The quick brown fox" args) numTokens
          benchT1 <- clockTime Monotonic
          let benchMs =
                let s  = cast {to=Double} (seconds benchT1 - seconds benchT0)
                    ns = cast {to=Double} (nanoseconds benchT1 - nanoseconds benchT0)
                in s * 1000.0 + ns / 1000000.0
          putStrLn ""
          putStrLn ("PERF_GENERATE_TOKENS=" ++ show numTokens)
          putStrLn ("PERF_GENERATE_WALL_MS=" ++ show benchMs)
