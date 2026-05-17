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
import System.File

import Array
import BuildConfig
import Checkpoint
import Device
import HfGpt2
import Tensor
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
modelDir = "packages/idris-transformers/models/" ++ ModelRepo

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
-- --dump-final-hidden mode: forward [15496, 995] once, dump last hidden
----------------------------------------------------------------------

runDumpHidden : Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                               ExampleDevice ExampleDType WithGrad
             -> IO ()
runDumpHidden model = do
  -- BPE("Hello world") = [15496, 995]. Same as save_oracle_gpt2.py.
  let inputIds = mkIds (the (Vect 2 Double) [15496.0, 995.0])
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
  lastRow <- trowSelect out 1
  printRow (cast {to=Int} Hidden) 0 lastRow.tensorPtr


----------------------------------------------------------------------
-- Greedy generation
----------------------------------------------------------------------

-- argmax over a [vocab]-shape row. Reads all values via primItem1d
-- (50257 FFI calls per token ≈ 50 ms) and picks the largest. Trades
-- a small constant cost vs primArgsort + index-0 read for code
-- simplicity; the forward pass dominates regardless.
argmaxRow : (vocab : Nat) -> AnyPtr -> IO Nat
argmaxRow vocab p = go (cast {to=Int} vocab) 0 0 (-1.0e300)
  where
    go : Int -> Int -> Int -> Double -> IO Nat
    go end i bestI bestV =
      if i >= end
        then pure (cast {to=Nat} bestI)
        else let v = primItem1d {d=ExampleDevice} p i
             in if v > bestV
                  then go end (i + 1) i v
                  else go end (i + 1) bestI bestV

-- Helper to convert a plain List to a dependent-pair Vect. The
-- explicit signature dodges the elaboration ambiguity Idris hits when
-- the DPair literal is written inline at a use site.
toExistVect : (xs : List a) -> (n : Nat ** Vect n a)
toExistVect xs = (length xs ** fromList xs)

-- One generation step: forward the current sequence, pick argmax of
-- the last position's LM-head logits, return the next token ID as a
-- Fin VocabSize. The accumulator is `List (Fin VocabSize)` rather
-- than `Vect n` to dodge the `(n + 1) ~ S n` rewrite tax inside the
-- generation loop — every step would otherwise need a
-- `plusCommutative`-style proof. We convert to `Vect` via fromList
-- once at each genOneStep call (the length is whatever the list is).
genOneStep : Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                            ExampleDevice ExampleDType WithGrad
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
      logits <- hfGpt2ForwardLm {d=ExampleDevice} {dt=ExampleDType}
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
                         ExampleDevice ExampleDType WithGrad
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
-- --prompt argv parsing
----------------------------------------------------------------------

-- Pull the value after `--prompt` from the argv list, or fall back to
-- the demo default.
extractPrompt : List String -> String
extractPrompt args = go args
  where
    go : List String -> String
    go ("--prompt" :: p :: _) = p
    go (_ :: rest)            = go rest
    go []                     = "The quick brown fox"

-- How many tokens to generate. Default is small to keep the demo
-- under a minute on tape; user can override with --num-tokens.
extractNumTokens : List String -> Nat
extractNumTokens args = go args
  where
    go : List String -> Nat
    go ("--num-tokens" :: n :: _) =
      fromMaybe 8 (parsePositive {a=Nat} n)
    go (_ :: rest)                = go rest
    go []                         = 8


----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : Tokenizer VocabSize
           -> Gpt2ModelState VocabSize Hidden NumLayers Intermediate MaxPos
                             ExampleDevice ExampleDType WithGrad
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
-- main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args

  -- Build a distilgpt2 model. Each param registers under the literal
  -- HF name (`transformer.wte.weight`, etc.).
  model <- hfGpt2Model {d=ExampleDevice} {dt=ExampleDType}
                       {vocab        = VocabSize}
                       {hidden       = Hidden}
                       {numLayers    = NumLayers}
                       {numHeads     = NumHeads}
                       {headDim      = HeadDim}
                       {intermediate = Intermediate}
                       {maxPos       = MaxPos}
                       ""
  -- Load the HF checkpoint. loadModelAllowCast handles dtype
  -- widening at the loader; distilgpt2 is F32 on disk so the cost is
  -- a copy on F32 backends (mlx-gpu / torch-mps) and a widen on F64
  -- backends (tape).
  ok <- loadModelAllowCast {d=ExampleDevice} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()

  if dumpHidden
    then runDumpHidden model
    else do
      tokR <- mkTokenizer ModelRepo VocabSize
      case tokR of
        Left err  => do
          putStrLn ("ERR: mkTokenizer: " ++ show err)
          exitFailure
        Right tok =>
          runGenerate tok model (extractPrompt args) (extractNumTokens args)
