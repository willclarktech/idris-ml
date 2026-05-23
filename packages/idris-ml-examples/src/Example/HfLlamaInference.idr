||| HfLlamaInference — load `meta-llama/Llama-3.2-1B` (base) and run
||| Llama through the typed-tensor / type-safe-dependent-shape stack.
|||
||| Llama 3.2 1B: vocab=128256, hidden=2048, n_layer=16, n_head=32,
||| n_kv_heads=8 (GQA 4:1), head_dim=64, intermediate=8192,
||| max_position=131072 (with NTK scaling factor=32 from the original
||| 8192 training context), rope_base=500000, rms_norm_eps=1e-5,
||| tie_word_embeddings=true. On-disk ~2.5 GB BF16; loadModelAllowCast
||| widens to F32 (mlx-gpu / torch-mps) or F64 (tape — but won't fit
||| in 16 GB).
|||
||| Two modes:
|||
|||   --dump-final-hidden     CI gate. Fixed prompt [a few tokens].
|||                           Forward once. Print the last-position
|||                           hidden state to stdout, one float per
|||                           line. Comparator in
|||                           scripts/compare_inference.py.
|||
|||   (default)               User-facing demo. Reads --prompt (default
|||                           "The capital of France is") and
|||                           --num-tokens (default 8). Tokenize via
|||                           the Phase-2 Tokenizer subprocess,
|||                           greedy-decode N tokens, detokenize,
|||                           print.
|||
||| Pre-requisites (CI handles these via the make targets):
|||   - HF_TOKEN with Llama 3.2 license accepted on huggingface.co
|||   - `packages/idris-transformers/models/meta-llama/Llama-3.2-1B/model.safetensors`
|||     — fetch with
|||         HF_TOKEN=hf_... bash packages/idris-transformers/scripts/hf-download.sh meta-llama/Llama-3.2-1B
|||   - Python `transformers` available via the pytorch venv (for the
|||     Tokenizer subprocess).
|||
||| Note: no KV cache in v1. Each generated token reruns forward on
||| the full growing sequence. ~16 generated tokens with hidden=2048 +
||| 16 layers is multiple seconds on tape and tens of seconds on
||| torch-mps; first-token latency dominated by the 76 param-load
||| operations. KV cache is filed as a follow-up.
module Example.HfLlamaInference

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
import Device
import HfLlama
import Layer.RoPE
import Tensor
import Tokenizer
import Util


----------------------------------------------------------------------
-- Llama 3.2 1B config, pinned at the type level
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 128256

Hidden : Nat
Hidden = 2048

NumLayers : Nat
NumLayers = 16

NumHeads : Nat
NumHeads = 32

NumKvHeads : Nat
NumKvHeads = 8

HeadDim : Nat
HeadDim = 64

QOut : Nat
QOut = NumHeads * HeadDim   -- = 2048

KvOut : Nat
KvOut = NumKvHeads * HeadDim  -- = 512

Intermediate : Nat
Intermediate = 8192

-- Clamped to the original training context (8192). The model's full
-- 131072 max with NTK scaling is supported by the table builder, but
-- the cos/sin tables at 131072 × 32 are 32 MB each; clamping keeps the
-- demo modest. Llama's positional behaviour at <8k is identical
-- regardless of `maxPos` chosen for the tables (positions m get the
-- same cos/sin values).
MaxPos : Nat
MaxPos = 8192

RopeBase : Double
RopeBase = 500000.0

RmsNormEps : Double
RmsNormEps = 1.0e-5

ModelRepo : String
ModelRepo = "meta-llama/Llama-3.2-1B"

modelDir : String
modelDir = "models/" ++ ModelRepo

hfWeightsPath : String
hfWeightsPath = modelDir ++ "/model.safetensors"


----------------------------------------------------------------------
-- Input-ID + position tensor helpers (mirror HfGpt2Inference)
----------------------------------------------------------------------

mkIds : {n : Nat} -> Vect n Double
     -> Tensor [n] ExampleDevice ExampleDType WithGrad
mkIds xs =
  let raw = bulkToTensor {d=ExampleDevice} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw


toExistVect : (xs : List a) -> (n : Nat ** Vect n a)
toExistVect xs = (length xs ** fromList xs)


----------------------------------------------------------------------
-- Dump a [hidden]-shape tensor row to stdout, one float per line
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
-- Greedy generation
----------------------------------------------------------------------

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


genOneStep : LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                             ExampleDevice ExampleDType WithGrad
          -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
          -> List (Fin VocabSize)
          -> IO (Maybe (Fin VocabSize))
genOneStep model tables toksList = do
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      let inputIds = mkIds idDoubles
      logits <- hfLlamaForwardLm {d=ExampleDevice} {dt=ExampleDType}
                                 {seq          = curLen}
                                 {vocab        = VocabSize}
                                 {hidden       = Hidden}
                                 {numLayers    = NumLayers}
                                 {numHeads     = NumHeads}
                                 {numKvHeads   = NumKvHeads}
                                 {headDim      = HeadDim}
                                 {intermediate = Intermediate}
                                 {maxPos       = MaxPos}
                                 RmsNormEps model tables inputIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow VocabSize lastRow.tensorPtr
      pure (natToFin nextN VocabSize)


genLoop : LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                          ExampleDevice ExampleDType WithGrad
       -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
       -> List (Fin VocabSize)
       -> (remaining : Nat)
       -> IO (List (Fin VocabSize))
genLoop _     _      tokens Z     = pure tokens
genLoop model tables tokens (S k) = do
  mNext <- genOneStep model tables tokens
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure tokens
    Just next => do
      -- Drop the previous forward's arena/autograd-tape entries before
      -- the next forward. On tape this prevents ~GB-per-forward arena
      -- accumulation that OOMs the VM on a no-KV-cache Llama decode.
      -- Params survive (they're persistent, re-registered on the fresh
      -- tape). Mild beneficial on torch + mlx; no-op-equivalent there.
      resetForEval {d=ExampleDevice}
      genLoop model tables (tokens ++ [next]) k


----------------------------------------------------------------------
-- --dump-final-hidden mode
----------------------------------------------------------------------

runDumpHidden : LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                                ExampleDevice ExampleDType WithGrad
             -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
             -> IO ()
runDumpHidden model tables = do
  -- Fixed input: BPE("Hello") = [9906] in Llama 3 vocab (verified by
  -- save_oracle_llama.py if/when added). Single token to keep the
  -- compute cheap on the first run.
  let inputIds = mkIds (the (Vect 1 Double) [9906.0])
  out <- hfLlamaForward {d=ExampleDevice} {dt=ExampleDType}
                        {seq          = 1}
                        {vocab        = VocabSize}
                        {hidden       = Hidden}
                        {numLayers    = NumLayers}
                        {numHeads     = NumHeads}
                        {numKvHeads   = NumKvHeads}
                        {headDim      = HeadDim}
                        {intermediate = Intermediate}
                        {maxPos       = MaxPos}
                        RmsNormEps model tables inputIds
  lastRow <- trowSelect out 0
  printRow (cast {to=Int} Hidden) 0 lastRow.tensorPtr


----------------------------------------------------------------------
-- --prompt + --num-tokens argv parsing
----------------------------------------------------------------------

extractPrompt : List String -> String
extractPrompt args = go args
  where
    go : List String -> String
    go ("--prompt" :: p :: _) = p
    go (_ :: rest)            = go rest
    go []                     = "The capital of France is"

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
           -> LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                              ExampleDevice ExampleDType WithGrad
           -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate tok model tables prompt numTokens = do
  Right (promptLen ** promptIds) <- tokenize tok prompt
    | Left err => putStrLn ("ERR: tokenize: " ++ show err)
  putStrLn ""
  putStrLn "Llama 3.2 1B greedy generation"
  putStrLn "=============================="
  putStrLn ""
  putStrLn ("Prompt:    " ++ prompt)
  putStrLn ("Tokens in: " ++ show promptLen ++ ", generating: " ++ show numTokens)
  putStrLn ""
  let promptList = toList promptIds
  finalList <- genLoop model tables promptList numTokens
  Right text <- detokenize tok (fromList finalList)
    | Left err => putStrLn ("ERR: detokenize: " ++ show err)
  putStrLn ("Output:    " ++ text)


----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

-- Stage timer for `main`. Llama setup goes through ~4 distinct stages
-- (tokenizer probe / model construction / checkpoint load / RoPE table
-- build) plus the forward / generation. Each can individually take
-- minutes at 1.24B params, so when a run looks hung from outside the
-- user needs to know WHICH stage to investigate. `formatElapsed` from
-- `Util` returns the cumulative `[hh:mm:ss]` since `t0`.
stageStamp : (label : String) -> Clock Monotonic -> IO ()
stageStamp label t0 = do
  now <- clockTime Monotonic
  putStrLn ("[stage] " ++ formatElapsed t0 now ++ " " ++ label)

main : IO ()
main = do
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args
  t0 <- clockTime Monotonic

  -- Probe the tokenizer up-front (~1s subprocess call) BEFORE the
  -- expensive model construction + 2.5 GB param load. If the
  -- tokenizer files aren't downloaded yet, fail in seconds instead
  -- of after 30+ seconds of model setup. The dump-hidden mode
  -- doesn't strictly need a tokenizer but probing in both branches
  -- keeps the failure semantics uniform.
  tokR <- mkTokenizer ModelRepo VocabSize
  case tokR of
    Left err =>
      if not dumpHidden
        then do
          putStrLn ("ERR: mkTokenizer failed: " ++ show err)
          putStrLn ("     Likely missing tokenizer files at models/" ++ ModelRepo ++ "/")
          putStrLn ("     Run: bash packages/idris-transformers/scripts/hf-download.sh " ++ ModelRepo)
          exitFailure
        else
          putStrLn ("WARN: mkTokenizer failed (continuing — dump-hidden doesn't need it): "
                    ++ show err)
    Right _ => pure ()
  stageStamp "tokenizer probe ok" t0

  -- Build the full Llama 3.2 1B state — 146 params, ~1.2B values at
  -- F64 = ~10 GB allocation. F32 backends (mlx-gpu / torch-mps) cut
  -- that to ~5 GB; that's the practical config for this VM. Tape
  -- (F64-only) doesn't fit in 16 GB; the example skips that lane.
  putStrLn "[stage] hfLlamaModel — constructing 146-param state (~5 GB at F32 / 10 GB at F64)..."
  model <- hfLlamaModel {d=ExampleDevice} {dt=ExampleDType}
                        {vocab        = VocabSize}
                        {hidden       = Hidden}
                        {numLayers    = NumLayers}
                        {qOut         = QOut}
                        {kvOut        = KvOut}
                        {intermediate = Intermediate}
                        "model"
  stageStamp "hfLlamaModel ok" t0

  -- Load the gated HF checkpoint. Requires HF_TOKEN with Llama 3.2
  -- license accepted on huggingface.co. ~2.5 GB BF16 on disk; the
  -- cast-on-load widens to F32 / F64 depending on backend.
  putStrLn ("[stage] loadModelAllowCast — reading " ++ hfWeightsPath ++ " (~2.5 GB BF16, casting to "
            ++ "F32/F64 host-side)...")
  ok <- loadModelAllowCast {d=ExampleDevice} hfWeightsPath
  if not ok
    then do
      putStrLn ("ERR: loadModelAllowCast failed for " ++ hfWeightsPath)
      exitFailure
    else pure ()
  stageStamp "loadModelAllowCast ok" t0

  -- Build RoPE tables once (reused across all forward passes /
  -- decode steps).
  putStrLn "[stage] buildLlamaRoPETables — precomputing cos/sin tables..."
  tables <- buildLlamaRoPETables {d=ExampleDevice} {dt=ExampleDType}
                                  {maxPos  = MaxPos}
                                  {headDim = HeadDim}
                                  RopeBase llama3Scaling
  stageStamp "buildLlamaRoPETables ok" t0

  if dumpHidden
    then do
      putStrLn "[stage] forward (dump-hidden mode) — single forward pass..."
      runDumpHidden model tables
      stageStamp "dump-hidden done" t0
      pure ()
    else case tokR of
      -- Unreachable in practice: dumpHidden=False + Left would have
      -- exitFailure'd at the top-of-main probe. Idris's type checker
      -- doesn't know that, so we handle Left defensively.
      Left err  => do
        putStrLn ("ERR: mkTokenizer (post-probe inconsistency): " ++ show err)
        exitFailure
      Right tok => do
        putStrLn "[stage] runGenerate — greedy decode loop..."
        runGenerate tok model tables (extractPrompt args) (extractNumTokens args)
        stageStamp "runGenerate done" t0
        -- Explicit pre-exit cleanup. Forces the backend's per-tensor
        -- destructor cascade (libtorch CPUAllocator releases on torch-
        -- cpu, mlx::array refcount drops on mlx-cpu) to run inside main
        -- where the cost is timed + bounded, rather than during the
        -- post-main C/OS teardown (where it took 20+ min on torch-cpu
        -- BF16 Llama). See TODO #394.
        _ <- drainManagedHandles
        forceMajorGc
        _ <- drainManagedHandles
        stageStamp "drain + GC done" t0
        releaseAllPersistent {d=ExampleDevice}
        stageStamp "releaseAllPersistent done" t0
        pure ()
