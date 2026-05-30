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
||| Three modes (all share the cache-aware decode path —
||| `genLoopCached`; the no-cache `genLoop` was dropped 2026-06-04
||| since it was correctness-equivalent and doubled Chez
||| elaboration cost):
|||
|||   --dump-final-hidden     CI gate. Fixed prompt [a few tokens].
|||                           Forward once. Print the last-position
|||                           hidden state to stdout, one float per
|||                           line. Comparator in
|||                           scripts/compare_inference.py.
|||
|||   --dump-tokens           CI gate (multi-step). Tokenize the
|||                           prompt (matches the default mode's path),
|||                           greedy-decode --num-tokens, print each
|||                           token id (Nat) one per line. Oracle is
|||                           `save_oracle_llama_generate.py` running
|||                           HF's `model.generate(do_sample=False,
|||                           use_cache=True)` on the same prompt;
|||                           comparator asserts exact sequence
|||                           match. Catches generation-path drift
|||                           that the single-forward
|||                           --dump-final-hidden gate can't see.
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
import Example.Common.HfInferenceHelper
import HfLlama
import KVCache
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
-- Cache-aware greedy generation
----------------------------------------------------------------------

||| Single decode step with KV cache. Takes the current per-layer
||| caches + the tokens to feed (the prompt on the seed call, then
||| [prevGenerated] each subsequent step), returns updated caches +
||| next-token id (or Nothing on out-of-range argmax).
genStepCached :
     LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                     ExampleDevice ExampleDType WithGrad
  -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
  -> Vect NumLayers (KVCache KvOut ExampleDevice ExampleDType)
  -> List (Fin VocabSize)
  -> IO (Vect NumLayers (KVCache KvOut ExampleDevice ExampleDType),
         Maybe (Fin VocabSize))
genStepCached model tables caches toksList = do
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      let inputIds = mkIds idDoubles
      (caches', logits) <- hfLlamaForwardLmStep
                                 {d=ExampleDevice} {dt=ExampleDType}
                                 {seq          = curLen}
                                 {vocab        = VocabSize}
                                 {hidden       = Hidden}
                                 {numLayers    = NumLayers}
                                 {numHeads     = NumHeads}
                                 {numKvHeads   = NumKvHeads}
                                 {headDim      = HeadDim}
                                 {intermediate = Intermediate}
                                 {maxPos       = MaxPos}
                                 RmsNormEps model tables caches inputIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow VocabSize lastRow.tensorPtr
      pure (caches', natToFin nextN VocabSize)


||| Cache-aware greedy decode — the canonical (and only) generation
||| path. Seed step feeds the full prompt into empty caches; each
||| subsequent step feeds only the previously-generated token, with
||| the per-layer caches threading through. The caches store
||| post-RoPE K and pre-RoPE V from earlier positions, so each new
||| step computes only the new token's K/V (constant per step)
||| instead of re-projecting the full growing prefix every time.
|||
||| Returns the full token list (prompt + generated). Caller is
||| responsible for any tokenizer-side decoding.
|||
||| Convention: this example has a single decode path. The legacy
||| no-cache `genLoop` / `genOneStep` were dropped 2026-06-04 — they
||| were correctness-equivalent (verified GREEN at Phase A baseline,
||| commit `b5443135`) but doubled Chez elaboration cost on this
||| file (the example's compile peak hit ~23 GB on a 16 GB VM with
||| both paths in scope; dropping the no-cache branch is the lever).
||| HfBert / HfGpt2 / HfBitNet follow the same single-decode-path
||| convention. Recover via `git show 70f5017c:...HfLlamaInference.idr`
||| if differential debugging is needed.
genLoopCached :
     LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                     ExampleDevice ExampleDType WithGrad
  -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
  -> List (Fin VocabSize)   -- prompt
  -> (remaining : Nat)
  -> IO (List (Fin VocabSize))
genLoopCached model tables prompt remaining =
  let initialCaches : Vect NumLayers (KVCache KvOut ExampleDevice ExampleDType)
      initialCaches = emptyKVCaches {numLayers=NumLayers} {kvOut=KvOut}
  in go initialCaches prompt prompt remaining
  where
    go : Vect NumLayers (KVCache KvOut ExampleDevice ExampleDType)
      -> List (Fin VocabSize)   -- accumulator (prompt + generated so far)
      -> List (Fin VocabSize)   -- new tokens to feed this step (prompt on seed, [prev] on steady)
      -> Nat                     -- remaining tokens to generate
      -> IO (List (Fin VocabSize))
    go _      acc _    Z         = pure acc
    go caches acc feed (S k)     = do
      perfReset {d=ExampleDevice}
      (caches', mNext) <- genStepCached model tables caches feed
      ops <- perfOpCount {d=ExampleDevice}
      putStrLn ("[perf] step " ++ show (length acc) ++ ": " ++ show ops ++ " ops")
      case mNext of
        Nothing => do
          putStrLn "  (argmax produced out-of-range token; stopping)"
          pure acc
        Just next => go caches' (acc ++ [next]) [next] k


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
  finalList <- genLoopCached model tables promptList numTokens
  Right text <- detokenize tok (fromList finalList)
    | Left err => putStrLn ("ERR: detokenize: " ++ show err)
  putStrLn ("Output:    " ++ text)


----------------------------------------------------------------------
-- --dump-tokens mode (multi-step generation CI gate)
----------------------------------------------------------------------

||| Greedy decode the same prompt the user-facing demo uses, then
||| dump every output token id (prompt + generated) one Nat per line
||| to stdout. The Python oracle (`save_oracle_llama_generate.py`)
||| runs HF `model.generate(do_sample=False, use_cache=True)` on the
||| same prompt and saves the resulting full id sequence under
||| key `"token_ids"`; the comparator
||| (`compare_inference.py --token-sequence`) asserts exact match.
|||
||| Banner / status text is suppressed (the comparator parses one
||| integer per line + filters `[stage]` / `[perf]` diagnostic
||| lines). `genLoop`'s `[perf] step N: K ops` lines are filtered;
||| no other text is emitted on success. On argmax-out-of-range
||| (unreachable in practice) `genLoop` emits a human-facing diag
||| line that would FAIL the comparator — that's the correct
||| failure mode.
runDumpTokens : Tokenizer VocabSize
             -> LlamaModelState VocabSize Hidden NumLayers QOut KvOut Intermediate
                                ExampleDevice ExampleDType WithGrad
             -> RoPETables MaxPos HeadDim ExampleDevice ExampleDType WithGrad
             -> (prompt : String) -> (numTokens : Nat) -> IO ()
runDumpTokens tok model tables prompt numTokens = do
  Right (_ ** promptIds) <- tokenize tok prompt
    | Left err => do
        putStrLn ("ERR: tokenize: " ++ show err)
        exitFailure
  let promptList = toList promptIds
  finalList <- genLoopCached model tables promptList numTokens
  traverse_ (putStrLn . show . finToNat) finalList


----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args
  let dumpTokens = elem "--dump-tokens" args
  t0 <- clockTime Monotonic

  -- Probe the tokenizer up-front (~1s subprocess call) BEFORE the
  -- expensive model construction + 2.5 GB param load. If the
  -- tokenizer files aren't downloaded yet, fail in seconds instead
  -- of after 30+ seconds of model setup. The dump-hidden mode
  -- doesn't strictly need a tokenizer but probing in both branches
  -- keeps the failure semantics uniform. dump-tokens mode DOES need
  -- it (tokenization is part of the gate).
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
      Right tok =>
        if dumpTokens
          then do
            putStrLn "[stage] runDumpTokens — greedy decode + dump token ids..."
            runDumpTokens tok model tables
                          (extractPrompt "The capital of France is" args)
                          (extractNumTokens 4 args)
            stageStamp "runDumpTokens done" t0
            _ <- drainManagedHandles
            forceMajorGc
            _ <- drainManagedHandles
            stageStamp "drain + GC done" t0
            releaseAllPersistent {d=ExampleDevice}
            stageStamp "releaseAllPersistent done" t0
            pure ()
          else do
            putStrLn "[stage] runGenerate — greedy decode loop..."
            let numTokens = extractNumTokens 8 args
            benchT0 <- clockTime Monotonic
            runGenerate tok model tables (extractPrompt "The capital of France is" args) numTokens
            benchT1 <- clockTime Monotonic
            stageStamp "runGenerate done" t0
            let benchMs =
                  let s  = cast {to=Double} (seconds benchT1 - seconds benchT0)
                      ns = cast {to=Double} (nanoseconds benchT1 - nanoseconds benchT0)
                  in s * 1000.0 + ns / 1000000.0
            putStrLn ""
            putStrLn ("PERF_GENERATE_TOKENS=" ++ show numTokens)
            putStrLn ("PERF_GENERATE_WALL_MS=" ++ show benchMs)
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
