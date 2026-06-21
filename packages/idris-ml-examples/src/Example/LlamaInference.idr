||| LlamaInference — load `unsloth/Llama-3.2-1B` (base; a public
||| mirror of `meta-llama/Llama-3.2-1B`'s weights, no `HF_TOKEN`) and run
||| Llama through the typed-tensor / type-safe-dependent-shape stack.
|||
||| The model is loaded with `Transformers.Llama.fromPretrained`: it reads
||| `<dir>/config.json` for the dims (GQA head counts, `rope_theta`,
||| `rms_norm_eps`), builds the model at the file's shapes, and fills
||| params from `model.safetensors`. The returned `(cfg ** model)` ties
||| the model's type to the file — nothing about Llama-3.2-1B is
||| hardcoded; `cfg.ropeBase` / `cfg.rmsNormEps` drive the RoPE tables +
||| RmsNorm eps below. (The model is built `NoGrad` — pure inference, so
||| no tape and no grad buffers; the RoPE tables are grad-mode-free
||| non-learnable constants — `RoPETables` carries no `g` index —
||| independent of the model's grad mode.)
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
|||                           match.
|||
|||   (default)               User-facing demo. Reads --prompt (default
|||                           "The capital of France is") and
|||                           --num-tokens (default 8). Tokenize via
|||                           the Tokenizer subprocess, greedy-decode N
|||                           tokens, detokenize, print.
|||
||| Pre-requisites (CI handles these via the make targets):
|||   - `models/unsloth/Llama-3.2-1B/{config.json,model.safetensors}`
|||     — fetch with
|||         bash packages/idris-transformers/scripts/hf-download.sh unsloth/Llama-3.2-1B
|||     (public mirror; no `HF_TOKEN` required).
|||   - Python `transformers` available via the pytorch venv (for the
|||     Tokenizer subprocess).
module Example.LlamaInference

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
import Example.Common.InferenceHelper
import Executor
import Nn.RoPE
import Tensor
import Transformers.KVCache
import Transformers.Llama
import Transformers.Tokenizer
import Util

----------------------------------------------------------------------
-- Model location (dims come from the file, not from here)
----------------------------------------------------------------------

-- Clamped RoPE-table context. The model's full 131072 max with NTK
-- scaling is supported by the table builder, but the cos/sin tables at
-- 131072 × 32 are 32 MB each; clamping to the original 8192 training
-- context keeps the demo modest. Llama's positional behaviour at <8k is
-- identical regardless of the `maxPos` chosen for the tables. This is a
-- table-size knob, separate from the model's `cfg.maxPosition`, so it
-- stays a local literal (it also appears at type position in
-- `RoPETables MaxPos (headDim cfg) ...`).
MaxPos : Nat
MaxPos = 8192

ModelRepo : String
ModelRepo = "unsloth/Llama-3.2-1B"

modelDir : String
modelDir = "models/" ++ ModelRepo

----------------------------------------------------------------------
-- Cache-aware greedy generation
----------------------------------------------------------------------

||| Single decode step with KV cache. Takes the current per-layer
||| caches + the tokens to feed (the prompt on the seed call, then
||| [prevGenerated] each subsequent step), returns updated caches +
||| next-token id (or Nothing on out-of-range argmax).
genStepCached :
     (cfg : LlamaConfig)
  -> LlamaModel cfg ExampleExecutor ExampleDType NoGrad
  -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
  -> Vect (numLayers cfg) (KVCache (numKvHeads cfg * headDim cfg) ExampleExecutor ExampleDType)
  -> List (Fin (vocabSize cfg))
  -> IO (Vect (numLayers cfg) (KVCache (numKvHeads cfg * headDim cfg) ExampleExecutor ExampleDType),
         Maybe (Fin (vocabSize cfg)))
genStepCached cfg model tables caches toksList = do
  let idsList = map (cast {to=Double} . finToNat) toksList
  case toExistVect idsList of
    (curLen ** idDoubles) => do
      let inputIds = retypeGrad (mkIds idDoubles)
      (caches', logits) <- hfLlamaForwardLmStep
                                 {ex=ExampleExecutor} {dt=ExampleDType}
                                 {seq          = curLen}
                                 {vocab        = vocabSize cfg}
                                 {hidden       = hidden cfg}
                                 {numLayers    = numLayers cfg}
                                 {numHeads     = numHeads cfg}
                                 {numKvHeads   = numKvHeads cfg}
                                 {headDim      = headDim cfg}
                                 {intermediate = intermediate cfg}
                                 {maxPos       = MaxPos}
                                 (rmsNormEps cfg) model tables caches inputIds
      lastRow <- trowSelect logits (cast {to=Int} curLen - 1)
      nextN   <- argmaxRow (vocabSize cfg) lastRow.tensorPtr
      pure (caches', natToFin nextN (vocabSize cfg))

||| Cache-aware greedy decode — the canonical (and only) generation
||| path. Seed step feeds the full prompt into empty caches; each
||| subsequent step feeds only the previously-generated token, with
||| the per-layer caches threading through.
|||
||| Convention: this example has a single decode path. The legacy
||| no-cache `genLoop` / `genOneStep` were dropped 2026-06-04 — they
||| were correctness-equivalent but doubled Chez elaboration cost on
||| this file. Recover via `git show 70f5017c:...LlamaInference.idr`
||| if differential debugging is needed.
genLoopCached :
     (cfg : LlamaConfig)
  -> LlamaModel cfg ExampleExecutor ExampleDType NoGrad
  -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
  -> List (Fin (vocabSize cfg))   -- prompt
  -> (remaining : Nat)
  -> IO (List (Fin (vocabSize cfg)))
genLoopCached cfg model tables prompt remaining =
  let initialCaches : Vect (numLayers cfg) (KVCache (numKvHeads cfg * headDim cfg) ExampleExecutor ExampleDType)
      initialCaches = emptyKVCaches {numLayers = numLayers cfg} {kvOut = numKvHeads cfg * headDim cfg}
  in go initialCaches prompt prompt remaining
  where
    go : Vect (numLayers cfg) (KVCache (numKvHeads cfg * headDim cfg) ExampleExecutor ExampleDType)
      -> List (Fin (vocabSize cfg))   -- accumulator (prompt + generated so far)
      -> List (Fin (vocabSize cfg))   -- new tokens to feed this step (prompt on seed, [prev] on steady)
      -> Nat                          -- remaining tokens to generate
      -> IO (List (Fin (vocabSize cfg)))
    go _      acc _    Z     = pure acc
    go caches acc feed (S k) = do
      perfReset {ex=ExampleExecutor}
      (caches', mNext) <- genStepCached cfg model tables caches feed
      ops <- perfOpCount {ex=ExampleExecutor}
      putStrLn ("[perf] step " ++ show (length acc) ++ ": " ++ show ops ++ " ops")
      case mNext of
        Nothing => do
          putStrLn "  (argmax produced out-of-range token; stopping)"
          pure acc
        Just next => go caches' (acc ++ [next]) [next] k

----------------------------------------------------------------------
-- --dump-final-hidden mode
----------------------------------------------------------------------

runDumpHidden : (cfg : LlamaConfig)
             -> LlamaModel cfg ExampleExecutor ExampleDType NoGrad
             -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
             -> IO ()
runDumpHidden cfg model tables = do
  -- Fixed input: BPE("Hello") = [9906] in Llama 3 vocab.
  let inputIds = retypeGrad (mkIds (the (Vect 1 Double) [9906.0]))
  out <- hfLlamaForward {ex=ExampleExecutor} {dt=ExampleDType}
                        {seq          = 1}
                        {vocab        = vocabSize cfg}
                        {hidden       = hidden cfg}
                        {numLayers    = numLayers cfg}
                        {numHeads     = numHeads cfg}
                        {numKvHeads   = numKvHeads cfg}
                        {headDim      = headDim cfg}
                        {intermediate = intermediate cfg}
                        {maxPos       = MaxPos}
                        (rmsNormEps cfg) model tables inputIds
  lastRow <- trowSelect out 0
  printRow (cast {to=Int} (hidden cfg)) 0 lastRow.tensorPtr

----------------------------------------------------------------------
-- Default mode: greedy generation demo
----------------------------------------------------------------------

runGenerate : (cfg : LlamaConfig)
           -> Tokenizer (vocabSize cfg)
           -> LlamaModel cfg ExampleExecutor ExampleDType NoGrad
           -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
           -> (prompt : String) -> (numTokens : Nat) -> IO ()
runGenerate cfg tok model tables prompt numTokens = do
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
  finalList <- genLoopCached cfg model tables promptList numTokens
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
||| same prompt; the comparator asserts exact match.
runDumpTokens : (cfg : LlamaConfig)
             -> Tokenizer (vocabSize cfg)
             -> LlamaModel cfg ExampleExecutor ExampleDType NoGrad
             -> RoPETables MaxPos (headDim cfg) ExampleExecutor ExampleDType
             -> (prompt : String) -> (numTokens : Nat) -> IO ()
runDumpTokens cfg tok model tables prompt numTokens = do
  Right (_ ** promptIds) <- tokenize tok prompt
    | Left err => do
        putStrLn ("ERR: tokenize: " ++ show err)
        exitFailure
  let promptList = toList promptIds
  finalList <- genLoopCached cfg model tables promptList numTokens
  traverse_ (putStrLn . show . finToNat) finalList

----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let dumpHidden = elem "--dump-final-hidden" args
  let dumpTokens = elem "--dump-tokens" args
  t0 <- clockTime Monotonic

  -- Load the full Llama 3.2 1B state straight from the HF checkpoint
  -- dir: fromPretrained reads config.json for the dims, builds the
  -- model (146 params, ~1.2B values), and fills them from
  -- model.safetensors (~2.5 GB BF16, cast-on-load to F32/F64). Tape
  -- (F64-only) doesn't fit in 16 GB; the example targets F32 backends.
  putStrLn "[stage] fromPretrained — reading config.json + 146-param state (~2.5 GB BF16)..."
  Right (cfg ** model) <- fromPretrained {ex=ExampleExecutor} {dt=ExampleDType} {g=NoGrad} modelDir
    | Left err => do
        putStrLn ("ERR: fromPretrained " ++ modelDir ++ ": " ++ show err)
        exitFailure
  stageStamp "fromPretrained ok" t0

  -- Probe the tokenizer (~1s subprocess). dump-hidden mode doesn't
  -- strictly need it, but probing keeps the failure semantics uniform.
  tokR <- mkTokenizer ModelRepo (vocabSize cfg)
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

  -- Build RoPE tables once (reused across all forward passes / decode
  -- steps). headDim + ropeBase come from the loaded config.
  putStrLn "[stage] buildLlamaRoPETables — precomputing cos/sin tables..."
  tables <- buildLlamaRoPETables {ex=ExampleExecutor} {dt=ExampleDType}
                                  {maxPos  = MaxPos}
                                  {headDim = headDim cfg}
                                  (ropeBase cfg) llama32_1B_RopeScaling
  stageStamp "buildLlamaRoPETables ok" t0

  if dumpHidden
    then do
      putStrLn "[stage] forward (dump-hidden mode) — single forward pass..."
      runDumpHidden cfg model tables
      stageStamp "dump-hidden done" t0
      pure ()
    else case tokR of
      -- Unreachable in practice: dumpHidden=False + Left would have
      -- exitFailure'd at the tokenizer probe. Idris's type checker
      -- doesn't know that, so we handle Left defensively.
      Left err  => do
        putStrLn ("ERR: mkTokenizer (post-probe inconsistency): " ++ show err)
        exitFailure
      Right tok =>
        if dumpTokens
          then do
            putStrLn "[stage] runDumpTokens — greedy decode + dump token ids..."
            runDumpTokens cfg tok model tables
                          (extractPrompt "The capital of France is" args)
                          (extractNumTokens 4 args)
            stageStamp "runDumpTokens done" t0
            _ <- drainManagedHandles
            forceMajorGc
            _ <- drainManagedHandles
            stageStamp "drain + GC done" t0
            releaseAllPersistent {ex=ExampleExecutor}
            stageStamp "releaseAllPersistent done" t0
            pure ()
          else do
            putStrLn "[stage] runGenerate — greedy decode loop..."
            let numTokens = extractNumTokens 8 args
            benchT0 <- clockTime Monotonic
            runGenerate cfg tok model tables (extractPrompt "The capital of France is" args) numTokens
            benchT1 <- clockTime Monotonic
            stageStamp "runGenerate done" t0
            let benchMs =
                  let s  = cast {to=Double} (seconds benchT1 - seconds benchT0)
                      ns = cast {to=Double} (nanoseconds benchT1 - nanoseconds benchT0)
                  in s * 1000.0 + ns / 1000000.0
            putStrLn ""
            putStrLn ("PERF_GENERATE_TOKENS=" ++ show numTokens)
            putStrLn ("PERF_GENERATE_WALL_MS=" ++ show benchMs)
            releaseAllPersistent {ex=ExampleExecutor}
            stageStamp "releaseAllPersistent done" t0
            pure ()
