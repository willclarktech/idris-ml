||| Cache-aware greedy generation for `Example.LlamaInference`, split into
||| its own module so the `hfLlamaForwardLmStep` implicit chain elaborates
||| in a separate idris2 pass from the rest of the example (the
||| elaboration-memory lever (a) from the "Reduce idris2/Chez elaboration
||| memory peak" TODO row: one process no longer resolves both this chain
||| and `runDumpHidden`'s plain `hfLlamaForward` chain).
module Example.LlamaCacheGen

import Data.Fin
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.RoPE
import Ml.Tensor
import Transformers.KVCache
import Transformers.Llama

import BuildConfig
import Example.Common.InferenceHelper

----------------------------------------------------------------------
-- RoPE-table context
----------------------------------------------------------------------

-- Clamped RoPE-table context. The model's full 131072 max with NTK
-- scaling is supported by the table builder, but the cos/sin tables at
-- 131072 × 32 are 32 MB each; clamping to the original 8192 training
-- context keeps the demo modest. Llama's positional behaviour at <8k is
-- identical regardless of the `maxPos` chosen for the tables. This is a
-- table-size knob, separate from the model's `cfg.maxPosition`, so it
-- stays a local literal (it also appears at type position in
-- `RoPETables MaxPos (headDim cfg) ...`).
public export
MaxPos : Nat
MaxPos = 8192

----------------------------------------------------------------------
-- Cache-aware greedy generation
----------------------------------------------------------------------

||| Single decode step with KV cache. Takes the current per-layer
||| caches + the tokens to feed (the prompt on the seed call, then
||| [prevGenerated] each subsequent step), returns updated caches +
||| next-token id (or Nothing on out-of-range argmax).
export
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
export
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
