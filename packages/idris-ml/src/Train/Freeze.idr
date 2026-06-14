||| Bulk freeze / unfreeze of registered parameters by paramId prefix.
|||
||| Composes with the existing single-optimizer training loop —
||| `freezeByPrefix opt "bert."` walks the registry and sets the
||| per-param LR override to 0 for every matching name, after which
||| the next `nativeTrainStep` leaves those params unchanged. Pair with
||| `unfreezeByPrefix` to thaw (LR -1 reverts to the optimizer's base
||| LR — see `setParamLR`).
|||
||| Designed for the warm-start fine-tune pattern: load a HF backbone
||| with `Checkpoint.loadModelPrefix`, freeze it by the same prefix,
||| train the fresh head on top.
module Train.Freeze

import Data.String

import Executor
import Tensor

-- Per-name worker: walks indices from k-1 down to 0; if a name starts
-- with `pfx`, applies `lr` via `setParamLR`.
applyIfPrefix : UserExecutorTraining ex => NativeOptimizer ex -> String -> Double -> Nat -> IO ()
applyIfPrefix opt pfx lr Z = pure ()
applyIfPrefix opt pfx lr (S k) = do
  let i = cast {to=Int} k
  isBuf <- getParamIsBuffer {ex} i
  name  <- getParamName {ex} i
  when (not isBuf && isPrefixOf pfx name) (setParamLR {ex} opt name lr)
  applyIfPrefix opt pfx lr k

-- Per-name worker: walks indices from k-1 down to 0; if a name ends
-- with `sfx`, applies `lr` via `setParamLR`.
applyIfSuffix : UserExecutorTraining ex => NativeOptimizer ex -> String -> Double -> Nat -> IO ()
applyIfSuffix opt sfx lr Z = pure ()
applyIfSuffix opt sfx lr (S k) = do
  let i = cast {to=Int} k
  isBuf <- getParamIsBuffer {ex} i
  name  <- getParamName {ex} i
  when (not isBuf && isSuffixOf sfx name) (setParamLR {ex} opt name lr)
  applyIfSuffix opt sfx lr k

||| Freeze every registered parameter whose paramId starts with
||| `pfx` by setting its per-param LR override to 0 on `opt`.
||| Subsequent training steps leave those parameters' weights
||| untouched. Safe to call multiple times; the underlying per-name
||| override is idempotent.
export
freezeByPrefix : UserExecutorTraining ex => NativeOptimizer ex -> (pfx : String) -> IO ()
freezeByPrefix opt pfx = do
  n <- getParamCount {ex}
  applyIfPrefix {ex} opt pfx 0.0 (cast {to=Nat} n)

||| Clear the per-param LR override on every registered parameter
||| whose paramId starts with `pfx`, restoring the optimizer's
||| base LR. Symmetric to `freezeByPrefix`.
export
unfreezeByPrefix : UserExecutorTraining ex => NativeOptimizer ex -> (pfx : String) -> IO ()
unfreezeByPrefix opt pfx = do
  n <- getParamCount {ex}
  applyIfPrefix {ex} opt pfx (-1.0) (cast {to=Nat} n)

||| Freeze every registered parameter whose paramId ends with `sfx`.
||| Mirror of `freezeByPrefix` for the cases where naming convention
||| puts the discriminating component at the tail (e.g. LoRA adapter
||| params live under `bert.*.lora_A` / `.lora_B` — a single
||| `freezeByPrefix opt "bert."` would freeze the adapters too, so
||| the canonical LoRA setup is `freezeByPrefix opt "bert."` followed
||| by `unfreezeBySuffix opt "lora_A"` + `unfreezeBySuffix opt "lora_B"`).
export
freezeBySuffix : UserExecutorTraining ex => NativeOptimizer ex -> (sfx : String) -> IO ()
freezeBySuffix opt sfx = do
  n <- getParamCount {ex}
  applyIfSuffix {ex} opt sfx 0.0 (cast {to=Nat} n)

||| Clear the per-param LR override on every registered parameter
||| whose paramId ends with `sfx`. Symmetric to `freezeBySuffix`.
||| Composes with `freezeByPrefix` to express the canonical LoRA
||| "freeze everything matching X, then unfreeze the adapters under
||| X" pattern in two cheap C calls.
export
unfreezeBySuffix : UserExecutorTraining ex => NativeOptimizer ex -> (sfx : String) -> IO ()
unfreezeBySuffix opt sfx = do
  n <- getParamCount {ex}
  applyIfSuffix {ex} opt sfx (-1.0) (cast {to=Nat} n)
