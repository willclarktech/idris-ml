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
  name <- getParamName {ex} (cast {to=Int} k)
  when (isPrefixOf pfx name) (setParamLR {ex} opt name lr)
  applyIfPrefix opt pfx lr k

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
