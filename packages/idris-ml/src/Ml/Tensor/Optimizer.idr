||| Native optimizer handle, Polyak soft-update, per-param LR control,
||| and the fused train-step shims.
module Ml.Tensor.Optimizer

import Data.List
import Data.Vect

import Ml.DType.Core
import Ml.Executor
import Ml.GradMode
import Ml.Schedule
import Ml.Tensor.Core

----------------------------------------------------------------------
-- Backpropagation: prims for native optimizer
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Native Optimizer
----------------------------------------------------------------------

||| Polyak soft update of one EXACTLY-named param pair (`onlineName` →
||| `targetName`):
|||   target_data ← (1 − tau) · target_data + tau · online_data
||| in-place. Returns 1 if blended, 0 if a name is absent or the shapes
||| differ. The per-pair primitive `polyakUpdatePaired` folds over.
|||
||| Per-backend: the registry storing the params lives in the backend
||| TU, so dispatch via `primPolyakBlendPair` from the in-scope
||| `UserExecutorTraining ex` instance.
export
polyakUpdate : UserExecutorTraining ex =>
               (tau : Double) -> (onlineName : String) -> (targetName : String) -> IO Int
polyakUpdate tau onlineName targetName =
  primIO (primPolyakBlendPair {ex} tau onlineName targetName)

||| Polyak soft-update by exact paired param names (structural, leak-free):
||| pairs `onlineNames[i]` with `targetNames[i]` positionally and blends each
||| `target ← (1 − tau)·target + tau·online`. Returns the count of pairs blended.
|||
||| The typed replacement for `polyakUpdate tau "q1_" "q1tgt_"`: feed it
||| `Nn.Group.reflectNames online` / `reflectNames target` so the source↔target
||| pairing is derived from the nets' structure, not a load-bearing naming
||| convention (a `"q1_"` prefix that also catches `"q1tgt_"` is the bug class
||| this removes). Pairing is positional, which is correct when the two nets
||| share a constructor — their `Params` traversal orders coincide.
|||
||| Each pair is blended by exact-name match (`primPolyakBlendPair`), so a
||| name that is a proper prefix of another can't over-match.
export
polyakUpdatePaired : UserExecutorTraining ex =>
                     (onlineNames : List String) -> (targetNames : List String) ->
                     (tau : Double) -> IO Int
polyakUpdatePaired ons tgs tau =
  sum <$> traverse (\(o, t) => polyakUpdate {ex} tau o t) (zip ons tgs)

public export
data ClipMode = NoClip | ValueClip Double | NormClip Double

||| Native optimizer handle. Single step() call updates all
||| parameters in the backend's registry. The `d` phantom pins the
||| optimizer to the backend whose registry it manages — a
||| `NativeOptimizer ex` can only step a loss `Tensor [] ex dt`.
|||
||| `schedule` is consumed by `Optimizer.tick` (Nothing = fixed LR);
||| attach one with `Optimizer.withSchedule`.
public export
record NativeOptimizer (0 ex : Executor) where
  constructor MkNativeOptimizer
  handle   : AnyPtr
  clipMode : ClipMode
  schedule : Maybe Schedule

||| Set a per-parameter learning rate override. Parameters matching the given
||| name will use this LR instead of the optimizer's base LR.
||| Use LR=0 to freeze a parameter. Set LR<0 to revert to base LR.
export
setParamLR : UserExecutorTraining ex => NativeOptimizer ex -> String -> Double -> IO ()
setParamLR opt name lr = primIO (primOptimizerSetParamLr {ex} opt.handle name lr)

||| Add one exact param name to the optimizer's owned-set. Once any name is
||| owned, the optimizer's step + clip touch ONLY owned params (true skip) —
||| every other registered param is left untouched. An empty owned-set (the
||| default) manages all params. The typed, leak-free replacement for the
||| deleted prefix scope: `Train.Freeze.restrictTo` feeds it exact names from
||| `Nn.Group.reflectNames`, so `q1_` can't leak into `q1tgt_`.
export
setOwnedParam : UserExecutorTraining ex => NativeOptimizer ex -> String -> IO ()
setOwnedParam opt name = primIO (primOptimizerOwnParam {ex} opt.handle name)

||| Update the optimizer's base (global) learning rate. Per-parameter
||| overrides set via `setParamLR` remain in effect; only un-overridden
||| params pick up the new base LR. Used to apply LR schedules per epoch.
export
setLearningRate : UserExecutorTraining ex => NativeOptimizer ex -> Double -> IO ()
setLearningRate opt lr = primIO (primOptimizerSetLr {ex} opt.handle lr)

-- Fused native train step: zero_grad → backward → clip → step.
-- Fused: zero_grad → backward → clip → step in single C call.
-- Returns loss value (read before step, so not stale).
--
-- After the C call returns, force a Chez minor GC + drain the
-- managed-handle guardian. This is the training-loop drain trigger
-- that lets the mlx refcount-driven lifecycle reclaim per-step
-- intermediate Tensors — without it, the wrap-and-retain on each
-- Tensor's creation keeps its refcount at >=1 indefinitely (Chez
-- doesn't auto-GC under foreign-side pressure alone, and drain is
-- only otherwise called at withNoGrad exit). On tape/torch the drain
-- is essentially a no-op (their retain/release are stubs).
-- After the step, force a Chez major GC then drain all dead wraps via
-- the per-backend dispatch helper `idris-drain-once` (installed by
-- prim__installDrainHelperC). This is the reclamation pump for hot
-- training loops where ops bypass `tape_append` and per-op refcount
-- bookkeeping doesn't fire — without it, the wrap-and-retain on each
-- new Tensor keeps refcounts at >=1 indefinitely.
-- The fused step itself dispatches per-backend via
-- `primNativeTrainStep {ex}` (see `UserExecutorTraining`); each backend's
-- Scheme wrap carries the same GC + drain epilogue.

-- Optimizer shim ------------------------------------------------------

||| Fused native train step on a Tensor loss: zero_grad → backward →
||| clip → step. Reads `prim__item` BEFORE the step so the returned
||| scalar is not stale. Mirrors `trainStep`.
export
trainStep : {0 ex : Executor} -> UserExecutorTraining ex => IsFloating dt =>
                  NativeOptimizer ex -> Tensor [] ex dt WithGrad -> IO Double
trainStep opt loss = ioRerun (\_ =>
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal  : Double
      clipVal = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
      lossVal = primItem {ex} loss.tensorPtr
  in primNativeTrainStep {ex} opt.handle clipMode clipVal loss.tensorPtr lossVal)

||| GradScaler-aware fused step (A3 of #410). The caller has already
||| multiplied the loss by `scale` so backward computes grads at the
||| scaled magnitude (avoiding F16 underflow). The C port unscales
||| grads by `1/scale`, checks for non-finite values, and either
||| steps + returns the unscaled loss, or returns NaN (= overflow
||| detected, step was skipped; caller halves its scale state).
|||
||| Wired across all three backends.
export
nativeTrainStepScaled : {0 ex : Executor} -> UserExecutorTraining ex => IsFloating dt =>
                        NativeOptimizer ex -> Tensor [] ex dt WithGrad ->
                        (scale : Double) -> IO Double
nativeTrainStepScaled opt loss scale = ioRerun (\_ =>
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal  : Double
      clipVal       = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
      scaledLossVal = primItem {ex} loss.tensorPtr
  in primNativeTrainStepScaled {ex} opt.handle clipMode clipVal loss.tensorPtr scaledLossVal scale)
