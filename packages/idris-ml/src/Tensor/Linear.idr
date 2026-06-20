||| The `L IO` op surface (the `*L` tensor ops) for the linear model
||| surface (Nn.Module / Nn.Recurrent).
module Tensor.Linear

import Control.Linear.LIO as LIO
import Data.Vect

import DType.Core
import Executor
import GradMode
import Tensor.Core
import Tensor.Handle
import Tensor.Internal
import Util

----------------------------------------------------------------------
-- `L IO` op surface (the `*L` tensor ops)
--
-- The `L IO` twins of the smart constructors above, for the model surface
-- (`Nn.Module.Module` / `Nn.Recurrent.Recurrent`). They let `forward` /
-- `recurStep` bodies sequence tensor ops directly in `L IO` instead of
-- wrapping each one in `liftIO1`. Built natively on `ioRerunL` (the single
-- lifting primitive), so there is no per-op `liftIO1` seam — the only lift is
-- centralized here. Tensors are *unrestricted* (reverse-mode AD shares them),
-- so these return `L IO` at the default `use = Unrestricted`.
--
-- The `IO` twins above are kept (the dual op surface is permanent): tensors
-- carry no single-owner footgun, so imperative `IO` callers — benchmarks,
-- hand-written HF forwards — keep using the `IO` ops, while model
-- `forward`/`recurStep` bodies (now including the mixed-precision
-- `ModuleMixed.forwardMixed`) use the `*L` ones. (Only the *model* surface
-- collapsed to a single linear form.)
----------------------------------------------------------------------

||| Lift a pure (FFI-side-effecting) expression into `L IO`, re-evaluated on
||| every sequencing (the `L IO` counterpart of `ioRerun`). The single seam
||| between the `PrimIO`/`IO` prims and the `L IO` op surface.
export %inline
ioRerunL : (() -> a) -> LIO.L IO a
ioRerunL f = liftIO1 (ioRerun f)

||| Born-linear construction from an `IO` action (the IO-constructor analog of
||| `runInitL`): run the action, re-emit its result at `use = 1` so the bound
||| value is **linear** and the caller must thread it. For model surfaces whose
||| constructors are plain `IO` (the HF `hf*` builders) rather than `Init` — a
||| `liftIO1` alone would hand back an unrestricted value, losing the
||| single-owner discipline at the construction seam.
export
bornL : {0 a : Type} -> (1 act : IO a) -> LIO.L IO {use = 1} a
bornL act = do
  x <- liftIO1 act
  pure1 x

export %inline
taddL : {0 ex : Executor} -> UserExecutorCore ex =>
        Tensor dims ex dt g -> Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
taddL a b = ioRerunL (\_ => MkTensor (primAdd {ex} a.tensorPtr b.tensorPtr) Nothing)

||| `L IO` twin of `tcastUnsafe`: the autograd-aware (unchecked) dtype cast on
||| the model surface, used by the mixed-precision `ModuleMixed.forwardMixed`
||| to materialise the `paramDt → computeDt` cast without a `liftIO1` seam.
export %inline
tcastUnsafeL : {0 ex : Executor} -> (0 to : DType) -> Backend ex to =>
               (IsDType from, IsDType to) =>
               Tensor dims ex from g -> LIO.L IO (Tensor dims ex to g)
tcastUnsafeL to v = ioRerunL (\_ => MkTensor (dtCastFrom {ex} {t=to} v.tensorPtr (deviceStreamTag {ex})) Nothing)

export %inline
tlinearL : {0 ex : Executor} -> UserExecutorTraining ex =>
           Tensor [o, i] ex dt g -> Tensor [i] ex dt g -> Tensor [o] ex dt g ->
           LIO.L IO (Tensor [o] ex dt g)
tlinearL w x bias = ioRerunL (\_ =>
  MkTensor (primLinear {ex} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

export %inline
tlinear2dL : {0 ex : Executor} -> UserExecutorTraining ex =>
             Tensor [o, i] ex dt g -> Tensor [b, i] ex dt g -> Tensor [o] ex dt g ->
             LIO.L IO (Tensor [b, o] ex dt g)
tlinear2dL w x bias = ioRerunL (\_ =>
  MkTensor (primLinear2d {ex} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

export %inline
tnllLossMeanL : {0 ex : Executor} -> UserExecutorNN ex => IsFloating dt => {b, n : Nat} ->
                Tensor [b, n] ex dt g -> Tensor [b, n] ex dt g -> LIO.L IO (Tensor [] ex dt g)
tnllLossMeanL {b} {n} p t = ioRerunL (\_ =>
  let logP = primLogSoftmax2d {ex} p.tensorPtr in
  let prod = primMul {ex} logP t.tensorPtr in
  let neg  = primNeg {ex} (primSum {ex} prod) in
  MkTensor (primMulScalar {ex} neg (1.0 / cast (b * n))) Nothing)

||| `withNoGrad` for the linear (`L IO`) surface: bracket a linear action with
||| the per-backend no-grad scope (push counter → run → drain → pop), threading
||| the action's linear result through. The `L IO` counterpart of `withNoGrad`,
||| used by linear eval / rollout loops that thread a `WithGrad` model through
||| `recurStepL` / `forwardSeqL` but want tape-free, memory-hygienic forwards
||| (essential on mlx per the per-sequence-withNoGrad note in CLAUDE.md). As
||| with plain `withNoGrad`, the linear result must not carry live *intermediate*
||| tensors created inside the bracket (registered params and scalar/Nat results
||| survive; a freshly-created output tensor would be freed by the exit drain).
export
withNoGradL : {0 ex : Executor} -> UserExecutorTraining ex => {0 a : Type} ->
              (1 act : LIO.L IO {use = 1} a) -> LIO.L IO {use = 1} a
withNoGradL act = do
  liftIO1 (primIO (primNoGradBegin {ex}))
  result <- act
  liftIO1 (do forceMajorGc; _ <- drainManagedHandles; pure ())
  liftIO1 (primIO (primNoGradEnd {ex}))
  pure1 result

||| `withGenFree` for the linear (`L IO`) surface: run a *grad-mode* linear
||| action inside a generation bracket (autograd stays ON), freeing the
||| wrap-only intermediates it created on exit. The `L IO` twin of
||| `withGenFree` — for fine-grained training inner loops (a DQN replay step, a
||| PPO rollout step) whose per-step grad intermediates would otherwise pile up
||| past the mlx buffer ceiling within one epoch. As with `withGenFree`, the
||| linear result must not carry fresh wrap-only intermediates created inside the
||| bracket (registered params — rc>1 — and scalar results are spared); the
||| threaded model handles it returns are registered, so they survive.
export
withGenFreeL : {0 ex : Executor} -> UserExecutorTraining ex => {0 a : Type} ->
               (1 act : LIO.L IO {use = 1} a) -> LIO.L IO {use = 1} a
withGenFreeL act = do
  liftIO1 (primIO (primEpochBegin {ex}))
  result <- act
  liftIO1 (primIO (primEpochEnd {ex}))
  pure1 result

export
tzeroState1dL : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> LIO.L IO (Tensor [n] ex dt g)
tzeroState1dL {n} = ioRerunL (\_ =>
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in MkTensor (dtCreateState1d {ex} {t=dt} nI buf (deviceStreamTag {ex})) Nothing)

export
tlstmGatesPairL : UserExecutorNN ex => {n : Nat} ->
                  TVec (4 * n) ex dt g -> TVec n ex dt g ->
                  LIO.L IO (TVec n ex dt g, TVec n ex dt g)
tlstmGatesPairL {n} combined prevCell = ioRerunL (\_ =>
  let nI = cast {to=Int} n
      pair = primLstmGatesPair {ex} combined.tensorPtr prevCell.tensorPtr nI
  in (MkTensor (primPairFirst {ex} pair) Nothing, MkTensor (primPairSecond {ex} pair) Nothing))

export
tgruCellL : UserExecutorNN ex => {n : Nat} ->
            TVec (3 * n) ex dt g -> TVec (3 * n) ex dt g -> TVec n ex dt g ->
            LIO.L IO (TVec n ex dt g)
tgruCellL {n} ih hh prevH = ioRerunL (\_ =>
  let nI = cast {to=Int} n
  in MkTensor (primGruCell {ex} ih.tensorPtr hh.tensorPtr prevH.tensorPtr nI) Nothing)

export %inline
ttanhL : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
ttanhL v = ioRerunL (\_ => MkTensor (primTanh {ex} v.tensorPtr) Nothing)

export %inline
tsigmoidL : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
tsigmoidL v = ioRerunL (\_ => MkTensor (primSigmoid {ex} v.tensorPtr) Nothing)

export %inline
treluL : {0 ex : Executor} -> UserExecutorCore ex => Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
treluL v = ioRerunL (\_ => MkTensor (primClampMin {ex} v.tensorPtr 0.0) Nothing)

export %inline
tgeluL : {0 ex : Executor} -> UserExecutorTraining ex => Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
tgeluL v = ioRerunL (\_ => MkTensor (primGelu {ex} v.tensorPtr) Nothing)

export %inline
tsiluL : {0 ex : Executor} -> UserExecutorTraining ex => Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
tsiluL v = ioRerunL (\_ => MkTensor (primSilu {ex} v.tensorPtr) Nothing)

export %inline
tleakyReluL : {0 ex : Executor} -> UserExecutorTraining ex => Double -> Tensor dims ex dt g -> LIO.L IO (Tensor dims ex dt g)
tleakyReluL slope v = ioRerunL (\_ => MkTensor (primLeakyRelu {ex} v.tensorPtr slope) Nothing)
