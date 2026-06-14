||| The v1 model abstraction (PyTorch `nn.Module` vocabulary). Models are
||| plain records of layers + a `forward`; this module holds the two
||| interfaces every model implements:
|||
|||   * `Module`  — a batched-first `forward`. No `applyVarBatch`, no
|||     `idris_crash` default: a layer that can't do batched forward
|||     simply isn't a `Module`, so it can't enter a batched `Seq` and
|||     there is no method to crash. (The old `LayerLike.applyVarBatch`
|||     crash hole dies structurally.)
|||   * `Params` — the parameter traversal (`groupOf`/freeze build on it).
|||
||| `GradMode` is ON the model type (`l i o ex dt g`): a model's params and
||| its activations share one `g`, so `forward` needs no `retypeGrad`, and a
||| trainable model (`WithGrad`) is a distinct type from an inference model
||| (`NoGrad`). This makes "train an inference model" / "carry a tape through
||| pure inference" unrepresentable rather than a runtime concern — the
||| library's "invalid programs unrepresentable" principle (reversing the
||| earlier "GradMode off model types" decision, 2026-06-14; see
||| design-decisions.md). Construction yields `WithGrad`; the optimizer path
||| requires a `WithGrad` loss, which only a `WithGrad` model can produce.
|||
||| The *loss-scaling* half of mixed precision is a `fit` config
||| (`fitSupervisedMixed` + `GradScaler`). The *master-weights* half —
||| store weights in `paramDt`, cast to `computeDt` inside the forward —
||| is inherently per-layer (the cast lives at the layer boundary), so it
||| gets `ModuleMixed` below: a sibling of `Module` adding the compute-dtype
||| slot. (The legacy `Layer/MixedCore.idr` needed an extra
||| `AsMixed`/`AnyLayerMixed`/`NetworkMixed`/`lift*` apparatus to chain
||| mixed + plain layers through existentials; models-as-records drops all
||| of it — a mixed model is just a record with a hand-written forward.)
|||
||| Coexists with the legacy `Layer/` surface (`LayerLike`/`Network`);
||| examples migrate at the later sweep.
module Nn.Module

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Tensor

||| A registered parameter, erased of shape/dtype: its C handle + the
||| registry name it was registered under (the `Tensor.paramId`). The
||| unit `Params`/`groupOf`/freeze operate on.
public export
record SomeParam where
  constructor MkSomeParam
  paramPtr  : AnyPtr
  paramName : Maybe String

||| Erase a tensor to its `SomeParam` (handle + registry name).
public export
toParam : {0 dims : Vect rank Nat} -> {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} ->
          Tensor dims ex dt g -> SomeParam
toParam t = MkSomeParam t.tensorPtr t.paramId

||| Mixed-precision model component (the master-weights half). Five-index
||| kind: `(in, out, executor, paramDt, computeDt)` — `paramDt` is where
||| weights are stored (the F32 "master"), `computeDt` where activations
||| flow (BF16 / F16). `forwardMixed` casts `paramDt → computeDt` inside
||| the layer, autograd-aware, so backward writes a `paramDt` gradient back
||| into the master. NO `GradMode` on the type (as `Module`); `g` lives on
||| the activation only.
|||
||| `IsDType paramDt` is required so the forward can call `tcastUnsafe` to
||| materialise the (typically lossy) `paramDt → computeDt` cast; the
||| activation side rides `Backend ex computeDt`. For `paramDt = computeDt`
||| the cast is a dtype-level no-op (the diagonal case).
public export
interface ModuleMixed (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  forwardMixed : {0 ex : Executor} -> Backend ex computeDt => IsDType paramDt => IsDType computeDt =>
                 {0 g : GradMode} -> {i, o, b : Nat} ->
                 l i o ex paramDt computeDt g -> Tensor [b, i] ex computeDt g ->
                 IO (Tensor [b, o] ex computeDt g)

||| Parameter traversal for a mixed-precision component. Identical erased
||| `SomeParam` list as `Params` (master weights live in `paramDt`, but
||| `SomeParam` is dtype-erased) — a separate interface only because the
||| kind carries the extra `computeDt` slot.
public export
interface ParamsMixed (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  paramsMixed : {0 ex : Executor} -> {0 paramDt, computeDt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
                l i o ex paramDt computeDt g -> List SomeParam

----------------------------------------------------------------------
-- The model surface (`L IO`): models are single-owner linear resources
----------------------------------------------------------------------

||| `Params` is the base capability every layer has, whether it is a batched
||| `Module` or a per-timestep `Recurrent` (`Params` is the base; `Module`/
||| `Recurrent` are separate capabilities on top). A model is a single-owner
||| resource: every method **consumes** the handle `(1 _ : l …)`, so a stale
||| alias (the classic "freeze a model, then reuse the old handle to train"
||| no-op) is a *compile-time* linearity error rather than a silent
||| afternoon-waster. (See `docs/develop/linear-types-and-effects.md`.)
|||
|||   * `params` — read the flat param list **without consuming** the model
||| (ω model arg; the read-only `.parameters()` analogue for registration /
||| checkpointing / prefix-freezing, where no linear threading is wanted). This
||| is a pure read of param handles, *not* the freeze/eval footgun (which
||| mutates `requires_grad` then threads the model on through `forward`/train).
|||   * `reflect` — the **linear** twin: expose the param list (for the C
||| `requires_grad` flips by `eval`/`freeze`) *without losing* the model —
||| returns `(!* params) # model` so the single-owner handle threads onward.
|||   * `castGrad` — the grad-mode retype (`g → g'`); pure, phantom retype.
|||   * `discard` — the explicit linear consumer: a use-once model that isn't
||| threaded onward must be discarded (it can't fall out of scope).
|||
||| Higher-kinded over the `(in, out, executor, dtype, gradmode)` type
||| constructor, so instances are written unapplied (`Params Linear where …`),
||| sidestepping the relevant-index-in-instance-head erasure trap a
||| `Params (m : Type)` head hits. The per-layer impl pattern-matches the
||| constructor (binding param fields at their ω constructor quantity — free
||| to reuse *and* rebuild the record); never `.field`-projects a linear value.
||| Hand-written 3-liner per layer (the spike chose this over `%runElab`
||| derivation — see design-decisions.md).
public export
interface Params (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  params : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
           l i o ex dt g -> List SomeParam
  reflect : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
            (1 _ : l i o ex dt g) -> LPair (!* (List SomeParam)) (l i o ex dt g)
  castGrad : {0 ex : Executor} -> {0 dt : DType} -> {0 g, g' : GradMode} -> {0 i, o : Nat} ->
             (1 _ : l i o ex dt g) -> l i o ex dt g'
  discard : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
            (1 _ : l i o ex dt g) -> L IO ()

||| `Module`: the batched-first `forward` on top of `Params`. `forward`
||| consumes the model and returns the (unrestricted) output tensor wrapped in
||| the `(!*)` bang (`MkBang`) — so it can ride the linear return pair — beside
||| the rebuilt model. Tensors stay unrestricted (reverse-mode AD shares them).
||| No `idris_crash` batched-forward hole: only `Module` layers (batched-first
||| by construction) can enter a `Seq`. The model's `g` and the activation's
||| `g` are one and the same — a `WithGrad` model maps a `WithGrad` activation
||| to a `WithGrad` result (training), a `NoGrad` model maps `NoGrad → NoGrad`
||| (genuinely tape-free inference); no `retypeGrad` in the body.
public export
interface Params l => Module (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  forward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i, o, b : Nat} ->
            (1 _ : l i o ex dt g) -> Tensor [b, i] ex dt g ->
            L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (l i o ex dt g))

||| A frozen model: a marker wrapper produced by `freeze`. Freezing flips the
||| C-side `requires_grad` of every param off (the optimizer then skips them —
||| gradients still flow THROUGH for downstream trainable layers, the fine-tune
||| backbone pattern). Its field is **linear** (`1 _`) so it can hold a
||| linearly-produced model (an ω field would demand an unrestricted value and
||| reject the linear handle); `unfreeze` consumes it to recover the threadable
||| model. (Optimizer-LR freezing via `groupOf … LR 0` is the other route.)
public export
record Frozen (m : Type) where
  constructor MkFrozen
  1 unFrozen : m

||| `eval`: consume a trainable model, flip C `requires_grad` off for every
||| param, and return the retyped `NoGrad` model (still linear, so the
||| inference handle threads through `forward`). The result runs genuinely
||| tape-free, and the optimiser can't accept it (it needs a `WithGrad` loss,
||| which a `NoGrad` model can't produce). Generic over any `Params` — it
||| reflects the param list rather than pattern-matching `l`. `l` is a
||| *relevant* implicit (method dispatch needs it). Inverse: `trainable`.
export
eval : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
       {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
       UserExecutorTraining ex => Params l =>
       (1 _ : l i o ex dt WithGrad) -> L IO {use=1} (l i o ex dt NoGrad)
eval m = do
  let (MkBang ps # m') = reflect m
  traverse_ (\p => liftIO1 (primIO (primSetRequiresGrad {ex} p.paramPtr 0))) ps
  pure1 (castGrad m')

||| `freeze`: consume the model, flip C `requires_grad` off for every param
||| (grads still flow THROUGH for downstream trainable layers — the fine-tune-
||| backbone pattern), and return it wrapped in `Frozen`. Same grad-mode in and
||| out; `unfreeze` inverts. Where `freeze` keeps a backbone inside a trainable
||| graph (grads flow THROUGH), `eval` takes the whole model out of training.
export
freeze : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         UserExecutorTraining ex => Params l =>
         (1 _ : l i o ex dt g) -> L IO {use=1} (Frozen (l i o ex dt g))
freeze m = do
  let (MkBang ps # m') = reflect m
  traverse_ (\p => liftIO1 (primIO (primSetRequiresGrad {ex} p.paramPtr 0))) ps
  pure1 (MkFrozen m')

||| `unfreeze`: flip `requires_grad` back on and unwrap `Frozen`.
export
unfreeze : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
           UserExecutorTraining ex => Params l =>
           (1 _ : Frozen (l i o ex dt g)) -> L IO {use=1} (l i o ex dt g)
unfreeze (MkFrozen m) = do
  let (MkBang ps # m') = reflect m
  traverse_ (\p => liftIO1 (primIO (primSetRequiresGrad {ex} p.paramPtr 1))) ps
  pure1 m'

||| `trainable`: inverse of `eval` — flip `requires_grad` back on and retype
||| `NoGrad → WithGrad`, making an inference model trainable again.
export
trainable : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
            {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
            UserExecutorTraining ex => Params l =>
            (1 _ : l i o ex dt NoGrad) -> L IO {use=1} (l i o ex dt WithGrad)
trainable m = do
  let (MkBang ps # m') = reflect m
  traverse_ (\p => liftIO1 (primIO (primSetRequiresGrad {ex} p.paramPtr 1))) ps
  pure1 (castGrad m')
