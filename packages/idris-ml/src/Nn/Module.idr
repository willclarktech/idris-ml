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

||| A model component with a batched-first forward. `l`'s five indices are
||| (in, out, executor, dtype, gradmode). The model's `g` and the
||| activation's `g` are one and the same — a `WithGrad` model maps a
||| `WithGrad` activation to a `WithGrad` result (training), a `NoGrad`
||| model maps `NoGrad → NoGrad` (genuinely tape-free inference). No
||| `retypeGrad` in the body: params and activation already share `g`.
public export
interface Module (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  forward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i, o, b : Nat} ->
            l i o ex dt g -> Tensor [b, i] ex dt g -> IO (Tensor [b, o] ex dt g)

||| Parameter traversal for a model component `l`. Higher-kinded over the
||| `(in, out, executor, dtype)` type constructor (same kind as `Module`,
||| same precedent as the legacy `LayerLike`): instances are written
||| unapplied (`Params Linear where …`), sidestepping the
||| relevant-index-in-instance-head erasure trap that a `Params (m : Type)`
||| head hits. Every layer and `Seq` fits this shape; composite records are
||| scoped per-field (`groupOf model.actor`), never as a whole.
|||
||| Hand-written 3-liner per layer (the spike chose this over `%runElab`
||| derivation — see design-decisions.md).
public export
interface Params (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  params : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
           l i o ex dt g -> List SomeParam
  ||| Retype the model's grad-mode index (`g → g'`). Pure — `g` is an erased
  ||| phantom, so this is field-wise `retypeGrad`; per-layer only because the
  ||| record shape varies. The C handles are unchanged; `eval`/`trainable`
  ||| pair it with the runtime `requires_grad` flip. Non-param fields
  ||| (buffers, config) are carried verbatim.
  castGrad : {0 ex : Executor} -> {0 dt : DType} -> {0 g, g' : GradMode} -> {0 i, o : Nat} ->
             l i o ex dt g -> l i o ex dt g'

||| A frozen model: a marker wrapper produced by `freeze`. Freezing
||| flips the C-side `requires_grad` of every param off (the optimizer
||| then skips them — gradients still flow THROUGH for downstream
||| trainable layers, the fine-tune backbone pattern). `Frozen` documents
||| intent at the type level; `unfreeze` flips them back. (Optimizer-LR
||| freezing via `groupOf … LR 0` is the other route.)
public export
record Frozen (m : Type) where
  constructor MkFrozen
  unFrozen : m

||| Freeze: flip `requires_grad` off for every param of `m`.
export
freeze : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         UserExecutorTraining ex => Params l =>
         l i o ex dt g -> IO (Frozen (l i o ex dt g))
freeze m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 0)) (params m)
  pure (MkFrozen m)

||| Unfreeze: flip `requires_grad` back on.
export
unfreeze : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
           UserExecutorTraining ex => Params l =>
           Frozen (l i o ex dt g) -> IO (l i o ex dt g)
unfreeze (MkFrozen m) = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 1)) (params m)
  pure m

||| Convert a trainable model into an inference one: flip C `requires_grad`
||| off for every param AND retype the model `WithGrad → NoGrad`. The result
||| runs genuinely tape-free, and the optimiser can't accept it (it needs a
||| `WithGrad` loss, which a `NoGrad` model can't produce). The inference
||| counterpart to `freeze` — where `freeze` keeps a backbone inside a
||| trainable graph (grads flow THROUGH), `eval` takes the whole model out
||| of training. Inverse: `trainable`.
export
eval : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
       {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
       UserExecutorTraining ex => Params l =>
       l i o ex dt WithGrad -> IO (l i o ex dt NoGrad)
eval m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 0)) (params m)
  pure (castGrad m)

||| Inverse of `eval`: flip `requires_grad` back on + retype `NoGrad →
||| WithGrad`, making an inference model trainable again.
export
trainable : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
            {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
            UserExecutorTraining ex => Params l =>
            l i o ex dt NoGrad -> IO (l i o ex dt WithGrad)
trainable m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 1)) (params m)
  pure (castGrad m)

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
