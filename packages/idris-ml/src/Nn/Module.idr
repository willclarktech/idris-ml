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
||| `GradMode` is OFF the model type: a layer owns its `WithGrad` params
||| by construction; `g` lives only on the activation `Tensor`. Mixed
||| precision is a `fit` config (no parallel `ModuleMixed` tree).
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

||| A model component with a batched-first forward. `l`'s four indices
||| are (in, out, executor, dtype) — NO `GradMode`; `forward` is
||| `g`-polymorphic on the activation tensor.
public export
interface Module (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type) where
  forward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i, o, b : Nat} ->
            l i o ex dt -> Tensor [b, i] ex dt g -> IO (Tensor [b, o] ex dt g)

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
interface Params (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type) where
  params : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
           l i o ex dt -> List SomeParam

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
freeze : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type} ->
         UserExecutorTraining ex => Params l =>
         l i o ex dt -> IO (Frozen (l i o ex dt))
freeze m = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 0)) (params m)
  pure (MkFrozen m)

||| Unfreeze: flip `requires_grad` back on.
export
unfreeze : {0 ex : Executor} -> {0 dt : DType} -> {0 i, o : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type} ->
           UserExecutorTraining ex => Params l =>
           Frozen (l i o ex dt) -> IO (l i o ex dt)
unfreeze (MkFrozen m) = do
  traverse_ (\p => primIO (primSetRequiresGrad {ex} p.paramPtr 1)) (params m)
  pure m
