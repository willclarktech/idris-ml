||| `GCast` — grad-mode retype + param traversal for *any* `g`-indexed
||| type, and (next increment) a `%runElab` deriver that generates it.
|||
||| `Nn.Module.Params` is leaf-kind only (`Nat -> Nat -> Executor -> DType
||| -> GradMode -> Type`): it fits `Linear`/`LayerNorm`/… but not the
||| composite model records (`BertModelState`, `Gpt2BlockState`, …) which
||| carry many config `Nat`s. Those needed hand-written field-wise
||| `castGrad`/`params` cascades (see `Transformers.Bert`). `GCast`
||| generalizes over the **`g`-applied form** `f : GradMode -> Type` — a
||| record `R a … z g` partially applied to everything *except* `g` is a
||| `GradMode -> Type` regardless of how many leading params it has — so a
||| single interface fits leaves AND composites.
|||
||| Two methods, the two halves `eval`/`trainable` need:
|||   * `gcastGrad : f g -> f g'` — retype the erased phantom `g` (pure;
|||     operationally `id`, field-wise `retypeGrad`).
|||   * `gparams   : f g -> List SomeParam` — collect every leaf param
|||     (for the C-side `requires_grad` flip).
|||
||| Foundation here (interface + `Tensor` leaf + `Params` bridge); the
||| `%runElab gcast` deriver for composites lands next (TODO: "%runElab
||| deriver for Params / castGrad"). Until then a composite gets a
||| hand-written `GCast` instance (3 lines, reusing `gcastGrad`/`gparams`
||| on each field — uniform whether the field is a leaf Nn layer or a
||| nested composite).
module Nn.Derive

import public Nn.Module

import Data.Vect

import Executor
import Tensor
import Nn.Linear
import Nn.LayerNorm
import Nn.Embedding
import Nn.RmsNorm

%default total

||| Grad-mode-structured: a type `f : GradMode -> Type` whose `g`-bearing
||| leaves can be retyped (`g -> g'`) and collected as params.
public export
interface GCast (0 f : (0 _ : GradMode) -> Type) where
  ||| Retype the erased phantom grad-mode of every leaf. Pure.
  gcastGrad : {0 g, g' : GradMode} -> f g -> f g'
  ||| Every leaf param, dtype/shape-erased into `SomeParam`.
  gparams : {0 g : GradMode} -> f g -> List SomeParam

||| A `Tensor` (applied to everything but `g`) is a `GCast` leaf: retype
||| the phantom, and it is exactly one param.
public export
GCast (Tensor dims ex dt) where
  gcastGrad = retypeGrad
  gparams t = [toParam t]

-- Leaf Nn layers as `GCast` (so a composite's `GCast` recurses uniformly
-- via `gcastGrad`/`gparams` over leaf-layer AND nested-composite fields
-- alike). These mirror the leaf `Params.castGrad`/`params` — a generic
-- `Params l => GCast (l i o ex dt)` bridge would be cleaner but hits an
-- erased-instance-param ("l is not accessible") wall; the `%runElab`
-- deriver will subsume these (and the leaf `Params` instances) later.

public export
GCast (Linear i o ex dt) where
  gcastGrad (MkLinear w b) = MkLinear (retypeGrad w) (retypeGrad b)
  gparams (MkLinear w b) = [toParam w, toParam b]

public export
GCast (LayerNorm n n ex dt) where
  gcastGrad (MkLayerNorm g b) = MkLayerNorm (retypeGrad g) (retypeGrad b)
  gparams (MkLayerNorm g b) = [toParam g, toParam b]

public export
GCast (Embedding vocab embedDim ex dt) where
  gcastGrad (MkEmbedding w) = MkEmbedding (retypeGrad w)
  gparams (MkEmbedding w) = [toParam w]

public export
GCast (RmsNorm n n ex dt) where
  gcastGrad (MkRmsNorm w) = MkRmsNorm (retypeGrad w)
  gparams (MkRmsNorm w) = [toParam w]
