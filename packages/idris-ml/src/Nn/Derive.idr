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

import Data.List
import Data.Vect

import Language.Reflection.Util

import Executor
import Nn.Embedding
import Nn.LayerNorm
import Nn.Linear
import public Nn.Module
import Nn.RmsNorm
import Tensor

%language ElabReflection

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
  gparams (MkLinear w b)   = [toParam w, toParam b]

public export
GCast (LayerNorm n n ex dt) where
  gcastGrad (MkLayerNorm g b) = MkLayerNorm (retypeGrad g) (retypeGrad b)
  gparams (MkLayerNorm g b)   = [toParam g, toParam b]

public export
GCast (Embedding vocab embedDim ex dt) where
  gcastGrad (MkEmbedding w) = MkEmbedding (retypeGrad w)
  gparams (MkEmbedding w)   = [toParam w]

public export
GCast (RmsNorm n n ex dt) where
  gcastGrad (MkRmsNorm w) = MkRmsNorm (retypeGrad w)
  gparams (MkRmsNorm w)   = [toParam w]

----------------------------------------------------------------------
-- `%runElab` deriver
----------------------------------------------------------------------

||| Factory for a `GCast` dictionary from its two methods. The derived
||| instances build their `%hint` via this, so we never spell the
||| compiler-generated interface-constructor name (mirrors
||| `Language.Reflection.Derive.mkEq` / `mkDecEq`).
public export %inline
mkGCast :
     {0 f : (0 _ : GradMode) -> Type}
  -> (cg : {0 g, g' : GradMode} -> f g -> f g')
  -> (gp : {0 g : GradMode} -> f g -> List SomeParam)
  -> GCast f
mkGCast = %runElab check (var $ singleCon "GCast")

||| Per-field `gcastGrad` transform: mentions-`g` direct → `gcastGrad`;
||| `Vect`/`Maybe` of such → `map gcastGrad`; otherwise pass through.
public export
gcArg : (gName : Name) -> BoundArg 1 Explicit -> TTImp
gcArg gName (BA a [x] _) =
  if not (rec [gName] a.type) then var x
  else case unApp a.type of
    (IVar _ h, [_,_]) =>
      if nameStr h == "Vect"  then `(map gcastGrad ~(var x)) else `(gcastGrad ~(var x))
    (IVar _ h, [_])   =>
      if nameStr h == "Maybe" then `(map gcastGrad ~(var x)) else `(gcastGrad ~(var x))
    _ => `(gcastGrad ~(var x))

||| Per-field `gparams` transform: mentions-`g` direct → `gparams`;
||| `Vect` → `concatMap gparams . toList`; `Maybe` → `foldMap gparams`;
||| otherwise contributes no params.
public export
gpArg : (gName : Name) -> BoundArg 1 Explicit -> TTImp
gpArg gName (BA a [x] _) =
  if not (rec [gName] a.type) then `(the (List SomeParam) Nil)
  else case unApp a.type of
    (IVar _ h, [_,_]) =>
      if nameStr h == "Vect"  then `(concatMap gparams (toList ~(var x))) else `(gparams ~(var x))
    (IVar _ h, [_])   =>
      if nameStr h == "Maybe" then `(foldMap gparams ~(var x)) else `(gparams ~(var x))
    _ => `(gparams ~(var x))

||| Concatenate per-field param lists with `(++)`, base `[]`.
public export
gpRhs : SnocList TTImp -> TTImp
gpRhs sx = foldr (\e,acc => `(~(e) ++ ~(acc))) `(the (List SomeParam) Nil) (sx <>> [])

||| The three `TopLevel`s (claim + def for `gcastGrad…`, `gparams…`, and
||| the `%hint impl…`) implementing `GCast` for a single-constructor
||| record `ti`/`con` whose final parameter is the grad-mode `gName`,
||| with remaining parameters `nonG`. Pure TTImp construction — safe to
||| call across package boundaries from an elaborator script.
public export
gcastTLs : (ti : TypeInfo) -> Con ti.arty ti.args -> (gName : Name) -> (nonG : List Name) -> List TopLevel
gcastTLs ti con gName nonG =
  let g0      = UN (Basic "gcDeriveG0")
      g1      = UN (Basic "gcDeriveG1")
      gradTy  = var "GradMode"
      headTy  = appNames ti.name nonG
      ratG0   = appNames ti.name (nonG ++ [g0])
      ratG1   = appNames ti.name (nonG ++ [g1])
      nonGI   = map erasedImplicit nonG
      g0bind  = MkArg M0 ImplicitArg (Just g0) gradTy
      g1bind  = MkArg M0 ImplicitArg (Just g1) gradTy
      gcastNm = funName ti "gcastGrad"
      gparmNm = funName ti "gparams"
      implNm  = implName ti "GCast"
      gcastTy = piAll `(~(ratG0) -> ~(ratG1)) (nonGI ++ [g0bind, g1bind])
      gparmTy = piAll `(~(ratG0) -> List SomeParam) (nonGI ++ [g0bind])
      implTy  = piAll `(GCast ~(headTy)) nonGI
      gcastCl = mapArgs explicit (\cx => `(~(var gcastNm) ~(cx))) (gcArg gName) con
      gparmCl = accumArgs explicit (\cx => `(~(var gparmNm) ~(cx))) gpRhs (gpArg gName) con
   in [ TL (simpleClaim Export gcastNm gcastTy) (def gcastNm [gcastCl])
      , TL (simpleClaim Export gparmNm gparmTy) (def gparmNm [gparmCl])
      , TL (implClaimVis Public implNm implTy)
           (def implNm [patClause (var implNm) `(mkGCast ~(var gcastNm) ~(var gparmNm))])
      ]

||| The elab-util deriving rule for `GCast` — a **pure** function from
||| the (already-introspected) `ParamTypeInfo` to the instance
||| `TopLevel`s. It splits off the record's final grad-mode parameter and
||| delegates to `gcastTLs`. Single-constructor records only; the final
||| type parameter must be the `GradMode g`.
|||
||| Usage (note: the rule passed to elab-util's `derive` must be a
||| *current-package* value, so wrap `GCastImpl` in a one-line local
||| function at the call site — an imported rule passed by value leaves a
||| stuck elaborator script. The Elab-monadic introspection runs inside
||| elab-util's `derive`, which is why this rule itself stays pure):
|||
||| ```idris
||| gcast : List Name -> ParamTypeInfo -> Res (List TopLevel)
||| gcast nms p = GCastImpl nms p
|||
||| %runElab derive `{MyRecord} [gcast]
||| ```
|||
||| Nested records must be derived before the composites that hold them.
||| `GradMode`, `SomeParam`, `GCast`, `mkGCast` and the `Vect`/`List`
||| combinators must be in scope at the call site.
public export
GCastImpl : List Name -> ParamTypeInfo -> Res (List TopLevel)
GCastImpl _ p =
  case (p.info.cons, reverse (toList p.info.argNames)) of
    ([con], (gName :: nonGrev)) => Right (gcastTLs p.info con gName (reverse nonGrev))
    ([_], [])                   => Left "deriveGCast: \{show p.info.name} has no type parameters"
    _                           => Left "deriveGCast: \{show p.info.name} must be a single-constructor record"
