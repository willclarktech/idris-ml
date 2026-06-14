||| `Residual` — a skip connection wrapping a same-width sublayer:
||| `forward x = x + sublayer(x)`. Demonstrates Module composition where a
||| layer holds another layer (the sublayer's `Module`+`Params` dicts are
||| packed in the constructor, same relevant-implicit treatment as `Seq`'s
||| `(::)`). The wrapped layer must map `n → n` (so the add type-checks).
module Nn.Residual

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| A residual block around a same-width sublayer `l n n`.
public export
data Residual : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkResidual : {n : Nat} ->
               {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
               (Module l, Params l) =>
               l n n ex dt g -> Residual n n ex dt g

public export
Module Residual where
  forward (MkResidual sub) x = do
    fx <- forward sub x
    tadd x fx

public export
Params Residual where
  params (MkResidual sub)   = params sub
  castGrad (MkResidual sub) = MkResidual (castGrad sub)

||| Wrap a sublayer in a residual connection.
public export
residual : {n : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
           (Module l, Params l) => l n n ex dt g -> Residual n n ex dt g
residual = MkResidual

----------------------------------------------------------------------
-- Linear-resource surface
----------------------------------------------------------------------

||| `ResidualL` — the linear-resource counterpart, holding its same-width
||| sublayer in a **linear `(1 _)`** field (so the rebuilt sublayer returned
||| by `forwardL` is accepted) plus the element's `ModuleL` dict. The
||| composite-with-one-sublayer exemplar (cf. `SeqL` for the list case).
public export
data ResidualL : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkResidualL : {n : Nat} ->
                {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
                ModuleL l =>
                (1 _ : l n n ex dt g) -> ResidualL n n ex dt g

||| `forward x = x + sublayer(x)`, threading the sublayer linearly. `x` is an
||| unrestricted tensor, so it feeds both the sublayer and the add.
public export
ModuleL ResidualL where
  forwardL (MkResidualL sub) x = do
    (MkBang fx # sub') <- forwardL sub x
    y <- liftIO1 (tadd x fx)
    pure1 (MkBang y # MkResidualL sub')
  reflectL (MkResidualL sub) =
    let (MkBang ps # sub') = reflectL sub in MkBang ps # MkResidualL sub'
  castGradL (MkResidualL sub) = MkResidualL (castGradL sub)
  discardL (MkResidualL sub)  = discardL sub

||| Wrap a sublayer in a linear residual connection.
public export
residualL : {n : Nat} ->
            {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
            ModuleL l => (1 _ : l n n ex dt g) -> ResidualL n n ex dt g
residualL = MkResidualL
