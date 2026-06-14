||| `Residual` — a skip connection wrapping a same-width sublayer:
||| `forward x = x + sublayer(x)`. Demonstrates Module composition where a
||| layer holds another layer in a **linear `(1 _)`** field (so the rebuilt
||| sublayer returned by `forward` is accepted) plus the element's `Module`
||| dict — the composite-with-one-sublayer exemplar (cf. `Seq` for the list
||| case). The wrapped layer must map `n → n` (so the add type-checks).
module Nn.Residual

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| A residual block around a same-width sublayer `l n n`, held linearly.
public export
data Residual : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkResidual : {n : Nat} ->
               {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
               Module l =>
               (1 _ : l n n ex dt g) -> Residual n n ex dt g

||| Params recurse into the sublayer.
public export
Params Residual where
  params (MkResidual sub) = params sub
  reflect (MkResidual sub) =
    let (MkBang ps # sub') = reflect sub in MkBang ps # MkResidual sub'
  castGrad (MkResidual sub) = MkResidual (castGrad sub)
  discard (MkResidual sub)  = discard sub

||| `forward x = x + sublayer(x)`, threading the sublayer linearly. `x` is an
||| unrestricted tensor, so it feeds both the sublayer and the add.
public export
Module Residual where
  forward (MkResidual sub) x = do
    (MkBang fx # sub') <- forward sub x
    y <- taddL x fx
    pure1 (MkBang y # MkResidual sub')

||| Wrap a sublayer in a residual connection.
public export
residual : {n : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
           Module l => (1 _ : l n n ex dt g) -> Residual n n ex dt g
residual = MkResidual
