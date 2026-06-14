||| `Residual` — a skip connection wrapping a same-width sublayer:
||| `forward x = x + sublayer(x)`. Demonstrates Module composition where a
||| layer holds another layer (the sublayer's `Module`+`Params` dicts are
||| packed in the constructor, same relevant-implicit treatment as `Seq`'s
||| `(::)`). The wrapped layer must map `n → n` (so the add type-checks).
module Nn.Residual

import Data.Vect

import Executor
import Tensor
import Nn.Module

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
  params (MkResidual sub) = params sub
  castGrad (MkResidual sub) = MkResidual (castGrad sub)

||| Wrap a sublayer in a residual connection.
public export
residual : {n : Nat} ->
           {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
           (Module l, Params l) => l n n ex dt g -> Residual n n ex dt g
residual = MkResidual
