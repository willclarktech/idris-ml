||| Heterogeneous sequential composition — the v1 successor to
||| `Network`/`OutputLayer`/`AnyLayer`. `Seq i o ex dt` is indexed by its
||| endpoints only; each `(::)` packs a layer's `Module`+`Params` dicts and
||| existentially hides the intermediate dimension `h`. The chain operator
||| `(~~>)` is a right-associative alias for `(::)` (roadmap decision 4);
||| list-literal sugar (`l1 :: l2 :: Nil`) coexists.
|||
||| `Seq` is itself a `Module` and `Params` (same `Nat -> Nat -> Executor ->
||| DType -> Type` kind), so a `Seq` nests inside another `Seq` like any
||| layer. No `idris_crash` batched-forward hole: only `Module` layers
||| (batched-first by construction) can enter a `Seq`.
module Nn.Seq

import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| A sequence of `Module` layers from `i` inputs to `o` outputs. The
||| intermediate width `h` is hidden by `(::)`; the constructor carries the
||| element's `Module` + `Params` dictionaries.
public export
data Seq : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  Nil  : Seq i i ex dt g
  -- `l` is bound as a *relevant* (non-erased) implicit so the element's
  -- type constructor stays accessible for `Module`/`Params` dispatch after
  -- the existential `h` is unpacked (the `AnyLayer` precedent — there `l`
  -- is an explicit constructor argument).
  (::) : {h : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         (Module l, Params l) =>
         l i h ex dt g -> Seq h o ex dt g -> Seq i o ex dt g

||| Right-associative chain alias for `(::)`.
public export
(~~>) : {h : Nat} ->
        {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
        (Module l, Params l) =>
        l i h ex dt g -> Seq h o ex dt g -> Seq i o ex dt g
(~~>) = (::)

export infixr 5 ~~>

||| Run a batched activation through the whole chain, left to right.
export
forwardSeq : {0 ex : Executor} -> Backend ex dt => {i, o, b : Nat} -> {0 g : GradMode} ->
             Seq i o ex dt g -> Tensor [b, i] ex dt g -> IO (Tensor [b, o] ex dt g)
forwardSeq Nil x = pure x
forwardSeq (l :: rest) x = do
  y <- forward l x
  forwardSeq rest y

||| A `Seq` is a `Module`: its `forward` is `forwardSeq`. Lets a `Seq` nest
||| inside another `Seq`.
public export
Module Seq where
  forward = forwardSeq

||| Parameters of a `Seq` are the concatenation of its elements' params, in
||| chain order.
public export
Params Seq where
  params Nil = []
  params (l :: rest) = params l ++ params rest
  castGrad Nil = Nil
  castGrad (l :: rest) = castGrad l :: castGrad rest
