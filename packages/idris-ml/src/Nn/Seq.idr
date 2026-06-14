||| `Seq` — heterogeneous sequential composition (the successor to
||| `Network`/`OutputLayer`/`AnyLayer`), threading a model linearly through
||| `Control.Linear.LIO.L IO`. `Seq i o ex dt g` is indexed by its endpoints
||| only; each `(::)` packs the element's `Module` dict and existentially hides
||| the intermediate width `h`. Both the layer field and the tail field are
||| **linear `(1 _)`** so the rebuilt linear sub-models (`l' :: rest'`) returned
||| by `forward` are accepted. (Leaf param fields stay ω — multiplicity differs
||| by role; see `docs/develop/linear-types-and-effects.md`.) The chain operator
||| `(~~>)` is a right-associative alias for `(::)`; list-literal sugar coexists.
|||
||| `Seq` is itself a `Module`, so it nests inside another `Seq` like any layer.
||| This is the composite case v0.8's checker is weakest on (existential under
||| linearity), proven viable by the Phase 0 spike. No `idris_crash` batched-
||| forward hole: only `Module` layers (batched-first by construction) enter.
module Nn.Seq

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| A sequence of `Module` layers from `i` to `o`. The intermediate width `h`
||| is hidden by `(::)`; both fields are linear so threaded sub-models re-pack
||| cleanly.
public export
data Seq : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  Nil  : Seq i i ex dt g
  (::) : {h : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         Module l =>
         (1 _ : l i h ex dt g) -> (1 _ : Seq h o ex dt g) -> Seq i o ex dt g

||| Right-associative chain alias for `(::)`.
public export
(~~>) : {h : Nat} ->
        {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
        Module l =>
        (1 _ : l i h ex dt g) -> (1 _ : Seq h o ex dt g) -> Seq i o ex dt g
(~~>) = (::)

public export infixr 5 ~~>

||| Run a batched activation through the whole chain, left to right, threading
||| each layer linearly. The output tensor rides the linear pair under `(!*)`.
export
forwardSeq : {0 ex : Executor} -> Backend ex dt => {i, o, b : Nat} -> {0 g : GradMode} ->
             (1 _ : Seq i o ex dt g) -> Tensor [b, i] ex dt g ->
             L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (Seq i o ex dt g))
forwardSeq Nil x         = pure1 (MkBang x # Nil)
forwardSeq (l :: rest) x = do
  (MkBang y # l')    <- forward l x
  (MkBang z # rest') <- forwardSeq rest y
  pure1 (MkBang z # (l' :: rest'))

||| Concatenated params of the chain (flat read, ω).
export
paramsSeq : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
            Seq i o ex dt g -> List SomeParam
paramsSeq Nil         = []
paramsSeq (l :: rest) = params l ++ paramsSeq rest

||| Concatenated params of the chain, threaded without consuming numerically.
export
reflectSeq : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
             (1 _ : Seq i o ex dt g) -> LPair (!* (List SomeParam)) (Seq i o ex dt g)
reflectSeq Nil         = MkBang [] # Nil
reflectSeq (l :: rest) =
  let (MkBang ps # l')       = reflect l
      (MkBang restPs # rest') = reflectSeq rest in
  MkBang (ps ++ restPs) # (l' :: rest')

||| Field-wise grad-mode retype of the whole chain.
export
castGradSeq : {0 ex : Executor} -> {0 dt : DType} -> {0 g, g' : GradMode} -> {0 i, o : Nat} ->
              (1 _ : Seq i o ex dt g) -> Seq i o ex dt g'
castGradSeq Nil         = Nil
castGradSeq (l :: rest) = castGrad l :: castGradSeq rest

||| Explicitly discard every element of the chain.
export
discardSeq : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
             (1 _ : Seq i o ex dt g) -> L IO ()
discardSeq Nil         = pure ()
discardSeq (l :: rest) = do
  discard l
  discardSeq rest

||| `Seq` is a `Params` + `Module`: lets a `Seq` nest inside another.
public export
Params Seq where
  params   = paramsSeq
  reflect  = reflectSeq
  castGrad = castGradSeq
  discard  = discardSeq

public export
Module Seq where
  forward = forwardSeq
