||| `SeqL` — the **linear-resource** counterpart of `Nn.Seq`, threading a
||| model linearly through `Control.Linear.LIO.L IO`. Each `(::)` packs the
||| element's `ModuleL` dict and existentially hides the intermediate width
||| `h`; both the layer field and the tail field are **linear `(1 _)`** so the
||| rebuilt linear sub-models (`l' :: rest'`) returned by `forwardL` are
||| accepted. (Leaf param fields stay ω — multiplicity differs by role; see
||| `docs/develop/linear-types-and-effects.md`.)
|||
||| `SeqL` is itself a `ModuleL`, so it nests inside another `SeqL` like any
||| layer. This is the composite half of the linear vertical slice — the
||| existential-under-linearity case v0.8's checker is weakest on, proven
||| viable by the Phase 0 spike.
|||
||| Coexists with the IO `Nn.Seq` during the migration; constructors live in
||| their own module to avoid `Nil`/`(::)` overload clashes with `Seq`.
module Nn.SeqL

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| A linear sequence of `ModuleL` layers from `i` to `o`. The intermediate
||| width `h` is hidden by `(::)`; both fields are linear so threaded sub-
||| models re-pack cleanly.
public export
data SeqL : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  Nil  : SeqL i i ex dt g
  (::) : {h : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         ModuleL l =>
         (1 _ : l i h ex dt g) -> (1 _ : SeqL h o ex dt g) -> SeqL i o ex dt g

||| Right-associative chain alias for `(::)`.
public export
(~~>) : {h : Nat} ->
        {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
        ModuleL l =>
        (1 _ : l i h ex dt g) -> (1 _ : SeqL h o ex dt g) -> SeqL i o ex dt g
(~~>) = (::)

export infixr 5 ~~>

||| Run a batched activation through the whole chain, left to right, threading
||| each layer linearly. The output tensor rides the linear pair under `(!*)`.
export
forwardSeqL : {0 ex : Executor} -> Backend ex dt => {i, o, b : Nat} -> {0 g : GradMode} ->
              (1 _ : SeqL i o ex dt g) -> Tensor [b, i] ex dt g ->
              L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (SeqL i o ex dt g))
forwardSeqL Nil x         = pure1 (MkBang x # Nil)
forwardSeqL (l :: rest) x = do
  (MkBang y # l')    <- forwardL l x
  (MkBang z # rest') <- forwardSeqL rest y
  pure1 (MkBang z # (l' :: rest'))

||| Concatenated params of the chain, threaded without consuming numerically.
export
reflectSeqL : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
              (1 _ : SeqL i o ex dt g) -> LPair (!* (List SomeParam)) (SeqL i o ex dt g)
reflectSeqL Nil         = MkBang [] # Nil
reflectSeqL (l :: rest) =
  let (MkBang ps # l')       = reflectL l
      (MkBang restPs # rest') = reflectSeqL rest in
  MkBang (ps ++ restPs) # (l' :: rest')

||| Field-wise grad-mode retype of the whole chain.
export
castGradSeqL : {0 ex : Executor} -> {0 dt : DType} -> {0 g, g' : GradMode} -> {0 i, o : Nat} ->
               (1 _ : SeqL i o ex dt g) -> SeqL i o ex dt g'
castGradSeqL Nil         = Nil
castGradSeqL (l :: rest) = castGradL l :: castGradSeqL rest

||| Explicitly discard every element of the chain.
export
discardSeqL : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {0 i, o : Nat} ->
              (1 _ : SeqL i o ex dt g) -> L IO ()
discardSeqL Nil         = pure ()
discardSeqL (l :: rest) = do
  discardL l
  discardSeqL rest

||| `SeqL` is a `ParamsL` + `ModuleL`: lets a `SeqL` nest inside another.
public export
ParamsL SeqL where
  reflectL  = reflectSeqL
  castGradL = castGradSeqL
  discardL  = discardSeqL

public export
ModuleL SeqL where
  forwardL = forwardSeqL
