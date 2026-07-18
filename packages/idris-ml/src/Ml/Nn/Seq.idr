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
module Ml.Nn.Seq

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Module
import Ml.Tensor

%default total

||| Chain-compatibility witness: the sole instance requires the layer's
||| out-dim to equal the next chain element's in-dim. It exists for error
||| quality only. With a single shared `h` index on `(::)`, a mis-sized chain
||| surfaces as the opaque `Can't find an implementation for Module ?l`
||| (higher-order unification postpones the layer-constructor inversion, and
||| the failed `Module` search wins error reporting — the 256-vs-128 conflict
||| is never printed). Splitting the index into `h`/`h'` and tying the halves
||| with this searchable witness makes the same mistake fail as
||| `Can't find an implementation for ChainFits 256 128` — both dims named.
||| Gate: `make test-integration-typegate-seq-shape`.
public export
interface ChainFits (layerOut : Nat) (nextIn : Nat) where
  constructor MkChainFits
  ||| The equality the witness carries (erased; `forwardSeq` rewrites the
  ||| activation's width across it).
  0 chainFitsPrf : layerOut = nextIn

||| The sole way to satisfy `ChainFits`: the two dims are equal. Provided as
||| a `%defaulthint` rather than a plain instance because default hints may
||| UNIFY an undetermined neighbour dim — identity layers (activations,
||| dropout, pools) receive their widths through this joint — while a plain
||| instance search refuses to bind goal metavariables and broke that
||| inference (`ChainFits (C1 * ConvOutDim ...) ?h'` in SeqClassify). A
||| genuinely mismatched pair still fails, naming both numbers.
%defaulthint
public export
chainFitsRefl : {n : Nat} -> ChainFits n n
chainFitsRefl = MkChainFits Refl

||| A sequence of `Module` layers from `i` to `o`. The intermediate width is
||| hidden by `(::)`; both fields are linear so threaded sub-models re-pack
||| cleanly. The layer's out-dim `h` and the tail's in-dim `h'` are separate
||| indices tied by `ChainFits` (see above — error quality only; the sole
||| instance forces `h = h'`).
public export
data Seq : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  Nil  : Seq i i ex dt g
  (::) : {h, h' : Nat} ->
         {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
         Module l => ChainFits h h' =>
         (1 _ : l i h ex dt g) -> (1 _ : Seq h' o ex dt g) -> Seq i o ex dt g

||| Right-associative chain alias for `(::)`.
public export
(~~>) : {h, h' : Nat} ->
        {l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type} ->
        Module l => ChainFits h h' =>
        (1 _ : l i h ex dt g) -> (1 _ : Seq h' o ex dt g) -> Seq i o ex dt g
(~~>) = (::)

public export infixr 5 ~~>

||| Run a batched activation through the whole chain, left to right, threading
||| each layer linearly. The output tensor rides the linear pair under `(!*)`.
export
forwardSeq : {0 ex : Executor} -> Backend ex dt => {i, o, b : Nat} -> {0 g : GradMode} ->
             (1 _ : Seq i o ex dt g) -> Tensor [b, i] ex dt g ->
             L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (Seq i o ex dt g))
forwardSeq Nil x                    = pure1 (MkBang x # Nil)
forwardSeq ((::) {h} {h'} l rest) x = do
  (MkBang y # l')    <- forward l x
  -- Carry the activation across the ChainFits-tied width split (h = h').
  let y' = replace {p = \n => Tensor [b, n] ex dt g} (chainFitsPrf {layerOut=h} {nextIn=h'}) y
  (MkBang z # rest') <- forwardSeq rest y'
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
