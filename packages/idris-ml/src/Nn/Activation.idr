||| `Activation` — the stateless-layer exemplar on the v1 `Nn` surface.
||| Holds an `ActivationKind` tag; `forward` dispatches to the matching
||| elementwise C op. Activation prims are shape-polymorphic, so the
||| batched `[b,n]` forward is the same call as the `[n]` one. No params,
||| so `Params` is empty and there is nothing to freeze.
module Nn.Activation

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

public export
data ActivationKind
  = ATanh
  | ASigmoid
  | ARelu
  | AGelu
  | ASilu
  | ALeakyRelu Double  -- slope

||| Stateless activation layer (`i = o = n`).
public export
data Activation : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkActivation : ActivationKind -> Activation n n ex dt g

||| Params (stateless — empty param list). The tag `k` is bound at its ω
||| constructor quantity, so it feeds the rebuild.
public export
Params Activation where
  params (MkActivation k)   = []
  reflect (MkActivation k)  = MkBang [] # MkActivation k
  castGrad (MkActivation k) = MkActivation k
  discard (MkActivation _)  = pure ()

||| `Module` — dispatches to the `L IO` activation ops.
public export
Module Activation where
  forward (MkActivation k) x = do
    y <- the (L IO (Tensor [b, o] ex dt g)) $ case k of
           ATanh          => ttanhL x
           ASigmoid       => tsigmoidL x
           ARelu          => treluL x
           AGelu          => tgeluL x
           ASilu          => tsiluL x
           (ALeakyRelu s) => tleakyReluL s x
    pure1 (MkBang y # MkActivation k)

-- Constructors (no Init needed — stateless, registers nothing).
public export
tanhA : Activation n n ex dt g
tanhA = MkActivation ATanh

public export
sigmoidA : Activation n n ex dt g
sigmoidA = MkActivation ASigmoid

public export
reluA : Activation n n ex dt g
reluA = MkActivation ARelu

public export
geluA : Activation n n ex dt g
geluA = MkActivation AGelu

public export
siluA : Activation n n ex dt g
siluA = MkActivation ASilu

public export
leakyReluA : Double -> Activation n n ex dt g
leakyReluA slope = MkActivation (ALeakyRelu slope)
