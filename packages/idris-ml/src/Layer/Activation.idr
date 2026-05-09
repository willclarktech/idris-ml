module Layer.Activation

import Data.Vect

import Device
import Layer.Core
import Variable


----------------------------------------------------------------------
-- Activation — typed-surface activation layer (Path C)
----------------------------------------------------------------------
--
-- Pure tensor-level dispatch: the activation kind is a simple tag,
-- and `applyVar` calls the matching prim__* via the Variable wrappers.
-- No scalar fallback, no `ActivationFunction` wrapping (that machinery
-- exists for the V1 `applyGeneric` Double-evaluation path, which
-- doesn't apply to the typed surface).

public export
data ActivationKind
  = ATanh
  | ASigmoid
  | ARelu
  | AGelu
  | ASilu
  | ALeakyRelu Double  -- slope

public export
data ActivationState : Nat -> Nat -> (0 _ : Device) -> Type where
  MkActivation : ActivationKind -> ActivationState n n d


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

%default partial

public export
LayerLike ActivationState where
  applyVar st@(MkActivation ATanh)         input = (st, ttanh input)
  applyVar st@(MkActivation ASigmoid)      input = (st, tsigmoid input)
  applyVar st@(MkActivation ARelu)         input = (st, trelu input)
  applyVar st@(MkActivation AGelu)         input = (st, tgelu input)
  applyVar st@(MkActivation ASilu)         input = (st, tsilu input)
  applyVar st@(MkActivation (ALeakyRelu s)) input = (st, tleakyRelu s input)

  -- Activation primitives are shape-polymorphic (operate elementwise),
  -- so the batched forward is identical to the single-sample form —
  -- just typed at `Variable [b, n] d` instead of `Variable [n] d`.
  applyVarBatch st@(MkActivation ATanh)         input = (st, ttanh input)
  applyVarBatch st@(MkActivation ASigmoid)      input = (st, tsigmoid input)
  applyVarBatch st@(MkActivation ARelu)         input = (st, trelu input)
  applyVarBatch st@(MkActivation AGelu)         input = (st, tgelu input)
  applyVarBatch st@(MkActivation ASilu)         input = (st, tsilu input)
  applyVarBatch st@(MkActivation (ALeakyRelu s)) input = (st, tleakyRelu s input)

  layerPrefix _ = "act"


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
tanhLayer : ActivationState n n d
tanhLayer = MkActivation ATanh

export
sigmoidLayer : ActivationState n n d
sigmoidLayer = MkActivation ASigmoid

export
reluLayer : ActivationState n n d
reluLayer = MkActivation ARelu

export
geluLayer : ActivationState n n d
geluLayer = MkActivation AGelu

export
siluLayer : ActivationState n n d
siluLayer = MkActivation ASilu

export
leakyReluLayer : Double -> ActivationState n n d
leakyReluLayer slope = MkActivation (ALeakyRelu slope)

||| Wrap an activation in `AnyLayer` for use in a `Network` chain.
export
tanhLayerAny : AnyLayer n n d
tanhLayerAny = MkAnyLayer ActivationState tanhLayer

export
sigmoidLayerAny : AnyLayer n n d
sigmoidLayerAny = MkAnyLayer ActivationState sigmoidLayer

export
reluLayerAny : AnyLayer n n d
reluLayerAny = MkAnyLayer ActivationState reluLayer

export
geluLayerAny : AnyLayer n n d
geluLayerAny = MkAnyLayer ActivationState geluLayer

export
siluLayerAny : AnyLayer n n d
siluLayerAny = MkAnyLayer ActivationState siluLayer

export
leakyReluLayerAny : Double -> AnyLayer n n d
leakyReluLayerAny slope = MkAnyLayer ActivationState (leakyReluLayer slope)
