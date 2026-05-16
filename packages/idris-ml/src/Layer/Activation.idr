module Layer.Activation

import Data.Vect

import Device
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Activation — typed-surface activation layer (Path C)
----------------------------------------------------------------------
--
-- Pure tensor-level dispatch: the activation kind is a simple tag,
-- and `applyVar` calls the matching prim__* via the Tensor wrappers.
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
data ActivationState : Nat -> Nat -> (0 _ : Device) -> (0 _ : GradMode) -> Type where
  MkActivation : ActivationKind -> ActivationState n n d g


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

%default partial

public export
LayerLike ActivationState where
  applyVar st@(MkActivation ATanh)         input = do out <- ttanh input;          pure (st, out)
  applyVar st@(MkActivation ASigmoid)      input = do out <- tsigmoid input;       pure (st, out)
  applyVar st@(MkActivation ARelu)         input = do out <- trelu input;          pure (st, out)
  applyVar st@(MkActivation AGelu)         input = do out <- tgelu input;          pure (st, out)
  applyVar st@(MkActivation ASilu)         input = do out <- tsilu input;          pure (st, out)
  applyVar st@(MkActivation (ALeakyRelu s)) input = do out <- tleakyRelu s input;  pure (st, out)

  -- Activation primitives are shape-polymorphic (operate elementwise),
  -- so the batched forward is identical to the single-sample form —
  -- just typed at `Tensor [b, n] d` instead of `Tensor [n] d`.
  applyVarBatch st@(MkActivation ATanh)         input = do out <- ttanh input;         pure (st, out)
  applyVarBatch st@(MkActivation ASigmoid)      input = do out <- tsigmoid input;      pure (st, out)
  applyVarBatch st@(MkActivation ARelu)         input = do out <- trelu input;         pure (st, out)
  applyVarBatch st@(MkActivation AGelu)         input = do out <- tgelu input;         pure (st, out)
  applyVarBatch st@(MkActivation ASilu)         input = do out <- tsilu input;         pure (st, out)
  applyVarBatch st@(MkActivation (ALeakyRelu s)) input = do out <- tleakyRelu s input; pure (st, out)

  layerPrefix _ = "act"

  -- Activation is stateless (no params); freeze/unfreeze just retypes.
  freezeLayer (MkActivation k) = pure (MkActivation k)
  unfreezeLayer (MkActivation k) = pure (MkActivation k)


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
tanhLayer : ActivationState n n d g
tanhLayer = MkActivation ATanh

export
sigmoidLayer : ActivationState n n d g
sigmoidLayer = MkActivation ASigmoid

export
reluLayer : ActivationState n n d g
reluLayer = MkActivation ARelu

export
geluLayer : ActivationState n n d g
geluLayer = MkActivation AGelu

export
siluLayer : ActivationState n n d g
siluLayer = MkActivation ASilu

export
leakyReluLayer : Double -> ActivationState n n d g
leakyReluLayer slope = MkActivation (ALeakyRelu slope)

||| Wrap an activation in `AnyLayer` for use in a `Network` chain.
export
tanhLayerAny : AnyLayer n n d g
tanhLayerAny = MkAnyLayer ActivationState tanhLayer

export
sigmoidLayerAny : AnyLayer n n d g
sigmoidLayerAny = MkAnyLayer ActivationState sigmoidLayer

export
reluLayerAny : AnyLayer n n d g
reluLayerAny = MkAnyLayer ActivationState reluLayer

export
geluLayerAny : AnyLayer n n d g
geluLayerAny = MkAnyLayer ActivationState geluLayer

export
siluLayerAny : AnyLayer n n d g
siluLayerAny = MkAnyLayer ActivationState siluLayer

export
leakyReluLayerAny : Double -> AnyLayer n n d g
leakyReluLayerAny slope = MkAnyLayer ActivationState (leakyReluLayer slope)
