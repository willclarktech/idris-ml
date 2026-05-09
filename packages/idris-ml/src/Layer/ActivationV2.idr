module Layer.ActivationV2

import Data.Vect

import Device
import Layer.CoreV2
import Variable


----------------------------------------------------------------------
-- ActivationV2 — typed-surface activation layer (Path C)
----------------------------------------------------------------------
--
-- Pure tensor-level dispatch: the activation kind is a simple tag,
-- and `applyTVar` calls the matching prim__* via the TVar wrappers.
-- No scalar fallback, no `ActivationFunction` wrapping (that machinery
-- exists for the V1 `applyGeneric` Double-evaluation path, which
-- doesn't apply to the typed surface).

public export
data ActivationKindV2
  = ATanh
  | ASigmoid
  | ARelu
  | AGelu
  | ASilu
  | ALeakyRelu Double  -- slope

public export
data ActivationStateV2 : Nat -> Nat -> (0 _ : Device) -> Type where
  MkActivationV2 : ActivationKindV2 -> ActivationStateV2 n n d


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

%default partial

public export
LayerLikeV2 ActivationStateV2 where
  applyTVar st@(MkActivationV2 ATanh)         input = (st, ttanh input)
  applyTVar st@(MkActivationV2 ASigmoid)      input = (st, tsigmoid input)
  applyTVar st@(MkActivationV2 ARelu)         input = (st, trelu input)
  applyTVar st@(MkActivationV2 AGelu)         input = (st, tgelu input)
  applyTVar st@(MkActivationV2 ASilu)         input = (st, tsilu input)
  applyTVar st@(MkActivationV2 (ALeakyRelu s)) input = (st, tleakyRelu s input)

  -- Activation primitives are shape-polymorphic (operate elementwise),
  -- so the batched forward is identical to the single-sample form —
  -- just typed at `TVar [b, n] d` instead of `TVar [n] d`.
  applyTVarBatch st@(MkActivationV2 ATanh)         input = (st, ttanh input)
  applyTVarBatch st@(MkActivationV2 ASigmoid)      input = (st, tsigmoid input)
  applyTVarBatch st@(MkActivationV2 ARelu)         input = (st, trelu input)
  applyTVarBatch st@(MkActivationV2 AGelu)         input = (st, tgelu input)
  applyTVarBatch st@(MkActivationV2 ASilu)         input = (st, tsilu input)
  applyTVarBatch st@(MkActivationV2 (ALeakyRelu s)) input = (st, tleakyRelu s input)

  layerPrefixV2 _ = "actV2"


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
tanhLayerV2 : ActivationStateV2 n n d
tanhLayerV2 = MkActivationV2 ATanh

export
sigmoidLayerV2 : ActivationStateV2 n n d
sigmoidLayerV2 = MkActivationV2 ASigmoid

export
reluLayerV2 : ActivationStateV2 n n d
reluLayerV2 = MkActivationV2 ARelu

export
geluLayerV2 : ActivationStateV2 n n d
geluLayerV2 = MkActivationV2 AGelu

export
siluLayerV2 : ActivationStateV2 n n d
siluLayerV2 = MkActivationV2 ASilu

export
leakyReluLayerV2 : Double -> ActivationStateV2 n n d
leakyReluLayerV2 slope = MkActivationV2 (ALeakyRelu slope)

||| Wrap an activation in `AnyLayerV2` for use in a `NetworkV2` chain.
export
tanhLayerV2Any : AnyLayerV2 n n d
tanhLayerV2Any = MkAnyLayerV2 ActivationStateV2 tanhLayerV2

export
sigmoidLayerV2Any : AnyLayerV2 n n d
sigmoidLayerV2Any = MkAnyLayerV2 ActivationStateV2 sigmoidLayerV2

export
reluLayerV2Any : AnyLayerV2 n n d
reluLayerV2Any = MkAnyLayerV2 ActivationStateV2 reluLayerV2

export
geluLayerV2Any : AnyLayerV2 n n d
geluLayerV2Any = MkAnyLayerV2 ActivationStateV2 geluLayerV2

export
siluLayerV2Any : AnyLayerV2 n n d
siluLayerV2Any = MkAnyLayerV2 ActivationStateV2 siluLayerV2

export
leakyReluLayerV2Any : Double -> AnyLayerV2 n n d
leakyReluLayerV2Any slope = MkAnyLayerV2 ActivationStateV2 (leakyReluLayerV2 slope)
