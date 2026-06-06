module Layer.BitLinear

import Data.Vect

import Executor
import Layer.Core
import Layer.MixedCore
import Tensor


----------------------------------------------------------------------
-- BitLinear: BitNet b1.58 quantized linear (#411 B2)
----------------------------------------------------------------------
--
-- Weights are ternary {-1, 0, +1} encoded 4-per-byte (2 bits each)
-- with two's-complement codes (see DType.Core "Ternary" + design-
-- decisions.md "Per-backend ternary storage" for layout). The
-- per-output-row dequantization scale + bias are float-typed in
-- `computeDt`; the forward computes
--
--   y = (W_ternary .* scale[:, None]) @ x + bias
--
-- decoding W inline (tape) or via int8-cast (torch/mlx). The weight
-- is `NoGrad` by construction — BitNet b1.58 freezes the ternary
-- params; the bias edge is grad-mode-parametric so the rest of the
-- chain can still flow gradients.
--
-- `paramDt` is a phantom on the record. The weight field type pins
-- `Ternary` directly; carrying the slot at the record's kind lets
-- `BitLinearState` slot into `LayerLikeMixed` (which expects a six-
-- argument kind: i, o, d, paramDt, computeDt, g).

public export
record BitLinearState (i : Nat) (o : Nat) (0 ex : Executor)
                      (0 paramDt : DType) (0 computeDt : DType) (0 g : GradMode) where
  constructor MkBitLinear
  weightT : Tensor [o, i] ex Ternary NoGrad
  scaleT  : Tensor [o] ex computeDt NoGrad
  biasT   : Tensor [o] ex computeDt g


----------------------------------------------------------------------
-- LayerLikeMixed instance
----------------------------------------------------------------------
--
-- `applyVarMixed` is generic over paramDt at the interface level,
-- but `tBitlinearFwd`'s signature pins `Tensor [o, i] ex Ternary
-- NoGrad` as the weight slot — so the BitLinear instance only
-- typechecks when callers instantiate `paramDt = Ternary`. The
-- type-level guard is the field annotation in `BitLinearState`'s
-- `weightT`, not a constraint on the instance head.

%default partial

public export
LayerLikeMixed BitLinearState where
  applyVarMixed st input = do
    out <- tBitlinearFwd st.weightT st.scaleT input st.biasT
    pure (st, out)

  layerPrefixMixed _ = "bitlinear"

  freezeLayerMixed (MkBitLinear w s b) = do
    b' <- weakenGrad b
    pure (MkBitLinear w s b')

  unfreezeLayerMixed (MkBitLinear w s b) = do
    primIO (primSetRequiresGrad {ex} b.tensorPtr 1)
    pure (MkBitLinear w s (retypeGrad b))


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Build a `BitLinearState i o ex Ternary computeDt g` from already-
||| materialised weight + scale + bias tensors. Pre-packed weight bytes
||| come from `prim__allocBytes` + `prim__setByte`; the typical caller
||| is HF BitNet checkpoint loading (B4) or the oracle test (B2/#424).
|||
||| No paramId registration in this commit — the BitNet inference path
||| is the first user; checkpoint loading via `loadModel` will need a
||| separate registration helper that handles the Ternary lifecycle
||| (the param registry's gradient surface assumes float dtypes today).
||| Filed under the #411 follow-up.
export
bitLinearFromTensors :
  {i, o : Nat} -> {0 ex : Executor} -> {0 cDt : DType} -> {0 g : GradMode} ->
  Tensor [o, i] ex Ternary NoGrad ->
  Tensor [o] ex cDt NoGrad ->
  Tensor [o] ex cDt g ->
  BitLinearState i o ex Ternary cDt g
bitLinearFromTensors w s b = MkBitLinear w s b

||| Wrap a `BitLinearState` in `AnyLayerMixed` for use in a
||| `NetworkMixed`. The chained network uses the standard
||| `forwardVarMixed` pipeline; BitLinear slots in alongside
||| `mixedLinearLayerAny` etc.
export
bitLinearFromTensorsAny :
  {i, o : Nat} -> {0 ex : Executor} -> {0 cDt : DType} -> {0 g : GradMode} ->
  Tensor [o, i] ex Ternary NoGrad ->
  Tensor [o] ex cDt NoGrad ->
  Tensor [o] ex cDt g ->
  AnyLayerMixed i o ex Ternary cDt g
bitLinearFromTensorsAny w s b =
  MkAnyLayerMixed BitLinearState (bitLinearFromTensors w s b)
