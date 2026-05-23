module Layer.BitLinear

import Data.Vect

import Device
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
record BitLinearState (i : Nat) (o : Nat) (0 d : Device)
                      (0 paramDt : DType) (0 computeDt : DType) (0 g : GradMode) where
  constructor MkBitLinear
  weightT : Tensor [o, i] d Ternary NoGrad
  scaleT  : Tensor [o] d computeDt NoGrad
  biasT   : Tensor [o] d computeDt g


----------------------------------------------------------------------
-- Forward + lifecycle helpers
----------------------------------------------------------------------

-- `LayerLikeMixed.applyVarMixed`'s constraint list doesn't include
-- `UserDeviceQuant d` (the surface is generic over all layers), so
-- the BitLinear instance can't resolve it from the interface's auto-
-- implicit list. Until LayerLikeMixed grows a quant escape hatch,
-- BitLinear is constructed + driven via its standalone forward
-- helpers (`tBitlinearFwd`) at the network-construction site that
-- has `UserDeviceQuant d` in scope. The unfreezable bias edge is
-- exposed via a separate helper so the standard freeze/unfreeze
-- workflow still works.
||| Freeze the bias edge (weight + scale are already NoGrad by
||| construction). Linear in input — parallel to `LayerLike.freezeLayer`.
export
freezeBitLinear : {0 d : Device} -> UserDeviceTraining d =>
                  {i, o : Nat} -> {0 g : GradMode} ->
                  (1 _ : BitLinearState i o d Ternary cDt g) ->
                  IO (BitLinearState i o d Ternary cDt NoGrad)
freezeBitLinear (MkBitLinear w s b) = do
  b' <- weakenGrad b
  pure (MkBitLinear w s b')

||| Unfreeze the bias edge (weight + scale stay NoGrad). Linear in
||| input — parallel to `LayerLike.unfreezeLayer`.
export
unfreezeBitLinear : {0 d : Device} -> UserDeviceTraining d =>
                    {i, o : Nat} ->
                    (1 _ : BitLinearState i o d Ternary cDt NoGrad) ->
                    IO (BitLinearState i o d Ternary cDt WithGrad)
unfreezeBitLinear (MkBitLinear w s b) = do
  primIO (primSetRequiresGrad {d} b.tensorPtr 1)
  pure (MkBitLinear w s (retypeGrad b))

||| Run a BitLinear forward step on a `BitLinearState`. Mirrors
||| `LayerLikeMixed.applyVarMixed`'s shape (returns the unchanged
||| state and the output) so the call site looks consistent with
||| other mixed-precision layers.
export
applyBitLinear : {0 d : Device} -> UserDeviceQuant d =>
                 {i, o : Nat} -> {0 g : GradMode} ->
                 BitLinearState i o d Ternary cDt g ->
                 Tensor [i] d cDt g ->
                 IO (BitLinearState i o d Ternary cDt g, Tensor [o] d cDt g)
applyBitLinear st input = do
  out <- tBitlinearFwd st.weightT st.scaleT input st.biasT
  pure (st, out)


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Build a `BitLinearState i o d Ternary computeDt g` from already-
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
  {i, o : Nat} -> {0 d : Device} -> {0 cDt : DType} -> {0 g : GradMode} ->
  Tensor [o, i] d Ternary NoGrad ->
  Tensor [o] d cDt NoGrad ->
  Tensor [o] d cDt g ->
  BitLinearState i o d Ternary cDt g
bitLinearFromTensors w s b = MkBitLinear w s b
