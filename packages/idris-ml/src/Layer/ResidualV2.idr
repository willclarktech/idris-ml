module Layer.ResidualV2

import Data.Vect

import Device
import Layer.CoreV2
import Variable


----------------------------------------------------------------------
-- ResidualV2 — typed-surface skip-connection wrapper (Path C)
----------------------------------------------------------------------
--
-- `output = input + inner(input)`. Inner layer must have same input
-- and output dim. GADT enforces `i = o = n`.

public export
data ResidualStateV2 : Nat -> Nat -> (0 _ : Device) -> Type where
  MkResidualV2 : AnyLayerV2 n n d -> ResidualStateV2 n n d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyResidualV2 : {n : Nat} ->
                  ResidualStateV2 n n d ->
                  TVec n d ->
                  (ResidualStateV2 n n d, TVec n d)
applyResidualV2 (MkResidualV2 inner) input =
  let (inner', innerOut) = applyTVarAny inner input
      sumT = tadd input innerOut
  in (MkResidualV2 inner', sumT)


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
LayerLikeV2 ResidualStateV2 where
  applyTVar st@(MkResidualV2 _) input = applyResidualV2 st input
  layerPrefixV2 _ = "resV2"

  resetStateV2 (MkResidualV2 (MkAnyLayerV2 l @{dict} inner)) =
    MkResidualV2 (MkAnyLayerV2 l @{dict} (resetStateV2 @{dict} inner))


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Wrap a same-dim AnyLayerV2 in a residual connection.
export
residualLayerV2 : {n : Nat} -> AnyLayerV2 n n d -> ResidualStateV2 n n d
residualLayerV2 inner = MkResidualV2 inner

||| Same, wrapped in `AnyLayerV2`.
export
residualLayerV2Any : {n : Nat} -> AnyLayerV2 n n d -> AnyLayerV2 n n d
residualLayerV2Any inner = MkAnyLayerV2 ResidualStateV2 (MkResidualV2 inner)
