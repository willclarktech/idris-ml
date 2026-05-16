module Layer.Residual

import Data.Vect

import Device
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Residual — typed-surface skip-connection wrapper (Path C)
----------------------------------------------------------------------
--
-- `output = input + inner(input)`. Inner layer must have same input
-- and output dim. GADT enforces `i = o = n`.

public export
data ResidualState : Nat -> Nat -> (0 _ : Type) -> (0 _ : GradMode) -> Type where
  MkResidual : AnyLayer n n d g -> ResidualState n n d g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyResidual : {n : Nat} ->
                  ResidualState n n d g ->
                  TVec n d g ->
                  (ResidualState n n d g, TVec n d g)
applyResidual (MkResidual inner) input =
  let (inner', innerOut) = applyVarAny inner input
      sumT = tadd input innerOut
  in (MkResidual inner', sumT)


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike ResidualState where
  applyVar st@(MkResidual _) input = applyResidual st input
  layerPrefix _ = "res"

  resetState (MkResidual (MkAnyLayer l @{dict} inner)) =
    MkResidual (MkAnyLayer l @{dict} (resetState @{dict} inner))

  freezeLayer (MkResidual inner) = do
    inner' <- freezeAnyLayer inner
    pure (MkResidual inner')

  unfreezeLayer (MkResidual inner) = do
    inner' <- unfreezeAnyLayer inner
    pure (MkResidual inner')


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Wrap a same-dim AnyLayer in a residual connection.
export
residualLayer : {n : Nat} -> AnyLayer n n d g -> ResidualState n n d g
residualLayer inner = MkResidual inner

||| Same, wrapped in `AnyLayer`.
export
residualLayerAny : {n : Nat} -> AnyLayer n n d g -> AnyLayer n n d g
residualLayerAny inner = MkAnyLayer ResidualState (MkResidual inner)
