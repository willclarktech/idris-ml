module Layer.BatchNorm

import Data.Vect

import Device
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- BatchNorm — typed-surface batch normalization (Path C)
----------------------------------------------------------------------
--
-- Per-channel normalization across spatial dim. Input treated as
-- `[channels, spatialDim]` flattened to `[channels * spatialDim]`.
-- Two learnable params (`gammaT`, `betaT` — both `[channels]`); two
-- persistent state tensors (running mean and var).
--
-- Training: use input stats, update running stats.
-- Eval: use running stats.
--
-- GADT pins `i = o = channels * spatialDim` so the layer fits
-- `LayerLike`'s `Nat -> Nat -> Device -> Type` arity (same trick
-- as Dropout / LayerNorm / EmbeddingWrap).

public export
data BatchNormState : (channels : Nat) -> (spatialDim : Nat) ->
                        Nat -> Nat -> (0 _ : Device) -> (0 _ : GradMode) -> Type where
  MkBatchNorm :
    TVec channels d g ->          -- gamma (learnable)
    TVec channels d g ->          -- beta (learnable)
    TVec channels d g ->          -- running mean (state)
    TVec channels d g ->          -- running var (state)
    (training : Bool) ->
    (momentum : Double) ->
    (eps : Double) ->
    BatchNormState channels spatialDim
                     (channels * spatialDim)
                     (channels * spatialDim) d g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyBatchNorm : {0 d : Device} -> UserDeviceNN d => {channels, spatialDim : Nat} ->
                   BatchNormState channels spatialDim
                     (channels * spatialDim)
                     (channels * spatialDim) d g ->
                   TVec (channels * spatialDim) d g ->
                   ( BatchNormState channels spatialDim
                       (channels * spatialDim)
                       (channels * spatialDim) d g
                   , TVec (channels * spatialDim) d g )
applyBatchNorm {channels} {spatialDim}
                 st@(MkBatchNorm gamma beta mean var training momentum eps)
                 input =
  let cI = cast {to=Int} channels
      sI = cast {to=Int} spatialDim
      tFlag : Int
      tFlag = if training then 1 else 0
      outPtr = prim__batchNorm input.tensorPtr gamma.tensorPtr beta.tensorPtr
                              mean.tensorPtr var.tensorPtr
                              cI sI tFlag momentum eps
  in (st, MkTensor outPtr Nothing)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Fill a buffer with a constant value.
fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v

||| Build a BatchNorm layer. Gamma initialised to 1.0, beta to 0.0,
||| running mean to 0.0, running var to 1.0. Defaults: momentum=0.1,
||| eps=1e-5. Starts in training mode (use `setTraining` to switch).
||| Params register as `<prefix>_gamma` / `<prefix>_beta`; state
||| tensors are persistent C tensors (non-learnable).
export
batchNormLayer : {channels, spatialDim : Nat} ->
                   (paramPrefix : String) ->
                   IO (BatchNormState channels spatialDim
                         (channels * spatialDim)
                         (channels * spatialDim) CPU WithGrad)
batchNormLayer paramPrefix = do
  let cI = cast {to=Int} channels
      gBuf = fillConst (prim__allocDoubles cI) 0 cI 1.0
      bBuf = fillConst (prim__allocDoubles cI) 0 cI 0.0
      mBuf = fillConst (prim__allocDoubles cI) 0 cI 0.0
      vBuf = fillConst (prim__allocDoubles cI) 0 cI 1.0
      gName = paramPrefix ++ "_gamma"
      bName = paramPrefix ++ "_beta"
      gPtr = prim__paramRegister gName (prim__createParam1d cI gBuf)
      bPtr = prim__paramRegister bName (prim__createParam1d cI bBuf)
      mPtr = prim__createState1d cI mBuf
      vPtr = prim__createState1d cI vBuf
      gTV : TVec channels CPU WithGrad
      gTV = MkTensor gPtr (Just gName)
      bTV : TVec channels CPU WithGrad
      bTV = MkTensor bPtr (Just bName)
      mTV : TVec channels CPU WithGrad
      mTV = MkTensor mPtr Nothing
      vTV : TVec channels CPU WithGrad
      vTV = MkTensor vPtr Nothing
  pure $ MkBatchNorm gTV bTV mTV vTV True 0.1 1.0e-5

||| Toggle training/eval mode.
export
setBatchNormTraining : Bool ->
  BatchNormState channels spatialDim i o d g ->
  BatchNormState channels spatialDim i o d g
setBatchNormTraining mode (MkBatchNorm g b m v _ mom eps) =
  MkBatchNorm g b m v mode mom eps


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
{channels, spatialDim : Nat} ->
  LayerLike (BatchNormState channels spatialDim) where
  applyVar st@(MkBatchNorm _ _ _ _ _ _ _) input = applyBatchNorm st input
  layerPrefix _ = "bn"

  freezeLayer (MkBatchNorm g b m v t mo e) = do
    g' <- weakenGrad g
    b' <- weakenGrad b
    m' <- weakenGrad m
    v' <- weakenGrad v
    pure (MkBatchNorm g' b' m' v' t mo e)

  unfreezeLayer (MkBatchNorm g b m v t mo e) = do
    primIO (prim__setRequiresGrad g.tensorPtr 1)
    primIO (prim__setRequiresGrad b.tensorPtr 1)
    primIO (prim__setRequiresGrad m.tensorPtr 1)
    primIO (prim__setRequiresGrad v.tensorPtr 1)
    pure (MkBatchNorm (retypeGrad g) (retypeGrad b)
                      (retypeGrad m) (retypeGrad v) t mo e)

||| Wrap in `AnyLayer`.
export
batchNormLayerAny : {channels, spatialDim : Nat} ->
                      (paramPrefix : String) ->
                      IO (AnyLayer (channels * spatialDim)
                                     (channels * spatialDim) CPU WithGrad)
batchNormLayerAny pid =
  map (MkAnyLayer (BatchNormState channels spatialDim))
      (batchNormLayer {channels} {spatialDim} pid)
