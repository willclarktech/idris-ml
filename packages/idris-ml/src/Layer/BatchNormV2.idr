module Layer.BatchNormV2

import Data.Vect

import Device
import Layer.CoreV2
import Variable


----------------------------------------------------------------------
-- BatchNormV2 — typed-surface batch normalization (Path C)
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
-- `LayerLikeV2`'s `Nat -> Nat -> Device -> Type` arity (same trick
-- as DropoutV2 / LayerNormV2 / EmbeddingV2Wrap).

public export
data BatchNormStateV2 : (channels : Nat) -> (spatialDim : Nat) ->
                        Nat -> Nat -> (0 _ : Device) -> Type where
  MkBatchNormV2 :
    TVec channels d ->          -- gamma (learnable)
    TVec channels d ->          -- beta (learnable)
    TVec channels d ->          -- running mean (state)
    TVec channels d ->          -- running var (state)
    (training : Bool) ->
    (momentum : Double) ->
    (eps : Double) ->
    BatchNormStateV2 channels spatialDim
                     (channels * spatialDim)
                     (channels * spatialDim) d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyBatchNormV2 : {channels, spatialDim : Nat} ->
                   BatchNormStateV2 channels spatialDim
                     (channels * spatialDim)
                     (channels * spatialDim) d ->
                   TVec (channels * spatialDim) d ->
                   ( BatchNormStateV2 channels spatialDim
                       (channels * spatialDim)
                       (channels * spatialDim) d
                   , TVec (channels * spatialDim) d )
applyBatchNormV2 {channels} {spatialDim}
                 st@(MkBatchNormV2 gamma beta mean var training momentum eps)
                 input =
  let cI = cast {to=Int} channels
      sI = cast {to=Int} spatialDim
      tFlag : Int
      tFlag = if training then 1 else 0
      outPtr = prim__batchNorm input.tensorPtr gamma.tensorPtr beta.tensorPtr
                              mean.tensorPtr var.tensorPtr
                              cI sI tFlag momentum eps
  in (st, MkTVar outPtr Nothing)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Fill a buffer with a constant value.
fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v

||| Build a BatchNormV2 layer. Gamma initialised to 1.0, beta to 0.0,
||| running mean to 0.0, running var to 1.0. Defaults: momentum=0.1,
||| eps=1e-5. Starts in training mode (use `setTrainingV2` to switch).
||| Params register as `<prefix>_gamma` / `<prefix>_beta`; state
||| tensors are persistent C tensors (non-learnable).
export
batchNormLayerV2 : {channels, spatialDim : Nat} ->
                   (paramPrefix : String) ->
                   IO (BatchNormStateV2 channels spatialDim
                         (channels * spatialDim)
                         (channels * spatialDim) CPU)
batchNormLayerV2 paramPrefix = do
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
      gTV : TVec channels CPU
      gTV = MkTVar gPtr (Just gName)
      bTV : TVec channels CPU
      bTV = MkTVar bPtr (Just bName)
      mTV : TVec channels CPU
      mTV = MkTVar mPtr Nothing
      vTV : TVec channels CPU
      vTV = MkTVar vPtr Nothing
  pure $ MkBatchNormV2 gTV bTV mTV vTV True 0.1 1.0e-5

||| Toggle training/eval mode.
export
setBatchNormTrainingV2 : Bool ->
  BatchNormStateV2 channels spatialDim i o d ->
  BatchNormStateV2 channels spatialDim i o d
setBatchNormTrainingV2 mode (MkBatchNormV2 g b m v _ mom eps) =
  MkBatchNormV2 g b m v mode mom eps


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
{channels, spatialDim : Nat} ->
  LayerLikeV2 (BatchNormStateV2 channels spatialDim) where
  applyTVar st@(MkBatchNormV2 _ _ _ _ _ _ _) input = applyBatchNormV2 st input
  layerPrefixV2 _ = "bnV2"

||| Wrap in `AnyLayerV2`.
export
batchNormLayerV2Any : {channels, spatialDim : Nat} ->
                      (paramPrefix : String) ->
                      IO (AnyLayerV2 (channels * spatialDim)
                                     (channels * spatialDim) CPU)
batchNormLayerV2Any pid =
  map (MkAnyLayerV2 (BatchNormStateV2 channels spatialDim))
      (batchNormLayerV2 {channels} {spatialDim} pid)
