-- | Batch normalization layer (instance norm when batch=1).
-- |
-- | Per-channel normalization across spatial dimensions.
-- | Input treated as [channels, spatialDim]. Normalizes each channel.
-- | gamma/beta are learnable. Running mean/var are state tensors.
-- | Training: use input stats, update running stats.
-- | Eval: use running stats.

module Layer.BatchNorm

import Data.Vect

import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- BatchNorm State
----------------------------------------------------------------------

public export
record BatchNormState (channels : Nat) (spatialDim : Nat)
                      (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkBatchNorm
  0 inputPrf  : inputSize = channels * spatialDim
  0 outputPrf : outputSize = channels * spatialDim
  gamma : Vector channels ty
  beta : Vector channels ty
  training : Bool
  momentum : Double
  eps : Double
  gammaTensor : Maybe AnyPtr
  betaTensor : Maybe AnyPtr
  meanTensor : Maybe AnyPtr   -- persistent state (running mean)
  varTensor : Maybe AnyPtr    -- persistent state (running var)


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
{channels, spatialDim : Nat} ->
  LayerLike (BatchNormState channels spatialDim) where

  applyGeneric _ _ = idris_crash "BatchNorm: use tensor path"
  applyVar _ _ = idris_crash "BatchNorm: use tensor path"

  applyVarTensor {i} {o} st inputT =
    case (st.gammaTensor, st.betaTensor, st.meanTensor, st.varTensor) of
      (Just gT, Just bT, Just mT, Just vT) =>
        let cI = cast {to=Int} channels
            sI = cast {to=Int} spatialDim
            tFlag : Int
            tFlag = if st.training then 1 else 0
            outT = prim__batchNorm inputT gT bT mT vT cI sI tFlag st.momentum st.eps
        in (st, outT)
      _ => idris_crash "BatchNorm: tensors not initialized (call autoName first)"

  emapLayer f (MkBatchNorm ip op g b t m e gt bt mt vt) =
    MkBatchNorm ip op (map f g) (map f b) t m e gt bt mt vt

  showLayer _ = "BatchNorm<" ++ show channels ++ ">"

  nameLayer {i} {o} prefx (MkBatchNorm ip op gamma beta t m e _ _ _ _) =
    if prim__backendSupportsTensorParams == 1
      then
        let cI = cast {to=Int} channels
            sI = cast {to=Int} spatialDim
            -- Gamma (learnable)
            gBuf = prim__allocDoubles cI
            (VTensor gElems) = gamma
            gBuf' = packScalarValues gBuf 0 gElems
            gT = prim__paramRegister (prefx ++ "_gamma")
                   (prim__createParam1d cI gBuf')
            -- Beta (learnable)
            bBuf = prim__allocDoubles cI
            (VTensor bElems) = beta
            bBuf' = packScalarValues bBuf 0 bElems
            bT = prim__paramRegister (prefx ++ "_beta")
                   (prim__createParam1d cI bBuf')
            -- Running mean (state, non-learnable)
            mBuf = prim__allocDoubles cI
            mBuf' = packZeros mBuf 0 cI
            mT = prim__createState1d cI mBuf'
            -- Running var (state, init to 1.0)
            vBuf = prim__allocDoubles cI
            vBuf' = packOnes vBuf 0 cI
            vT = prim__createState1d cI vBuf'
        in MkBatchNorm ip op gamma beta t m e (Just gT) (Just bT) (Just mT) (Just vT)
      else idris_crash "BatchNorm: scalar path not supported"
    where
      packZeros : AnyPtr -> Int -> Int -> AnyPtr
      packZeros buf idx n = if idx >= n then buf
        else packZeros (prim__setDouble buf idx 0.0) (idx + 1) n

      packOnes : AnyPtr -> Int -> Int -> AnyPtr
      packOnes buf idx n = if idx >= n then buf
        else packOnes (prim__setDouble buf idx 1.0) (idx + 1) n

  layerPrefix _ = "bn"

  toDoubleLayer (MkBatchNorm ip op g b _ m e _ _ _ _) =
    MkBatchNorm ip op (map value g) (map value b) False m e Nothing Nothing Nothing Nothing

  setTraining mode (MkBatchNorm ip op g b _ m e gt bt mt vt) =
    MkBatchNorm ip op g b mode m e gt bt mt vt

  debugApply _ _ = idris_crash "BatchNorm: use tensor path"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a batch norm layer for the given number of channels.
||| Gamma initialized to 1, beta to 0.
export
batchNormLayer : {channels, spatialDim : Nat} ->
                 (Num ty, FromDouble ty) =>
                 IO (AnyLayer (channels * spatialDim)
                              (channels * spatialDim) ty)
batchNormLayer = do
  let gammaVals = the (Vector channels ty) (map (const (fromDouble 1.0)) zeros)
      betaVals = the (Vector channels ty) zeros
  pure $ MkAnyLayer (BatchNormState channels spatialDim)
    (MkBatchNorm Refl Refl gammaVals betaVals True 0.1 1.0e-5
                 Nothing Nothing Nothing Nothing)
