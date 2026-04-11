module Layer.LayerNorm

import Data.Vect

import Endofunctor
import Floating
import Layer.Core
import Layer.Linear
import Tensor
import Variable


----------------------------------------------------------------------
-- Layer Norm State
----------------------------------------------------------------------

||| Layer normalization with learnable scale (gamma) and shift (beta).
||| Not a standalone LayerLike — used as a sub-component of transformers.
public export
record LayerNormState (dim : Nat) (ty : Type) where
  constructor MkLayerNorm
  gamma : Vector dim ty         -- scale (init: all 1.0)
  beta : Vector dim ty          -- shift (init: all 0.0)
  gammaTensor : Maybe AnyPtr    -- consolidated tensor for C path
  betaTensor : Maybe AnyPtr


----------------------------------------------------------------------
-- Construction
----------------------------------------------------------------------

export
mkLayerNorm : {dim : Nat} -> (Num ty, FromDouble ty) => IO (LayerNormState dim ty)
mkLayerNorm {dim} = do
  let gammaVec = replicate dim (STensor (fromDouble 1.0))
      betaVec  = replicate dim (STensor (fromDouble 0.0))
  pure $ MkLayerNorm (VTensor gammaVec) (VTensor betaVec) Nothing Nothing


----------------------------------------------------------------------
-- Sub-component operations (parallel to LayerLike methods)
----------------------------------------------------------------------

export
emapLayerNorm : (ty -> ty) -> LayerNormState dim ty -> LayerNormState dim ty
emapLayerNorm f (MkLayerNorm g b gt bt) = MkLayerNorm (map f g) (map f b) gt bt

export
nameLayerNorm : {dim : Nat} -> String -> LayerNormState dim Variable -> LayerNormState dim Variable
nameLayerNorm {dim} prefx (MkLayerNorm gamma beta _ _) =
  if prim__backendSupportsTensorParams == 1
    then
      let dI = cast {to=Int} dim
          (VTensor gElems) = gamma
          gBuf = prim__allocDoubles dI
          gBuf' = packScalarValues gBuf 0 gElems
          gammaT = prim__paramRegister (prefx ++ "_gamma") (prim__createParam1d dI gBuf')
          (VTensor bElems) = beta
          bBuf = prim__allocDoubles dI
          bBuf' = packScalarValues bBuf 0 bElems
          betaT = prim__paramRegister (prefx ++ "_beta") (prim__createParam1d dI bBuf')
      in MkLayerNorm (VTensor $ buildViewVector (prefx ++ "_g") gammaT 0 dim)
                     (VTensor $ buildViewVector (prefx ++ "_b") betaT 0 dim)
                     (Just gammaT) (Just betaT)
    else MkLayerNorm gamma beta Nothing Nothing

export
toDoubleLayerNorm : {dim : Nat} -> LayerNormState dim Variable -> LayerNormState dim Double
toDoubleLayerNorm {dim} (MkLayerNorm _ _ (Just gt) (Just bt)) =
  let gVec = VTensor $ map (\i => STensor (prim__item1d gt (cast (finToNat i))))
                           (Data.Vect.Fin.range {len=dim})
      bVec = VTensor $ map (\i => STensor (prim__item1d bt (cast (finToNat i))))
                           (Data.Vect.Fin.range {len=dim})
  in MkLayerNorm gVec bVec Nothing Nothing
toDoubleLayerNorm (MkLayerNorm g b _ _) =
  MkLayerNorm (map (\v => case v of Var _ _ x => x) g)
              (map (\v => case v of Var _ _ x => x) b)
              Nothing Nothing

export
getLayerNormParamIds : LayerNormState dim Variable -> List String
getLayerNormParamIds (MkLayerNorm (VTensor gElems) (VTensor bElems) _ _) =
  let getIds : Vect k (Scalar Variable) -> List String
      getIds [] = []
      getIds (STensor (Var _ (Just pid) _) :: rest) = pid :: getIds rest
      getIds (_ :: rest) = getIds rest
  in getIds gElems ++ getIds bElems

||| Extract the gamma AnyPtr handle.
export
extractGammaTensor : LayerNormState dim Variable -> Maybe AnyPtr
extractGammaTensor st = st.gammaTensor

||| Extract the beta AnyPtr handle.
export
extractBetaTensor : LayerNormState dim Variable -> Maybe AnyPtr
extractBetaTensor st = st.betaTensor
