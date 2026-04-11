module Layer.Transformer

import Data.Vect

import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Transformer State
----------------------------------------------------------------------

||| Single-head Transformer block with causal masking.
||| seqLen and dModel are compile-time Nat parameters.
||| Input/output size = seqLen * dModel (type-safe reshape).
public export
record TransformerState (seqLen : Nat) (dModel : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkTransformer
  queryW : LinearState dModel dModel ty
  keyW : LinearState dModel dModel ty
  valueW : LinearState dModel dModel ty
  ff1 : LinearState dModel (4 * dModel) ty
  ff2 : LinearState (4 * dModel) dModel ty


----------------------------------------------------------------------
-- FFI
----------------------------------------------------------------------

%foreign "C:tensor_causal_mask,libidrisml"
prim__causalMask : Int -> AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
{seqLen : Nat} -> {dModel : Nat} -> LayerLike (TransformerState seqLen dModel) where

  applyGeneric {i} {o} st xs = (st, believe_me xs)  -- TODO: proper Double-level impl

  applyVar {i} {o} st xs =
    case extractWeightTensor (queryW st) of
      Just qW =>
        let Just kW = extractWeightTensor (keyW st) | Nothing => (st, believe_me xs)
            Just vW = extractWeightTensor (valueW st) | Nothing => (st, believe_me xs)
            Just f1W = extractWeightTensor (ff1 st) | Nothing => (st, believe_me xs)
            Just f2W = extractWeightTensor (ff2 st) | Nothing => (st, believe_me xs)
        in
        let sI = cast {to=Int} seqLen
            dI = cast {to=Int} dModel
            -- Pack input: Vector (seqLen*dModel) -> [seqLen, dModel] matrix
            (VTensor xElems) = xs
            inputFlat = vecStackTensor xElems
            inputMat = prim__reshape2d inputFlat sI dI

            -- Q, K, V: [seqLen, dModel] @ [dModel, dModel]^T
            q = prim__mm inputMat (prim__transpose2d qW)
            k = prim__mm inputMat (prim__transpose2d kW)
            v = prim__mm inputMat (prim__transpose2d vW)

            -- Attention: softmax(Q @ K^T / sqrt(d)) @ V
            scores = prim__mulScalar (prim__mm q (prim__transpose2d k))
                       (1.0 / sqrt (cast {to=Double} dModel))
            mask = prim__causalMask sI
            masked = prim__maskedFill scores mask (-1.0e20)
            attn = prim__softmax2d masked
            attnOut = prim__mm attn v  -- [seqLen, dModel]

            -- Residual 1
            residual1 = tensorAdd attnOut inputMat

            -- Feedforward: ReLU(x @ W1^T) @ W2^T
            ffHidden = prim__clampMin (prim__mm residual1 (prim__transpose2d f1W)) 0.0
            ffOut = prim__mm ffHidden (prim__transpose2d f2W)

            -- Residual 2
            residual2 = tensorAdd ffOut residual1

            -- Flatten: [seqLen, dModel] -> Vector (seqLen*dModel)
            flatSize = sI * dI
            flat = prim__reshape2d residual2 1 flatSize
            output = VTensor (tensorToScalars (prim__narrow flat 0 0 flatSize) 0 o)
        in (st, output)
      Nothing => (st, believe_me xs)

  emapLayer f (MkTransformer qw kw vw f1 f2) =
    MkTransformer (emapLayer f qw) (emapLayer f kw) (emapLayer f vw)
                  (emapLayer f f1) (emapLayer f f2)

  showLayer _ = "Transformer<" ++ show seqLen ++ "x" ++ show dModel ++ ">"

  nameLayer prefx (MkTransformer qw kw vw f1 f2) =
    MkTransformer (nameLayer (prefx ++ "_q") qw) (nameLayer (prefx ++ "_k") kw)
                  (nameLayer (prefx ++ "_v") vw) (nameLayer (prefx ++ "_ff1") f1)
                  (nameLayer (prefx ++ "_ff2") f2)

  layerPrefix _ = "tfm"

  toDoubleLayer (MkTransformer qw kw vw f1 f2) =
    MkTransformer (toDoubleLayer qw) (toDoubleLayer kw) (toDoubleLayer vw)
                  (toDoubleLayer f1) (toDoubleLayer f2)

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry (showLayer @{%search} st) [])

  getParamIds (MkTransformer qw kw vw f1 f2) =
    getParamIds qw ++ getParamIds kw ++ getParamIds vw ++
    getParamIds f1 ++ getParamIds f2


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
mkTransformer : {seqLen, dModel : Nat} -> (Num ty, FromDouble ty) =>
                IO (TransformerState seqLen dModel (seqLen * dModel) (seqLen * dModel) ty)
mkTransformer = do
  qw <- mkLinear {i=dModel, o=dModel}
  kw <- mkLinear {i=dModel, o=dModel}
  vw <- mkLinear {i=dModel, o=dModel}
  f1 <- mkLinear {i=dModel, o=4*dModel}
  f2 <- mkLinear {i=4*dModel, o=dModel}
  pure $ MkTransformer qw kw vw f1 f2

export
transformerLayer : {seqLen, dModel : Nat} -> (Num ty, FromDouble ty) =>
                   IO (AnyLayer (seqLen * dModel) (seqLen * dModel) ty)
transformerLayer = map (MkAnyLayer (TransformerState seqLen dModel)) mkTransformer
