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

||| Single-head Transformer block with causal masking and output projection.
||| Type parameters: seqLen, dModel, vocabSize — all checked at compile time.
||| Input: seqLen * dModel (embedded sequence)
||| Output: seqLen * vocabSize (per-position class logits)
public export
record TransformerState (seqLen : Nat) (dModel : Nat) (vocabSize : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkTransformer
  -- Proofs that input/output sizes match the computed values
  0 inputPrf : inputSize = seqLen * dModel
  0 outputPrf : outputSize = seqLen * vocabSize
  queryW : LinearState dModel dModel ty
  keyW : LinearState dModel dModel ty
  valueW : LinearState dModel dModel ty
  ff1 : LinearState dModel (4 * dModel) ty
  ff2 : LinearState (4 * dModel) dModel ty
  outProj : LinearState dModel vocabSize ty  -- per-position output projection


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

-- applyVar uses idris_crash for unreachable fallbacks (weights always initialized after autoName)
%default partial
export
{seqLen : Nat} -> {dModel : Nat} -> {vocabSize : Nat} -> LayerLike (TransformerState seqLen dModel vocabSize) where

  -- Pure Idris attention for Double-level evaluation (no C tensors needed).
  -- Same algorithm as applyVar but using type-safe Math operations.
  applyGeneric {i} {o} st xs =
    let input = reshapeToMatrix {m=seqLen, n=dModel} (rewrite sym st.inputPrf in xs)
        qWeights = weights (queryW st)                        -- Matrix dModel dModel ty
        kWeights = weights (keyW st)
        vWeights = weights (valueW st)
        f1Weights = weights (ff1 st)                          -- Matrix (4*dModel) dModel ty
        f2Weights = weights (ff2 st)                          -- Matrix dModel (4*dModel) ty
        opWeights = weights (outProj st)                      -- Matrix vocabSize dModel ty

        -- Q, K, V projections: input @ W^T
        q = matrixMultiply input (transpose qWeights)         -- [seqLen, dModel]
        k = matrixMultiply input (transpose kWeights)
        v = matrixMultiply input (transpose vWeights)

        -- Scaled dot-product attention
        scores = scaleMatrix (1 / sqrt (fromInteger (natToInteger dModel)))
                             (matrixMultiply q (transpose k))  -- [seqLen, seqLen]
        masked = causalMaskMatrix scores
        attn = softmaxMatrix masked                            -- [seqLen, seqLen]
        attnOut = matrixMultiply attn v                        -- [seqLen, dModel]

        -- Residual 1
        residual1 = attnOut + input

        -- Feedforward: ReLU(x @ W1^T) @ W2^T
        ffHidden = clampMinTensor 0 (matrixMultiply residual1 (transpose f1Weights))
        ffOut = matrixMultiply ffHidden (transpose f2Weights)

        -- Residual 2
        residual2 = ffOut + residual1

        -- Per-position output projection
        output = matrixMultiply residual2 (transpose opWeights) -- [seqLen, vocabSize]

    in let result = flattenMatrix output  -- : Vector (seqLen * vocabSize) ty
       in (st, replace {p = \x => Vector x ty} (sym st.outputPrf) result)

  applyVar {i} {o} st xs =
    case extractWeightTensor (queryW st) of
      Just qW =>
        let Just kW = extractWeightTensor (keyW st) | Nothing => idris_crash "Transformer: weight tensor not initialized (call autoName first)"
            Just vW = extractWeightTensor (valueW st) | Nothing => idris_crash "Transformer: weight tensor not initialized (call autoName first)"
            Just f1W = extractWeightTensor (ff1 st) | Nothing => idris_crash "Transformer: weight tensor not initialized (call autoName first)"
            Just f2W = extractWeightTensor (ff2 st) | Nothing => idris_crash "Transformer: weight tensor not initialized (call autoName first)"
            Just opW = extractWeightTensor (outProj st) | Nothing => idris_crash "Transformer: weight tensor not initialized (call autoName first)"
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

            -- Per-position output projection: [seqLen, dModel] @ [dModel, vocabSize]^T -> [seqLen, vocabSize]
            outT = prim__mm residual2 (prim__transpose2d opW)  -- [seqLen, vocabSize]

            -- Flatten: [seqLen, vocabSize] -> 1D [seqLen*vocabSize]
            vI = cast {to=Int} vocabSize
            flat1d = prim__narrow outT 0 0 (sI * vI)
            output = VTensor (tensorToScalars flat1d 0 o)
        in (st, output)
      Nothing => idris_crash "Transformer: queryW not initialized"

  emapLayer f (MkTransformer iprf oprf qw kw vw f1 f2 op) =
    MkTransformer iprf oprf (emapLayer f qw) (emapLayer f kw) (emapLayer f vw)
                  (emapLayer f f1) (emapLayer f f2) (emapLayer f op)

  showLayer _ = "Transformer<" ++ show seqLen ++ "x" ++ show dModel ++ ">"

  nameLayer prefx (MkTransformer iprf oprf qw kw vw f1 f2 op) =
    MkTransformer iprf oprf (nameLayer (prefx ++ "_q") qw) (nameLayer (prefx ++ "_k") kw)
                  (nameLayer (prefx ++ "_v") vw) (nameLayer (prefx ++ "_ff1") f1)
                  (nameLayer (prefx ++ "_ff2") f2) (nameLayer (prefx ++ "_out") op)

  layerPrefix _ = "tfm"

  toDoubleLayer (MkTransformer iprf oprf qw kw vw f1 f2 op) =
    MkTransformer iprf oprf (toDoubleLayer qw) (toDoubleLayer kw) (toDoubleLayer vw)
                  (toDoubleLayer f1) (toDoubleLayer f2) (toDoubleLayer op)

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry (showLayer @{%search} st) [])

  getParamIds (MkTransformer _ _ qw kw vw f1 f2 op) =
    getParamIds qw ++ getParamIds kw ++ getParamIds vw ++
    getParamIds f1 ++ getParamIds f2 ++ getParamIds op


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
mkTransformer : {seqLen, dModel, vocabSize : Nat} -> (Num ty, FromDouble ty) =>
                IO (TransformerState seqLen dModel vocabSize (seqLen * dModel) (seqLen * vocabSize) ty)
mkTransformer = do
  qw <- mkLinear {i=dModel, o=dModel}
  kw <- mkLinear {i=dModel, o=dModel}
  vw <- mkLinear {i=dModel, o=dModel}
  f1 <- mkLinear {i=dModel, o=4*dModel}
  f2 <- mkLinear {i=4*dModel, o=dModel}
  op <- mkLinear {i=dModel, o=vocabSize}
  pure $ MkTransformer Refl Refl qw kw vw f1 f2 op

export
transformerLayer : {seqLen, dModel, vocabSize : Nat} -> (Num ty, FromDouble ty) =>
                   IO (AnyLayer (seqLen * dModel) (seqLen * vocabSize) ty)
transformerLayer = map (MkAnyLayer (TransformerState seqLen dModel vocabSize)) mkTransformer
