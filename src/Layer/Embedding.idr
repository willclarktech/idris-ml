-- | Embedding layer: lookup table for token indices.
-- |
-- | Input: [seqLen] flat tensor of token indices (doubles cast to ints).
-- | Output: [seqLen * embedDim] flat tensor of embedding vectors.
-- | Weight: [vocabSize, embedDim] learnable parameter.
-- |
-- | Replaces one-hot + linear, reducing O(vocab) to O(1) per token.

module Layer.Embedding

import Data.Vect

import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Sampler
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Embedding State
----------------------------------------------------------------------

public export
record EmbeddingState (vocabSize : Nat) (embedDim : Nat) (seqLen : Nat)
                      (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkEmbedding
  0 inputPrf  : inputSize = seqLen
  0 outputPrf : outputSize = seqLen * embedDim
  weight : Vector (vocabSize * embedDim) ty
  weightTensor : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
{vocabSize, embedDim, seqLen : Nat} ->
  LayerLike (EmbeddingState vocabSize embedDim seqLen) where

  applyGeneric _ _ = idris_crash "Embedding: use tensor path"
  applyVar _ _ = idris_crash "Embedding: use tensor path"

  applyVarTensor {i} {o} st inputT =
    case st.weightTensor of
      Just wT =>
        let nI = cast {to=Int} seqLen
            dI = cast {to=Int} embedDim
        in (st, prim__embedding wT inputT nI dI)
      Nothing => idris_crash "Embedding: weight tensor not initialized"

  emapLayer f (MkEmbedding ip op w wt) = MkEmbedding ip op (map f w) wt

  showLayer _ = "Embedding<" ++ show vocabSize ++ "x" ++ show embedDim ++ ">"

  nameLayer {i} {o} prefx (MkEmbedding ip op w _) =
    if prim__backendSupportsTensorParams == 1
      then
        let nI = cast {to=Int} (vocabSize * embedDim)
            buf = prim__allocDoubles nI
            (VTensor elems) = w
            buf' = packScalarValues buf 0 elems
            wT = prim__paramRegister (prefx ++ "_weight")
                   (prim__createParam2d (cast vocabSize) (cast embedDim) buf')
        in MkEmbedding ip op w (Just wT)
      else idris_crash "Embedding: scalar path not supported"

  layerPrefix _ = "emb"

  toDoubleLayer (MkEmbedding ip op w _) =
    MkEmbedding ip op (map value w) Nothing

  debugApply _ _ = idris_crash "Embedding: use tensor path"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create an embedding layer.
||| Input: seqLen token indices -> Output: seqLen * embedDim embedded vectors.
export
embeddingLayer : {vocabSize, embedDim, seqLen : Nat} ->
                 (Num ty, FromDouble ty) =>
                 IO (AnyLayer seqLen (seqLen * embedDim) ty)
embeddingLayer = do
  weightVals <- traverse (\_ => map fromDouble (map (* 0.02) normalSample))
                         (the (Vector (vocabSize * embedDim) ty) zeros)
  pure $ MkAnyLayer (EmbeddingState vocabSize embedDim seqLen)
    (MkEmbedding Refl Refl weightVals Nothing)
