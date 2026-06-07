module Layer.Embedding

import Data.Vect

import Executor
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Embedding — typed-surface lookup table (Path C)
----------------------------------------------------------------------
--
-- Maps a `[seqLen]` tensor of token indices (encoded as doubles) to a
-- flattened `[seqLen * embedDim]` tensor of embedding vectors. The
-- vocab × embedDim weight is a learnable param.
--
-- Provided as a standalone op + a `LayerLike` adapter that fits a
-- specific (seqLen, embedDim) pair into the `Nat -> Nat -> Executor ->
-- Type` interface.

public export
record EmbeddingState (vocab : Nat) (embedDim : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkEmbedding
  weightT : TMat vocab embedDim ex dt g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Embedding lookup forward. Indices `tokens : TVec seqLen ex` are
||| token IDs encoded as doubles; output `[seqLen * embedDim]` is
||| the flattened embedding vectors. Wraps `primEmbedding {ex}`.
export
applyEmbedding : {0 ex : Executor} -> Backend ex dt => {seqLen, embedDim, vocab : Nat} ->
                   EmbeddingState vocab embedDim ex dt g ->
                   TVec seqLen ex dt g ->
                   IO (TVec (seqLen * embedDim) ex dt g)
applyEmbedding {seqLen} {embedDim} (MkEmbedding w) tokens = ioRerun (\_ =>
  let nI = cast {to=Int} seqLen
      dI = cast {to=Int} embedDim
      outPtr = primEmbedding {ex} w.tensorPtr tokens.tensorPtr nI dI
  in MkTensor outPtr Nothing)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build an `EmbeddingState vocab embedDim TapeExecutor` with weights
||| sampled from N(0, 0.02) — HF default for token / position embeddings.
||| Weight registers as one C param under `<prefix>_weight`.
export
embeddingLayer : Backend ex dt => {vocab, embedDim : Nat} -> (paramPrefix : String) ->
                   IO (EmbeddingState vocab embedDim ex dt WithGrad)
embeddingLayer paramPrefix = do
  let wName = paramPrefix ++ "_weight"
  weight <- tparam2dNormal {ex} {dt} {o=vocab} {i=embedDim} wName 0.0 0.02
  pure $ MkEmbedding weight


----------------------------------------------------------------------
-- LayerLike adapter (specific seqLen × embedDim)
----------------------------------------------------------------------
--
-- `LayerLike` requires `(l : Nat -> Nat -> Executor -> Type)`. To fit
-- Embedding (which has 3 Nat params), we wrap it for a specific
-- (vocab, embedDim) pair. The seqLen is the input dim; output is
-- `seqLen * embedDim`.
--
-- The `EmbeddingWrap vocab embedDim seqLen out d` GADT pattern
-- pins `out = seqLen * embedDim`, so the constructor enforces the
-- shape relationship and `LayerLike`'s `i / o` interpretation
-- gives `i = seqLen, o = seqLen * embedDim`.

public export
data EmbeddingWrap : (vocab : Nat) -> (embedDim : Nat) ->
                      Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkEmbeddingWrap : EmbeddingState vocab embedDim ex dt g ->
                     EmbeddingWrap vocab embedDim seqLen (seqLen * embedDim) ex dt g

public export
{vocab, embedDim : Nat} ->
  LayerLike (EmbeddingWrap vocab embedDim) where
  applyVar (MkEmbeddingWrap st) input = do
    out <- applyEmbedding st input
    pure (MkEmbeddingWrap st, out)
  layerPrefix _ = "emb"

  freezeLayer (MkEmbeddingWrap (MkEmbedding w)) = do
    w' <- weakenGrad w
    pure (MkEmbeddingWrap (MkEmbedding w'))

  unfreezeLayer (MkEmbeddingWrap (MkEmbedding w)) = do
    primIO (primSetRequiresGrad {ex} w.tensorPtr 1)
    pure (MkEmbeddingWrap (MkEmbedding (retypeGrad w)))

||| Wrap a fresh embedding into `AnyLayer` for a specific seqLen.
export
embeddingLayerAny : Backend ex dt => {vocab, embedDim, seqLen : Nat} ->
                      (paramPrefix : String) ->
                      IO (AnyLayer seqLen (seqLen * embedDim) ex dt WithGrad)
embeddingLayerAny pid = do
  st <- embeddingLayer {vocab} {embedDim} pid
  pure $ MkAnyLayer (EmbeddingWrap vocab embedDim) (MkEmbeddingWrap st)
