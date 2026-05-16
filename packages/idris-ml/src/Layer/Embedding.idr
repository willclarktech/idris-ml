module Layer.Embedding

import Data.Vect

import Compat.Random
import Device
import Layer.Core
import Sampler
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
-- specific (seqLen, embedDim) pair into the `Nat -> Nat -> Device ->
-- Type` interface.

public export
record EmbeddingState (vocab : Nat) (embedDim : Nat) (0 d : Device) (0 g : GradMode) where
  constructor MkEmbedding
  weightT : TMat vocab embedDim d g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Embedding lookup forward. Indices `tokens : TVec seqLen d` are
||| token IDs encoded as doubles; output `[seqLen * embedDim]` is
||| the flattened embedding vectors. Wraps `prim__embedding`.
export
applyEmbedding : {0 d : Device} -> UserDeviceNN d => {seqLen, embedDim, vocab : Nat} ->
                   EmbeddingState vocab embedDim d g ->
                   TVec seqLen d g ->
                   TVec (seqLen * embedDim) d g
applyEmbedding {seqLen} {embedDim} (MkEmbedding w) tokens =
  let nI = cast {to=Int} seqLen
      dI = cast {to=Int} embedDim
      outPtr = prim__embedding w.tensorPtr tokens.tensorPtr nI dI
  in MkTensor outPtr Nothing


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

||| Build an `EmbeddingState vocab embedDim CPU` with weights
||| sampled from N(0, 0.02) — same init as V1 `embeddingLayer`.
||| Weight registers as one C param under `<prefix>_weight`.
export
embeddingLayer : {vocab, embedDim : Nat} -> (paramPrefix : String) ->
                   IO (EmbeddingState vocab embedDim CPU WithGrad)
embeddingLayer paramPrefix = do
  let vI = cast {to=Int} vocab
      eI = cast {to=Int} embedDim
      n = vocab * embedDim
  vals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate n ())
  let buf = prim__allocDoubles (cast {to=Int} n)
      buf' = packDoubles buf 0 vals
      wName = paramPrefix ++ "_weight"
      wPtr = prim__paramRegister wName (prim__createParam2d vI eI buf')
      wTV : TMat vocab embedDim CPU WithGrad
      wTV = MkTensor wPtr (Just wName)
  pure $ MkEmbedding wTV


----------------------------------------------------------------------
-- LayerLike adapter (specific seqLen × embedDim)
----------------------------------------------------------------------
--
-- `LayerLike` requires `(l : Nat -> Nat -> Device -> Type)`. To fit
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
                      Nat -> Nat -> (0 _ : Device) -> (0 _ : GradMode) -> Type where
  MkEmbeddingWrap : EmbeddingState vocab embedDim d g ->
                     EmbeddingWrap vocab embedDim seqLen (seqLen * embedDim) d g

public export
{vocab, embedDim : Nat} ->
  LayerLike (EmbeddingWrap vocab embedDim) where
  applyVar (MkEmbeddingWrap st) input =
    (MkEmbeddingWrap st, applyEmbedding st input)
  layerPrefix _ = "emb"

  freezeLayer (MkEmbeddingWrap (MkEmbedding w)) = do
    w' <- weakenGrad w
    pure (MkEmbeddingWrap (MkEmbedding w'))

  unfreezeLayer (MkEmbeddingWrap (MkEmbedding w)) = do
    primIO (prim__setRequiresGrad w.tensorPtr 1)
    pure (MkEmbeddingWrap (MkEmbedding (retypeGrad w)))

||| Wrap a fresh embedding into `AnyLayer` for a specific seqLen.
export
embeddingLayerAny : {vocab, embedDim, seqLen : Nat} ->
                      (paramPrefix : String) ->
                      IO (AnyLayer seqLen (seqLen * embedDim) CPU WithGrad)
embeddingLayerAny pid = do
  st <- embeddingLayer {vocab} {embedDim} pid
  pure $ MkAnyLayer (EmbeddingWrap vocab embedDim) (MkEmbeddingWrap st)
