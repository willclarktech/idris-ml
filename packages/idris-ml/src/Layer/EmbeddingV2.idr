module Layer.EmbeddingV2

import Data.Vect

import Compat.Random
import Device
import Layer.CoreV2
import Sampler
import Variable


----------------------------------------------------------------------
-- EmbeddingV2 — typed-surface lookup table (Path C)
----------------------------------------------------------------------
--
-- Maps a `[seqLen]` tensor of token indices (encoded as doubles) to a
-- flattened `[seqLen * embedDim]` tensor of embedding vectors. The
-- vocab × embedDim weight is a learnable param.
--
-- Provided as a standalone op + a `LayerLikeV2` adapter that fits a
-- specific (seqLen, embedDim) pair into the `Nat -> Nat -> Device ->
-- Type` interface.

public export
record EmbeddingStateV2 (vocab : Nat) (embedDim : Nat) (0 d : Device) where
  constructor MkEmbeddingV2
  weightT : TMat vocab embedDim d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Embedding lookup forward. Indices `tokens : TVec seqLen d` are
||| token IDs encoded as doubles; output `[seqLen * embedDim]` is
||| the flattened embedding vectors. Wraps `prim__embedding`.
export
applyEmbeddingV2 : {seqLen, embedDim, vocab : Nat} ->
                   EmbeddingStateV2 vocab embedDim d ->
                   TVec seqLen d ->
                   TVec (seqLen * embedDim) d
applyEmbeddingV2 {seqLen} {embedDim} (MkEmbeddingV2 w) tokens =
  let nI = cast {to=Int} seqLen
      dI = cast {to=Int} embedDim
      outPtr = prim__embedding w.tensorPtr tokens.tensorPtr nI dI
  in MkTVar outPtr Nothing


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

||| Build an `EmbeddingStateV2 vocab embedDim CPU` with weights
||| sampled from N(0, 0.02) — same init as V1 `embeddingLayer`.
||| Weight registers as one C param under `<prefix>_weight`.
export
embeddingLayerV2 : {vocab, embedDim : Nat} -> (paramPrefix : String) ->
                   IO (EmbeddingStateV2 vocab embedDim CPU)
embeddingLayerV2 paramPrefix = do
  let vI = cast {to=Int} vocab
      eI = cast {to=Int} embedDim
      n = vocab * embedDim
  vals <- traverse (\_ => map (* 0.02) normalSample) (Vect.replicate n ())
  let buf = prim__allocDoubles (cast {to=Int} n)
      buf' = packDoubles buf 0 vals
      wName = paramPrefix ++ "_weight"
      wPtr = prim__paramRegister wName (prim__createParam2d vI eI buf')
      wTV : TMat vocab embedDim CPU
      wTV = MkTVar wPtr (Just wName)
  pure $ MkEmbeddingV2 wTV


----------------------------------------------------------------------
-- LayerLikeV2 adapter (specific seqLen × embedDim)
----------------------------------------------------------------------
--
-- `LayerLikeV2` requires `(l : Nat -> Nat -> Device -> Type)`. To fit
-- Embedding (which has 3 Nat params), we wrap it for a specific
-- (vocab, embedDim) pair. The seqLen is the input dim; output is
-- `seqLen * embedDim`.
--
-- The `EmbeddingV2Wrap vocab embedDim seqLen out d` GADT pattern
-- pins `out = seqLen * embedDim`, so the constructor enforces the
-- shape relationship and `LayerLikeV2`'s `i / o` interpretation
-- gives `i = seqLen, o = seqLen * embedDim`.

public export
data EmbeddingV2Wrap : (vocab : Nat) -> (embedDim : Nat) ->
                      Nat -> Nat -> (0 _ : Device) -> Type where
  MkEmbeddingV2Wrap : EmbeddingStateV2 vocab embedDim d ->
                     EmbeddingV2Wrap vocab embedDim seqLen (seqLen * embedDim) d

public export
{vocab, embedDim : Nat} ->
  LayerLikeV2 (EmbeddingV2Wrap vocab embedDim) where
  applyTVar (MkEmbeddingV2Wrap st) input =
    (MkEmbeddingV2Wrap st, applyEmbeddingV2 st input)
  layerPrefixV2 _ = "embV2"

||| Wrap a fresh embedding into `AnyLayerV2` for a specific seqLen.
export
embeddingLayerV2Any : {vocab, embedDim, seqLen : Nat} ->
                      (paramPrefix : String) ->
                      IO (AnyLayerV2 seqLen (seqLen * embedDim) CPU)
embeddingLayerV2Any pid = do
  st <- embeddingLayerV2 {vocab} {embedDim} pid
  pure $ MkAnyLayerV2 (EmbeddingV2Wrap vocab embedDim) (MkEmbeddingV2Wrap st)
