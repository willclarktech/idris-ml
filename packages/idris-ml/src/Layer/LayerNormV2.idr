module Layer.LayerNormV2

import Data.Vect

import Device
import Layer.CoreV2
import Variable


----------------------------------------------------------------------
-- LayerNormV2 — typed-surface layer normalisation (Path C)
----------------------------------------------------------------------
--
-- Normalises along the last (only) dim of a 1D `TVec n d` input,
-- then applies a learnable scale (gamma) + shift (beta).
--
-- The C backend currently exposes only `prim__layerNorm2d` (operates
-- on `[B, N]` shape). For 1D input we reshape `[n]` → `[1, n]`,
-- normalise, reshape back. ~3 tape entries per call (still much
-- cheaper than computing mean/var/sqrt manually).
--
-- GADT shape `i = o = n` lets the layer fit `LayerLikeV2`'s arity
-- (mirrors DropoutV2's pattern).

public export
data LayerNormStateV2 : Nat -> Nat -> (0 _ : Device) -> Type where
  MkLayerNormV2 : TVec n d -> TVec n d -> LayerNormStateV2 n n d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyLayerNormV2 : {n : Nat} ->
                   LayerNormStateV2 n n d ->
                   TVec n d ->
                   (LayerNormStateV2 n n d, TVec n d)
applyLayerNormV2 {n} st@(MkLayerNormV2 gamma beta) input =
  let nI = cast {to=Int} n
      input2d = prim__reshape2d input.tensorPtr 1 nI
      norm2d = prim__layerNorm2d input2d gamma.tensorPtr beta.tensorPtr 1.0e-5
      norm1d = prim__reshape1d norm2d nI
  in (st, MkTVar norm1d Nothing)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

-- Fill a buffer with a constant value.
fillConst : AnyPtr -> Int -> Int -> Double -> AnyPtr
fillConst buf _ 0 _ = buf
fillConst buf off n v =
  fillConst (prim__setDouble buf off v) (off + 1) (n - 1) v

||| Build a `LayerNormStateV2 n n CPU` with gamma initialised to 1.0
||| and beta to 0.0. Both register as C params under
||| `<prefix>_gamma` / `<prefix>_beta`.
export
layerNormLayerV2 : {n : Nat} -> (paramPrefix : String) ->
                   IO (LayerNormStateV2 n n CPU)
layerNormLayerV2 paramPrefix = do
  let nI = cast {to=Int} n
      gBuf = prim__allocDoubles nI
      gBuf' = fillConst gBuf 0 nI 1.0
      bBuf = prim__allocDoubles nI
      bBuf' = fillConst bBuf 0 nI 0.0
      gName = paramPrefix ++ "_gamma"
      bName = paramPrefix ++ "_beta"
      gPtr = prim__paramRegister gName (prim__createParam1d nI gBuf')
      bPtr = prim__paramRegister bName (prim__createParam1d nI bBuf')
  pure $ MkLayerNormV2 (MkTVar gPtr (Just gName)) (MkTVar bPtr (Just bName))


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
LayerLikeV2 LayerNormStateV2 where
  applyTVar st@(MkLayerNormV2 _ _) input = applyLayerNormV2 st input
  layerPrefixV2 _ = "lnV2"

||| Wrap a LayerNormV2 in `AnyLayerV2`.
export
layerNormLayerV2Any : {n : Nat} -> (paramPrefix : String) ->
                      IO (AnyLayerV2 n n CPU)
layerNormLayerV2Any pid = map (MkAnyLayerV2 LayerNormStateV2) (layerNormLayerV2 pid)
