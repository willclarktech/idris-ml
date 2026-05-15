module Layer.LayerNorm

import Data.Vect

import Device
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- LayerNorm — typed-surface layer normalisation (Path C)
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
-- GADT shape `i = o = n` lets the layer fit `LayerLike`'s arity
-- (mirrors Dropout's pattern).

public export
data LayerNormState : Nat -> Nat -> (0 _ : Device) -> (0 _ : GradMode) -> Type where
  MkLayerNorm : TVec n d g -> TVec n d g -> LayerNormState n n d g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyLayerNorm : {0 d : Device} -> UserDeviceConv d => {n : Nat} ->
                   LayerNormState n n d g ->
                   TVec n d g ->
                   (LayerNormState n n d g, TVec n d g)
applyLayerNorm {n} st@(MkLayerNorm gamma beta) input =
  let nI = cast {to=Int} n
      input2d = prim__reshape2d input.tensorPtr 1 nI
      norm2d = prim__layerNorm2d input2d gamma.tensorPtr beta.tensorPtr 1.0e-5
      norm1d = prim__reshape1d norm2d nI
  in (st, MkTensor norm1d Nothing)


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

||| Build a `LayerNormState n n CPU` with gamma initialised to 1.0
||| and beta to 0.0. Both register as C params under
||| `<prefix>_gamma` / `<prefix>_beta`.
export
layerNormLayer : {n : Nat} -> (paramPrefix : String) ->
                   IO (LayerNormState n n CPU WithGrad)
layerNormLayer paramPrefix = do
  let nI = cast {to=Int} n
      gBuf = prim__allocDoubles nI
      gBuf' = fillConst gBuf 0 nI 1.0
      bBuf = prim__allocDoubles nI
      bBuf' = fillConst bBuf 0 nI 0.0
      gName = paramPrefix ++ "_gamma"
      bName = paramPrefix ++ "_beta"
      gPtr = prim__paramRegister gName (prim__createParam1d nI gBuf')
      bPtr = prim__paramRegister bName (prim__createParam1d nI bBuf')
  pure $ MkLayerNorm (MkTensor gPtr (Just gName)) (MkTensor bPtr (Just bName))


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike LayerNormState where
  applyVar st@(MkLayerNorm _ _) input = applyLayerNorm st input
  layerPrefix _ = "ln"

  freezeLayer (MkLayerNorm g b) = do
    g' <- weakenGrad g
    b' <- weakenGrad b
    pure (MkLayerNorm g' b')

  unfreezeLayer (MkLayerNorm g b) = do
    primIO (prim__setRequiresGrad g.tensorPtr 1)
    primIO (prim__setRequiresGrad b.tensorPtr 1)
    pure (MkLayerNorm (retypeGrad g) (retypeGrad b))

||| Wrap a LayerNorm in `AnyLayer`.
export
layerNormLayerAny : {n : Nat} -> (paramPrefix : String) ->
                      IO (AnyLayer n n CPU WithGrad)
layerNormLayerAny pid = map (MkAnyLayer LayerNormState) (layerNormLayer pid)
