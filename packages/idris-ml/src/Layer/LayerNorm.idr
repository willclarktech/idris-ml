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
-- The C backend currently exposes only `primLayerNorm2d {d}` (operates
-- on `[B, N]` shape). For 1D input we reshape `[n]` → `[1, n]`,
-- normalise, reshape back. ~3 tape entries per call (still much
-- cheaper than computing mean/var/sqrt manually).
--
-- GADT shape `i = o = n` lets the layer fit `LayerLike`'s arity
-- (mirrors Dropout's pattern).

public export
data LayerNormState : Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLayerNorm : TVec n d dt g -> TVec n d dt g -> LayerNormState n n d dt g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyLayerNorm : {0 d : Device} -> UserDeviceTape d => UserDeviceCore d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} ->
                   LayerNormState n n d dt g ->
                   TVec n d dt g ->
                   IO (LayerNormState n n d dt g, TVec n d dt g)
applyLayerNorm {n} st@(MkLayerNorm gamma beta) input = ioRerun (\_ =>
  let nI = cast {to=Int} n
      input2d = primReshape2d {d} input.tensorPtr 1 nI
      norm2d = primLayerNorm2d {d} input2d gamma.tensorPtr beta.tensorPtr 1.0e-5
      norm1d = primReshape1d {d} norm2d nI
  in (st, MkTensor norm1d Nothing))


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

||| Build a `LayerNormState n n TapeDev` with gamma initialised to 1.0
||| and beta to 0.0. Both register as C params under
||| `<prefix>_gamma` / `<prefix>_beta`.
export
layerNormLayer : UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> (paramPrefix : String) ->
                   IO (LayerNormState n n d dt WithGrad)
layerNormLayer paramPrefix = do
  let nI = cast {to=Int} n
      gBuf = prim__allocDoubles nI
      gBuf' = fillConst gBuf 0 nI 1.0
      bBuf = prim__allocDoubles nI
      bBuf' = fillConst bBuf 0 nI 0.0
      gName = paramPrefix ++ "_gamma"
      bName = paramPrefix ++ "_beta"
      gPtr = primParamRegister {d} gName (dtCreateParam1d {d} {t=dt} nI gBuf' (deviceStreamTag {d}))
      bPtr = primParamRegister {d} bName (dtCreateParam1d {d} {t=dt} nI bBuf' (deviceStreamTag {d}))
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
    primIO (primSetRequiresGrad {d} g.tensorPtr 1)
    primIO (primSetRequiresGrad {d} b.tensorPtr 1)
    pure (MkLayerNorm (retypeGrad g) (retypeGrad b))

||| Wrap a LayerNorm in `AnyLayer`.
export
layerNormLayerAny : UserDeviceTape d => RuntimeDType dt => Linked d => Compatible d dt => {n : Nat} -> (paramPrefix : String) ->
                      IO (AnyLayer n n d dt WithGrad)
layerNormLayerAny pid = map (MkAnyLayer LayerNormState) (layerNormLayer pid)
