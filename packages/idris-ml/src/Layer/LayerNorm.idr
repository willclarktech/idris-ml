module Layer.LayerNorm

import Data.Vect

import Executor
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- LayerNorm — typed-surface layer normalisation (Path C)
----------------------------------------------------------------------
--
-- Normalises along the last (only) dim of a 1D `TVec n ex` input,
-- then applies a learnable scale (gamma) + shift (beta).
--
-- The C backend currently exposes only `primLayerNorm2d {ex}` (operates
-- on `[B, N]` shape). For 1D input we reshape `[n]` → `[1, n]`,
-- normalise, reshape back. ~3 tape entries per call (still much
-- cheaper than computing mean/var/sqrt manually).
--
-- GADT shape `i = o = n` lets the layer fit `LayerLike`'s arity
-- (mirrors Dropout's pattern).

public export
data LayerNormState : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkLayerNorm : TVec n ex dt g -> TVec n ex dt g -> LayerNormState n n ex dt g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyLayerNorm : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {n : Nat} ->
                   LayerNormState n n ex dt g ->
                   TVec n ex dt g ->
                   IO (LayerNormState n n ex dt g, TVec n ex dt g)
applyLayerNorm {n} st@(MkLayerNorm gamma beta) input = ioRerun (\_ =>
  let nI = cast {to=Int} n
      input2d = primReshape2d {ex} input.tensorPtr 1 nI
      norm2d = primLayerNorm2d {ex} input2d gamma.tensorPtr beta.tensorPtr 1.0e-5
      norm1d = primReshape1d {ex} norm2d nI
  in (st, MkTensor norm1d Nothing))


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `LayerNormState n n TapeExecutor` with gamma initialised to 1.0
||| and beta to 0.0. Both register as C params under
||| `<prefix>_gamma` / `<prefix>_beta`.
export
layerNormLayer : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => {n : Nat} -> (paramPrefix : String) ->
                   IO (LayerNormState n n ex dt WithGrad)
layerNormLayer paramPrefix = do
  let gName = paramPrefix ++ "_gamma"
      bName = paramPrefix ++ "_beta"
  gamma <- tparam1dConst {ex} {dt} {n} gName 1.0
  beta  <- tparam1dConst {ex} {dt} {n} bName 0.0
  pure $ MkLayerNorm gamma beta


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
    primIO (primSetRequiresGrad {ex} g.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} b.tensorPtr 1)
    pure (MkLayerNorm (retypeGrad g) (retypeGrad b))

||| Wrap a LayerNorm in `AnyLayer`.
export
layerNormLayerAny : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => {n : Nat} -> (paramPrefix : String) ->
                      IO (AnyLayer n n ex dt WithGrad)
layerNormLayerAny pid = map (MkAnyLayer LayerNormState) (layerNormLayer pid)
