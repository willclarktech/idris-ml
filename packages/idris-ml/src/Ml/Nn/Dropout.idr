||| `Dropout` — stateless inverted-dropout layer on the v1 `Nn` surface.
||| Training mode zeros elements with probability `p` and scales survivors
||| by `1/(1-p)`; eval mode is identity. No params.
|||
||| The mask comes from an injected `MaskSource` (`Ml.Rng`) — the JAX/Flax
||| shape, landed at construction because `Module.forward`'s signature is
||| fixed by the typeclass. `dropout p` is the ordinary live layer (a fresh
||| seed per forward, the fused rank-agnostic `primDropout` kernel);
||| `dropoutWith src p` accepts a recorded source, whose masks apply as a
||| no-grad constant tensor times the input. That multiply IS dropout: the
||| mask holds exactly `0` and `1/(1-p)`, so `tmul` against a no-grad factor
||| has dropout's backward on every backend, and multiplying by exact 0/1
||| values is bit-identical however the reference grouped its own product.
module Ml.Nn.Dropout

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Module
import Ml.Rng
import Ml.Tensor

%default total

||| Inverted dropout (`i = o = n`); `training` toggles drop vs identity.
public export
data Dropout : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkDropout : (p : Double) -> (training : Bool) -> MaskSource -> Dropout n n ex dt g

||| Params (stateless — empty param list). `p`/`training`/the source ride at
||| ω quantity through the rebuild.
public export
Params Dropout where
  params (MkDropout p t src)           = []
  reflect (MkDropout p t src)          = MkBang [] # MkDropout p t src
  castGrad (MkDropout p t src)         = MkDropout p t src
  discard (MkDropout _ _ _)            = pure ()
  setTraining mode (MkDropout p _ src) = MkDropout p mode src

||| A recorded mask, applied as data: bits (`True` = kept) become a no-grad
||| constant `[b, i]` tensor of `0` / `1/(1-p)`, multiplied elementwise into
||| the activation. `recordedMasks` already checked the bit count, so the
||| length hole here is unreachable unless the two drifted.
givenMask : {b, i : Nat} -> {0 ex : Executor} -> Backend ex dt =>
            (p : Double) -> List Bool -> IO (Tensor [b, i] ex dt NoGrad)
givenMask p bits =
  let scale = 1.0 / (1.0 - p)
      vals  = map (\keep => if keep then scale else 0.0) bits
  in case exactLength (b * i) (fromList vals) of
       Just v  => tensor {dims = [b, i]} (FromVect v)
       Nothing => pure (assert_total $ idris_crash
                          "Ml.Nn.Dropout.givenMask: mask bits do not fill [b, i]")

||| `Module` — sequences the `L IO` dropout op directly (identity in eval
||| mode). A `FreshSeed` runs the fused C kernel; `GivenBits` applies the
||| recorded mask as a no-grad factor.
public export
Module Dropout where
  -- Matching `MkDropout : Dropout n n …` substitutes the method's `i` to `o`
  -- inside `x`'s type, but the name `i` stays bound to the pre-substitution
  -- rigid — a `[b, i]`-typed mask will not unify with `x`. Everything below
  -- is pinned to `o` for that reason.
  forward (MkDropout p t src) x = do
    y <- the (L IO (Tensor [b, i] ex dt g)) $
           if t
             then do
               spec <- liftIO (src.nextMask (b * o))
               case spec of
                 FreshSeed s    =>
                   ioRerunL (\_ => MkTensor (primDropout {ex} x.tensorPtr p 1 s) Nothing)
                 GivenBits bits => do
                   mask <- liftIO (givenMask {b} {i = o} p bits)
                   res  <- liftIO (x *. retypeGrad mask)
                   pure res
             else pure x
    pure1 (MkBang y # MkDropout p t src)

||| Dropout with drop probability `p`, starting in training mode, drawing
||| live masks (a fresh seed per forward from the process-global generator).
public export
dropout : Double -> Dropout n n ex dt g
dropout p = MkDropout p True liveMasks

||| Dropout drawing its masks from `src` — pass `replay.masks` to reproduce
||| a recorded run's exact drop pattern.
public export
dropoutWith : MaskSource -> Double -> Dropout n n ex dt g
dropoutWith src p = MkDropout p True src
