||| `Dropout` — stateless inverted-dropout layer on the v1 `Nn` surface.
||| Training mode zeros elements with probability `p` and scales survivors
||| by `1/(1-p)`; eval mode is identity. `primDropout` is rank-agnostic, so
||| the batched `[b,n]` forward is the same call as `[n]`. No params.
module Ml.Nn.Dropout

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Module
import Ml.Tensor

%default total

-- Per-call mask seed. Re-declared here rather than imported from
-- Layer.Dropout to keep the Nn surface independent.
--
-- `PrimIO Int`, not `Int -> Int`: the C side advances the process-global
-- `rand()`, and every call site is the identical expression `dropoutSeed 0`,
-- so a pure-typed FFI is free to be common-subexpression-eliminated down to
-- one evaluation, pinning every forward in a run to a single mask
-- (gotchas.md "Zero-arg FFI CSE trap"). The IO typing is what guarantees the
-- call fires per forward; the dummy argument alone does not.
%foreign "C:dropout_random_seed,libidrisml"
prim__dropoutSeed : Int -> PrimIO Int

dropoutSeed : HasIO io => io Int
dropoutSeed = primIO (prim__dropoutSeed 0)

||| Inverted dropout (`i = o = n`); `training` toggles drop vs identity.
public export
data Dropout : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkDropout : (p : Double) -> (training : Bool) -> Dropout n n ex dt g

||| Params (stateless — empty param list). `p`/`training` ride at ω quantity
||| through the rebuild.
public export
Params Dropout where
  params (MkDropout p t)           = []
  reflect (MkDropout p t)          = MkBang [] # MkDropout p t
  castGrad (MkDropout p t)         = MkDropout p t
  discard (MkDropout _ _)          = pure ()
  setTraining mode (MkDropout p _) = MkDropout p mode

||| `Module` — sequences the `L IO` dropout op directly (identity in eval
||| mode).
public export
Module Dropout where
  forward (MkDropout p t) x = do
    y <- the (L IO (Tensor [b, i] ex dt g)) $
           if t
             then do
               s <- liftIO dropoutSeed
               ioRerunL (\_ => MkTensor (primDropout {ex} x.tensorPtr p 1 s) Nothing)
             else pure x
    pure1 (MkBang y # MkDropout p t)

||| Dropout with drop probability `p`, starting in training mode.
public export
dropout : Double -> Dropout n n ex dt g
dropout p = MkDropout p True
