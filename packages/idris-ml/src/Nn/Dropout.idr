||| `Dropout` — stateless inverted-dropout layer on the v1 `Nn` surface.
||| Training mode zeros elements with probability `p` and scales survivors
||| by `1/(1-p)`; eval mode is identity. `primDropout` is rank-agnostic, so
||| the batched `[b,n]` forward is the same call as `[n]`. No params.
module Nn.Dropout

import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

-- Per-call seed (dummy arg defeats CSE). Re-declared here rather than
-- imported from Layer.Dropout to keep the Nn surface independent.
%foreign "C:dropout_random_seed,libidrisml"
dropoutSeed : Int -> Int

||| Inverted dropout (`i = o = n`); `training` toggles drop vs identity.
public export
data Dropout : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkDropout : (p : Double) -> (training : Bool) -> Dropout n n ex dt g

public export
Module Dropout where
  forward (MkDropout p training) x =
    if training
      then ioRerun (\_ => MkTensor (primDropout {ex} x.tensorPtr p 1 (dropoutSeed 0)) Nothing)
      else pure x

public export
Params Dropout where
  params _ = []
  castGrad (MkDropout p training) = MkDropout p training

||| Dropout with drop probability `p`, starting in training mode.
public export
dropout : Double -> Dropout n n ex dt g
dropout p = MkDropout p True

||| Toggle training/eval mode.
public export
setTraining : Bool -> Dropout n n ex dt g -> Dropout n n ex dt g
setTraining mode (MkDropout p _) = MkDropout p mode
