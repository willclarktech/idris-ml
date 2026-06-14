||| `LoraLinear` — low-rank adaptation (Hu et al. 2021) wrapping an
||| `Nn.Linear`: `y = W·x + b + (α/r)·B·(A·x)`. Params-only (NOT a batched
||| `Module`): the forward is 1-D (`tlinear` + two `tmv`s keep the rank-r
||| factoring) and the layer carries a third Nat (the rank, an implicit
||| field). LoRA is composed inside attention forwards, not dropped into a
||| generic `Seq`, so the Module-shape miss doesn't bite.
|||
||| At init B = 0, so the LoRA delta is identically zero and the wrapped
||| layer is bit-identical to the bare base.
module Nn.LoraLinear

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module
import Nn.Linear

%default total

||| A LoRA-adapted linear. `rank` is an implicit (runtime-available) field;
||| `base`'s params keep their own registry names, `loraA`/`loraB` add two.
public export
record LoraLinear (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLoraLinear
  {rank : Nat}
  base  : Linear i o ex dt g
  loraA : Tensor [rank, i] ex dt g
  loraB : Tensor [o, rank] ex dt g
  alpha : Double

public export
Params LoraLinear where
  params (MkLoraLinear base a b _) = params base ++ [toParam a, toParam b]
  castGrad (MkLoraLinear base a b alpha) = MkLoraLinear (castGrad base) (retypeGrad a) (retypeGrad b) alpha

||| 1-D LoRA forward: `W·x + b + (α/r)·B·(A·x)`.
export
loraForward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i, o : Nat} ->
              LoraLinear i o ex dt g -> Tensor [i] ex dt g -> IO (Tensor [o] ex dt g)
loraForward (MkLoraLinear {rank} base a b alpha) input = do
  baseOut <- tlinear base.weightT input base.biasT
  aOut    <- tmv a input
  bOut    <- tmv b aOut
  scaled  <- tmulScalar bOut (alpha / cast {to=Double} rank)
  tadd baseOut scaled

||| Wrap an `Nn.Linear` with trainable LoRA adapters inside an `Init`
||| derivation. Registers `<scope>.lora_<n>.lora_A` (N(0, 1/√rank)) /
||| `.lora_B` (zero). The base is taken as-is (its params stay registered).
export
loraLinear : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
             (rank : Nat) -> (alpha : Double) -> Linear i o ex dt WithGrad ->
             Init (LoraLinear i o ex dt WithGrad)
loraLinear rank alpha base = do
  name <- freshChild "lora"
  a <- liftIO $ tparam2dNormal {ex} {dt} {o=rank} {i=i} (name ++ ".lora_A") 0.0 (1.0 / sqrt (cast {to=Double} rank))
  b <- liftIO $ tparam2dConst  {ex} {dt} {o=o} {i=rank} (name ++ ".lora_B") 0.0
  pure (MkLoraLinear {rank} base a b alpha)
