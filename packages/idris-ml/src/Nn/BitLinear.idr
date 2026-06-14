||| `BitLinear` — BitNet b1.58 ternary-quantized linear on the v1 `Nn`
||| surface. Params-only, and the dual-dtype is collapsed to the single
||| surface `dt` (= the compute dtype): the ternary weight is a fixed
||| `Ternary`-typed internal field, not a type parameter. `y = (W_ternary ⊙
||| scale) · x + bias`, via the fused `tBitlinearFwd` (needs
||| `UserExecutorQuant`, beyond the `Backend` bundle — so this is a
||| standalone forward, not a `Module`).
|||
||| BitNet freezes the ternary weight + scale (`NoGrad`); only the bias
||| trains. There is no random `Init` smart constructor — the ternary
||| weight comes from a checkpoint (packed bytes via
||| `tCreateTernaryPacked2d`); `bitLinear` builds one from ready tensors.
module Nn.BitLinear

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| Ternary linear. `dt` is the compute dtype (scale/bias/activations);
||| the weight is fixed `Ternary`. Weight + scale frozen `NoGrad`; bias
||| trainable.
public export
record BitLinear (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkBitLinear
  weightT : Tensor [o, i] ex Ternary NoGrad
  scaleT  : TVec o ex dt NoGrad
  biasT   : TVec o ex dt g

||| All three tensors (frozen ternary weight, frozen scale, trainable bias)
||| — dtype-erased into `SomeParam`, so the mixed Ternary/float dtypes
||| coexist. The frozen pair is grad-gated out of the optimizer.
public export
Params BitLinear where
  params (MkBitLinear w s b) = [toParam w, toParam s, toParam b]
  -- Only the bias carries `g`; the ternary weight + scale are frozen `NoGrad`.
  castGrad (MkBitLinear w s b) = MkBitLinear w s (retypeGrad b)

||| Linear-resource params. All three handles are ω fields (reflected +
||| rebuilt); only the bias carries `g`.
public export
ParamsL BitLinear where
  reflectL (MkBitLinear w s b)  = MkBang [toParam w, toParam s, toParam b] # MkBitLinear w s b
  castGradL (MkBitLinear w s b) = MkBitLinear w s (retypeGrad b)
  discardL (MkBitLinear _ _ _)  = pure ()

||| Build a `BitLinear` from ready tensors (the ternary weight typically
||| from a checkpoint's packed bytes).
public export
bitLinear : {0 ex : Executor} -> {0 dt : DType} -> {i, o : Nat} ->
            Tensor [o, i] ex Ternary NoGrad -> TVec o ex dt NoGrad -> TVec o ex dt WithGrad ->
            BitLinear i o ex dt WithGrad
bitLinear = MkBitLinear

||| Standalone quantized forward `(W ⊙ scale)·x + bias` (1-D). Needs the
||| quant capability; not a `Module`.
export
bitLinearForward : {0 ex : Executor} -> UserExecutorQuant ex => {0 dt : DType} -> {0 g : GradMode} -> {i, o : Nat} ->
                   BitLinear i o ex dt g -> Tensor [i] ex dt g -> IO (Tensor [o] ex dt g)
bitLinearForward (MkBitLinear w s b) x = tBitlinearFwd w s x b
