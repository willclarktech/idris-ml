||| `MaxPool2D` / `MaxPool1D` — parameter-free pooling layers on the v1
||| `Nn` surface. Both are batched `Module`s with no learnable params (so
||| `Params` is empty and `castGrad` is identity). `MaxPool1D` has no
||| dedicated batched prim, so it runs the batched 2-D pool with a unit
||| height (`[b, c, 1, len]`, poolH=1), reusing the tested
||| `primMaxPool2dBatched`. Input/output are flattened, matching `Conv2D`
||| so a pool slots into a flattened `Seq`.
module Nn.Pool

import Data.Nat
import Data.Vect

import Executor
import Nn.Module
import Tensor

%default total

||| `PoolOutDim inDim kernel stride = (inDim − kernel) / stride + 1`. The
||| stride is pattern-matched so the divide uses `divNatNZ` with a `NonZero`
||| proof — total, covering, and (unlike interface `div`) reducing at the
||| type level, which the Module's flattened o-index needs. Zero stride is
||| degenerate (0); real pools always stride ≥ 1.
public export
PoolOutDim : Nat -> Nat -> Nat -> Nat
PoolOutDim _     _      Z     = 0
PoolOutDim inDim kernel (S s) = divNatNZ (inDim `minus` kernel) (S s) ItIsSucc + 1

----------------------------------------------------------------------
-- MaxPool2D
----------------------------------------------------------------------

public export
data MaxPool2D :
  (c : Nat) -> (inH : Nat) -> (inW : Nat) ->
  (poolH : Nat) -> (poolW : Nat) -> (strH : Nat) -> (strW : Nat) ->
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkMaxPool2D :
    MaxPool2D c inH inW poolH poolW strH strW
              (c * (inH * inW))
              (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
              ex dt g

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  Module (MaxPool2D c inH inW poolH poolW strH strW) where
  forward MkMaxPool2D input = ioRerun (\_ =>
    let bI    = cast {to=Int} b
        cI    = cast {to=Int} c
        hI    = cast {to=Int} inH
        wI    = cast {to=Int} inW
        inp4d = primReshape4d {ex} input.tensorPtr bI cI hI wI
        outT  = primMaxPool2dBatched {ex} inp4d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                            (cast {to=Int} strH) (cast {to=Int} strW)
        outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
        out2d = primReshape2d {ex} outT bI (cast {to=Int} outFlat)
    in MkTensor out2d Nothing)

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  Params (MaxPool2D c inH inW poolH poolW strH strW) where
  params _ = []
  castGrad MkMaxPool2D = MkMaxPool2D

||| MaxPool2D with the given window + stride (no params, nothing to init).
public export
maxPool2d : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
            MaxPool2D c inH inW poolH poolW strH strW
                      (c * (inH * inW))
                      (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                      ex dt g
maxPool2d = MkMaxPool2D

----------------------------------------------------------------------
-- MaxPool1D
----------------------------------------------------------------------

public export
data MaxPool1D :
  (c : Nat) -> (len : Nat) -> (poolK : Nat) -> (str : Nat) ->
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkMaxPool1D :
    MaxPool1D c len poolK str
              (c * len)
              (c * PoolOutDim len poolK str)
              ex dt g

public export
{c, len, poolK, str : Nat} -> Module (MaxPool1D c len poolK str) where
  forward MkMaxPool1D input = ioRerun (\_ =>
    let bI    = cast {to=Int} b
        cI    = cast {to=Int} c
        lenI  = cast {to=Int} len
        inp4d = primReshape4d {ex} input.tensorPtr bI cI 1 lenI
        outT  = primMaxPool2dBatched {ex} inp4d 1 (cast {to=Int} poolK) 1 (cast {to=Int} str)
        outFlat = c * PoolOutDim len poolK str
        out2d = primReshape2d {ex} outT bI (cast {to=Int} outFlat)
    in MkTensor out2d Nothing)

public export
{c, len, poolK, str : Nat} -> Params (MaxPool1D c len poolK str) where
  params _ = []
  castGrad MkMaxPool1D = MkMaxPool1D

||| MaxPool1D with the given window + stride (no params, nothing to init).
public export
maxPool1d : {c, len, poolK, str : Nat} ->
            MaxPool1D c len poolK str
                      (c * len)
                      (c * PoolOutDim len poolK str)
                      ex dt g
maxPool1d = MkMaxPool1D
