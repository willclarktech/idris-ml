||| `Conv2D` — 2-D convolution on the v1 `Nn` surface, and the first
||| genuinely 4-D layer. Unlike most ports it IS a batched `Module`: it has
||| a real batched forward (`primConv2dBatched`), and partially applying the
||| eight conv type-params (`Conv2D inC outC h w kH kW padH padW`) yields the
||| `Nat → Nat → Executor → DType → Type` kind `Module`/`Params` need — the
||| flattened `i = inC·h·w` / `o = outC·outH·outW` are the last two indices
||| (same trick the legacy `LayerLike (Conv2DState …)` used). Input/output
||| are flattened (`[b, inC·h·w] → [b, outC·outH·outW]`); the forward
||| reshapes to `[b,c,h,w]` internally. `ConvOutDim` computes the output
||| spatial size per axis, so a `Conv2D` slots into a flattened `Seq`
||| without a hand-written shape constant.
module Nn.Conv

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| `ConvOutDim inDim kernel pad = inDim + 2·pad − kernel + 1`.
public export
ConvOutDim : Nat -> Nat -> Nat -> Nat
ConvOutDim inDim kernel pad = ((inDim + 2 * pad) `minus` kernel) + 1

||| 2-D conv: kernel `[outC, inC, kH, kW]`, bias `[outC]`. The trailing two
||| indices are the flattened in/out sizes (so the kind fits Module/Params).
public export
data Conv2D :
  (inC : Nat) -> (outC : Nat) -> (h : Nat) -> (w : Nat) ->
  (kH : Nat) -> (kW : Nat) -> (padH : Nat) -> (padW : Nat) ->
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkConv2D :
    Tensor [outC, inC, kH, kW] ex dt g ->
    TVec outC ex dt g ->
    Conv2D inC outC h w kH kW padH padW
           (inC * (h * w))
           (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
           ex dt g

public export
{inC, outC, h, w, kH, kW, padH, padW : Nat} ->
  Module (Conv2D inC outC h w kH kW padH padW) where
  forward (MkConv2D ker bias) input = ioRerun (\_ =>
    let bI    = cast {to=Int} b
        inCI  = cast {to=Int} inC
        hI    = cast {to=Int} h
        wI    = cast {to=Int} w
        inp4d = primReshape4d {ex} input.tensorPtr bI inCI hI wI
        padHI = cast {to=Int} padH
        padWI = cast {to=Int} padW
        outT  = primConv2dBatched {ex} inp4d ker.tensorPtr bias.tensorPtr padHI padWI 1 1
        outFlat = outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)
        out2d = primReshape2d {ex} outT bI (cast {to=Int} outFlat)
    in MkTensor out2d Nothing)

public export
{inC, outC, h, w, kH, kW, padH, padW : Nat} ->
  Params (Conv2D inC outC h w kH kW padH padW) where
  params (MkConv2D ker bias) = [toParam ker, toParam bias]
  castGrad (MkConv2D ker bias) = MkConv2D (retypeGrad ker) (retypeGrad bias)

||| Construct a `Conv2D` inside an `Init` derivation. He-normal kernel
||| (std = √(2/fan_in), fan_in = inC·kH·kW), zero bias. Registers
||| `<scope>.conv2d_<n>.weight` (kernel) / `.bias` (PyTorch Conv2d names).
export partial
conv2d : {0 ex : Executor} -> Backend ex dt =>
         {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
         Init (Conv2D inC outC h w kH kW padH padW
                      (inC * (h * w))
                      (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                      ex dt WithGrad)
conv2d = do
  name <- freshChild "conv2d"
  let kerStd = sqrt (2.0 / cast {to=Double} (inC * kH * kW))
  ker  <- liftIO $ tparam4dNormal {ex} {dt} {a=outC} {b=inC} {c=kH} {e=kW} (name ++ ".weight") 0.0 kerStd
  bias <- liftIO $ tparam1dConst  {ex} {dt} {n=outC} (name ++ ".bias") 0.0
  pure (MkConv2D ker bias)

----------------------------------------------------------------------
-- Conv1D
----------------------------------------------------------------------

||| 1-D conv: kernel `[outC, inC, kL]`, bias `[outC]`. There is no batched
||| 1-D conv prim, so the batched forward runs the existing batched 2-D op
||| with a unit height (`[b, inC, 1, len]` × `[outC, inC, 1, kL]`, padH=0),
||| reusing the tested `primConv2dBatched`. Trailing two indices are the
||| flattened in/out sizes (so the kind fits Module/Params).
public export
data Conv1D :
  (inC : Nat) -> (outC : Nat) -> (len : Nat) -> (kL : Nat) -> (pad : Nat) ->
  Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkConv1D :
    Tensor [outC, inC, kL] ex dt g ->
    TVec outC ex dt g ->
    Conv1D inC outC len kL pad
           (inC * len)
           (outC * ConvOutDim len kL pad)
           ex dt g

public export
{inC, outC, len, kL, pad : Nat} -> Module (Conv1D inC outC len kL pad) where
  forward (MkConv1D ker bias) input = ioRerun (\_ =>
    let bI     = cast {to=Int} b
        inCI   = cast {to=Int} inC
        lenI   = cast {to=Int} len
        outCI  = cast {to=Int} outC
        kLI    = cast {to=Int} kL
        inp4d  = primReshape4d {ex} input.tensorPtr bI inCI 1 lenI
        ker4d  = primReshape4d {ex} ker.tensorPtr outCI inCI 1 kLI
        outT   = primConv2dBatched {ex} inp4d ker4d bias.tensorPtr 0 (cast {to=Int} pad) 1 1
        outFlat = outC * ConvOutDim len kL pad
        out2d  = primReshape2d {ex} outT bI (cast {to=Int} outFlat)
    in MkTensor out2d Nothing)

public export
{inC, outC, len, kL, pad : Nat} -> Params (Conv1D inC outC len kL pad) where
  params (MkConv1D ker bias) = [toParam ker, toParam bias]
  castGrad (MkConv1D ker bias) = MkConv1D (retypeGrad ker) (retypeGrad bias)

||| Construct a `Conv1D` inside an `Init` derivation. He-normal kernel
||| (std = √(2/fan_in), fan_in = inC·kL), zero bias. Registers
||| `<scope>.conv1d_<n>.weight` (kernel) / `.bias`.
export partial
conv1d : {0 ex : Executor} -> Backend ex dt =>
         {inC, outC, len, kL, pad : Nat} ->
         Init (Conv1D inC outC len kL pad
                      (inC * len)
                      (outC * ConvOutDim len kL pad)
                      ex dt WithGrad)
conv1d = do
  name <- freshChild "conv1d"
  let kerStd = sqrt (2.0 / cast {to=Double} (inC * kL))
  ker  <- liftIO $ tparam3dNormal {ex} {dt} {a=outC} {b=inC} {c=kL} (name ++ ".weight") 0.0 kerStd
  bias <- liftIO $ tparam1dConst  {ex} {dt} {n=outC} (name ++ ".bias") 0.0
  pure (MkConv1D ker bias)
