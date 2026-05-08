module Layer.ConvV2

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Layer.Conv  -- reuse `ConvOutDim` and `PoolOutDim` from V1
import Sampler
import Variable


----------------------------------------------------------------------
-- ConvV2 — typed-surface conv + pool layers (Path C)
----------------------------------------------------------------------
--
-- Six layers in one file (matches the V1 `Layer/Conv.idr` shape):
--   - Conv2DV2 / Conv1DV2  (learnable kernel + bias)
--   - MaxPool2DV2 / MaxPool1DV2 (no params)
--   - AvgPool2DV2 / AvgPool1DV2 (no params)
--
-- All take flattened input shape `[c * spatial]` and produce
-- flattened output `[outC * spatialOut]`. The forward path reshapes
-- to `[c, spatial]` (1D) or `[c, h, w]` (2D), calls the C op, then
-- flattens. Type-level `i / o` indices are computed with V1's
-- `ConvOutDim` / `PoolOutDim` helpers.


----------------------------------------------------------------------
-- Conv2DV2
----------------------------------------------------------------------

public export
data Conv2DStateV2 :
  (inC : Nat) -> (outC : Nat) -> (h : Nat) -> (w : Nat) ->
  (kH : Nat) -> (kW : Nat) -> (padH : Nat) -> (padW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkConv2DV2 :
    TVar [outC, inC, kH, kW] d ->                       -- kernel
    TVec outC d ->                                      -- bias
    Conv2DStateV2 inC outC h w kH kW padH padW
                  (inC * (h * w))
                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                  d

%default partial

export
applyConv2DV2 : {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                Conv2DStateV2 inC outC h w kH kW padH padW
                              (inC * (h * w))
                              (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                              d ->
                TVec (inC * (h * w)) d ->
                TVec (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)) d
applyConv2DV2 {inC} {outC} {h} {w} {kH} {kW} {padH} {padW}
              (MkConv2DV2 ker bias) input =
  let inCI = cast {to=Int} inC
      hI = cast {to=Int} h
      wI = cast {to=Int} w
      inp3d = prim__reshape3d input.tensorPtr inCI hI wI
      padHI = cast {to=Int} padH
      padWI = cast {to=Int} padW
      outT = prim__conv2d inp3d ker.tensorPtr bias.tensorPtr padHI padWI 1 1
      outFlat = outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)
      flatPtr = prim__reshape1d outT (cast {to=Int} outFlat)
  in MkTVar flatPtr Nothing

-- Pack a Vect of Doubles into a buffer.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build a Conv2DV2 layer with He-normal kernel init and zero bias.
export
conv2dLayerV2 : {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                (paramPrefix : String) ->
                IO (Conv2DStateV2 inC outC h w kH kW padH padW
                                  (inC * (h * w))
                                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                                  CPU)
conv2dLayerV2 paramPrefix = do
  let kerCount = outC * inC * kH * kW
  kerVals <- traverse (\_ => he normal (inC * kH * kW) outC)
                      (Vect.replicate kerCount ())
  let kerBuf = prim__allocDoubles (cast {to=Int} kerCount)
      kerBuf' = packDoubles kerBuf 0 kerVals
      biasBuf = prim__allocDoubles (cast {to=Int} outC)
      biasBuf' = zeroBuf biasBuf 0 (cast {to=Int} outC)
      kerName = paramPrefix ++ "_kernel"
      biasName = paramPrefix ++ "_bias"
      kerPtr = prim__paramRegister kerName
        (prim__createParam4d (cast outC) (cast inC) (cast kH) (cast kW) kerBuf')
      biasPtr = prim__paramRegister biasName
        (prim__createParam1d (cast outC) biasBuf')
      kerTV : TVar [outC, inC, kH, kW] CPU
      kerTV = MkTVar kerPtr (Just kerName)
      biasTV : TVec outC CPU
      biasTV = MkTVar biasPtr (Just biasName)
  pure $ MkConv2DV2 kerTV biasTV

public export
{inC, outC, h, w, kH, kW, padH, padW : Nat} ->
  LayerLikeV2 (Conv2DStateV2 inC outC h w kH kW padH padW) where
  applyTVar st@(MkConv2DV2 _ _) input = (st, applyConv2DV2 st input)
  layerPrefixV2 _ = "convV2"

export
conv2dLayerV2Any : {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                   (paramPrefix : String) ->
                   IO (AnyLayerV2 (inC * (h * w))
                                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                                  CPU)
conv2dLayerV2Any pid =
  map (MkAnyLayerV2 (Conv2DStateV2 inC outC h w kH kW padH padW))
      (conv2dLayerV2 {inC} {outC} {h} {w} {kH} {kW} {padH} {padW} pid)


----------------------------------------------------------------------
-- Conv1DV2
----------------------------------------------------------------------

public export
data Conv1DStateV2 :
  (inC : Nat) -> (outC : Nat) -> (len : Nat) -> (kL : Nat) -> (pad : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkConv1DV2 :
    TVar [outC, inC, kL] d ->
    TVec outC d ->
    Conv1DStateV2 inC outC len kL pad
                  (inC * len)
                  (outC * ConvOutDim len kL pad)
                  d

export
applyConv1DV2 : {inC, outC, len, kL, pad : Nat} ->
                Conv1DStateV2 inC outC len kL pad
                              (inC * len)
                              (outC * ConvOutDim len kL pad) d ->
                TVec (inC * len) d ->
                TVec (outC * ConvOutDim len kL pad) d
applyConv1DV2 {inC} {outC} {len} {kL} {pad} (MkConv1DV2 ker bias) input =
  let inCI = cast {to=Int} inC
      lenI = cast {to=Int} len
      inp2d = prim__reshape2d input.tensorPtr inCI lenI
      outT = prim__conv1d inp2d ker.tensorPtr bias.tensorPtr (cast {to=Int} pad) 1
      outFlat = outC * ConvOutDim len kL pad
  in MkTVar (prim__reshape1d outT (cast {to=Int} outFlat)) Nothing

export
conv1dLayerV2 : {inC, outC, len, kL, pad : Nat} ->
                (paramPrefix : String) ->
                IO (Conv1DStateV2 inC outC len kL pad
                                  (inC * len)
                                  (outC * ConvOutDim len kL pad) CPU)
conv1dLayerV2 paramPrefix = do
  let kerCount = outC * inC * kL
  kerVals <- traverse (\_ => he normal (inC * kL) outC)
                      (Vect.replicate kerCount ())
  let kerBuf = prim__allocDoubles (cast {to=Int} kerCount)
      kerBuf' = packDoubles kerBuf 0 kerVals
      biasBuf = prim__allocDoubles (cast {to=Int} outC)
      biasBuf' = zeroBuf biasBuf 0 (cast {to=Int} outC)
      kerName = paramPrefix ++ "_kernel"
      biasName = paramPrefix ++ "_bias"
      kerPtr = prim__paramRegister kerName
        (prim__createParam3d (cast outC) (cast inC) (cast kL) kerBuf')
      biasPtr = prim__paramRegister biasName
        (prim__createParam1d (cast outC) biasBuf')
      kerTV : TVar [outC, inC, kL] CPU
      kerTV = MkTVar kerPtr (Just kerName)
      biasTV : TVec outC CPU
      biasTV = MkTVar biasPtr (Just biasName)
  pure $ MkConv1DV2 kerTV biasTV

public export
{inC, outC, len, kL, pad : Nat} ->
  LayerLikeV2 (Conv1DStateV2 inC outC len kL pad) where
  applyTVar st@(MkConv1DV2 _ _) input = (st, applyConv1DV2 st input)
  layerPrefixV2 _ = "conv1dV2"

export
conv1dLayerV2Any : {inC, outC, len, kL, pad : Nat} ->
                   (paramPrefix : String) ->
                   IO (AnyLayerV2 (inC * len) (outC * ConvOutDim len kL pad) CPU)
conv1dLayerV2Any pid =
  map (MkAnyLayerV2 (Conv1DStateV2 inC outC len kL pad))
      (conv1dLayerV2 {inC} {outC} {len} {kL} {pad} pid)


----------------------------------------------------------------------
-- MaxPool2DV2 / AvgPool2DV2 (no learnable params)
----------------------------------------------------------------------

public export
data MaxPool2DStateV2 :
  (c : Nat) -> (inH : Nat) -> (inW : Nat) ->
  (poolH : Nat) -> (poolW : Nat) -> (strH : Nat) -> (strW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkMaxPool2DV2 :
    MaxPool2DStateV2 c inH inW poolH poolW strH strW
                     (c * (inH * inW))
                     (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                     d

export
applyMaxPool2DV2 : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   MaxPool2DStateV2 c inH inW poolH poolW strH strW
                                    (c * (inH * inW))
                                    (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                                    d ->
                   TVec (c * (inH * inW)) d ->
                   TVec (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)) d
applyMaxPool2DV2 {c} {inH} {inW} {poolH} {poolW} {strH} {strW} _ input =
  let cI = cast {to=Int} c
      hI = cast {to=Int} inH
      wI = cast {to=Int} inW
      inp3d = prim__reshape3d input.tensorPtr cI hI wI
      outT = prim__maxPool2d inp3d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                   (cast {to=Int} strH) (cast {to=Int} strW)
      outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
  in MkTVar (prim__reshape1d outT (cast {to=Int} outFlat)) Nothing

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  LayerLikeV2 (MaxPool2DStateV2 c inH inW poolH poolW strH strW) where
  applyTVar st@MkMaxPool2DV2 input = (st, applyMaxPool2DV2 st input)
  layerPrefixV2 _ = "maxpool2dV2"

export
maxPool2dLayerV2 : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AnyLayerV2 (c * (inH * inW))
                              (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                              d
maxPool2dLayerV2 =
  MkAnyLayerV2 (MaxPool2DStateV2 c inH inW poolH poolW strH strW)
               MkMaxPool2DV2

public export
data AvgPool2DStateV2 :
  (c : Nat) -> (inH : Nat) -> (inW : Nat) ->
  (poolH : Nat) -> (poolW : Nat) -> (strH : Nat) -> (strW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkAvgPool2DV2 :
    AvgPool2DStateV2 c inH inW poolH poolW strH strW
                     (c * (inH * inW))
                     (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                     d

export
applyAvgPool2DV2 : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AvgPool2DStateV2 c inH inW poolH poolW strH strW
                                    (c * (inH * inW))
                                    (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                                    d ->
                   TVec (c * (inH * inW)) d ->
                   TVec (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)) d
applyAvgPool2DV2 {c} {inH} {inW} {poolH} {poolW} {strH} {strW} _ input =
  let cI = cast {to=Int} c
      hI = cast {to=Int} inH
      wI = cast {to=Int} inW
      inp3d = prim__reshape3d input.tensorPtr cI hI wI
      outT = prim__avgPool2d inp3d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                   (cast {to=Int} strH) (cast {to=Int} strW)
      outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
  in MkTVar (prim__reshape1d outT (cast {to=Int} outFlat)) Nothing

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  LayerLikeV2 (AvgPool2DStateV2 c inH inW poolH poolW strH strW) where
  applyTVar st@MkAvgPool2DV2 input = (st, applyAvgPool2DV2 st input)
  layerPrefixV2 _ = "avgpool2dV2"

export
avgPool2dLayerV2 : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AnyLayerV2 (c * (inH * inW))
                              (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                              d
avgPool2dLayerV2 =
  MkAnyLayerV2 (AvgPool2DStateV2 c inH inW poolH poolW strH strW)
               MkAvgPool2DV2


----------------------------------------------------------------------
-- MaxPool1DV2 / AvgPool1DV2
----------------------------------------------------------------------

public export
data MaxPool1DStateV2 :
  (c : Nat) -> (len : Nat) -> (poolK : Nat) -> (str : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkMaxPool1DV2 :
    MaxPool1DStateV2 c len poolK str
                     (c * len)
                     (c * PoolOutDim len poolK str) d

export
applyMaxPool1DV2 : {c, len, poolK, str : Nat} ->
                   MaxPool1DStateV2 c len poolK str
                                    (c * len) (c * PoolOutDim len poolK str) d ->
                   TVec (c * len) d ->
                   TVec (c * PoolOutDim len poolK str) d
applyMaxPool1DV2 {c} {len} {poolK} {str} _ input =
  let cI = cast {to=Int} c
      lenI = cast {to=Int} len
      inp2d = prim__reshape2d input.tensorPtr cI lenI
      outT = prim__maxPool1d inp2d (cast {to=Int} poolK) (cast {to=Int} str)
      outFlat = c * PoolOutDim len poolK str
  in MkTVar (prim__reshape1d outT (cast {to=Int} outFlat)) Nothing

public export
{c, len, poolK, str : Nat} ->
  LayerLikeV2 (MaxPool1DStateV2 c len poolK str) where
  applyTVar st@MkMaxPool1DV2 input = (st, applyMaxPool1DV2 st input)
  layerPrefixV2 _ = "maxpool1dV2"

export
maxPool1dLayerV2 : {c, len, poolK, str : Nat} ->
                   AnyLayerV2 (c * len) (c * PoolOutDim len poolK str) d
maxPool1dLayerV2 =
  MkAnyLayerV2 (MaxPool1DStateV2 c len poolK str) MkMaxPool1DV2

public export
data AvgPool1DStateV2 :
  (c : Nat) -> (len : Nat) -> (poolK : Nat) -> (str : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> Type
  where
  MkAvgPool1DV2 :
    AvgPool1DStateV2 c len poolK str
                     (c * len)
                     (c * PoolOutDim len poolK str) d

export
applyAvgPool1DV2 : {c, len, poolK, str : Nat} ->
                   AvgPool1DStateV2 c len poolK str
                                    (c * len) (c * PoolOutDim len poolK str) d ->
                   TVec (c * len) d ->
                   TVec (c * PoolOutDim len poolK str) d
applyAvgPool1DV2 {c} {len} {poolK} {str} _ input =
  let cI = cast {to=Int} c
      lenI = cast {to=Int} len
      inp2d = prim__reshape2d input.tensorPtr cI lenI
      outT = prim__avgPool1d inp2d (cast {to=Int} poolK) (cast {to=Int} str)
      outFlat = c * PoolOutDim len poolK str
  in MkTVar (prim__reshape1d outT (cast {to=Int} outFlat)) Nothing

public export
{c, len, poolK, str : Nat} ->
  LayerLikeV2 (AvgPool1DStateV2 c len poolK str) where
  applyTVar st@MkAvgPool1DV2 input = (st, applyAvgPool1DV2 st input)
  layerPrefixV2 _ = "avgpool1dV2"

export
avgPool1dLayerV2 : {c, len, poolK, str : Nat} ->
                   AnyLayerV2 (c * len) (c * PoolOutDim len poolK str) d
avgPool1dLayerV2 =
  MkAnyLayerV2 (AvgPool1DStateV2 c len poolK str) MkAvgPool1DV2
