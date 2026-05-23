module Layer.Conv

import Data.Vect

import Device
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Type-level conv/pool output-dimension helpers
----------------------------------------------------------------------

||| ConvOutDim inDim kernel pad = inDim + 2*pad - kernel + 1
public export
ConvOutDim : Nat -> Nat -> Nat -> Nat
ConvOutDim inDim kernel pad = ((inDim + 2 * pad) `minus` kernel) + 1

||| PoolOutDim inDim kernel stride = (inDim - kernel) / stride + 1
public export
PoolOutDim : Nat -> Nat -> Nat -> Nat
PoolOutDim inDim kernel stride = div (inDim `minus` kernel) stride + 1


----------------------------------------------------------------------
-- Conv — typed-surface conv + pool layers (Path C)
----------------------------------------------------------------------
--
-- Six layers in one file (matches the V1 `Layer/Conv.idr` shape):
--   - Conv2D / Conv1D  (learnable kernel + bias)
--   - MaxPool2D / MaxPool1D (no params)
--   - AvgPool2D / AvgPool1D (no params)
--
-- All take flattened input shape `[c * spatial]` and produce
-- flattened output `[outC * spatialOut]`. The forward path reshapes
-- to `[c, spatial]` (1D) or `[c, h, w]` (2D), calls the C op, then
-- flattens. Type-level `i / o` indices are computed with V1's
-- `ConvOutDim` / `PoolOutDim` helpers.


----------------------------------------------------------------------
-- Conv2D
----------------------------------------------------------------------

public export
data Conv2DState :
  (inC : Nat) -> (outC : Nat) -> (h : Nat) -> (w : Nat) ->
  (kH : Nat) -> (kW : Nat) -> (padH : Nat) -> (padW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkConv2D :
    Tensor [outC, inC, kH, kW] d dt g ->                       -- kernel
    TVec outC d dt g ->                                      -- bias
    Conv2DState inC outC h w kH kW padH padW
                  (inC * (h * w))
                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                  d dt g

%default partial

export
applyConv2D : {0 d : Device} -> UserDeviceTraining d => {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                Conv2DState inC outC h w kH kW padH padW
                              (inC * (h * w))
                              (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                              d dt g ->
                TVec (inC * (h * w)) d dt g->
                TVec (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)) d dt g
applyConv2D {inC} {outC} {h} {w} {kH} {kW} {padH} {padW}
              (MkConv2D ker bias) input =
  let inCI = cast {to=Int} inC
      hI = cast {to=Int} h
      wI = cast {to=Int} w
      inp3d = primReshape3d {d} input.tensorPtr inCI hI wI
      padHI = cast {to=Int} padH
      padWI = cast {to=Int} padW
      outT = primConv2d {d} inp3d ker.tensorPtr bias.tensorPtr padHI padWI 1 1
      outFlat = outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)
      flatPtr = primReshape1d {d} outT (cast {to=Int} outFlat)
  in MkTensor flatPtr Nothing

-- Batched forward: input [b, inC * h * w], reshape to [b, inC, h, w] for
-- the batched primitive, then flatten back. One conv2d call per batched
-- forward (vs B single-sample calls in `applyConv2D`).
export
applyConv2DBatched : {0 d : Device} -> UserDeviceTraining d => {inC, outC, h, w, kH, kW, padH, padW : Nat} -> {b : Nat} ->
                       Conv2DState inC outC h w kH kW padH padW
                                     (inC * (h * w))
                                     (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                                     d dt g ->
                       Tensor [b, inC * (h * w)] d dt g ->
                       Tensor [b, outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)] d dt g
applyConv2DBatched {inC} {outC} {h} {w} {kH} {kW} {padH} {padW} {b}
                     (MkConv2D ker bias) input =
  let bI    = cast {to=Int} b
      inCI  = cast {to=Int} inC
      hI    = cast {to=Int} h
      wI    = cast {to=Int} w
      inp4d = primReshape4d {d} input.tensorPtr bI inCI hI wI
      padHI = cast {to=Int} padH
      padWI = cast {to=Int} padW
      outT  = primConv2dBatched {d} inp4d ker.tensorPtr bias.tensorPtr padHI padWI 1 1
      outFlat = outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)
      out2d = primReshape2d {d} outT bI (cast {to=Int} outFlat)
  in MkTensor out2d Nothing


||| Build a Conv2D layer with He-normal kernel init and zero bias.
export
conv2dLayer : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                (paramPrefix : String) ->
                IO (Conv2DState inC outC h w kH kW padH padW
                                  (inC * (h * w))
                                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                                  d dt WithGrad)
conv2dLayer paramPrefix = do
  -- He-normal kernel init: std = sqrt(2 / fan_in) where fan_in for a
  -- conv kernel is inC * kH * kW. Zero bias.
  let kerStd = sqrt (2.0 / cast {to=Double} (inC * kH * kW))
      kerName  = paramPrefix ++ "_kernel"
      biasName = paramPrefix ++ "_bias"
  kernel <- tparam4dNormal {a=outC} {b=inC} {c=kH} {e=kW} kerName 0.0 kerStd
  bias   <- tparam1dConst {n=outC} biasName 0.0
  pure $ MkConv2D kernel bias

public export
{inC, outC, h, w, kH, kW, padH, padW : Nat} ->
  LayerLike (Conv2DState inC outC h w kH kW padH padW) where
  applyVar st@(MkConv2D _ _) input = ioRerun (\_ => (st, applyConv2D st input))
  applyVarBatch st@(MkConv2D _ _) input = ioRerun (\_ => (st, applyConv2DBatched st input))
  layerPrefix _ = "conv"

  freezeLayer (MkConv2D k b) = do
    k' <- weakenGrad k
    b' <- weakenGrad b
    pure (MkConv2D k' b')

  unfreezeLayer (MkConv2D k b) = do
    primIO (primSetRequiresGrad {d} k.tensorPtr 1)
    primIO (primSetRequiresGrad {d} b.tensorPtr 1)
    pure (MkConv2D (retypeGrad k) (retypeGrad b))

export
conv2dLayerAny : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
                   (paramPrefix : String) ->
                   IO (AnyLayer (inC * (h * w))
                                  (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                                  d dt WithGrad)
conv2dLayerAny pid =
  map (MkAnyLayer (Conv2DState inC outC h w kH kW padH padW))
      (conv2dLayer {inC} {outC} {h} {w} {kH} {kW} {padH} {padW} pid)


----------------------------------------------------------------------
-- Conv1D
----------------------------------------------------------------------

public export
data Conv1DState :
  (inC : Nat) -> (outC : Nat) -> (len : Nat) -> (kL : Nat) -> (pad : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkConv1D :
    Tensor [outC, inC, kL] d dt g ->
    TVec outC d dt g ->
    Conv1DState inC outC len kL pad
                  (inC * len)
                  (outC * ConvOutDim len kL pad)
                  d dt g

export
applyConv1D : {0 d : Device} -> UserDeviceTraining d => {inC, outC, len, kL, pad : Nat} ->
                Conv1DState inC outC len kL pad
                              (inC * len)
                              (outC * ConvOutDim len kL pad) d dt g ->
                TVec (inC * len) d dt g ->
                TVec (outC * ConvOutDim len kL pad) d dt g
applyConv1D {inC} {outC} {len} {kL} {pad} (MkConv1D ker bias) input =
  let inCI = cast {to=Int} inC
      lenI = cast {to=Int} len
      inp2d = primReshape2d {d} input.tensorPtr inCI lenI
      outT = primConv1d {d} inp2d ker.tensorPtr bias.tensorPtr (cast {to=Int} pad) 1
      outFlat = outC * ConvOutDim len kL pad
  in MkTensor (primReshape1d {d} outT (cast {to=Int} outFlat)) Nothing

export
conv1dLayer : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {inC, outC, len, kL, pad : Nat} ->
                (paramPrefix : String) ->
                IO (Conv1DState inC outC len kL pad
                                  (inC * len)
                                  (outC * ConvOutDim len kL pad) d dt WithGrad)
conv1dLayer paramPrefix = do
  -- He-normal kernel init: std = sqrt(2 / fan_in) with fan_in = inC * kL.
  -- Zero bias.
  let kerStd = sqrt (2.0 / cast {to=Double} (inC * kL))
      kerName  = paramPrefix ++ "_kernel"
      biasName = paramPrefix ++ "_bias"
  kernel <- tparam3dNormal {a=outC} {b=inC} {c=kL} kerName 0.0 kerStd
  bias   <- tparam1dConst {n=outC} biasName 0.0
  pure $ MkConv1D kernel bias

public export
{inC, outC, len, kL, pad : Nat} ->
  LayerLike (Conv1DState inC outC len kL pad) where
  applyVar st@(MkConv1D _ _) input = ioRerun (\_ => (st, applyConv1D st input))
  layerPrefix _ = "conv1d"

  freezeLayer (MkConv1D k b) = do
    k' <- weakenGrad k
    b' <- weakenGrad b
    pure (MkConv1D k' b')

  unfreezeLayer (MkConv1D k b) = do
    primIO (primSetRequiresGrad {d} k.tensorPtr 1)
    primIO (primSetRequiresGrad {d} b.tensorPtr 1)
    pure (MkConv1D (retypeGrad k) (retypeGrad b))

export
conv1dLayerAny : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {inC, outC, len, kL, pad : Nat} ->
                   (paramPrefix : String) ->
                   IO (AnyLayer (inC * len) (outC * ConvOutDim len kL pad) d dt WithGrad)
conv1dLayerAny pid =
  map (MkAnyLayer (Conv1DState inC outC len kL pad))
      (conv1dLayer {inC} {outC} {len} {kL} {pad} pid)


----------------------------------------------------------------------
-- MaxPool2D / AvgPool2D (no learnable params)
----------------------------------------------------------------------

public export
data MaxPool2DState :
  (c : Nat) -> (inH : Nat) -> (inW : Nat) ->
  (poolH : Nat) -> (poolW : Nat) -> (strH : Nat) -> (strW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkMaxPool2D :
    MaxPool2DState c inH inW poolH poolW strH strW
                     (c * (inH * inW))
                     (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                     d dt g

export
applyMaxPool2D : {0 d : Device} -> UserDeviceTraining d => {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   MaxPool2DState c inH inW poolH poolW strH strW
                                    (c * (inH * inW))
                                    (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                                    d dt g ->
                   TVec (c * (inH * inW)) d dt g->
                   TVec (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)) d dt g
applyMaxPool2D {c} {inH} {inW} {poolH} {poolW} {strH} {strW} _ input =
  let cI = cast {to=Int} c
      hI = cast {to=Int} inH
      wI = cast {to=Int} inW
      inp3d = primReshape3d {d} input.tensorPtr cI hI wI
      outT = primMaxPool2d {d} inp3d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                   (cast {to=Int} strH) (cast {to=Int} strW)
      outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
  in MkTensor (primReshape1d {d} outT (cast {to=Int} outFlat)) Nothing

-- Batched: input [b, c * inH * inW], reshape to [b, c, inH, inW], pool,
-- flatten back to [b, c * outH * outW].
export
applyMaxPool2DBatched : {0 d : Device} -> UserDeviceTraining d => {c, inH, inW, poolH, poolW, strH, strW : Nat} -> {b : Nat} ->
                          MaxPool2DState c inH inW poolH poolW strH strW
                                           (c * (inH * inW))
                                           (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                                           d dt g ->
                          Tensor [b, c * (inH * inW)] d dt g ->
                          Tensor [b, c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)] d dt g
applyMaxPool2DBatched {c} {inH} {inW} {poolH} {poolW} {strH} {strW} {b} _ input =
  let bI = cast {to=Int} b
      cI = cast {to=Int} c
      hI = cast {to=Int} inH
      wI = cast {to=Int} inW
      inp4d = primReshape4d {d} input.tensorPtr bI cI hI wI
      outT = primMaxPool2dBatched {d} inp4d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                          (cast {to=Int} strH) (cast {to=Int} strW)
      outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
      out2d = primReshape2d {d} outT bI (cast {to=Int} outFlat)
  in MkTensor out2d Nothing

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  LayerLike (MaxPool2DState c inH inW poolH poolW strH strW) where
  applyVar st@MkMaxPool2D input = ioRerun (\_ => (st, applyMaxPool2D st input))
  applyVarBatch st@MkMaxPool2D input = ioRerun (\_ => (st, applyMaxPool2DBatched st input))
  layerPrefix _ = "maxpool2d"

  -- Stateless: freeze/unfreeze just retypes.
  freezeLayer MkMaxPool2D = pure MkMaxPool2D
  unfreezeLayer MkMaxPool2D = pure MkMaxPool2D

export
maxPool2dLayer : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AnyLayer (c * (inH * inW))
                              (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                              d dt g
maxPool2dLayer =
  MkAnyLayer (MaxPool2DState c inH inW poolH poolW strH strW)
               MkMaxPool2D

public export
data AvgPool2DState :
  (c : Nat) -> (inH : Nat) -> (inW : Nat) ->
  (poolH : Nat) -> (poolW : Nat) -> (strH : Nat) -> (strW : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkAvgPool2D :
    AvgPool2DState c inH inW poolH poolW strH strW
                     (c * (inH * inW))
                     (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                     d dt g

export
applyAvgPool2D : {0 d : Device} -> UserDeviceTraining d => {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AvgPool2DState c inH inW poolH poolW strH strW
                                    (c * (inH * inW))
                                    (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                                    d dt g ->
                   TVec (c * (inH * inW)) d dt g->
                   TVec (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)) d dt g
applyAvgPool2D {c} {inH} {inW} {poolH} {poolW} {strH} {strW} _ input =
  let cI = cast {to=Int} c
      hI = cast {to=Int} inH
      wI = cast {to=Int} inW
      inp3d = primReshape3d {d} input.tensorPtr cI hI wI
      outT = primAvgPool2d {d} inp3d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                   (cast {to=Int} strH) (cast {to=Int} strW)
      outFlat = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)
  in MkTensor (primReshape1d {d} outT (cast {to=Int} outFlat)) Nothing

public export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  LayerLike (AvgPool2DState c inH inW poolH poolW strH strW) where
  applyVar st@MkAvgPool2D input = ioRerun (\_ => (st, applyAvgPool2D st input))
  layerPrefix _ = "avgpool2d"

  freezeLayer MkAvgPool2D = pure MkAvgPool2D
  unfreezeLayer MkAvgPool2D = pure MkAvgPool2D

export
avgPool2dLayer : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                   AnyLayer (c * (inH * inW))
                              (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                              d dt g
avgPool2dLayer =
  MkAnyLayer (AvgPool2DState c inH inW poolH poolW strH strW)
               MkAvgPool2D


----------------------------------------------------------------------
-- MaxPool1D / AvgPool1D
----------------------------------------------------------------------

public export
data MaxPool1DState :
  (c : Nat) -> (len : Nat) -> (poolK : Nat) -> (str : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkMaxPool1D :
    MaxPool1DState c len poolK str
                     (c * len)
                     (c * PoolOutDim len poolK str) d dt g

export
applyMaxPool1D : {0 d : Device} -> UserDeviceTraining d => {c, len, poolK, str : Nat} ->
                   MaxPool1DState c len poolK str
                                    (c * len) (c * PoolOutDim len poolK str) d dt g ->
                   TVec (c * len) d dt g ->
                   TVec (c * PoolOutDim len poolK str) d dt g
applyMaxPool1D {c} {len} {poolK} {str} _ input =
  let cI = cast {to=Int} c
      lenI = cast {to=Int} len
      inp2d = primReshape2d {d} input.tensorPtr cI lenI
      outT = primMaxPool1d {d} inp2d (cast {to=Int} poolK) (cast {to=Int} str)
      outFlat = c * PoolOutDim len poolK str
  in MkTensor (primReshape1d {d} outT (cast {to=Int} outFlat)) Nothing

public export
{c, len, poolK, str : Nat} ->
  LayerLike (MaxPool1DState c len poolK str) where
  applyVar st@MkMaxPool1D input = ioRerun (\_ => (st, applyMaxPool1D st input))
  layerPrefix _ = "maxpool1d"

  freezeLayer MkMaxPool1D = pure MkMaxPool1D
  unfreezeLayer MkMaxPool1D = pure MkMaxPool1D

export
maxPool1dLayer : {c, len, poolK, str : Nat} ->
                   AnyLayer (c * len) (c * PoolOutDim len poolK str) d dt g
maxPool1dLayer =
  MkAnyLayer (MaxPool1DState c len poolK str) MkMaxPool1D

public export
data AvgPool1DState :
  (c : Nat) -> (len : Nat) -> (poolK : Nat) -> (str : Nat) ->
  Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
  where
  MkAvgPool1D :
    AvgPool1DState c len poolK str
                     (c * len)
                     (c * PoolOutDim len poolK str) d dt g

export
applyAvgPool1D : {0 d : Device} -> UserDeviceTraining d => {c, len, poolK, str : Nat} ->
                   AvgPool1DState c len poolK str
                                    (c * len) (c * PoolOutDim len poolK str) d dt g ->
                   TVec (c * len) d dt g ->
                   TVec (c * PoolOutDim len poolK str) d dt g
applyAvgPool1D {c} {len} {poolK} {str} _ input =
  let cI = cast {to=Int} c
      lenI = cast {to=Int} len
      inp2d = primReshape2d {d} input.tensorPtr cI lenI
      outT = primAvgPool1d {d} inp2d (cast {to=Int} poolK) (cast {to=Int} str)
      outFlat = c * PoolOutDim len poolK str
  in MkTensor (primReshape1d {d} outT (cast {to=Int} outFlat)) Nothing

public export
{c, len, poolK, str : Nat} ->
  LayerLike (AvgPool1DState c len poolK str) where
  applyVar st@MkAvgPool1D input = ioRerun (\_ => (st, applyAvgPool1D st input))
  layerPrefix _ = "avgpool1d"

  freezeLayer MkAvgPool1D = pure MkAvgPool1D
  unfreezeLayer MkAvgPool1D = pure MkAvgPool1D

export
avgPool1dLayer : {c, len, poolK, str : Nat} ->
                   AnyLayer (c * len) (c * PoolOutDim len poolK str) d dt g
avgPool1dLayer =
  MkAnyLayer (AvgPool1DState c len poolK str) MkAvgPool1D
