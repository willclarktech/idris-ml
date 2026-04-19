-- | Conv2D layer with type-safe spatial dimensions.
-- |
-- | Input: flat [inC * h * w], internally reshaped to [inC, h, w].
-- | Output: flat [outC * oH * oW] where oH = h + 2*padH - kH + 1 (stride=1).
-- | Kernel: [outC, inC, kH, kW], bias: [outC].
-- |
-- | Carries erased proofs linking flat dimensions to spatial params,
-- | following the TransformerState pattern.

module Layer.Conv

import Data.Vect

import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Type-level dimension functions
----------------------------------------------------------------------

||| Output spatial dimension for stride=1 convolution.
||| ConvOutDim inDim kernel pad = inDim + 2*pad - kernel + 1
public export
ConvOutDim : Nat -> Nat -> Nat -> Nat
ConvOutDim inDim kernel pad = ((inDim + 2 * pad) `minus` kernel) + 1

||| Output spatial dimension for pooling.
||| PoolOutDim inDim kernel stride = (inDim - kernel) / stride + 1
public export
PoolOutDim : Nat -> Nat -> Nat -> Nat
PoolOutDim inDim kernel stride = div (inDim `minus` kernel) stride + 1


----------------------------------------------------------------------
-- Conv2D State
----------------------------------------------------------------------

||| Conv2D layer state. Extra spatial params are fixed at construction;
||| inputSize and outputSize are the flat dimensions for the Network chain.
public export
record Conv2DState (inC : Nat) (outC : Nat) (h : Nat) (w : Nat)
                   (kH : Nat) (kW : Nat) (padH : Nat) (padW : Nat)
                   (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkConv2D
  0 inputPrf  : inputSize = inC * (h * w)
  0 outputPrf : outputSize = outC * (ConvOutDim h kH padH * ConvOutDim w kW padW)
  kernelFlat  : Vector (outC * inC * kH * kW) ty
  bias        : Vector outC ty
  kernelTensor : Maybe AnyPtr
  biasTensor   : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
{inC, outC, h, w, kH, kW, padH, padW : Nat} ->
  LayerLike (Conv2DState inC outC h w kH kW padH padW) where

  applyGeneric _ _ = idris_crash "Conv2D: generic forward not implemented (use tensor path)"

  applyVar _ _ = idris_crash "Conv2D: scalar Variable forward not implemented (use tensor path)"

  applyVarTensor {i} {o} st inputT =
    case (st.kernelTensor, st.biasTensor) of
      (Just kerT, Just biasT) =>
        let hI  = cast {to=Int} h
            wI  = cast {to=Int} w
            inCI = cast {to=Int} inC
            inp3d = prim__reshape3d inputT inCI hI wI
            padHI = cast {to=Int} padH
            padWI = cast {to=Int} padW
            outT = prim__conv2d inp3d kerT biasT padHI padWI 1 1
            -- Flatten output [outC, oH, oW] -> [outC * oH * oW]
            oI = cast {to=Int} o
            flatOut = prim__reshape1d outT oI
        in (st, flatOut)
      _ => idris_crash "Conv2D: weight tensors not initialized (call autoName first)"

  emapLayer f (MkConv2D ip op k b kt bt) = MkConv2D ip op (map f k) (map f b) kt bt

  showLayer _ = "Conv2D<" ++ show inC ++ "->" ++ show outC
             ++ " k=" ++ show kH ++ "x" ++ show kW ++ ">"

  nameLayer {i} {o} prefx (MkConv2D ip op kernelFlat bias _ _) =
    if prim__backendSupportsTensorParams == 1
      then
        let kerI = cast {to=Int} (outC * inC * kH * kW)
            kerBuf = prim__allocDoubles kerI
            (VTensor kerElems) = kernelFlat
            kerBuf' = packScalarValues kerBuf 0 kerElems
            kerT = prim__paramRegister (prefx ++ "_kernel")
                     (prim__createParam4d (cast outC) (cast inC) (cast kH) (cast kW) kerBuf')
            biasI = cast {to=Int} outC
            biasBuf = prim__allocDoubles biasI
            (VTensor biasElems) = bias
            biasBuf' = packScalarValues biasBuf 0 biasElems
            biasT = prim__paramRegister (prefx ++ "_bias")
                      (prim__createParam1d biasI biasBuf')
        in MkConv2D ip op kernelFlat bias (Just kerT) (Just biasT)
      else idris_crash "Conv2D: scalar path not supported"

  layerPrefix _ = "conv"

  toDoubleLayer (MkConv2D ip op k b _ _) =
    MkConv2D ip op (map value k) (map value b) Nothing Nothing

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry "Conv2D" [])


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a Conv2D layer with He initialization.
||| Returns AnyLayer (inC*h*w) (outC*oH*oW) ty.
export
conv2dLayer : {inC, outC, h, w, kH, kW, padH, padW : Nat} ->
              (Num ty, FromDouble ty) =>
              IO (AnyLayer (inC * (h * w))
                           (outC * (ConvOutDim h kH padH * ConvOutDim w kW padW))
                           ty)
conv2dLayer = do
  kernelVals <- traverse (\_ => map fromDouble (he normal (inC * kH * kW) outC))
                         (the (Vector (outC * inC * kH * kW) ty) zeros)
  let biasVals = the (Vector outC ty) zeros
  pure $ MkAnyLayer (Conv2DState inC outC h w kH kW padH padW)
    (MkConv2D Refl Refl kernelVals biasVals Nothing Nothing)


----------------------------------------------------------------------
-- MaxPool2D State
----------------------------------------------------------------------

||| MaxPool2D layer state. No learnable parameters.
public export
record MaxPool2DState (c : Nat) (inH : Nat) (inW : Nat)
                      (poolH : Nat) (poolW : Nat) (strH : Nat) (strW : Nat)
                      (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkMaxPool2D
  0 inputPrf  : inputSize = c * (inH * inW)
  0 outputPrf : outputSize = c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW)


----------------------------------------------------------------------
-- MaxPool2D LayerLike Instance
----------------------------------------------------------------------

export
{c, inH, inW, poolH, poolW, strH, strW : Nat} ->
  LayerLike (MaxPool2DState c inH inW poolH poolW strH strW) where

  applyGeneric _ _ = idris_crash "MaxPool2D: generic forward not implemented"
  applyVar _ _ = idris_crash "MaxPool2D: scalar forward not implemented"

  applyVarTensor {i} {o} st inputT =
    let cI  = cast {to=Int} c
        hI  = cast {to=Int} inH
        wI  = cast {to=Int} inW
        inp3d = prim__reshape3d inputT cI hI wI
        outT = prim__maxPool2d inp3d (cast {to=Int} poolH) (cast {to=Int} poolW)
                                     (cast {to=Int} strH) (cast {to=Int} strW)
        oI = cast {to=Int} o
        flatOut = prim__reshape1d outT oI
    in (st, flatOut)

  emapLayer _ st = st
  showLayer _ = "MaxPool2D<k=" ++ show poolH ++ " s=" ++ show strH ++ ">"
  nameLayer _ st = st
  layerPrefix _ = "pool"
  toDoubleLayer (MkMaxPool2D ip op) = MkMaxPool2D ip op

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry "MaxPool2D" [])


----------------------------------------------------------------------
-- MaxPool2D Constructor
----------------------------------------------------------------------

||| Create a MaxPool2D layer.
export
maxPool2dLayer : {c, inH, inW, poolH, poolW, strH, strW : Nat} ->
                 AnyLayer (c * (inH * inW))
                          (c * (PoolOutDim inH poolH strH * PoolOutDim inW poolW strW))
                          ty
maxPool2dLayer = MkAnyLayer (MaxPool2DState c inH inW poolH poolW strH strW)
                  (MkMaxPool2D Refl Refl)


----------------------------------------------------------------------
-- Conv1D State
----------------------------------------------------------------------

public export
record Conv1DState (inC : Nat) (outC : Nat) (len : Nat)
                   (kL : Nat) (pad : Nat)
                   (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkConv1D
  0 inputPrf  : inputSize = inC * len
  0 outputPrf : outputSize = outC * (ConvOutDim len kL pad)
  kernelFlat  : Vector (outC * inC * kL) ty
  bias        : Vector outC ty
  kernelTensor : Maybe AnyPtr
  biasTensor   : Maybe AnyPtr


----------------------------------------------------------------------
-- Conv1D LayerLike Instance
----------------------------------------------------------------------

export
{inC, outC, len, kL, pad : Nat} ->
  LayerLike (Conv1DState inC outC len kL pad) where

  applyGeneric _ _ = idris_crash "Conv1D: use tensor path"
  applyVar _ _ = idris_crash "Conv1D: use tensor path"

  applyVarTensor {i} {o} st inputT =
    case (st.kernelTensor, st.biasTensor) of
      (Just kerT, Just biasT) =>
        let inCI = cast {to=Int} inC
            lenI = cast {to=Int} len
            inp2d = prim__reshape2d inputT inCI lenI
            padI = cast {to=Int} pad
            outT = prim__conv1d inp2d kerT biasT padI 1
            oI = cast {to=Int} o
            flatOut = prim__reshape1d outT oI
        in (st, flatOut)
      _ => idris_crash "Conv1D: weight tensors not initialized"

  emapLayer f (MkConv1D ip op k b kt bt) = MkConv1D ip op (map f k) (map f b) kt bt
  showLayer _ = "Conv1D<" ++ show inC ++ "->" ++ show outC ++ " k=" ++ show kL ++ ">"

  nameLayer {i} {o} prefx (MkConv1D ip op kernelFlat bias _ _) =
    if prim__backendSupportsTensorParams == 1
      then
        let kerI = cast {to=Int} (outC * inC * kL)
            kerBuf = prim__allocDoubles kerI
            (VTensor kerElems) = kernelFlat
            kerBuf' = packScalarValues kerBuf 0 kerElems
            kerT = prim__paramRegister (prefx ++ "_kernel")
                     (prim__createParam3d (cast outC) (cast inC) (cast kL) kerBuf')
            biasI = cast {to=Int} outC
            biasBuf = prim__allocDoubles biasI
            (VTensor biasElems) = bias
            biasBuf' = packScalarValues biasBuf 0 biasElems
            biasT = prim__paramRegister (prefx ++ "_bias")
                      (prim__createParam1d biasI biasBuf')
        in MkConv1D ip op kernelFlat bias (Just kerT) (Just biasT)
      else idris_crash "Conv1D: scalar path not supported"

  layerPrefix _ = "conv1d"
  toDoubleLayer (MkConv1D ip op k b _ _) =
    MkConv1D ip op (map value k) (map value b) Nothing Nothing
  debugApply _ _ = idris_crash "Conv1D: use tensor path"


----------------------------------------------------------------------
-- Conv1D Constructor
----------------------------------------------------------------------

export
conv1dLayer : {inC, outC, len, kL, pad : Nat} ->
              (Num ty, FromDouble ty) =>
              IO (AnyLayer (inC * len)
                           (outC * (ConvOutDim len kL pad))
                           ty)
conv1dLayer = do
  kernelVals <- traverse (\_ => map fromDouble (he normal (inC * kL) outC))
                         (the (Vector (outC * inC * kL) ty) zeros)
  let biasVals = the (Vector outC ty) zeros
  pure $ MkAnyLayer (Conv1DState inC outC len kL pad)
    (MkConv1D Refl Refl kernelVals biasVals Nothing Nothing)


----------------------------------------------------------------------
-- MaxPool1D State
----------------------------------------------------------------------

public export
record MaxPool1DState (c : Nat) (len : Nat) (poolK : Nat) (str : Nat)
                      (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkMaxPool1D
  0 inputPrf  : inputSize = c * len
  0 outputPrf : outputSize = c * (PoolOutDim len poolK str)


----------------------------------------------------------------------
-- MaxPool1D LayerLike Instance
----------------------------------------------------------------------

export
{c, len, poolK, str : Nat} ->
  LayerLike (MaxPool1DState c len poolK str) where

  applyGeneric _ _ = idris_crash "MaxPool1D: use tensor path"
  applyVar _ _ = idris_crash "MaxPool1D: use tensor path"

  applyVarTensor {i} {o} st inputT =
    let cI = cast {to=Int} c
        lenI = cast {to=Int} len
        inp2d = prim__reshape2d inputT cI lenI
        outT = prim__maxPool1d inp2d (cast {to=Int} poolK) (cast {to=Int} str)
        oI = cast {to=Int} o
        flatOut = prim__reshape1d outT oI
    in (st, flatOut)

  emapLayer _ st = st
  showLayer _ = "MaxPool1D<k=" ++ show poolK ++ " s=" ++ show str ++ ">"
  nameLayer _ st = st
  layerPrefix _ = "pool1d"
  toDoubleLayer (MkMaxPool1D ip op) = MkMaxPool1D ip op
  debugApply _ _ = idris_crash "MaxPool1D: use tensor path"


----------------------------------------------------------------------
-- MaxPool1D Constructor
----------------------------------------------------------------------

export
maxPool1dLayer : {c, len, poolK, str : Nat} ->
                 AnyLayer (c * len)
                          (c * (PoolOutDim len poolK str))
                          ty
maxPool1dLayer = MkAnyLayer (MaxPool1DState c len poolK str)
                  (MkMaxPool1D Refl Refl)
