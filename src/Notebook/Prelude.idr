-- | Notebook Prelude: re-exports all idris-ml modules for interactive use.
-- |
-- | Loaded automatically by the Jupyter kernel so users don't need
-- | manual :module imports for common operations.

module Notebook.Prelude

import public Data.List
import public Data.String
import public Data.Vect
import public Decidable.Equality
import public System
import public System.Random

import public Backprop
import public Checkpoint
import public DataLoader
import public DataPoint
import public Device
import public Endofunctor
import public Floating
import public Generate
import public Init
import public Layer
import public Layer.Activation
import public Layer.BatchNorm
import public Layer.Conv
import public Layer.Core
import public Layer.Dropout
import public Layer.Embedding
import public Layer.Gru
import public Layer.LayerNorm
import public Layer.Linear
import public Layer.Lstm
import public Layer.Normalization
import public Layer.Residual
import public Layer.Rnn
import public Layer.Transformer
import public Math
import public Memory
import public Optimizer
import public Sampler
import public Schedule
import public Tensor
import public Train
import public Util
import public Variable

----------------------------------------------------------------------
-- Notebook convenience functions
----------------------------------------------------------------------

||| Convert a Vector of Doubles to a persistent 1D C tensor.
||| Persistent tensors survive tape resets across training epochs.
export
vectorToTensor : {n : Nat} -> Vector n Double -> AnyPtr
vectorToTensor {n} (VTensor elems) =
  let nI = cast {to=Int} n
      buf = packBuf (prim__allocDoubles nI) 0 elems
  in prim__createState1d nI buf
  where
    packBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packBuf buf _ [] = buf
    packBuf buf off (STensor v :: rest) = packBuf (prim__setDouble buf off v) (off + 1) rest

||| Convert a DataPoint with Doubles to a TensorDataPoint for C-level training.
export
toTDP : {i, o : Nat} -> DataPoint i o Double -> TensorDataPoint i o
toTDP dp = MkTensorDataPoint (vectorToTensor (x dp)) (vectorToTensor (y dp))

||| Tensor-level cross-entropy loss: -mean(target * logSoftmax(pred)).
||| Works for any number of output classes.
export
crossEntropyTensor : LossFnTensor CPU
crossEntropyTensor predT targetT =
  let logP = prim__logSoftmax predT 0
      product = prim__mul logP targetT
      loss = prim__neg (prim__mean product)
      val = prim__item loss
  in Var loss Nothing val

||| Tensor-level binary cross-entropy with logits (numerically stable).
||| Formula: mean(max(x,0) - x*y + log(1+exp(-|x|)))
export
bceTensor : LossFnTensor CPU
bceTensor predT targetT =
  let relu_x = prim__clampMin predT 0.0
      xy = prim__mul predT targetT
      abs_x = prim__abs predT
      neg_abs_x = prim__neg abs_x
      exp_neg = prim__exp neg_abs_x
      one_plus_exp = tensorAdd exp_neg (prim__createScalar 1.0 0)
      log_term = prim__log one_plus_exp
      loss = tensorAdd (prim__sub relu_x xy) log_term
      result = prim__mean loss
      val = prim__item result
  in Var result Nothing val
