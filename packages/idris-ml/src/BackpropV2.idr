module BackpropV2

import Data.List
import Data.Vect

import DataPoint
import Device
import Layer.CoreV2
import Tensor
import Variable


----------------------------------------------------------------------
-- BackpropV2 — typed-surface training loops for NetworkV2
----------------------------------------------------------------------
--
-- Mirrors `Backprop.idr`'s `epoch*Tensor*` runners but with TVar
-- surfaces. The two runners cover the common training shapes:
--   - `epochTVar`           : feed-forward supervised
--   - `epochRecurrentTVar`  : recurrent (sequence-to-sequence)
--
-- Both return `(network, lossDouble)` so the existing `runTraining`
-- runner from `Train.idr` works unchanged.

public export
0 LossFnV2 : (0 _ : Device) -> Nat -> Type
LossFnV2 d n = TVec n d -> TVec n d -> TVar [] d


----------------------------------------------------------------------
-- Helpers (top-level so let-blocks below don't need where-clauses)
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoublesIntoBuf : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoublesIntoBuf buf _ [] = buf
packDoublesIntoBuf buf off (x :: rest) =
  packDoublesIntoBuf (prim__setDouble buf off x) (off + 1) rest

-- Non-persistent input/target tensor from Vector n Double.
-- Mirrors V1's `Variable.bulkToTensor` (uses `prim__create1d nI buf' 0`,
-- not `prim__createState1d`). MLX requires non-grad tensors to be
-- non-persistent — `prim__createState1d` marks them persistent and
-- the lazy graphs that reference them survive tape_reset and dangle
-- after the next epoch starts. The example crashes with "invalid
-- memory reference" on epoch 2.
bulkToPersistent : {n : Nat} -> Vector n Double -> AnyPtr
bulkToPersistent {n} (VTensor elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packScalars buf 0 elems
  in prim__create1d nI buf' 0
  where
    packScalars : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packScalars b _ [] = b
    packScalars b o (STensor v :: rest) =
      packScalars (prim__setDouble b o v) (o + 1) rest

-- Scalar TVar holding 0.0.
zeroLossT : {0 d : Device} -> TVar [] d
zeroLossT = MkTVar (prim__createScalar 0.0 0) Nothing

-- Add two scalar TVars (bypasses the implicit-resolution overhead of
-- the polymorphic `tadd`).
taddScalar : {0 d : Device} -> TVar [] d -> TVar [] d -> TVar [] d
taddScalar a b = MkTVar (prim__add a.tensorPtr b.tensorPtr) Nothing

-- Scale a scalar TVar by a Double.
scaleLoss : {0 d : Device} -> TVar [] d -> Double -> TVar [] d
scaleLoss v s = MkTVar (prim__mulScalar v.tensorPtr s) Nothing


----------------------------------------------------------------------
-- Supervised epoch (feed-forward)
----------------------------------------------------------------------

%default partial

-- Per-point loss closure factored out to avoid let-block elaboration
-- weirdness in epochTVar's body.
perPointLoss : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
               LossFnV2 d o ->
               NetworkV2 i hs o d ->
               DataPoint i o Double ->
               TVar [] d
perPointLoss lossFn model dp =
  let inT = bulkToPersistent (x dp)
      tgtT = bulkToPersistent (y dp)
      inV = the (TVec i d) (MkTVar inT Nothing)
      tgtV = the (TVec o d) (MkTVar tgtT Nothing)
      (_, predV) = forwardTVar model inV
  in lossFn predV tgtV

||| One supervised epoch: forward each data point, accumulate per-
||| sample losses, mean-reduce, native train step. Returns the
||| (unchanged) network and the loss scalar.
export
epochTVar : {d : Device} -> {i, o, n : Nat} -> {hs : List Nat} ->
            NativeOptimizer ->
            Vect n (DataPoint i o Double) ->
            LossFnV2 d o ->
            NetworkV2 i hs o d ->
            (NetworkV2 i hs o d, Double)
epochTVar opt dataPoints lossFn model =
  let losses = map (perPointLoss lossFn model) dataPoints in
  let totalLoss = foldl taddScalar zeroLossT losses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStepTVar opt mean)


----------------------------------------------------------------------
-- Recurrent epoch (sequence per data point)
----------------------------------------------------------------------

-- One step of a sequence: forward, compute loss against target,
-- accumulate.
recurStep : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
            LossFnV2 d o ->
            (NetworkV2 i hs o d, TVar [] d) ->
            (Vector i Double, Vector o Double) ->
            (NetworkV2 i hs o d, TVar [] d)
recurStep lossFn (net, accLoss) (xVec, yVec) =
  let inV = the (TVec i d) (MkTVar (bulkToPersistent xVec) Nothing)
      tgtV = the (TVec o d) (MkTVar (bulkToPersistent yVec) Nothing)
      (net', predV) = forwardTVar net inV
      stepL = lossFn predV tgtV
  in (net', taddScalar accLoss stepL)

-- Per-sequence loss: reset state, walk timesteps, mean-reduce.
perSeqLoss : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
             LossFnV2 d o ->
             NetworkV2 i hs o d ->
             RecurrentDataPoint i o Double ->
             TVar [] d
perSeqLoss lossFn model dp =
  let pairs = zip (xs dp) (ys dp)
      startNet = resetNetworkV2 model
      (_, totalLoss) = foldl (recurStep lossFn) (startNet, zeroLossT) pairs
      stepCount = length pairs
  in if stepCount == 0
       then totalLoss
       else scaleLoss totalLoss (1.0 / cast stepCount)

||| One recurrent epoch: per data point, reset state, walk the
||| sequence, mean per-step loss, mean across sequences, native train
||| step. Returns the (unchanged) network and the loss scalar.
export
epochRecurrentTVar : {d : Device} -> {i, o, n : Nat} -> {hs : List Nat} ->
                     NativeOptimizer ->
                     Vect n (RecurrentDataPoint i o Double) ->
                     LossFnV2 d o ->
                     NetworkV2 i hs o d ->
                     (NetworkV2 i hs o d, Double)
epochRecurrentTVar opt dataPoints lossFn model =
  let seqLosses = map (perSeqLoss lossFn model) dataPoints in
  let totalLoss = foldl taddScalar zeroLossT seqLosses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStepTVar opt mean)


----------------------------------------------------------------------
-- Two-phase epoch (NTM/DNC pattern: encode then decode)
----------------------------------------------------------------------

-- Forward zeros for `numSteps` (the decode phase), accumulating per-step loss.
decodeStep : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
             LossFnV2 d o ->
             AnyPtr ->                                    -- zero input tensor (reused)
             (NetworkV2 i hs o d, TVar [] d) ->
             Vector o Double ->
             (NetworkV2 i hs o d, TVar [] d)
decodeStep lossFn zeroInPtr (net, accLoss) tgtVec =
  let inV = the (TVec i d) (MkTVar zeroInPtr Nothing)
      tgtV = the (TVec o d) (MkTVar (bulkToPersistent tgtVec) Nothing)
      (net', predV) = forwardTVar net inV
      stepL = lossFn predV tgtV
  in (net', taddScalar accLoss stepL)

-- Encode phase: forward each input, discard output, thread state.
encodeStep : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
             NetworkV2 i hs o d ->
             Vector i Double ->
             NetworkV2 i hs o d
encodeStep net xVec =
  let inV = the (TVec i d) (MkTVar (bulkToPersistent xVec) Nothing)
      (net', _) = forwardTVar net inV
  in net'

-- Per-sequence two-phase loss: reset state, encode all inputs, decode
-- with zeros for `length targets` steps, mean-reduce per-step loss.
perSeqLossTwoPhase : {0 d : Device} -> {i, o : Nat} -> {hs : List Nat} ->
                     LossFnV2 d o ->
                     NetworkV2 i hs o d ->
                     TwoPhaseDataPoint i o Double ->
                     TVar [] d
perSeqLossTwoPhase lossFn model dp =
  let startNet = resetNetworkV2 model
      encNet = foldl encodeStep startNet (encodingInputs dp)
      iI = cast {to=Int} i
      zeroIn = prim__create1d iI (prim__allocDoubles iI) 0
      (_, totalLoss) = foldl (decodeStep lossFn zeroIn) (encNet, zeroLossT) (targets dp)
      stepCount = length (targets dp)
  in if stepCount == 0
       then totalLoss
       else scaleLoss totalLoss (1.0 / cast stepCount)

||| One two-phase epoch: encode-then-decode pattern (NTM/DNC). Per
||| data point: reset state, walk the encoding inputs (discarding
||| outputs), then forward zeros for each target step computing loss.
||| Mean across sequences, native train step.
export
epochTwoPhaseTVar : {d : Device} -> {i, o, n : Nat} -> {hs : List Nat} ->
                    NativeOptimizer ->
                    Vect n (TwoPhaseDataPoint i o Double) ->
                    LossFnV2 d o ->
                    NetworkV2 i hs o d ->
                    (NetworkV2 i hs o d, Double)
epochTwoPhaseTVar opt dataPoints lossFn model =
  let seqLosses = map (perSeqLossTwoPhase lossFn model) dataPoints in
  let totalLoss = foldl taddScalar zeroLossT seqLosses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStepTVar opt mean)
