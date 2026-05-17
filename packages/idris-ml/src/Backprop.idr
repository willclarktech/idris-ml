module Backprop

import Data.List
import Data.Vect

import DataPoint
import Device
import Layer.Core
import Array
import Tensor


----------------------------------------------------------------------
-- Backprop — typed-surface training loops for Network
----------------------------------------------------------------------
--
-- Mirrors `Backprop.idr`'s `epoch*Array*` runners but with Tensor
-- surfaces. The two runners cover the common training shapes:
--   - `epochVar`           : feed-forward supervised
--   - `epochRecurrentVar`  : recurrent (sequence-to-sequence)
--
-- Both return `IO (network, lossDouble)` so the existing `runTraining`
-- runner from `Train.idr` works unchanged.

public export
0 LossFn : (0 _ : Device) -> (0 _ : DType) -> Nat -> Type
LossFn d F64 n = TVec n d F64 WithGrad -> TVec n d F64 WithGrad -> IO (Tensor [] d F64 WithGrad)


----------------------------------------------------------------------
-- Helpers (top-level so let-blocks below don't need where-clauses)
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoublesIntoBuf : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoublesIntoBuf buf _ [] = buf
packDoublesIntoBuf buf off (x :: rest) =
  packDoublesIntoBuf (prim__setDouble buf off x) (off + 1) rest

-- Non-persistent input/target tensor from Vector n Double.
bulkToPersistent : {n : Nat} -> Vector n Double -> AnyPtr
bulkToPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packScalars buf 0 elems
  in prim__create1d nI buf' 0
  where
    packScalars : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packScalars b _ [] = b
    packScalars b o (SArray v :: rest) =
      packScalars (prim__setDouble b o v) (o + 1) rest

-- Scalar Tensor holding 0.0. IO so its FFI side effect happens at
-- sequence-time rather than at call-time.
freshZeroLossT : {0 d : Device} -> Double -> IO (Tensor [] d F64 WithGrad)
freshZeroLossT seed = ioRerun (\_ => MkTensor (prim__createScalar seed 0) Nothing)

-- Add two scalar TVars (bypasses the implicit-resolution overhead of
-- the polymorphic `tadd`).
taddScalar : {0 d : Device} -> Tensor [] d F64 WithGrad -> Tensor [] d F64 WithGrad -> IO (Tensor [] d F64 WithGrad)
taddScalar a b = ioRerun (\_ => MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing)

-- Scale a scalar Tensor by a Double.
scaleLoss : {0 d : Device} -> Tensor [] d F64 WithGrad -> Double -> IO (Tensor [] d F64 WithGrad)
scaleLoss v s = ioRerun (\_ => MkTensor (prim__mulScalar v.tensorPtr s) Nothing)

-- Sum a list of scalar tensors starting from a fresh zero. Replaces
-- the old `foldl taddScalar (freshZeroLossT 0.0) losses` pattern under
-- the IO-typed surface.
sumLosses : {0 d : Device} -> List (Tensor [] d F64 WithGrad) -> IO (Tensor [] d F64 WithGrad)
sumLosses losses = do
  zero <- freshZeroLossT 0.0
  foldlM taddScalar zero losses
  where
    foldlM : (Tensor [] d F64 WithGrad -> Tensor [] d F64 WithGrad -> IO (Tensor [] d F64 WithGrad)) ->
             Tensor [] d F64 WithGrad -> List (Tensor [] d F64 WithGrad) ->
             IO (Tensor [] d F64 WithGrad)
    foldlM _ acc [] = pure acc
    foldlM f acc (x :: rest) = do
      acc' <- f acc x
      foldlM f acc' rest


----------------------------------------------------------------------
-- Supervised epoch (feed-forward)
----------------------------------------------------------------------

%default partial

-- Per-point loss closure factored out to avoid let-block elaboration
-- weirdness in epochVar's body.
perPointLoss : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
               LossFn d F64 o ->
               Network i hs o d F64 WithGrad ->
               DataPoint i o Double ->
               IO (Tensor [] d F64 WithGrad)
perPointLoss lossFn model dp = do
  let inT = bulkToPersistent (x dp)
      tgtT = bulkToPersistent (y dp)
      inV = the (TVec i d F64 WithGrad) (MkTensor inT Nothing)
      tgtV = the (TVec o d F64 WithGrad) (MkTensor tgtT Nothing)
  (_, predV) <- forwardVar model inV
  lossFn predV tgtV

||| One supervised epoch: forward each data point, accumulate per-
||| sample losses, mean-reduce, native train step. Returns the
||| (unchanged) network and the loss scalar.
export
epochVar : {0 d : Device} -> UserDeviceTape d => {i, o, n : Nat} -> {hs : List Nat} ->
            NativeOptimizer ->
            Vect n (DataPoint i o Double) ->
            LossFn d F64 o ->
            Network i hs o d F64 WithGrad ->
            IO (Network i hs o d F64 WithGrad, Double)
epochVar opt dataPoints lossFn model = do
  losses <- traverse (perPointLoss lossFn model) dataPoints
  totalLoss <- sumLosses (toList losses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)


-- Per-point loss for already-tensor-pre-built inputs (TensorDataPoint).
perPointLossTensor : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
                     LossFn d F64 o ->
                     Network i hs o d F64 WithGrad ->
                     TensorDataPoint i o ->
                     IO (Tensor [] d F64 WithGrad)
perPointLossTensor lossFn model dp = do
  let inV = the (TVec i d F64 WithGrad) (MkTensor (inputTensor dp) Nothing)
      tgtV = the (TVec o d F64 WithGrad) (MkTensor (targetTensor dp) Nothing)
  (_, predV) <- forwardVar model inV
  lossFn predV tgtV

||| Supervised epoch over already-tensor-pre-built data points.
export
epochVarTensor : {0 d : Device} -> UserDeviceTape d => {i, o, n : Nat} -> {hs : List Nat} ->
                  NativeOptimizer ->
                  Vect n (TensorDataPoint i o) ->
                  LossFn d F64 o ->
                  Network i hs o d F64 WithGrad ->
                  IO (Network i hs o d F64 WithGrad, Double)
epochVarTensor opt dataPoints lossFn model = do
  losses <- traverse (perPointLossTensor lossFn model) dataPoints
  totalLoss <- sumLosses (toList losses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)


-- Concatenate a vector of per-sample [k] tensors into a single [n, k]
catAllTensors : List AnyPtr -> AnyPtr
catAllTensors [] = idris_crash "catAllTensors: empty list"
catAllTensors [x] = x
catAllTensors (x :: y :: rest) = catAllTensors (prim__cat2 x y :: rest)

-- Per-sample loss for batched-forward shape.
perRowLoss : {0 d : Device} -> UserDeviceTape d => {n, o : Nat} ->
             LossFn d F64 o ->
             Tensor [n, o] d F64 WithGrad ->
             Tensor [n, o] d F64 WithGrad ->
             Int ->
             IO (Tensor [] d F64 WithGrad)
perRowLoss lossFn predB tgtB k = do
  predRow <- trowSelect predB k
  tgtRow <- trowSelect tgtB k
  lossFn predRow tgtRow

||| Batched supervised epoch over `TensorDataPoint`s.
export
epochVarTensorBatch : {0 d : Device} -> UserDeviceTape d => {i, o, n : Nat} -> {hs : List Nat} ->
                       NativeOptimizer ->
                       Vect n (TensorDataPoint i o) ->
                       LossFn d F64 o ->
                       Network i hs o d F64 WithGrad ->
                       IO (Network i hs o d F64 WithGrad, Double)
epochVarTensorBatch opt dataPoints lossFn model = do
  let inputs = toList (map inputTensor dataPoints)
      targets = toList (map targetTensor dataPoints)
      stackedIn = catAllTensors inputs
      stackedTgt = catAllTensors targets
      iI = cast {to=Int} i
      oI = cast {to=Int} o
      nI = cast {to=Int} n
      stackedInReshaped = prim__reshape2d stackedIn nI iI
      stackedTgtReshaped = prim__reshape2d stackedTgt nI oI
      inV = the (Tensor [n, i] d F64 WithGrad) (MkTensor stackedInReshaped Nothing)
      tgtV = the (Tensor [n, o] d F64 WithGrad) (MkTensor stackedTgtReshaped Nothing)
  (_, predB) <- forwardVarBatch model inV
  losses <- go predB tgtV 0 n
  totalLoss <- sumLosses losses
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)
  where
    go : Tensor [n, o] d F64 WithGrad -> Tensor [n, o] d F64 WithGrad -> Int -> Nat ->
         IO (List (Tensor [] d F64 WithGrad))
    go _ _ _ Z = pure []
    go predB tgtV k (S rest) = do
      l <- perRowLoss lossFn predB tgtV k
      ls <- go predB tgtV (k + 1) rest
      pure (l :: ls)


----------------------------------------------------------------------
-- Recurrent epoch (sequence per data point)
----------------------------------------------------------------------

-- One step of a sequence: forward, compute loss against target,
-- accumulate.
recurStep : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
            LossFn d F64 o ->
            (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad) ->
            (Vector i Double, Vector o Double) ->
            IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
recurStep lossFn (net, accLoss) (xVec, yVec) = do
  let inV = the (TVec i d F64 WithGrad) (MkTensor (bulkToPersistent xVec) Nothing)
      tgtV = the (TVec o d F64 WithGrad) (MkTensor (bulkToPersistent yVec) Nothing)
  (net', predV) <- forwardVar net inV
  stepL <- lossFn predV tgtV
  newAcc <- taddScalar accLoss stepL
  pure (net', newAcc)

-- Per-sequence loss: reset state, walk timesteps, mean-reduce.
perSeqLoss : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
             LossFn d F64 o ->
             Network i hs o d F64 WithGrad ->
             RecurrentDataPoint i o Double ->
             IO (Tensor [] d F64 WithGrad)
perSeqLoss lossFn model dp = do
  let pairs = zip (xs dp) (ys dp)
      startNet = resetNetwork model
  zero <- freshZeroLossT 0.0
  (_, totalLoss) <- foldlIO (recurStep lossFn) (startNet, zero) pairs
  let stepCount = length pairs
  if stepCount == 0
     then pure totalLoss
     else scaleLoss totalLoss (1.0 / cast stepCount)
  where
    foldlIO : ((Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
                -> (Vector i Double, Vector o Double)
                -> IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad))
            -> (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
            -> List (Vector i Double, Vector o Double)
            -> IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
    foldlIO _ acc [] = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

||| One recurrent epoch.
export
epochRecurrentVar : {0 d : Device} -> UserDeviceTape d => {i, o, n : Nat} -> {hs : List Nat} ->
                     NativeOptimizer ->
                     Vect n (RecurrentDataPoint i o Double) ->
                     LossFn d F64 o ->
                     Network i hs o d F64 WithGrad ->
                     IO (Network i hs o d F64 WithGrad, Double)
epochRecurrentVar opt dataPoints lossFn model = do
  seqLosses <- traverse (perSeqLoss lossFn model) dataPoints
  totalLoss <- sumLosses (toList seqLosses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)


----------------------------------------------------------------------
-- Two-phase epoch (NTM/DNC pattern: encode then decode)
----------------------------------------------------------------------

decodeStep : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
             LossFn d F64 o ->
             AnyPtr ->
             (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad) ->
             Vector o Double ->
             IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
decodeStep lossFn zeroInPtr (net, accLoss) tgtVec = do
  let inV = the (TVec i d F64 WithGrad) (MkTensor zeroInPtr Nothing)
      tgtV = the (TVec o d F64 WithGrad) (MkTensor (bulkToPersistent tgtVec) Nothing)
  (net', predV) <- forwardVar net inV
  stepL <- lossFn predV tgtV
  newAcc <- taddScalar accLoss stepL
  pure (net', newAcc)

encodeStep : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
             Network i hs o d F64 WithGrad ->
             Vector i Double ->
             IO (Network i hs o d F64 WithGrad)
encodeStep net xVec = do
  let inV = the (TVec i d F64 WithGrad) (MkTensor (bulkToPersistent xVec) Nothing)
  (net', _) <- forwardVar net inV
  pure net'

perSeqLossTwoPhase : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
                     LossFn d F64 o ->
                     Network i hs o d F64 WithGrad ->
                     TwoPhaseDataPoint i o Double ->
                     IO (Tensor [] d F64 WithGrad)
perSeqLossTwoPhase lossFn model dp = do
  let startNet = resetNetwork model
  encNet <- foldlIO encodeStep startNet (encodingInputs dp)
  let iI = cast {to=Int} i
      zeroIn = prim__create1d iI (prim__allocDoubles iI) 0
  zero <- freshZeroLossT 0.0
  (_, totalLoss) <- foldlIO2 (decodeStep lossFn zeroIn) (encNet, zero) (targets dp)
  let stepCount = length (targets dp)
  if stepCount == 0
     then pure totalLoss
     else scaleLoss totalLoss (1.0 / cast stepCount)
  where
    foldlIO : (Network i hs o d F64 WithGrad -> Vector i Double -> IO (Network i hs o d F64 WithGrad))
            -> Network i hs o d F64 WithGrad
            -> List (Vector i Double)
            -> IO (Network i hs o d F64 WithGrad)
    foldlIO _ acc [] = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

    foldlIO2 : ((Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
                 -> Vector o Double
                 -> IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad))
             -> (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
             -> List (Vector o Double)
             -> IO (Network i hs o d F64 WithGrad, Tensor [] d F64 WithGrad)
    foldlIO2 _ acc [] = pure acc
    foldlIO2 f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO2 f acc' rest

||| One two-phase epoch.
export
epochTwoPhaseVar : {0 d : Device} -> UserDeviceTape d => {i, o, n : Nat} -> {hs : List Nat} ->
                    NativeOptimizer ->
                    Vect n (TwoPhaseDataPoint i o Double) ->
                    LossFn d F64 o ->
                    Network i hs o d F64 WithGrad ->
                    IO (Network i hs o d F64 WithGrad, Double)
epochTwoPhaseVar opt dataPoints lossFn model = do
  seqLosses <- traverse (perSeqLossTwoPhase lossFn model) dataPoints
  totalLoss <- sumLosses (toList seqLosses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)


----------------------------------------------------------------------
-- Two-phase eval helpers (no autograd consumption)
----------------------------------------------------------------------

export
tvecToVector : {n : Nat} -> AnyPtr -> Vector n Double
tvecToVector {n} ptr = VArray (build 0 n)
  where
    build : Int -> (k : Nat) -> Vect k (Scalar Double)
    build _ Z = []
    build off (S k) = SArray (prim__item1d ptr off) :: build (off + 1) k

export
forwardTwoPhase : {0 d : Device} -> UserDeviceTape d => {i, o : Nat} -> {hs : List Nat} ->
                      Network i hs o d F64 WithGrad ->
                      TwoPhaseDataPoint i o Double ->
                      IO (Network i hs o d F64 WithGrad, List (Vector o Double))
forwardTwoPhase model dp = do
  let startNet = resetNetwork model
  encNet <- foldlIO encodeStep startNet (encodingInputs dp)
  let iI = cast {to=Int} i
      zeroIn = prim__create1d iI (prim__allocDoubles iI) 0
  foldlIO2 (decodeOnce zeroIn) (encNet, []) (targets dp)
  where
    decodeOnce : AnyPtr ->
                 (Network i hs o d F64 WithGrad, List (Vector o Double)) ->
                 Vector o Double ->
                 IO (Network i hs o d F64 WithGrad, List (Vector o Double))
    decodeOnce zeroIn (net, preds) _ = do
      let inV = the (TVec i d F64 WithGrad) (MkTensor zeroIn Nothing)
      (net', predV) <- forwardVar net inV
      let predVec = the (Vector o Double) (tvecToVector {n = o} predV.tensorPtr)
      pure (net', preds ++ [predVec])

    foldlIO : (Network i hs o d F64 WithGrad -> Vector i Double -> IO (Network i hs o d F64 WithGrad))
            -> Network i hs o d F64 WithGrad
            -> List (Vector i Double)
            -> IO (Network i hs o d F64 WithGrad)
    foldlIO _ acc [] = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

    foldlIO2 : ((Network i hs o d F64 WithGrad, List (Vector o Double))
                 -> Vector o Double
                 -> IO (Network i hs o d F64 WithGrad, List (Vector o Double)))
             -> (Network i hs o d F64 WithGrad, List (Vector o Double))
             -> List (Vector o Double)
             -> IO (Network i hs o d F64 WithGrad, List (Vector o Double))
    foldlIO2 _ acc [] = pure acc
    foldlIO2 f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO2 f acc' rest
