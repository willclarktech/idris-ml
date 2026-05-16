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
-- Both return `(network, lossDouble)` so the existing `runTraining`
-- runner from `Train.idr` works unchanged.

public export
0 LossFn : (0 _ : Type) -> Nat -> Type
LossFn d n = TVec n d WithGrad -> TVec n d WithGrad -> Tensor [] d WithGrad


----------------------------------------------------------------------
-- Helpers (top-level so let-blocks below don't need where-clauses)
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoublesIntoBuf : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoublesIntoBuf buf _ [] = buf
packDoublesIntoBuf buf off (x :: rest) =
  packDoublesIntoBuf (prim__setDouble buf off x) (off + 1) rest

-- Non-persistent input/target tensor from Vector n Double.
-- Mirrors V1's `Tensor.bulkToTensor` (uses `prim__create1d nI buf' 0`,
-- not `prim__createState1d`). MLX requires non-grad tensors to be
-- non-persistent — `prim__createState1d` marks them persistent and
-- the lazy graphs that reference them survive tape_reset and dangle
-- after the next epoch starts. The example crashes with "invalid
-- memory reference" on epoch 2.
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

-- Scalar Tensor holding 0.0. Takes a `Double` argument that flows through
-- to the FFI call so the Idris/Chez compiler does not memoise the FFI
-- result as a module-level constant. (A zero-arg top-level def whose body
-- is `prim__createScalar 0.0 0` is evaluated ONCE at module load and
-- cached — the cached AnyPtr is non-persistent and is freed by the first
-- `tape_reset` at optimizer step, leaving every subsequent epoch reading
-- a dangling pointer. MLX surfaces this as `invalid memory reference`.)
freshZeroLossT : {0 d : Type} -> Double -> Tensor [] d WithGrad
freshZeroLossT seed = MkTensor (prim__createScalar seed 0) Nothing

-- Add two scalar TVars (bypasses the implicit-resolution overhead of
-- the polymorphic `tadd`).
taddScalar : {0 d : Type} -> Tensor [] d WithGrad -> Tensor [] d WithGrad -> Tensor [] d WithGrad
taddScalar a b = MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing

-- Scale a scalar Tensor by a Double.
scaleLoss : {0 d : Type} -> Tensor [] d WithGrad -> Double -> Tensor [] d WithGrad
scaleLoss v s = MkTensor (prim__mulScalar v.tensorPtr s) Nothing


----------------------------------------------------------------------
-- Supervised epoch (feed-forward)
----------------------------------------------------------------------

%default partial

-- Per-point loss closure factored out to avoid let-block elaboration
-- weirdness in epochVar's body.
perPointLoss : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
               LossFn d o ->
               Network i hs o d WithGrad ->
               DataPoint i o Double ->
               Tensor [] d WithGrad
perPointLoss lossFn model dp =
  let inT = bulkToPersistent (x dp)
      tgtT = bulkToPersistent (y dp)
      inV = the (TVec i d WithGrad) (MkTensor inT Nothing)
      tgtV = the (TVec o d WithGrad) (MkTensor tgtT Nothing)
      (_, predV) = forwardVar model inV
  in lossFn predV tgtV

||| One supervised epoch: forward each data point, accumulate per-
||| sample losses, mean-reduce, native train step. Returns the
||| (unchanged) network and the loss scalar.
export
epochVar : {0 d : Type} -> UserDeviceCore d => {i, o, n : Nat} -> {hs : List Nat} ->
            NativeOptimizer ->
            Vect n (DataPoint i o Double) ->
            LossFn d o ->
            Network i hs o d WithGrad ->
            (Network i hs o d WithGrad, Double)
epochVar opt dataPoints lossFn model =
  let losses = map (perPointLoss lossFn model) dataPoints in
  let totalLoss = foldl taddScalar (freshZeroLossT 0.0) losses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStep opt mean)


-- Per-point loss for already-tensor-pre-built inputs (TensorDataPoint).
-- Used by examples whose data pipeline already produces tensor pointers
-- (e.g. MNIST loaded via prim__mnistGetImage).
perPointLossTensor : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
                     LossFn d o ->
                     Network i hs o d WithGrad ->
                     TensorDataPoint i o ->
                     Tensor [] d WithGrad
perPointLossTensor lossFn model dp =
  let inV = the (TVec i d WithGrad) (MkTensor (inputTensor dp) Nothing)
      tgtV = the (TVec o d WithGrad) (MkTensor (targetTensor dp) Nothing)
      (_, predV) = forwardVar model inV
  in lossFn predV tgtV

||| Supervised epoch over already-tensor-pre-built data points (mirrors
||| V1 `epochNativeTensorPre`). Use when the data pipeline produces
||| tensor pointers directly (MNIST, on-disk indexed loaders) and you
||| do not want to round-trip through `Vector ... Double`.
export
epochVarTensor : {0 d : Type} -> UserDeviceCore d => {i, o, n : Nat} -> {hs : List Nat} ->
                  NativeOptimizer ->
                  Vect n (TensorDataPoint i o) ->
                  LossFn d o ->
                  Network i hs o d WithGrad ->
                  (Network i hs o d WithGrad, Double)
epochVarTensor opt dataPoints lossFn model =
  let losses = map (perPointLossTensor lossFn model) dataPoints in
  let totalLoss = foldl taddScalar (freshZeroLossT 0.0) losses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStep opt mean)


-- Concatenate a vector of per-sample [k] tensors into a single [n, k]
-- tensor via repeated `prim__cat2`. Mirrors V1 `transformerForwardBatch`'s
-- `catAll`. Used by `epochVarTensorBatch` to stack inputs and targets
-- for a single batched forward pass.
catAllTensors : List AnyPtr -> AnyPtr
catAllTensors [] = idris_crash "catAllTensors: empty list"
catAllTensors [x] = x
catAllTensors (x :: y :: rest) = catAllTensors (prim__cat2 x y :: rest)

-- Per-sample loss for batched-forward shape: forward once, then
-- extract per-row predictions, build per-sample loss against per-row
-- targets, mean-reduce.
perRowLoss : {0 d : Type} -> UserDeviceCore d => {n, o : Nat} ->
             LossFn d o ->
             Tensor [n, o] d WithGrad ->                         -- batched predictions
             Tensor [n, o] d WithGrad ->                         -- batched targets
             Int ->                                    -- row index
             Tensor [] d WithGrad
perRowLoss lossFn predB tgtB k =
  let predRow = the (TVec o d WithGrad) (trowSelect predB k)
      tgtRow = the (TVec o d WithGrad) (trowSelect tgtB k)
  in lossFn predRow tgtRow

||| Batched supervised epoch over `TensorDataPoint`s: stacks per-sample
||| inputs and targets into [n, i] / [n, o], runs ONE `forwardVarBatch`,
||| extracts per-row predictions, builds per-sample loss, mean-reduces.
||| Mirrors V1 `epochNativeTensorBatch` for layers that benefit from a
||| single batched forward (notably Transformer).
export
epochVarTensorBatch : {0 d : Type} -> UserDeviceCore d => {i, o, n : Nat} -> {hs : List Nat} ->
                       NativeOptimizer ->
                       Vect n (TensorDataPoint i o) ->
                       LossFn d o ->
                       Network i hs o d WithGrad ->
                       (Network i hs o d WithGrad, Double)
epochVarTensorBatch opt dataPoints lossFn model =
  let inputs = toList (map inputTensor dataPoints)
      targets = toList (map targetTensor dataPoints)
      stackedIn = catAllTensors inputs
      stackedTgt = catAllTensors targets
      iI = cast {to=Int} i
      oI = cast {to=Int} o
      nI = cast {to=Int} n
      stackedInReshaped = prim__reshape2d stackedIn nI iI
      stackedTgtReshaped = prim__reshape2d stackedTgt nI oI
      inV = the (Tensor [n, i] d WithGrad) (MkTensor stackedInReshaped Nothing)
      tgtV = the (Tensor [n, o] d WithGrad) (MkTensor stackedTgtReshaped Nothing)
      (_, predB) = forwardVarBatch model inV
      losses = the (List (Tensor [] d WithGrad)) (go predB tgtV 0 n)
      totalLoss = foldl taddScalar (freshZeroLossT 0.0) losses
      mean = scaleLoss totalLoss (1.0 / cast n)
  in (model, nativeTrainStep opt mean)
  where
    go : Tensor [n, o] d WithGrad -> Tensor [n, o] d WithGrad -> Int -> Nat -> List (Tensor [] d WithGrad)
    go _ _ _ Z = []
    go predB tgtV k (S rest) =
      perRowLoss lossFn predB tgtV k :: go predB tgtV (k + 1) rest


----------------------------------------------------------------------
-- Recurrent epoch (sequence per data point)
----------------------------------------------------------------------

-- One step of a sequence: forward, compute loss against target,
-- accumulate.
recurStep : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
            LossFn d o ->
            (Network i hs o d WithGrad, Tensor [] d WithGrad) ->
            (Vector i Double, Vector o Double) ->
            (Network i hs o d WithGrad, Tensor [] d WithGrad)
recurStep lossFn (net, accLoss) (xVec, yVec) =
  let inV = the (TVec i d WithGrad) (MkTensor (bulkToPersistent xVec) Nothing)
      tgtV = the (TVec o d WithGrad) (MkTensor (bulkToPersistent yVec) Nothing)
      (net', predV) = forwardVar net inV
      stepL = lossFn predV tgtV
  in (net', taddScalar accLoss stepL)

-- Per-sequence loss: reset state, walk timesteps, mean-reduce.
perSeqLoss : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
             LossFn d o ->
             Network i hs o d WithGrad ->
             RecurrentDataPoint i o Double ->
             Tensor [] d WithGrad
perSeqLoss lossFn model dp =
  let pairs = zip (xs dp) (ys dp)
      startNet = resetNetwork model
      (_, totalLoss) = foldl (recurStep lossFn) (startNet, freshZeroLossT 0.0) pairs
      stepCount = length pairs
  in if stepCount == 0
       then totalLoss
       else scaleLoss totalLoss (1.0 / cast stepCount)

||| One recurrent epoch: per data point, reset state, walk the
||| sequence, mean per-step loss, mean across sequences, native train
||| step. Returns the (unchanged) network and the loss scalar.
export
epochRecurrentVar : {0 d : Type} -> UserDeviceCore d => {i, o, n : Nat} -> {hs : List Nat} ->
                     NativeOptimizer ->
                     Vect n (RecurrentDataPoint i o Double) ->
                     LossFn d o ->
                     Network i hs o d WithGrad ->
                     (Network i hs o d WithGrad, Double)
epochRecurrentVar opt dataPoints lossFn model =
  let seqLosses = map (perSeqLoss lossFn model) dataPoints in
  let totalLoss = foldl taddScalar (freshZeroLossT 0.0) seqLosses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStep opt mean)


----------------------------------------------------------------------
-- Two-phase epoch (NTM/DNC pattern: encode then decode)
----------------------------------------------------------------------

-- Forward zeros for `numSteps` (the decode phase), accumulating per-step loss.
decodeStep : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
             LossFn d o ->
             AnyPtr ->                                    -- zero input tensor (reused)
             (Network i hs o d WithGrad, Tensor [] d WithGrad) ->
             Vector o Double ->
             (Network i hs o d WithGrad, Tensor [] d WithGrad)
decodeStep lossFn zeroInPtr (net, accLoss) tgtVec =
  let inV = the (TVec i d WithGrad) (MkTensor zeroInPtr Nothing)
      tgtV = the (TVec o d WithGrad) (MkTensor (bulkToPersistent tgtVec) Nothing)
      (net', predV) = forwardVar net inV
      stepL = lossFn predV tgtV
  in (net', taddScalar accLoss stepL)

-- Encode phase: forward each input, discard output, thread state.
encodeStep : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
             Network i hs o d WithGrad ->
             Vector i Double ->
             Network i hs o d WithGrad
encodeStep net xVec =
  let inV = the (TVec i d WithGrad) (MkTensor (bulkToPersistent xVec) Nothing)
      (net', _) = forwardVar net inV
  in net'

-- Per-sequence two-phase loss: reset state, encode all inputs, decode
-- with zeros for `length targets` steps, mean-reduce per-step loss.
perSeqLossTwoPhase : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
                     LossFn d o ->
                     Network i hs o d WithGrad ->
                     TwoPhaseDataPoint i o Double ->
                     Tensor [] d WithGrad
perSeqLossTwoPhase lossFn model dp =
  let startNet = resetNetwork model
      encNet = foldl encodeStep startNet (encodingInputs dp)
      iI = cast {to=Int} i
      zeroIn = prim__create1d iI (prim__allocDoubles iI) 0
      (_, totalLoss) = foldl (decodeStep lossFn zeroIn) (encNet, freshZeroLossT 0.0) (targets dp)
      stepCount = length (targets dp)
  in if stepCount == 0
       then totalLoss
       else scaleLoss totalLoss (1.0 / cast stepCount)

||| One two-phase epoch: encode-then-decode pattern (NTM/DNC). Per
||| data point: reset state, walk the encoding inputs (discarding
||| outputs), then forward zeros for each target step computing loss.
||| Mean across sequences, native train step.
export
epochTwoPhaseVar : {0 d : Type} -> UserDeviceCore d => {i, o, n : Nat} -> {hs : List Nat} ->
                    NativeOptimizer ->
                    Vect n (TwoPhaseDataPoint i o Double) ->
                    LossFn d o ->
                    Network i hs o d WithGrad ->
                    (Network i hs o d WithGrad, Double)
epochTwoPhaseVar opt dataPoints lossFn model =
  let seqLosses = map (perSeqLossTwoPhase lossFn model) dataPoints in
  let totalLoss = foldl taddScalar (freshZeroLossT 0.0) seqLosses in
  let mean = scaleLoss totalLoss (1.0 / cast n) in
  (model, nativeTrainStep opt mean)


----------------------------------------------------------------------
-- Two-phase eval helpers (no autograd consumption)
----------------------------------------------------------------------

-- Read the elements of a 1D tensor pointer back into a Vector.
-- Used by `forwardTwoPhase` to convert per-step predictions to
-- pure Doubles for evaluation metrics like `bitAccuracy`.
export
tvecToVector : {n : Nat} -> AnyPtr -> Vector n Double
tvecToVector {n} ptr = VArray (build 0 n)
  where
    build : Int -> (k : Nat) -> Vect k (Scalar Double)
    build _ Z = []
    build off (S k) = SArray (prim__item1d ptr off) :: build (off + 1) k

-- Encode-then-decode forward pass, returning the per-step decode
-- predictions as Doubles. Mirrors V1's `forwardTwoPhase` for eval
-- purposes; tape entries from each forward accumulate (no train_step
-- is called) but do not affect correctness — only memory growth on
-- long eval runs.
export
forwardTwoPhase : {0 d : Type} -> UserDeviceCore d => {i, o : Nat} -> {hs : List Nat} ->
                      Network i hs o d WithGrad ->
                      TwoPhaseDataPoint i o Double ->
                      (Network i hs o d WithGrad, List (Vector o Double))
forwardTwoPhase model dp =
  let startNet = resetNetwork model
      encNet = foldl encodeStep startNet (encodingInputs dp)
      iI = cast {to=Int} i
      zeroIn = prim__create1d iI (prim__allocDoubles iI) 0
      decodeOnce : (Network i hs o d WithGrad, List (Vector o Double)) ->
                   Vector o Double ->
                   (Network i hs o d WithGrad, List (Vector o Double))
      decodeOnce (net, preds) _ =
        let inV = the (TVec i d WithGrad) (MkTensor zeroIn Nothing)
            (net', predV) = forwardVar net inV
            predVec = the (Vector o Double) (tvecToVector {n = o} predV.tensorPtr)
        in (net', preds ++ [predVec])
  in foldl decodeOnce (encNet, []) (targets dp)
