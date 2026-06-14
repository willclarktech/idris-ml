module Backprop

import Data.List
import Data.Vect

import Array
import DataPoint
import Executor
import GradScaler
import Layer.Core
import Layer.MixedCore
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
0 LossFn : (0 _ : Executor) -> (0 _ : DType) -> Nat -> Type
LossFn ex dt n = TVec n ex dt WithGrad -> TVec n ex dt WithGrad -> IO (Tensor [] ex dt WithGrad)

----------------------------------------------------------------------
-- Helpers (top-level so let-blocks below don't need where-clauses)
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a buffer at offset.
packDoublesIntoBuf : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoublesIntoBuf buf _ []            = buf
packDoublesIntoBuf buf off (x :: rest) =
  packDoublesIntoBuf (prim__setDouble buf off x) (off + 1) rest

-- Non-persistent input/target tensor from Vector n Double.
bulkToPersistent : {0 ex : Executor} -> Backend ex dt => {n : Nat} -> Vector n Double -> AnyPtr
bulkToPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf  = prim__allocDoubles nI
      buf' = packScalars buf 0 elems
  in dtCreate1d {ex} {t=dt} nI buf' 0 (deviceStreamTag {ex})
  where
    packScalars : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packScalars b _ []                 = b
    packScalars b o (SArray v :: rest) =
      packScalars (prim__setDouble b o v) (o + 1) rest

-- Scalar Tensor holding 0.0. IO so its FFI side effect happens at
-- sequence-time rather than at call-time.
freshZeroLossT : {0 ex : Executor} -> Backend ex dt => Double -> IO (Tensor [] ex dt WithGrad)
freshZeroLossT seed = ioRerun (\_ => MkTensor (dtCreateScalar {ex} {t=dt} seed 0 (deviceStreamTag {ex})) Nothing)

-- Add two scalar TVars. Dispatches via `primAdd {ex}` so the
-- type-level device tag drives MLX stream selection.
taddScalar : {0 ex : Executor} -> UserExecutorCore ex =>
             Tensor [] ex dt WithGrad -> Tensor [] ex dt WithGrad -> IO (Tensor [] ex dt WithGrad)
taddScalar a b = ioRerun (\_ => MkTensor (primAdd {ex} a.tensorPtr b.tensorPtr) Nothing)

-- Scale a scalar Tensor by a Double.
scaleLoss : {0 ex : Executor} -> UserExecutorCore ex =>
            Tensor [] ex dt WithGrad -> Double -> IO (Tensor [] ex dt WithGrad)
scaleLoss v s = ioRerun (\_ => MkTensor (primMulScalar {ex} v.tensorPtr s) Nothing)

-- Sum a list of scalar tensors starting from a fresh zero. Replaces
-- the old `foldl taddScalar (freshZeroLossT 0.0) losses` pattern under
-- the IO-typed surface.
sumLosses : {0 ex : Executor} -> Backend ex dt =>
            List (Tensor [] ex dt WithGrad) -> IO (Tensor [] ex dt WithGrad)
sumLosses losses = do
  zero <- freshZeroLossT 0.0
  foldlM taddScalar zero losses
  where
    foldlM : (Tensor [] ex dt WithGrad -> Tensor [] ex dt WithGrad -> IO (Tensor [] ex dt WithGrad)) ->
             Tensor [] ex dt WithGrad -> List (Tensor [] ex dt WithGrad) ->
             IO (Tensor [] ex dt WithGrad)
    foldlM _ acc []          = pure acc
    foldlM f acc (x :: rest) = do
      acc' <- f acc x
      foldlM f acc' rest

----------------------------------------------------------------------
-- Supervised epoch (feed-forward)
----------------------------------------------------------------------

%default partial

-- Per-point loss closure factored out to avoid let-block elaboration
-- weirdness in epochVar's body.
perPointLoss : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
               LossFn ex dt o ->
               Network i hs o ex dt WithGrad ->
               DataPoint i o Double ->
               IO (Tensor [] ex dt WithGrad)
perPointLoss lossFn model dp = do
  let inT = bulkToPersistent {ex} {dt} (x dp)
      tgtT = bulkToPersistent {ex} {dt} (y dp)
      inV  = the (TVec i ex dt WithGrad) (MkTensor inT Nothing)
      tgtV = the (TVec o ex dt WithGrad) (MkTensor tgtT Nothing)
  (_, predV) <- forwardVar model inV
  lossFn predV tgtV

||| One supervised epoch: forward each data point, accumulate per-
||| sample losses, mean-reduce, native train step. Returns the
||| (unchanged) network and the loss scalar.
export
epochVar : {0 ex : Executor} -> Backend ex dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
            NativeOptimizer ex ->
            Vect n (DataPoint i o Double) ->
            LossFn ex dt o ->
            Network i hs o ex dt WithGrad ->
            IO (Network i hs o ex dt WithGrad, Double)
epochVar opt dataPoints lossFn model = do
  losses <- traverse (perPointLoss lossFn model) dataPoints
  totalLoss <- sumLosses (toList losses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)

----------------------------------------------------------------------
-- Mixed-precision supervised epoch (A4 of #410)
----------------------------------------------------------------------

-- Per-point loss for mixed-precision networks. Mirrors `perPointLoss`
-- but threads the cDt (compute / activation dtype) through both the
-- bulkToPersistent material​isation of inputs/targets and the
-- forwardVarMixed call. The paramDt slot on the network is unused
-- at the forward boundary — params get cast paramDt → cDt inside
-- the layer's `applyVarMixed`.
perPointLossMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                    UserExecutorQuant ex =>
                    IsDType pDt          => IsDType cDt =>
                    RuntimeDType pDt     => RuntimeDType cDt =>
                    Linked ex            => Compatible ex pDt => Compatible ex cDt =>
                    {i, o : Nat} -> {hs : List Nat} ->
                    LossFn ex cDt o ->
                    NetworkMixed i hs o ex pDt cDt WithGrad ->
                    DataPoint i o Double ->
                    IO (Tensor [] ex cDt WithGrad)
perPointLossMixed lossFn model dp = do
  let inT  = bulkToPersistent {ex} {dt=cDt} (x dp)
      tgtT = bulkToPersistent {ex} {dt=cDt} (y dp)
      inV  = the (TVec i ex cDt WithGrad) (MkTensor inT  Nothing)
      tgtV = the (TVec o ex cDt WithGrad) (MkTensor tgtT Nothing)
  (_, predV) <- forwardVarMixed model inV
  lossFn predV tgtV

||| One supervised epoch in mixed precision: forward each data point
||| in `cDt` (activations / compute), accumulate per-sample losses,
||| mean-reduce, scale by the GradScaler's current factor, and run
||| `trainStepScaled` which (a) backwards at the scaled magnitude,
||| (b) divides grads by the scale, (c) checks for non-finite values
||| and skips the step + halves the scaler on overflow, (d) clips +
||| steps + advances the scaler's growth/backoff state machine.
|||
||| Returns the loss scalar from `trainStepScaled` (which is NaN if
||| the step was skipped due to overflow — callers should treat NaN
||| epoch losses as "skip" rather than "diverged").
export
epochVarMixed : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex =>
                UserExecutorQuant ex =>
                IsDType pDt          => IsDType cDt =>
                RuntimeDType pDt     => RuntimeDType cDt =>
                Linked ex            => Compatible ex pDt => Compatible ex cDt =>
                IsFloating cDt       =>
                {i, o, n : Nat} -> {hs : List Nat} ->
                NativeOptimizer ex ->
                GradScaler ex cDt ->
                Vect n (DataPoint i o Double) ->
                LossFn ex cDt o ->
                NetworkMixed i hs o ex pDt cDt WithGrad ->
                IO (NetworkMixed i hs o ex pDt cDt WithGrad, Double)
epochVarMixed opt gs dataPoints lossFn model = do
  losses <- traverse (perPointLossMixed lossFn model) dataPoints
  totalLoss <- sumLosses (toList losses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  scaledMean <- applyScale gs mean
  loss <- trainStepScaled opt gs scaledMean
  pure (model, loss)

-- `catAllTensors` (per-sample [k] handles -> [n*k]) now lives in
-- Tensor.idr next to `bulkToTensor2d`; imported via `import Tensor`.

-- Per-sample loss for batched-forward shape.
perRowLoss : {0 ex : Executor} -> UserExecutorTraining ex => {n, o : Nat} ->
             LossFn ex dt o ->
             Tensor [n, o] ex dt WithGrad ->
             Tensor [n, o] ex dt WithGrad ->
             Int ->
             IO (Tensor [] ex dt WithGrad)
perRowLoss lossFn predB tgtB k = do
  predRow <- trowSelect predB k
  tgtRow <- trowSelect tgtB k
  lossFn predRow tgtRow

||| Batched supervised epoch over `TensorDataPoint`s.
export
epochVarTensorBatch : {0 ex : Executor} -> Backend ex dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
                       NativeOptimizer ex ->
                       Vect n (TensorDataPoint i o) ->
                       LossFn ex dt o ->
                       Network i hs o ex dt WithGrad ->
                       IO (Network i hs o ex dt WithGrad, Double)
epochVarTensorBatch opt dataPoints lossFn model = do
  let inputs = toList (map inputTensor dataPoints)
      targets            = toList (map targetTensor dataPoints)
      stackedIn          = catAllTensors {ex} inputs
      stackedTgt         = catAllTensors {ex} targets
      iI                 = cast {to=Int} i
      oI                 = cast {to=Int} o
      nI                 = cast {to=Int} n
      stackedInReshaped  = primReshape2d {ex} stackedIn nI iI
      stackedTgtReshaped = primReshape2d {ex} stackedTgt nI oI
      inV                = the (Tensor [n, i] ex dt WithGrad) (MkTensor stackedInReshaped Nothing)
      tgtV               = the (Tensor [n, o] ex dt WithGrad) (MkTensor stackedTgtReshaped Nothing)
  (_, predB) <- forwardVarBatch model inV
  losses <- go predB tgtV 0 n
  totalLoss <- sumLosses losses
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)
  where
    go : Tensor [n, o] ex dt WithGrad -> Tensor [n, o] ex dt WithGrad -> Int -> Nat ->
         IO (List (Tensor [] ex dt WithGrad))
    go _ _ _ Z               = pure []
    go predB tgtV k (S rest) = do
      l <- perRowLoss lossFn predB tgtV k
      ls <- go predB tgtV (k + 1) rest
      pure (l :: ls)

----------------------------------------------------------------------
-- Recurrent epoch (sequence per data point)
----------------------------------------------------------------------

-- One step of a sequence: forward, compute loss against target,
-- accumulate.
recurStep : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
            LossFn ex dt o ->
            (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad) ->
            (Vector i Double, Vector o Double) ->
            IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
recurStep lossFn (net, accLoss) (xVec, yVec) = do
  let inV = the (TVec i ex dt WithGrad) (MkTensor (bulkToPersistent {ex} {dt} xVec) Nothing)
      tgtV = the (TVec o ex dt WithGrad) (MkTensor (bulkToPersistent {ex} {dt} yVec) Nothing)
  (net', predV) <- forwardVar net inV
  stepL <- lossFn predV tgtV
  newAcc <- taddScalar accLoss stepL
  pure (net', newAcc)

-- Per-sequence loss: reset state, walk timesteps, mean-reduce.
perSeqLoss : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
             LossFn ex dt o ->
             Network i hs o ex dt WithGrad ->
             RecurrentDataPoint i o Double ->
             IO (Tensor [] ex dt WithGrad)
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
    foldlIO : ((Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
                -> (Vector i Double, Vector o Double)
                -> IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad))
            -> (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
            -> List (Vector i Double, Vector o Double)
            -> IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
    foldlIO _ acc []          = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

||| One recurrent epoch.
export
epochRecurrentVar : {0 ex : Executor} -> Backend ex dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
                     NativeOptimizer ex ->
                     Vect n (RecurrentDataPoint i o Double) ->
                     LossFn ex dt o ->
                     Network i hs o ex dt WithGrad ->
                     IO (Network i hs o ex dt WithGrad, Double)
epochRecurrentVar opt dataPoints lossFn model = do
  seqLosses <- traverse (perSeqLoss lossFn model) dataPoints
  totalLoss <- sumLosses (toList seqLosses)
  mean <- scaleLoss totalLoss (1.0 / cast n)
  loss <- nativeTrainStep opt mean
  pure (model, loss)

----------------------------------------------------------------------
-- Two-phase epoch (NTM/DNC pattern: encode then decode)
----------------------------------------------------------------------

decodeStep : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
             LossFn ex dt o ->
             AnyPtr ->
             (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad) ->
             Vector o Double ->
             IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
decodeStep lossFn zeroInPtr (net, accLoss) tgtVec = do
  let inV = the (TVec i ex dt WithGrad) (MkTensor zeroInPtr Nothing)
      tgtV = the (TVec o ex dt WithGrad) (MkTensor (bulkToPersistent {ex} {dt} tgtVec) Nothing)
  (net', predV) <- forwardVar net inV
  stepL <- lossFn predV tgtV
  newAcc <- taddScalar accLoss stepL
  pure (net', newAcc)

encodeStep : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
             Network i hs o ex dt WithGrad ->
             Vector i Double ->
             IO (Network i hs o ex dt WithGrad)
encodeStep net xVec = do
  let inV = the (TVec i ex dt WithGrad) (MkTensor (bulkToPersistent {ex} {dt} xVec) Nothing)
  (net', _) <- forwardVar net inV
  pure net'

perSeqLossTwoPhase : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
                     LossFn ex dt o ->
                     Network i hs o ex dt WithGrad ->
                     TwoPhaseDataPoint i o Double ->
                     IO (Tensor [] ex dt WithGrad)
perSeqLossTwoPhase lossFn model dp = do
  let startNet = resetNetwork model
  encNet <- foldlIO encodeStep startNet (encodingInputs dp)
  let iI = cast {to=Int} i
      zeroIn = dtCreate1d {ex} {t=dt} iI (prim__allocDoubles iI) 0 (deviceStreamTag {ex})
  zero <- freshZeroLossT 0.0
  (_, totalLoss) <- foldlIO2 (decodeStep lossFn zeroIn) (encNet, zero) (targets dp)
  let stepCount = length (targets dp)
  if stepCount == 0
     then pure totalLoss
     else scaleLoss totalLoss (1.0 / cast stepCount)
  where
    foldlIO : (Network i hs o ex dt WithGrad -> Vector i Double -> IO (Network i hs o ex dt WithGrad))
            -> Network i hs o ex dt WithGrad
            -> List (Vector i Double)
            -> IO (Network i hs o ex dt WithGrad)
    foldlIO _ acc []          = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

    foldlIO2 : ((Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
                 -> Vector o Double
                 -> IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad))
             -> (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
             -> List (Vector o Double)
             -> IO (Network i hs o ex dt WithGrad, Tensor [] ex dt WithGrad)
    foldlIO2 _ acc []          = pure acc
    foldlIO2 f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO2 f acc' rest

||| One two-phase epoch.
export
epochTwoPhaseVar : {0 ex : Executor} -> Backend ex dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
                    NativeOptimizer ex ->
                    Vect n (TwoPhaseDataPoint i o Double) ->
                    LossFn ex dt o ->
                    Network i hs o ex dt WithGrad ->
                    IO (Network i hs o ex dt WithGrad, Double)
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
tvecToVector : {0 ex : Executor} -> UserExecutorCore ex => {n : Nat} -> AnyPtr -> Vector n Double
tvecToVector {n} ptr = VArray (build 0 n)
  where
    build : Int -> (k : Nat) -> Vect k (Scalar Double)
    build _ Z       = []
    build off (S k) = SArray (primItem1d {ex} ptr off) :: build (off + 1) k

export
forwardTwoPhase : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> {hs : List Nat} ->
                      Network i hs o ex dt WithGrad ->
                      TwoPhaseDataPoint i o Double ->
                      IO (Network i hs o ex dt WithGrad, List (Vector o Double))
forwardTwoPhase model dp = do
  let startNet = resetNetwork model
  encNet <- foldlIO encodeStep startNet (encodingInputs dp)
  let iI = cast {to=Int} i
      zeroIn = dtCreate1d {ex} {t=dt} iI (prim__allocDoubles iI) 0 (deviceStreamTag {ex})
  foldlIO2 (decodeOnce zeroIn) (encNet, []) (targets dp)
  where
    decodeOnce : AnyPtr ->
                 (Network i hs o ex dt WithGrad, List (Vector o Double)) ->
                 Vector o Double ->
                 IO (Network i hs o ex dt WithGrad, List (Vector o Double))
    decodeOnce zeroIn (net, preds) _ = do
      let inV = the (TVec i ex dt WithGrad) (MkTensor zeroIn Nothing)
      (net', predV) <- forwardVar net inV
      let predVec = the (Vector o Double) (tvecToVector {ex} {n = o} predV.tensorPtr)
      pure (net', preds ++ [predVec])

    foldlIO : (Network i hs o ex dt WithGrad -> Vector i Double -> IO (Network i hs o ex dt WithGrad))
            -> Network i hs o ex dt WithGrad
            -> List (Vector i Double)
            -> IO (Network i hs o ex dt WithGrad)
    foldlIO _ acc []          = pure acc
    foldlIO f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO f acc' rest

    foldlIO2 : ((Network i hs o ex dt WithGrad, List (Vector o Double))
                 -> Vector o Double
                 -> IO (Network i hs o ex dt WithGrad, List (Vector o Double)))
             -> (Network i hs o ex dt WithGrad, List (Vector o Double))
             -> List (Vector o Double)
             -> IO (Network i hs o ex dt WithGrad, List (Vector o Double))
    foldlIO2 _ acc []          = pure acc
    foldlIO2 f acc (x :: rest) = do
      acc' <- f acc x
      foldlIO2 f acc' rest
