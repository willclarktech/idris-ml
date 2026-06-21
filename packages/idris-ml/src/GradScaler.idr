||| GradScaler — the state machine that wraps `nativeTrainStepScaled`
||| with PyTorch's `cuda.amp.GradScaler` growth/backoff policy.
|||
||| Purpose: when training in low-precision compute (BF16, F16), grads
||| at the F16/BF16 magnitude underflow to zero before they reach the
||| optimizer. Pre-multiplying the loss by a `scale` factor shifts the
||| grad magnitude up — backward computes scaled grads, the C-side
||| `native_train_step_scaled` divides them back down by the same
||| factor before the optimizer step. The scale needs to stay high
||| enough to avoid underflow but not so high that grads overflow the
||| target dtype's range. PyTorch's policy: start high (default 2^16),
||| halve on observed non-finite grads (overflow), grow by 2× after
||| `growthInterval` consecutive successful steps.
|||
||| Wraps the state in two `IORef`s — current scale and consecutive-
||| success counter — keyed by `(d, dt)` at the type level so the
||| scaler is a first-class value the user passes through their
||| training loop alongside the optimizer.
|||
||| A3 of the type-safe mixed-precision plan (#410). Built on top of
||| `nativeTrainStepScaled` (Tensor.idr) which lands the NaN-sentinel
||| overflow signal across all three backends.
module GradScaler

import Data.IORef
import Data.Vect

import Executor
import Tensor

public export
record GradScaler (0 ex : Executor) (0 dt : DType) where
  constructor MkGradScaler
  scaleRef       : IORef Double
  consecutiveRef : IORef Nat
  growthFactor   : Double
  backoffFactor  : Double
  growthInterval : Nat

||| Construct a GradScaler with explicit policy parameters. PyTorch's
||| `cuda.amp.GradScaler` defaults: initScale = 2^16 = 65536.0,
||| growthFactor = 2.0, backoffFactor = 0.5, growthInterval = 2000.
||| Use these for F16; for BF16 the initial scale can be 1.0 because
||| BF16's 8-bit exponent matches F32's range (the underflow problem
||| only really bites in F16).
export
gradScaler : (initScale, growthFactor, backoffFactor : Double) ->
             (growthInterval : Nat) ->
             IO (GradScaler ex dt)
gradScaler initScale gf bf gi = do
  s <- newIORef initScale
  c <- newIORef Z
  pure (MkGradScaler s c gf bf gi)

||| PyTorch's `cuda.amp.GradScaler` defaults — 2^16 init scale, 2×
||| growth every 2000 successful steps, 0.5× backoff on overflow.
||| The right starting point for F16 training; BF16 mostly doesn't
||| need scaling (its 8-bit exponent matches F32's range), so for
||| BF16 use `gradScaler 1.0 1.0 1.0 1000000` for a no-op scaler.
export
defaultGradScaler : IO (GradScaler ex dt)
defaultGradScaler = gradScaler 65536.0 2.0 0.5 2000

||| Read the scaler's current scale (after any growth/backoff updates
||| from prior steps).
export
currentScale : GradScaler ex dt -> IO Double
currentScale gs = readIORef gs.scaleRef

||| Apply the scaler's current scale to a loss tensor (multiplies
||| by `currentScale gs`). The caller feeds the returned tensor
||| into backward + `trainStepScaled` (which unscales the grads).
||| Named `applyScale` rather than `scaleLoss` — it scales the loss
||| tensor for the mixed-precision step, not a mean reduction.
export
applyScale : {0 ex : Executor} -> UserExecutorCore ex =>
             GradScaler ex dt -> Tensor [] ex dt WithGrad ->
             IO (Tensor [] ex dt WithGrad)
applyScale gs loss = do
  s <- readIORef gs.scaleRef
  tmulScalar loss s

private
isNaN : Double -> Bool
isNaN x = x /= x

||| Run the GradScaler-aware fused train step. Reads the current
||| scale, calls `nativeTrainStepScaled`, advances the state machine
||| based on the NaN sentinel:
|||
||| * Non-NaN return → step succeeded; increment the consecutive-
|||   success counter, grow scale by `growthFactor` if the counter
|||   hit `growthInterval`.
||| * NaN return → overflow detected, step was skipped; reset the
|||   counter and shrink scale by `backoffFactor`.
|||
||| Returns the same Double the underlying primitive returned —
||| NaN for skipped steps, the unscaled loss otherwise. Callers
||| that want to distinguish should check `isNaN` themselves.
export
trainStepScaled : {0 ex : Executor} -> UserExecutorTraining ex => IsFloating dt =>
                  NativeOptimizer ex -> GradScaler ex dt ->
                  Tensor [] ex dt WithGrad -> IO Double
trainStepScaled opt gs scaledLoss = do
  scale <- readIORef gs.scaleRef
  result <- nativeTrainStepScaled opt scaledLoss scale
  if isNaN result
    then do
      modifyIORef gs.scaleRef (* gs.backoffFactor)
      writeIORef gs.consecutiveRef Z
      pure result
    else do
      c <- readIORef gs.consecutiveRef
      let c' = S c
      if c' >= gs.growthInterval
        then do
          modifyIORef gs.scaleRef (* gs.growthFactor)
          writeIORef gs.consecutiveRef Z
        else writeIORef gs.consecutiveRef c'
      pure result
