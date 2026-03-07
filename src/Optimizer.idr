module Optimizer

import Data.Maybe
import Data.SortedMap

import Variable


----------------------------------------------------------------------
-- State
----------------------------------------------------------------------

||| Optimizer state shared across algorithms.
||| SGD ignores all fields; Adam uses m (first moment), v (second moment), t (timestep).
public export
record OptimizerState where
  constructor MkOptimizerState
  m : SortedMap String Double
  v : SortedMap String Double
  t : Int

export
initState : OptimizerState
initState = MkOptimizerState empty empty 0

----------------------------------------------------------------------
-- Optimizer Record
----------------------------------------------------------------------

||| Bundles a step function with its initial state.
||| step: takes per-parameter gradients -> current state -> (per-param deltas, new state)
public export
record Optimizer where
  constructor MkOptimizer
  step : SortedMap String Double -> OptimizerState ->
         (SortedMap String Double, OptimizerState)

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

clipGrad : Double -> Double -> Double
clipGrad maxAbs g = max (-maxAbs) (min maxAbs g)

export
applyDeltas : SortedMap String Double -> Variable -> Variable
applyDeltas deltas v = case v.paramId of
  Just pid => case lookup pid deltas of
    Just d  => { value := v.value - d } v
    Nothing => v
  Nothing  => v

----------------------------------------------------------------------
-- SGD
----------------------------------------------------------------------

||| Stochastic gradient descent: delta = lr * clipped_grad
export
sgd : (lr : Double) -> (maxGrad : Double) -> Optimizer
sgd lr maxGrad = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st =
      let deltas = map (\g => lr * clipGrad maxGrad g) grads
      in (deltas, st)

----------------------------------------------------------------------
-- Global Gradient Norm Clipping
----------------------------------------------------------------------

||| Scale all gradients so the global L2 norm does not exceed maxNorm.
||| Preserves gradient direction — standard for attention/recurrent models.
export
clipGlobalNorm : Double -> SortedMap String Double -> SortedMap String Double
clipGlobalNorm maxNorm grads =
  let norm = Prelude.sqrt $ foldl (\acc, g => acc + g * g) 0 (values grads)
      scale = if norm > maxNorm then maxNorm / norm else 1.0
  in map (* scale) grads

----------------------------------------------------------------------
-- Value Clipping
----------------------------------------------------------------------

||| Clip each gradient element to [-maxVal, maxVal].
||| Matches PyTorch's clip_grad_value_.
export
clipGradValue : Double -> SortedMap String Double -> SortedMap String Double
clipGradValue maxVal = map (clipGrad maxVal)

----------------------------------------------------------------------
-- RMSprop
----------------------------------------------------------------------

rmspropStep : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
              SortedMap String Double -> OptimizerState ->
              (SortedMap String Double, OptimizerState)
rmspropStep lr alpha eps grads st =
  foldl (\(ds, s), (pid, g) =>
    let vPrev = fromMaybe 0 $ lookup pid s.v
        vNew = alpha * vPrev + (1 - alpha) * g * g
        delta = lr * g / (Prelude.sqrt vNew + eps)
    in (insert pid delta ds,
        { v := insert pid vNew s.v } s))
    (empty, st) (Data.SortedMap.toList grads)

||| RMSprop optimizer (Hinton, 2012).
||| v_t = alpha * v_{t-1} + (1 - alpha) * g^2
||| delta = lr * g / (sqrt(v_t) + eps)
export
rmsprop : (lr : Double) -> (alpha : Double) -> (eps : Double) -> Optimizer
rmsprop lr alpha eps = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st = rmspropStep lr alpha eps grads st

||| RMSprop with per-element value clipping (matches PyTorch reference).
export
rmspropValueClip : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                   (maxVal : Double) -> Optimizer
rmspropValueClip lr alpha eps maxVal = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st =
      let clipped = clipGradValue maxVal grads
      in rmspropStep lr alpha eps clipped st

----------------------------------------------------------------------
-- Adam
----------------------------------------------------------------------

adamStep : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
           (eps : Double) ->
           SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
adamStep lr beta1 beta2 eps grads st =
  let t' = st.t + 1
      tf = cast {to=Double} t'
  in foldl (\(ds, s), (pid, g) =>
    let mPrev = fromMaybe 0 $ lookup pid s.m
        vPrev = fromMaybe 0 $ lookup pid s.v
        mNew = beta1 * mPrev + (1 - beta1) * g
        vNew = beta2 * vPrev + (1 - beta2) * g * g
        mHat = mNew / (1 - Prelude.pow beta1 tf)
        vHat = vNew / (1 - Prelude.pow beta2 tf)
        delta = lr * mHat / (Prelude.sqrt vHat + eps)
    in (insert pid delta ds,
        { m := insert pid mNew s.m,
          v := insert pid vNew s.v,
          t := t' } s))
    (empty, st) (Data.SortedMap.toList grads)

||| Adam optimizer (Kingma & Ba, 2014) with per-parameter gradient clipping.
export
adam : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
       (eps : Double) -> (maxGrad : Double) -> Optimizer
adam lr beta1 beta2 eps maxGrad = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st =
      let clipped = map (clipGrad maxGrad) grads
      in adamStep lr beta1 beta2 eps clipped st

||| Adam optimizer with global gradient norm clipping.
||| Preserves gradient direction — preferred for attention/recurrent models.
export
adamGlobalClip : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                 (eps : Double) -> (maxNorm : Double) -> Optimizer
adamGlobalClip lr beta1 beta2 eps maxNorm = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st =
      let clipped = clipGlobalNorm maxNorm grads
      in adamStep lr beta1 beta2 eps clipped st


----------------------------------------------------------------------
-- Dense Optimizer (C arrays, no SortedMap)
----------------------------------------------------------------------

||| Dense optimizer state: C arrays indexed by pid_id.
public export
record DenseOptimizerState where
  constructor MkDenseOptimizerState
  v : AnyPtr     -- C double array (RMSprop v / Adam v)
  m : AnyPtr     -- C double array (Adam m, unused for RMSprop/SGD)
  buf : AnyPtr   -- reusable grads/deltas buffer
  t : Int
  n : Int        -- number of parameters

||| Dense optimizer: operates on C arrays via FFI.
||| step takes the grads buffer (filled by collectGradsDense),
||| transforms grads -> deltas in-place, updates state.
public export
record DenseOptimizer where
  constructor MkDenseOptimizer
  step : AnyPtr -> DenseOptimizerState -> DenseOptimizerState

||| Initialize dense optimizer state for n parameters.
export
initDenseState : Int -> DenseOptimizerState
initDenseState numPids = MkDenseOptimizerState
  (prim__denseAlloc numPids) (prim__denseAlloc numPids)
  (prim__denseAlloc numPids) 0 numPids

||| RMSprop with value clipping (dense C arrays).
export
rmspropValueClipDense : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                        (maxVal : Double) -> DenseOptimizer
rmspropValueClipDense lr alpha eps maxVal = MkDenseOptimizer step
  where
    step : AnyPtr -> DenseOptimizerState -> DenseOptimizerState
    step grads st =
      let _ = prim__rmspropVcStep grads st.v st.n lr alpha eps maxVal
      in st  -- v updated in-place, grads now contains deltas

||| SGD with per-param clipping (dense C arrays).
export
sgdDense : (lr : Double) -> (maxGrad : Double) -> DenseOptimizer
sgdDense lr maxGrad = MkDenseOptimizer step
  where
    step : AnyPtr -> DenseOptimizerState -> DenseOptimizerState
    step grads st =
      let _ = prim__sgdStep grads st.n lr maxGrad
      in st

||| Adam with global gradient norm clipping (dense C arrays).
export
adamGlobalClipDense : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                      (eps : Double) -> (maxNorm : Double) -> DenseOptimizer
adamGlobalClipDense lr beta1 beta2 eps maxNorm = MkDenseOptimizer step
  where
    step : AnyPtr -> DenseOptimizerState -> DenseOptimizerState
    step grads st =
      let _ = prim__adamGcStep grads st.m st.v st.n lr beta1 beta2 eps maxNorm st.t
      in { t := st.t + 1 } st
