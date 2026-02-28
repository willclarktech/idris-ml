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
    Just d  => let val = v.value - d
               in Var (nextNodeId val) (Just pid) val 0 (const []) []
    Nothing => Var (nextNodeId v.value) v.paramId v.value 0 (const []) []
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
-- Adam
----------------------------------------------------------------------

||| Adam optimizer (Kingma & Ba, 2014)
export
adam : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
       (eps : Double) -> (maxGrad : Double) -> Optimizer
adam lr beta1 beta2 eps maxGrad = MkOptimizer step
  where
    step : SortedMap String Double -> OptimizerState ->
           (SortedMap String Double, OptimizerState)
    step grads st =
      let t' = st.t + 1
          tf = cast {to=Double} t'
      in foldl (\(ds, s), (pid, g) =>
        let gc = clipGrad maxGrad g
            mPrev = fromMaybe 0 $ lookup pid s.m
            vPrev = fromMaybe 0 $ lookup pid s.v
            mNew = beta1 * mPrev + (1 - beta1) * gc
            vNew = beta2 * vPrev + (1 - beta2) * gc * gc
            mHat = mNew / (1 - Prelude.pow beta1 tf)
            vHat = vNew / (1 - Prelude.pow beta2 tf)
            delta = lr * mHat / (Prelude.sqrt vHat + eps)
        in (insert pid delta ds,
            { m := insert pid mNew s.m,
              v := insert pid vNew s.v,
              t := t' } s))
        (empty, st) (Data.SortedMap.toList grads)
