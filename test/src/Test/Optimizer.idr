module Test.Optimizer

import Data.Maybe
import Data.SortedMap

import Harness
import Optimizer
import Variable


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- SGD: delta = lr * grad
    let grads = fromList [("w", 2.0)]
        opt = sgd 0.1 100.0
        (deltas, _) = opt.step grads initState
    in checkClose "sgd delta = lr * grad" 0.2 (fromMaybe 0.0 (lookup "w" deltas)) tol

  -- SGD clipping
  , let grads = fromList [("w", 200.0)]
        opt = sgd 0.1 10.0
        (deltas, _) = opt.step grads initState
    in checkClose "sgd clips grad" 1.0 (fromMaybe 0.0 (lookup "w" deltas)) tol

  -- Adam first step: verify it produces a delta
  , let grads = fromList [("w", 1.0)]
        opt = adamGlobalClip 0.001 0.9 0.999 1.0e-8 50.0
        (deltas, st') = opt.step grads initState
        d = fromMaybe 0.0 (lookup "w" deltas)
    in check "adam produces delta" (d > 0.0 && st'.t == 1)

  -- clipGlobalNorm: scales when norm > maxNorm
  , let grads = fromList [("a", 3.0), ("b", 4.0)]  -- norm = 5
        clipped = clipGlobalNorm 2.5 grads
        a = fromMaybe 0.0 (lookup "a" clipped)
        b = fromMaybe 0.0 (lookup "b" clipped)
    in check "clipGlobalNorm scales"
       (abs (a - 3.0 * 2.5/5.0) < tol && abs (b - 4.0 * 2.5/5.0) < tol)

  -- clipGlobalNorm: no-op when under
  , let grads = fromList [("a", 0.1), ("b", 0.1)]
        clipped = clipGlobalNorm 50.0 grads
        a = fromMaybe 0.0 (lookup "a" clipped)
    in checkClose "clipGlobalNorm no-op when under" 0.1 a tol

  -- RMSprop first step: v = (1-alpha)*g^2, delta = lr*g/sqrt(v+eps)
  , let grads = fromList [("w", 2.0)]
        opt = rmsprop 0.01 0.99 1.0e-8
        (deltas, st') = opt.step grads initState
        d = fromMaybe 0.0 (lookup "w" deltas)
        -- v = 0.01*4 = 0.04, delta = 0.01*2/sqrt(0.04+1e-8) = 0.02/0.2 = 0.1
    in checkClose "rmsprop step" 0.1 d tol

  -- RMSprop accumulates running average across steps
  , let grads = fromList [("w", 2.0)]
        opt = rmsprop 0.01 0.99 1.0e-8
        (_, st1) = opt.step grads initState
        (deltas2, _) = opt.step grads st1
        d = fromMaybe 0.0 (lookup "w" deltas2)
        -- v1 = 0.04, v2 = 0.99*0.04 + 0.01*4 = 0.0396+0.04 = 0.0796
        -- delta = 0.01*2/sqrt(0.0796) = 0.02/0.28213.. ≈ 0.0708988..
    in checkClose "rmsprop accumulates" 0.07088811799 d 1.0e-5

  -- clipGradValue clips to range
  , let grads = fromList [("a", 15.0), ("b", -20.0), ("c", 5.0)]
        clipped = clipGradValue 10.0 grads
        a = fromMaybe 0.0 (lookup "a" clipped)
        b = fromMaybe 0.0 (lookup "b" clipped)
        c = fromMaybe 0.0 (lookup "c" clipped)
    in check "clipGradValue" (abs (a - 10.0) < tol && abs (b - (-10.0)) < tol && abs (c - 5.0) < tol)

  -- rmspropValueClip clips before computing step
  , let grads = fromList [("w", 200.0)]
        opt = rmspropValueClip 0.01 0.99 1.0e-8 10.0
        (deltas, _) = opt.step grads initState
        d = fromMaybe 0.0 (lookup "w" deltas)
        -- clipped to 10, v = 0.01*100=1.0, delta = 0.01*10/sqrt(1+1e-8) ≈ 0.1
    in checkClose "rmspropValueClip clips" 0.1 d 1.0e-5

  -- applyDeltas updates named variable
  , let deltas = fromList [("w", 0.5)]
        v = param "w" 3.0
        v' = applyDeltas deltas v
    in checkClose "applyDeltas updates" 2.5 v'.value tol
  ]
