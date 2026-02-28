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

  -- applyDeltas updates named variable
  , let deltas = fromList [("w", 0.5)]
        v = param "w" 3.0
        v' = applyDeltas deltas v
    in checkClose "applyDeltas updates" 2.5 v'.value tol
  ]
