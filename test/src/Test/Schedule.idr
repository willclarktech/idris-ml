module Test.Schedule

import Harness
import Schedule


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- constant returns same value at any epoch
    checkClose "constant epoch 0" 0.01 (constant 0.01 0) tol
  , checkClose "constant epoch 100" 0.01 (constant 0.01 100) tol

  -- cosineAnnealing: epoch 0 = lrMax
  , checkClose "cosine epoch 0 = lrMax" 0.1 (cosineAnnealing 0.1 0.001 1000 0) tol

  -- cosineAnnealing: epoch total = lrMin
  , checkClose "cosine epoch total = lrMin" 0.001 (cosineAnnealing 0.1 0.001 1000 1000) tol

  -- oneCycle: epoch 0 = lrMax / div
  , checkClose "oneCycle epoch 0" (0.001 / 25.0) (oneCycle 0.001 25.0 1.0e5 0.25 6000 0) tol

  -- oneCycle: past total = lrMax / divFinal
  , checkClose "oneCycle past total" (0.001 / 1.0e5) (oneCycle 0.001 25.0 1.0e5 0.25 6000 6000) tol
  ]
