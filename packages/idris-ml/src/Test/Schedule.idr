module Test.Schedule

import Ml.Schedule
import Test.Harness

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

  -- cosineWithWarmup: pinned against the reference's nanoGPT-verbatim
  -- cosine_lr (torch_ref/scripts/gpt.py) at lr=0.001 min=0.0001
  -- warmup=3 total=30 — the gpt example's defaults, which the step
  -- oracle compares one update under.
  , checkClose "cosineWithWarmup epoch 0" 0.00025 (cosineWithWarmup 0.001 0.0001 3 30 0) tol
  , checkClose "cosineWithWarmup epoch 2" 0.00075 (cosineWithWarmup 0.001 0.0001 3 30 2) tol
  , checkClose "cosineWithWarmup warmup end = lrMax" 0.001 (cosineWithWarmup 0.001 0.0001 3 30 3) tol
  , checkClose "cosineWithWarmup mid-decay epoch 16" 0.0005761651730097141
               (cosineWithWarmup 0.001 0.0001 3 30 16) tol
  , checkClose "cosineWithWarmup epoch total = lrMin" 0.0001 (cosineWithWarmup 0.001 0.0001 3 30 30) tol
  , checkClose "cosineWithWarmup past total = lrMin" 0.0001 (cosineWithWarmup 0.001 0.0001 3 30 45) tol

  -- oneCycle: epoch 0 = lrMax / div
  , checkClose "oneCycle epoch 0" (0.001 / 25.0) (oneCycle 0.001 25.0 1.0e5 0.25 6000 0) tol

  -- oneCycle: past total = lrMax / divFinal
  , checkClose "oneCycle past total" (0.001 / 1.0e5) (oneCycle 0.001 25.0 1.0e5 0.25 6000 6000) tol

  -- stepLR: epoch 0 = baseLR
  , checkClose "stepLR epoch 0" 0.1 (stepLR 0.1 10 0.5 0) tol

  -- stepLR: epoch 9 = baseLR (still in first step)
  , checkClose "stepLR epoch 9" 0.1 (stepLR 0.1 10 0.5 9) tol

  -- stepLR: epoch 10 = baseLR * gamma
  , checkClose "stepLR epoch 10" 0.05 (stepLR 0.1 10 0.5 10) tol

  -- stepLR: epoch 20 = baseLR * gamma^2
  , checkClose "stepLR epoch 20" 0.025 (stepLR 0.1 10 0.5 20) tol

  -- exponentialLR: epoch 0 = baseLR
  , checkClose "exponentialLR epoch 0" 0.1 (exponentialLR 0.1 0.95 0) tol

  -- exponentialLR: epoch 1 = baseLR * gamma
  , checkClose "exponentialLR epoch 1" (0.1 * 0.95) (exponentialLR 0.1 0.95 1) tol

  -- exponentialLR: epoch 10 = baseLR * gamma^10
  , checkClose "exponentialLR epoch 10" (0.1 * pow 0.95 10.0) (exponentialLR 0.1 0.95 10) tol
  ]
