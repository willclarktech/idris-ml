module Test.Init

import Compat.Random
import Init
import Test.Harness

tol : Double
tol = 1.0e-12

-- Init strategies draw from Compat.Random (libc rand), so seed via
-- Compat.Random.srand and assert properties (variance-derived bounds,
-- determinism, sampler scaling) — not exact values, which encode one
-- platform's rand() sequence (glibc and macOS libc differ; CI run
-- 27373449876 failed every hardcoded check on Ubuntu).

uniformBound : (var : Double) -> Double -> Bool
uniformBound var v = let limit = sqrt (3.0 * var) in v >= -limit && v <= limit

export
tests : List (IO Bool)
tests =
  [ -- xavier uniform: var = 2/(10+10) = 0.1
    do srand 42
       v <- xavier uniform 10 10
       check "xavier uniform 10 10 within bounds" (uniformBound 0.1 v)

  -- xavier normal scales a standard normal draw by sqrt(2/(fanIn+fanOut))
  , do srand 42
       n <- normalSample
       srand 42
       v <- xavier normal 10 10
       checkClose "xavier normal 10 10 = normalSample * sqrt(0.1)" (n * sqrt 0.1) v tol

  -- he uniform: var = 2/10 = 0.2
  , do srand 42
       v <- he uniform 10 5
       check "he uniform 10 5 within bounds" (uniformBound 0.2 v)

  -- he normal: var = 2/10 = 0.2
  , do srand 42
       n <- normalSample
       srand 42
       v <- he normal 10 5
       checkClose "he normal 10 5 = normalSample * sqrt(0.2)" (n * sqrt 0.2) v tol

  -- lecun uniform: var = 1/10 = 0.1
  , do srand 42
       v <- lecun uniform 10 5
       check "lecun uniform 10 5 within bounds" (uniformBound 0.1 v)

  -- lecun normal: var = 1/10 = 0.1
  , do srand 42
       n <- normalSample
       srand 42
       v <- lecun normal 10 5
       checkClose "lecun normal 10 5 = normalSample * sqrt(0.1)" (n * sqrt 0.1) v tol

  -- fixedRange ignores dimensions, samples U(-bound, bound)
  , do srand 42
       v <- fixedRange 2.0 10 10
       check "fixedRange 2.0 within bounds" (v >= -2.0 && v <= 2.0)

  -- determinism: same seed reproduces the same init sample
  , do srand 42
       v1 <- xavier uniform 10 10
       srand 42
       v2 <- xavier uniform 10 10
       checkClose "xavier uniform deterministic under same seed" v1 v2 tol
  ]
