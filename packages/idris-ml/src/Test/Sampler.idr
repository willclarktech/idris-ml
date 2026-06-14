module Test.Sampler

import Compat.Random
import Sampler
import Test.Harness

tol : Double
tol = 1.0e-12

-- The samplers draw from Compat.Random (libc rand), so determinism
-- tests must seed via Compat.Random.srand. Property checks only —
-- exact expected values encode one platform's rand() sequence
-- (glibc and macOS libc differ, so hardcoded numbers can never pass
-- on both; CI run 27373449876 failed every such check on Ubuntu).

export
tests : List (IO Bool)
tests =
  [ -- uniform var=0.1 stays inside U(-sqrt(0.3), sqrt(0.3))
    do srand 42
       v <- uniform 0.1
       let limit = sqrt 0.3
       check "uniform 0.1 within bounds" (v >= -limit && v <= limit)

  -- the RNG advances: consecutive samples differ
  , do srand 42
       v1 <- uniform 0.1
       v2 <- uniform 0.1
       check "uniform consecutive samples differ" (v1 /= v2)

  -- same seed reproduces the same sample
  , do srand 42
       v1 <- uniform 0.1
       srand 42
       v2 <- uniform 0.1
       checkClose "uniform deterministic under same seed" v1 v2 tol

  -- different seeds give a different sequence
  , do srand 42
       v1 <- uniform 0.1
       srand 1337
       v2 <- uniform 0.1
       check "uniform seed-sensitive" (v1 /= v2)

  -- normalSample is finite and deterministic under a fixed seed
  , do srand 42
       v <- normalSample
       check "normalSample finite" (v == v && v /= (1.0 / 0.0) && v /= (-1.0 / 0.0))

  , do srand 42
       v1 <- normalSample
       srand 42
       v2 <- normalSample
       checkClose "normalSample deterministic under same seed" v1 v2 tol

  -- normal var scales the standard normal draw by sqrt(var)
  , do srand 42
       n <- normalSample
       srand 42
       v <- normal 0.1
       checkClose "normal 0.1 = normalSample * sqrt(0.1)" (n * sqrt 0.1) v tol
  ]
