module Test.Rng

import Test.Harness
import Gym.Rng

-- Generate n uniform samples starting from a seed.
samples : Seed -> Nat -> List Double
samples _ Z     = []
samples s (S k) =
  let (d, s') = nextDouble s
  in d :: samples s' k

mean : List Double -> Double
mean xs = sum xs / cast (cast {to=Integer} (length xs))

natBounds : Seed -> Nat -> (Nat, Nat) -> (Nat, Nat)
natBounds _ Z acc = acc
natBounds s (S k) (lo, hi) =
  let (n, s') = nextNat s 10
      lo' = if n < lo then n else lo
      hi' = if n > hi then n else hi
  in natBounds s' k (lo', hi')

range : List Double -> (Double, Double)
range [] = (0.0, 0.0)
range (x :: xs) = go x x xs
  where
    go : Double -> Double -> List Double -> (Double, Double)
    go lo hi []        = (lo, hi)
    go lo hi (y :: ys) =
      let lo' = if y < lo then y else lo
          hi' = if y > hi then y else hi
      in go lo' hi' ys

export
tests : List (IO Bool)
tests =
  [ -- Reproducibility: same seed produces same first sample.
    let (d1, _) = nextDouble 42
        (d2, _) = nextDouble 42
    in check "nextDouble reproducible" (d1 == d2)

  , -- First sample is in [0, 1).
    let (d, _) = nextDouble 42
    in check "nextDouble in range" (d >= 0.0 && d < 1.0)

  , -- Successive samples differ.
    let (d1, s1) = nextDouble 42
        (d2, _)  = nextDouble s1
    in check "nextDouble advances" (d1 /= d2)

  , -- 1000 samples: all in [0, 1).
    let xs = samples 123 1000
        (lo, hi) = range xs
    in check "1000 samples in [0,1)" (lo >= 0.0 && hi < 1.0)

  , -- 1000 samples: mean roughly 0.5.
    let xs = samples 123 1000
        m  = mean xs
    in check "uniform mean ~0.5" (abs (m - 0.5) < 0.05)

  , -- nextNat in [0, 10) for all draws.
    let (lo, hi) = natBounds 7 500 (999, 0)
    in check "nextNat in [0, 10)" (hi < 10)

  , -- nextNat 0 returns 0.
    let (n, _) = nextNat 42 0
    in check "nextNat 0 returns 0" (n == 0)

  , -- nextNormal produces a Double (no NaN).
    let (z, _) = nextNormal 42
    in check "nextNormal non-NaN" (z == z)
  ]
