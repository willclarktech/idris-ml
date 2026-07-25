module Test.Dist

import Random.Dist
import Random.Source
import Test.Harness

%default total

collect : (Source -> (a, Source)) -> Source -> Nat -> List a
collect _ _ Z     = []
collect f s (S k) = let (v, s') = f s in v :: collect f s' k

mean : List Double -> Double
mean xs = sum xs / cast (length xs)

countOf : Nat -> List Nat -> Nat
countOf n = foldl (\acc, x => if x == n then S acc else acc) 0

export
tests : List (IO Bool)
tests =
  [ let xs = collect (\s => uniform s (-2.0) 5.0) (Seeded 42) 512
    in check "uniform stays in [lo, hi)" (all (\d => d >= -2.0 && d < 5.0) xs)

  , let xs = collect (\s => uniform s (-1.0) 1.0) (Seeded 7) 4000
    in check "uniform mean near the midpoint" (abs (mean xs) < 0.05)

  , -- Exactness is the point: this is what lets a recording be built from
    -- observed values rather than from the draws behind them.
    let x = 0.37
    in check "uniformInverse inverts uniform"
             (abs (fst (uniform (Recorded [uniformInverse x (-0.05) 0.05]) (-0.05) 0.05) - x)
                < 1.0e-12)

  , check "uniformInverse survives a degenerate range"
          (uniformInverse 1.0 2.0 2.0 == 0.0)

  , let xs = collect (\s => boundedNat s 5) (Seeded 42) 500
    in check "boundedNat stays under the bound" (all (\n => n < 5) xs)

  , check "boundedNat of 0 is 0" (fst (boundedNat (Seeded 1) 0) == 0)

  , -- A draw of exactly 1.0 would scale to the bound itself without the cap.
    check "boundedNat caps a draw at the top of the range"
          (fst (boundedNat (Recorded [1.0]) 4) == 3)

  , let xs = collect normal (Seeded 42) 4000
    in check "normal mean near 0" (abs (mean xs) < 0.06)

  , let xs = collect normal (Seeded 42) 4000
        v  = mean (map (\z => z * z) xs)
    in check "normal variance near 1" (abs (v - 1.0) < 0.1)

  , let xs = collect (\s => normalWith s 3.0 0.5) (Seeded 11) 4000
    in check "normalWith shifts and scales" (abs (mean xs - 3.0) < 0.06)

  , -- log 0 is -inf; one unlucky draw must not poison the result. NaN is the
    -- only Double that fails self-equality, so this is the portable check.
    let z = fst (normal (Recorded [0.0, 0.5]))
    in check "normal survives a zero draw" (z == z)

  , let xs = collect (\s => categorical s [0.25, 0.75]) (Seeded 42) 2000
        p1 = cast {to = Double} (countOf 1 xs) / 2000.0
    in check "categorical respects the weights" (abs (p1 - 0.75) < 0.05)

  , check "categorical: a draw below the first bucket picks index 0"
          (fst (categorical (Recorded [0.1]) [0.25, 0.75]) == 0)

  , check "categorical: a draw above it picks index 1"
          (fst (categorical (Recorded [0.9]) [0.25, 0.75]) == 1)

  , -- Probabilities that fall short of 1 must still be total: the last index
    -- absorbs the remainder rather than running off the end.
    check "categorical absorbs a shortfall in the last bucket"
          (fst (categorical (Recorded [0.99]) [0.1, 0.1, 0.1]) == 2)

  , check "categorical of a certain outcome picks it"
          (fst (categorical (Recorded [0.5]) [0.0, 1.0]) == 1)
  ]
