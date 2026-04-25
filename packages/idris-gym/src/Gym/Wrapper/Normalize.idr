module Gym.Wrapper.Normalize

import Data.Vect


----------------------------------------------------------------------
-- Running statistics
----------------------------------------------------------------------

||| Running mean, variance, and count for a single scalar channel.
||| Uses Welford's online algorithm for numerical stability.
public export
record RunStats where
  constructor MkRunStats
  runMean : Double
  runM2   : Double   -- sum of squared deviations
  runCount : Nat

export
emptyStats : RunStats
emptyStats = MkRunStats 0.0 0.0 Z

||| Update running stats with one new sample (Welford's method).
export
updateStats : RunStats -> Double -> RunStats
updateStats (MkRunStats mean m2 n) x =
  let n' = S n
      nD = cast {to=Double} (natToInteger n')
      delta = x - mean
      mean' = mean + delta / nD
      delta2 = x - mean'
      m2' = m2 + delta * delta2
  in MkRunStats mean' m2' n'

||| Running variance from accumulated stats. Returns 1.0 if count < 2.
export
variance : RunStats -> Double
variance (MkRunStats _ m2 n) =
  case n of
    Z => 1.0
    S Z => 1.0
    S (S _) => m2 / cast {to=Double} (natToInteger n)


----------------------------------------------------------------------
-- NormObs wrapper (vectorized per-dim running stats)
----------------------------------------------------------------------

||| Per-dimension running stats for a Vect n Double observation.
public export
record NormObs (n : Nat) where
  constructor MkNormObs
  stats : Vect n RunStats

export
emptyNormObs : {n : Nat} -> NormObs n
emptyNormObs = MkNormObs (replicate n emptyStats)

||| Normalize an observation and update the running stats.
||| eps prevents division by zero when variance is near zero.
export
normalizeObs : NormObs n -> Vect n Double -> (NormObs n, Vect n Double)
normalizeObs (MkNormObs stats) obs =
  let stats' = zipWith updateStats stats obs
      out = zipWith (\s, x => (x - s.runMean) / (prim__doubleSqrt (variance s) + 1.0e-8))
                    stats' obs
  in (MkNormObs stats', out)


----------------------------------------------------------------------
-- NormReward wrapper (scalar running stats)
----------------------------------------------------------------------

||| Normalize a reward by dividing by the running standard deviation.
||| Does not subtract the mean (matches Gymnasium's NormalizeReward).
export
normalizeReward : RunStats -> Double -> (RunStats, Double)
normalizeReward stats r =
  let stats' = updateStats stats r
      out    = r / (prim__doubleSqrt (variance stats') + 1.0e-8)
  in (stats', out)
