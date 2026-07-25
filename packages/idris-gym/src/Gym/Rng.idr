||| Gymnasium-style seeded randomness for environments.
|||
||| The generator and the distributions live in `idris-random`; this module is
||| the gym-facing spelling of them, so environment code keeps reading in terms
||| of a `Seed` it threads. Every function here is arithmetically identical to
||| the package's.
module Gym.Rng

import Random.Dist
import Random.Source

-- `Seed` and `nextDouble` come straight through: they mean exactly what they
-- mean in the package, and re-declaring them here would only make the name
-- ambiguous at every use site in this module.
import public Random.SplitMix

||| One SplitMix64 step. Returns (random 64-bit value, next seed). The gym
||| spelling of `Random.SplitMix.next`.
export
splitMix64 : Seed -> (Bits64, Seed)
splitMix64 = SplitMix.next

||| Integer in [0, n). n == 0 returns 0 (caller's responsibility).
|||
||| Reduces the raw 64-bit draw rather than scaling a Double — a different
||| algorithm from `Random.Dist.boundedNat`, and kept because the toy-text envs
||| (Blackjack's deck, FrozenLake's slip) have always drawn this way.
export
nextNat : Seed -> Nat -> (Nat, Seed)
nextNat s Z       = (Z, s)
nextNat s n@(S _) =
  let (r, s')  = SplitMix.next s
      rInt   = cast {to = Integer} r
      result = rInt `mod` cast n
  in (cast {to = Nat} result, s')

||| Standard normal sample N(0,1) via Box-Muller.
export
nextNormal : Seed -> (Double, Seed)
nextNormal s =
  let (u1, s1) = SplitMix.nextDouble s
      (u2, s2) = SplitMix.nextDouble s1
  in (fst (Dist.normal (Recorded [u1, u2])), s2)

||| Uniform Double in [lo, hi).
export
nextUniform : Seed -> Double -> Double -> (Double, Seed)
nextUniform s lo hi =
  let (d, s') = SplitMix.nextDouble s
  in (fst (Dist.uniform (Recorded [d]) lo hi), s')
