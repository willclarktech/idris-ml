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

-- `Source` is what environments thread, so it comes through too.
import public Random.Source

||| One SplitMix64 step. Returns (random 64-bit value, next seed). The gym
||| spelling of `Random.SplitMix.next`.
export
splitMix64 : Seed -> (Bits64, Seed)
splitMix64 = SplitMix.next

||| Integer in [0, n). n == 0 returns 0 (caller's responsibility).
|||
||| The seeded arm reduces the raw 64-bit draw, which is what the toy-text envs
||| (Blackjack's deck, FrozenLake's slip) have always done — changing it would
||| move their streams. A recording has no raw word to reduce, so that arm
||| scales the recorded uniform instead. The two therefore differ by
||| construction, and only the seeded one is a compatibility surface.
export
nextNat : Source -> Nat -> (Nat, Source)
nextNat s Z                = (Z, s)
nextNat (Seeded s) n@(S _) =
  let (r, s')  = SplitMix.next s
      result   = cast {to = Integer} r `mod` cast n
  in (cast {to = Nat} result, Seeded s')
nextNat rec n@(S _) = Dist.boundedNat rec n

||| Standard normal sample N(0,1) via Box-Muller.
export
nextNormal : Source -> (Double, Source)
nextNormal = Dist.normal

||| Uniform Double in [lo, hi).
export
nextUniform : Source -> Double -> Double -> (Double, Source)
nextUniform = Dist.uniform
