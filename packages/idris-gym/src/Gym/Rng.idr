module Gym.Rng


----------------------------------------------------------------------
-- Seed
----------------------------------------------------------------------

||| Pure PRNG state. Thread through step to keep env evolution deterministic.
public export
Seed : Type
Seed = Bits64


----------------------------------------------------------------------
-- SplitMix64 (Steele, Lea, Flood 2014)
----------------------------------------------------------------------

-- Golden-gamma constant (golden ratio * 2^64).
gammaK : Bits64
gammaK = 0x9E3779B97F4A7C15

-- MurmurHash3 / Stafford Mix-14 multipliers.
mix1K : Bits64
mix1K = 0xBF58476D1CE4E5B9

mix2K : Bits64
mix2K = 0x94D049BB133111EB

||| One SplitMix64 step. Returns (random 64-bit value, next seed).
export
splitMix64 : Seed -> (Bits64, Seed)
splitMix64 s =
  let s' = s + gammaK
      z0 = s'
      z1 = (prim__xor_Bits64 z0 (prim__shr_Bits64 z0 30)) * mix1K
      z2 = (prim__xor_Bits64 z1 (prim__shr_Bits64 z1 27)) * mix2K
      z3 = prim__xor_Bits64 z2 (prim__shr_Bits64 z2 31)
  in (z3, s')


----------------------------------------------------------------------
-- Derived distributions
----------------------------------------------------------------------

||| Uniform Double in [0, 1). Uses top 53 bits (full Double mantissa).
export
nextDouble : Seed -> (Double, Seed)
nextDouble s =
  let (r, s') = splitMix64 s
      top53   = prim__shr_Bits64 r 11
      -- 2^53 = 9007199254740992
      d       = cast {to=Double} (cast {to=Integer} top53) / 9007199254740992.0
  in (d, s')

||| Integer in [0, n). n == 0 returns 0 (caller's responsibility).
export
nextNat : Seed -> Nat -> (Nat, Seed)
nextNat s Z     = (Z, s)
nextNat s n@(S _) =
  let (r, s') = splitMix64 s
      rInt    = cast {to=Integer} r
      result  = rInt `mod` cast n
  in (cast {to=Nat} result, s')

||| Standard normal sample N(0,1) via Box-Muller.
export
nextNormal : Seed -> (Double, Seed)
nextNormal s =
  let (u1raw, s1) = nextDouble s
      (u2, s2)    = nextDouble s1
      u1          = if u1raw < 1.0e-10 then 1.0e-10 else u1raw
      z           = prim__doubleSqrt (-2.0 * prim__doubleLog u1)
                  * prim__doubleCos (2.0 * 3.141592653589793 * u2)
  in (z, s2)

||| Uniform Double in [lo, hi).
export
nextUniform : Seed -> Double -> Double -> (Double, Seed)
nextUniform s lo hi =
  let (d, s') = nextDouble s
  in (lo + d * (hi - lo), s')
