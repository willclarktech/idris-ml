||| SplitMix64 (Steele, Lea, Flood 2014) — a fast, small-state generator.
|||
||| One 64-bit word of state advanced by a fixed increment, then run through a
||| finalizing mix. Cheap, and adequate wherever the consumer is not itself
||| statistically demanding: seeding another generator, drawing a handful of
||| values per call, per-element masks. `Random.Xoshiro` is the choice when the
||| stream itself has to hold up.
module Random.SplitMix

import Data.Vect

%default total

||| Generator state. One word, so it is cheap to carry and to store.
public export
Seed : Type
Seed = Bits64

-- Golden-gamma constant (the golden ratio scaled to 2^64).
gammaK : Bits64
gammaK = 0x9E3779B97F4A7C15

-- MurmurHash3 / Stafford Mix-14 multipliers.
mix1K : Bits64
mix1K = 0xBF58476D1CE4E5B9

mix2K : Bits64
mix2K = 0x94D049BB133111EB

||| One step: the next 64-bit value and the advanced seed.
export
next : Seed -> (Bits64, Seed)
next s =
  let s' = s + gammaK
      z0 = s'
      z1 = (prim__xor_Bits64 z0 (prim__shr_Bits64 z0 30)) * mix1K
      z2 = (prim__xor_Bits64 z1 (prim__shr_Bits64 z1 27)) * mix2K
      z3 = prim__xor_Bits64 z2 (prim__shr_Bits64 z2 31)
  in (z3, s')

||| A uniform Double in [0, 1), and the advanced seed.
|||
||| Takes the top 53 bits — exactly a Double's mantissa width — so every
||| representable value in the interval is reachable and none is favoured.
export
nextDouble : Seed -> (Double, Seed)
nextDouble s =
  let (r, s') = next s
      top53   = prim__shr_Bits64 r 11
      -- 2^53 = 9007199254740992
      d       = cast {to = Double} (cast {to = Integer} top53) / 9007199254740992.0
  in (d, s')

||| Expand one seed into `n` words. Used to fill a wider generator's state —
||| `Random.Xoshiro` seeds its four words this way, as the reference
||| implementation recommends, so a caller only ever supplies one number.
export
expand : Seed -> (n : Nat) -> (Vect n Bits64, Seed)
expand s Z     = ([], s)
expand s (S k) =
  let (v, s')   = next s
      (vs, s'') = expand s' k
  in (v :: vs, s'')
