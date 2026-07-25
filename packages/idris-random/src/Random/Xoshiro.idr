||| xoshiro256 (Blackman, Vigna 2018) — four words of state, a long period and
||| good equidistribution.
|||
||| The choice when the *stream* itself has to hold up rather than merely being
||| cheap: shuffling, sampling without replacement, anything where a whole
||| permutation depends on the generator's quality. `Random.SplitMix` is the
||| small fast alternative, and is what seeds this one's four words.
|||
||| Two output scramblers share one state update. `**` returns
||| `rotl(s1 * 5, 7) * 9`; `++` returns `rotl(s0 + s3, 23) + s0`. That single
||| line is the whole difference, which makes them very easy to conflate in
||| prose and impossible to substitute in practice — the streams diverge from
||| the first draw. They are spelled out separately here for that reason.
module Random.Xoshiro

import Data.Vect

import Random.SplitMix

%default total

||| Generator state: four 64-bit words.
public export
record Gen where
  constructor MkGen
  words : Vect 4 Bits64

||| Seed all four words from one number, via SplitMix64. The reference
||| implementation recommends exactly this, so a caller supplying a small or
||| low-entropy seed still starts from well-spread state.
export
seed : Bits64 -> Gen
seed s = MkGen (fst (SplitMix.expand s 4))

-- `k` is a rotation amount, so it is `Bits64` rather than `Int` (negatives are
-- meaningless) or `Nat` (the `64 - k` complement would be a Peano walk on
-- every draw — gotchas.md, "`Data.Nat` stdlib functions are recursive at
-- runtime too"). It is also the type the shift primitives take, so no
-- conversion happens at the call site.
rotl : Bits64 -> Bits64 -> Bits64
rotl x k = prim__or_Bits64 (prim__shl_Bits64 x k) (prim__shr_Bits64 x (64 - k))

||| The state update, identical for both scramblers.
export
step : Gen -> Gen
step (MkGen [s0, s1, s2, s3]) =
  let t   = prim__shl_Bits64 s1 17
      s2a = prim__xor_Bits64 s2 s0
      s3a = prim__xor_Bits64 s3 s1
      s1b = prim__xor_Bits64 s1 s2a
      s0b = prim__xor_Bits64 s0 s3a
      s2b = prim__xor_Bits64 s2a t
      s3b = rotl s3a 45
  in MkGen [s0b, s1b, s2b, s3b]

||| xoshiro256**: output `rotl(s1 * 5, 7) * 9`.
export
nextStarStar : Gen -> (Bits64, Gen)
nextStarStar g@(MkGen [_, s1, _, _]) = (rotl (s1 * 5) 7 * 9, step g)

||| xoshiro256++: output `rotl(s0 + s3, 23) + s0`.
export
nextPlusPlus : Gen -> (Bits64, Gen)
nextPlusPlus g@(MkGen [s0, _, _, s3]) = (rotl (s0 + s3) 23 + s0, step g)

||| The `**` scrambler, which is what the shuffle below uses.
export
next : Gen -> (Bits64, Gen)
next = nextStarStar

||| An index in [0, n), by remainder. Slightly biased for n that do not divide
||| 2^64, which is negligible at shuffle-sized bounds and is what the C-side
||| shuffler this serves as reference for also does.
export
boundedNat : Gen -> (n : Nat) -> (Nat, Gen)
boundedNat g Z       = (Z, g)
boundedNat g n@(S _) =
  let (v, g') = next g
      r       = cast {to = Integer} v `mod` cast n
  in (cast {to = Nat} r, g')

setAt : Nat -> a -> List a -> List a
setAt _     _ []        = []
setAt Z     v (_ :: xs) = v :: xs
setAt (S k) v (x :: xs) = x :: setAt k v xs

swapAt : Nat -> Nat -> List a -> List a
swapAt i j xs =
  case (Prelude.getAt i xs, Prelude.getAt j xs) of
    (Just vi, Just vj) => setAt i vj (setAt j vi xs)
    _                  => xs

||| Fisher-Yates: walk from the last index down to 1, swapping each with a
||| uniformly chosen index at or below it.
|||
||| The swap order matters, not just the algorithm's name — a variant that
||| walked upward, or drew its bound differently, would be an equally valid
||| shuffle and an entirely different permutation. This one matches
||| `seeded_index_array_shuffle` in the idris-ml backends, which it is the
||| readable reference for.
export
shuffle : Gen -> List a -> (List a, Gen)
shuffle g xs = go g (pred (length xs)) xs
  where
    go : Gen -> Nat -> List a -> (List a, Gen)
    go g' Z     acc = (acc, g')
    go g' (S k) acc =
      let (j, g'') = boundedNat g' (S (S k))
      in go g'' k (swapAt (S k) j acc)
