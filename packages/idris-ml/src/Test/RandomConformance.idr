||| The Idris generators in `idris-random` and the C copies in
||| `packages/backends/shared_utils.c` must agree bit for bit.
|||
||| Neither copy can be deleted. The C side shuffles a raw `int*` in a tight
||| loop and draws the dropout mask seed, so it cannot call Idris; the Idris
||| side is what a caller threading a `Source` gets. So the duplication is
||| permanent, and the point of this suite is to make it *gated* rather than
||| merely tolerated — the Idris implementations act as the executable
||| specification the C ones are held to.
|||
||| Lives here rather than in idris-random's own suite because it needs the
||| dylib, and that package must not depend on the backends.
module Test.RandomConformance

import Data.List

import Random.SplitMix
import Random.Xoshiro

import Test.Harness

%foreign "C:idrisml_srand,libidrisml"
prim__srand : Bits64 -> PrimIO ()

%foreign "C:idrisml_rand64,libidrisml"
prim__rand64 : PrimIO Bits64

%foreign "C:create_seeded_index_array,libidrisml"
prim__createSeededIndexArray : Int -> Bits64 -> PrimIO AnyPtr

%foreign "C:seeded_index_array_shuffle,libidrisml"
prim__seededIndexArrayShuffle : AnyPtr -> PrimIO AnyPtr

%foreign "C:seeded_index_array_get,libidrisml"
prim__seededIndexArrayGet : AnyPtr -> Int -> PrimIO Int

-- `idrisml_rand64` advances process-global state, so the draws must be
-- sequenced rather than mapped over a pure list.
cDraws : Nat -> IO (List Bits64)
cDraws Z     = pure []
cDraws (S k) = do
  v  <- primIO prim__rand64
  vs <- cDraws k
  pure (v :: vs)

idrisDraws : SplitMix.Seed -> Nat -> List Bits64
idrisDraws _ Z     = []
idrisDraws s (S k) = let (v, s') = SplitMix.next s in v :: idrisDraws s' k

-- The C shuffler's permutation of [0 .. n-1] for a given seed.
cPermutation : Nat -> Bits64 -> IO (List Nat)
cPermutation n s = do
  h  <- primIO (prim__createSeededIndexArray (cast n) s)
  h' <- primIO (prim__seededIndexArrayShuffle h)
  readBack h' 0
  where
    readBack : AnyPtr -> Nat -> IO (List Nat)
    readBack ptr i =
      if i >= n
        then pure []
        else do v  <- primIO (prim__seededIndexArrayGet ptr (cast i))
                vs <- readBack ptr (S i)
                pure (cast v :: vs)

splitMixMatches : Nat -> Bits64 -> IO Bool
splitMixMatches n s = do
  primIO (prim__srand s)
  c <- cDraws n
  let i = idrisDraws s n
  check ("SplitMix64 matches C at seed " ++ show s) (c == i)

shuffleMatches : Nat -> Bits64 -> IO Bool
shuffleMatches n s = do
  c <- cPermutation n s
  let (i, _) = Xoshiro.shuffle (Xoshiro.seed s) [0 .. n `minus` 1]
  check ("xoshiro256** shuffle matches C at seed " ++ show s ++ ", n=" ++ show n)
        (c == i)

export
tests : List (IO Bool)
tests =
  [ -- Several seeds, including the degenerate one: SplitMix64's increment
    -- means seed 0 is unremarkable, and a copy that special-cased it would
    -- diverge here.
    splitMixMatches 16 0
  , splitMixMatches 16 1
  , splitMixMatches 16 42
  , splitMixMatches 64 987654321

  , -- The shuffle exercises xoshiro's stream, its seeding from SplitMix64,
    -- and the Fisher-Yates swap order all at once. A copy that walked the
    -- array the other way would still shuffle, and would fail here.
    --
    -- This is also what caught the C shuffler being xoshiro256** while three
    -- comments called it ++: the two share a state update and differ only in
    -- the output scrambler, so nothing but a stream comparison notices.
    shuffleMatches 8 42
  , shuffleMatches 32 7
  , shuffleMatches 64 2026

  , -- n = 1 and n = 2 pin the loop bounds: the C side runs `i` from n-1 down
    -- to 1, so n = 1 must perform no swap at all.
    shuffleMatches 1 5
  , shuffleMatches 2 5
  ]
