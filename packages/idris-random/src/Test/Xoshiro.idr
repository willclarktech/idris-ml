module Test.Xoshiro

import Data.List
import Data.Vect

import Random.Xoshiro
import Test.Harness

%default total

draws : Gen -> Nat -> List Bits64
draws _ Z     = []
draws g (S k) = let (v, g') = next g in v :: draws g' k

export
tests : List (IO Bool)
tests =
  [ let a = draws (seed 42) 8
        b = draws (seed 42) 8
    in check "same seed, same stream" (a == b)

  , let a = draws (seed 42) 8
        b = draws (seed 43) 8
    in check "different seed, different stream" (a /= b)

  , let vs = draws (seed 7) 64
    in check "64 draws are distinct" (length (nub vs) == 64)

  , -- The two scramblers share a state update but must not agree on output:
    -- conflating them is the easy mistake, and it is silent.
    let (a, _) = nextStarStar (seed 42)
        (b, _) = nextPlusPlus (seed 42)
    in check "** and ++ differ from the first draw" (a /= b)

  , let ga = snd (nextStarStar (seed 42))
        gb = snd (nextPlusPlus (seed 42))
    in check "** and ++ share the state update" (words ga == words gb)

  , -- All four state words must be seeded, not just the first. A generator
    -- left with three zero words still produces output, just a far worse
    -- stream, and nothing above would notice.
    check "seeding fills all four words"
          (length (nub (toList (words (seed 1)))) == 4)

  , let (xs, _) = shuffle (seed 42) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
    in check "shuffle is a permutation" (sort xs == [0, 1, 2, 3, 4, 5, 6, 7])

  , let (a, _) = shuffle (seed 42) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
        (b, _) = shuffle (seed 42) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
    in check "shuffle is reproducible" (a == b)

  , let (a, _) = shuffle (seed 42) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
        (b, _) = shuffle (seed 43) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
    in check "shuffle depends on the seed" (a /= b)

  , let (xs, _) = shuffle (seed 9) [the Nat 0, 1, 2, 3, 4, 5, 6, 7]
    in check "shuffle actually moves elements" (xs /= [0, 1, 2, 3, 4, 5, 6, 7])

  , -- Degenerate sizes: the loop runs from `pred (length xs)` down, so both
    -- must terminate without touching anything.
    let (e, _) = shuffle (seed 1) (the (List Nat) [])
        (o, _) = shuffle (seed 1) [the Nat 42]
    in check "shuffle handles empty and singleton" (e == [] && o == [42])
  ]
