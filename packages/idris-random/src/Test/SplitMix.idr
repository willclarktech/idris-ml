module Test.SplitMix

import Data.List
import Data.Vect

import Random.SplitMix
import Test.Harness

%default total

draws : Seed -> Nat -> List Bits64
draws _ Z     = []
draws s (S k) = let (v, s') = next s in v :: draws s' k

export
tests : List (IO Bool)
tests =
  [ let a = draws 42 8
        b = draws 42 8
    in check "same seed, same stream" (a == b)

  , let a = draws 42 8
        b = draws 43 8
    in check "different seed, different stream" (a /= b)

  , let vs = draws 7 64
    in check "64 draws are distinct" (List.length (List.nub vs) == 64)

  , -- A generator that failed to advance, or advanced by a constant the mix
    -- undid, would repeat. Distinctness above catches that; this pins that
    -- successive states differ too.
    let (_, s1) = next 0
        (_, s2) = next s1
    in check "state advances" (s1 /= s2 && s1 /= 0)

  , -- `expand` is what seeds wider generators, so its words must not collide.
    let (ws, _) = expand 12345 4
    in check "expand gives 4 distinct words" (List.length (List.nub (toList ws)) == 4)

  , let (ws, _)  = expand 99 4
        direct   = draws 99 4
    in check "expand agrees with repeated next" (toList ws == direct)
  ]
