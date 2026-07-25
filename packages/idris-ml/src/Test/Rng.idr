module Test.Rng

import Data.List
import Data.String
import System.File

import Ml.Rng
import Random.Source
import Test.Harness

tol : Double
tol = 0.0

fixture : String
fixture = "/tmp/idris-ml-test-rng.replay"

-- A replay file interleaves channels freely; each channel's draws come
-- back in file order. `#` lines and blank lines are ignored.
fixtureText : String
fixtureText = unlines
  [ "# recorded by a test"
  , "choice 1"
  , "mask 0110"
  , "env 0.25"
  , "uniform 0.5"
  , ""
  , "env 0.75"
  , "choice 0"
  , "normal -1.5"
  , "uniform 0.125"
  , "mask 11"
  , "normal -0.25"
  , "uniform 3.2e-05"
  ]

drawEnv : Source -> (Double, Source)
drawEnv = Random.Source.next

export
tests : List (IO Bool)
tests =
  [ do Right () <- writeFile fixture fixtureText
         | Left err => check ("write " ++ fixture ++ ": " ++ show err) False
       replay <- loadReplay fixture
       -- choice replays recorded decisions, ignoring the probabilities
       c1 <- replay.rng.choice [0.9, 0.1]
       c2 <- replay.rng.choice [0.9, 0.1]
       check "choice channel replays decisions in order" (c1 == 1 && c2 == 0)

  , do replay <- loadReplay fixture
       u1 <- replay.rng.uniform
       u2 <- replay.rng.uniform
       checkClose "uniform channel replays draws in order" (u1 * 1000 + u2) 500.125 tol

  , do replay <- loadReplay fixture
       n1 <- replay.rng.normal
       checkClose "normal channel replays its draw" n1 (-1.5) tol

  , do replay <- loadReplay fixture
       let (e1, s1) = drawEnv replay.envSource
           (e2, s2) = drawEnv s1
           (e3, _)  = drawEnv s2
       checkClose "env channel is a Recorded Source (0.0 past the end)"
                  (e1 * 100 + e2 * 10 + e3) 32.5 tol

  -- natRange shares the recorded decision channel with choice, so a
  -- recording's discrete outcomes replay in one consumption order.
  , do replay <- loadReplay fixture
       n1 <- replay.rng.natRange 0 5
       c1 <- replay.rng.choice [0.9, 0.1]
       check "natRange replays from the choice channel" (n1 == 1 && c1 == 0)

  -- The stdlib parseDouble drops the sign when the integer part is zero
  -- ("-0.25" parses as +0.25 on the pinned toolchain, "-1.5" is fine), so
  -- loadReplay handles the sign itself. A normal draw is the one channel
  -- where negatives are routine.
  , do replay <- loadReplay fixture
       _  <- replay.rng.normal
       n2 <- replay.rng.normal
       checkClose "negative zero-integer-part draw keeps its sign" n2 (-0.25) tol

  -- parseDouble's exponent path can land one ulp off the correctly
  -- rounded double, so exponent-form draws replay to parser rounding, not
  -- bit-for-bit (see loadReplay's doc).
  , do replay <- loadReplay fixture
       _  <- replay.rng.uniform
       _  <- replay.rng.uniform
       u3 <- replay.rng.uniform
       checkClose "exponent-form draw parses" u3 3.2e-5 1.0e-18

  -- Each mask line is one dropout call's whole keep-mask (1 = kept), and
  -- calls consume lines in file order. Exhaustion and a bits/numel length
  -- mismatch are loud crashes (`recordedMasks`), so neither is testable
  -- in-process — same as the other channels' exhaustion.
  , do replay <- loadReplay fixture
       GivenBits b1 <- replay.masks.nextMask 4
         | _ => check "mask channel replays keep-bits in order" False
       GivenBits b2 <- replay.masks.nextMask 2
         | _ => check "mask channel replays keep-bits in order" False
       check "mask channel replays keep-bits in order"
             (b1 == [False, True, True, False] && b2 == [True, True])

  , do FreshSeed a <- liveMasks.nextMask 10
         | _ => check "live masks draw fresh seeds" False
       FreshSeed b <- liveMasks.nextMask 10
         | _ => check "live masks draw fresh seeds" False
       check "live masks draw fresh seeds" (a /= b)

  , do rng <- liveRng
       ns <- sequence (List.replicate 20 (rng.natRange 3 7))
       check "live natRange stays in [lo, hi]" (all (\n => n >= 3 && n <= 7) ns)
  ]
