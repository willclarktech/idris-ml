module Test.Rng

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
  , "env 0.25"
  , "uniform 0.5"
  , ""
  , "env 0.75"
  , "choice 0"
  , "normal -1.5"
  , "uniform 0.125"
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
  ]
