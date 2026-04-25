module Test.Sampler

import System.Random
import Harness
import Sampler


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- uniform sampler: two consecutive samples with seed 42
    do srand 42
       v <- uniform 0.1
       checkClose "uniform 0.1 #1" (-0.051447930172099976) v tol

  , do srand 42
       _ <- uniform 0.1
       v <- uniform 0.1
       checkClose "uniform 0.1 #2" (-0.09812017528773276) v tol

  -- normalSample: two consecutive samples with seed 42
  , do srand 42
       v <- normalSample
       checkClose "normalSample #1" (-1.06431657638792) v tol

  , do srand 42
       _ <- normalSample
       v <- normalSample
       checkClose "normalSample #2" 1.464963054474271 v tol

  -- normal sampler: two consecutive samples with seed 42
  , do srand 42
       v <- normal 0.1
       checkClose "normal 0.1 #1" (-0.3365664532858412) v tol

  , do srand 42
       _ <- normal 0.1
       v <- normal 0.1
       checkClose "normal 0.1 #2" 0.463261994013602 v tol
  ]
