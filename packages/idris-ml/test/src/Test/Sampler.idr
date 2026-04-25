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
       checkClose "uniform 0.1 #1" (-0.12761708012258566) v tol

  , do srand 42
       _ <- uniform 0.1
       v <- uniform 0.1
       checkClose "uniform 0.1 #2" 0.36255437163649473 v tol

  -- normalSample: two consecutive samples with seed 42
  , do srand 42
       v <- normalSample
       checkClose "normalSample #1" 2.4491293398339544 v tol

  , do srand 42
       _ <- normalSample
       v <- normalSample
       checkClose "normalSample #2" (-2.3195889837233232) v tol

  -- normal sampler: two consecutive samples with seed 42
  , do srand 42
       v <- normal 0.1
       checkClose "normal 0.1 #1" (-0.6389194414221542) v tol

  , do srand 42
       _ <- normal 0.1
       v <- normal 0.1
       checkClose "normal 0.1 #2" 0.06822387931732488 v tol
  ]
