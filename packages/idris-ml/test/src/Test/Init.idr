module Test.Init

import System.Random
import Test.Harness
import Init


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- xavier uniform: same as uniform with var = 2/(10+10) = 0.1
    do srand 42
       v <- xavier uniform 10 10
       checkClose "xavier uniform 10 10" (-0.5477139841471926) v tol

  -- xavier normal
  , do srand 42
       v <- xavier normal 10 10
       checkClose "xavier normal 10 10" 0.022427920237363546 v tol

  -- he uniform: var = 2/10 = 0.2
  , do srand 42
       v <- he uniform 10 5
       checkClose "he uniform 10 5" (-0.06405894017134184) v tol

  -- he normal: var = 2/10 = 0.2
  , do srand 42
       v <- he normal 10 5
       checkClose "he normal 10 5" 0.0972619232531175 v tol

  -- lecun uniform: var = 1/10 = 0.1
  , do srand 42
       v <- lecun uniform 10 5
       checkClose "lecun uniform 10 5" (-0.4961877624854715) v tol

  -- lecun normal: var = 1/10 = 0.1
  , do srand 42
       v <- lecun normal 10 5
       checkClose "lecun normal 10 5" (-0.11961862807950828) v tol

  -- fixedRange ignores dimensions
  , do srand 42
       v <- fixedRange 2.0 10 10
       checkClose "fixedRange 2.0" 1.7387715837633104 v tol
  ]
