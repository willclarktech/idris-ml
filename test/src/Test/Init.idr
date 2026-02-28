module Test.Init

import System.Random
import Harness
import Init


tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- xavier uniform: same as uniform with var = 2/(10+10) = 0.1
    do srand 42
       v <- xavier uniform 10 10
       checkClose "xavier uniform 10 10" (-0.051447930172099976) v tol

  -- xavier normal
  , do srand 42
       v <- xavier normal 10 10
       checkClose "xavier normal 10 10" (-0.3365664532858412) v tol

  -- he uniform: var = 2/10 = 0.2
  , do srand 42
       v <- he uniform 10 5
       checkClose "he uniform 10 5" (-0.07275836060540775) v tol

  -- he normal: var = 2/10 = 0.2
  , do srand 42
       v <- he normal 10 5
       checkClose "he normal 10 5" (-0.4759768428766474) v tol

  -- lecun uniform: var = 1/10 = 0.1
  , do srand 42
       v <- lecun uniform 10 5
       checkClose "lecun uniform 10 5" (-0.051447930172099976) v tol

  -- lecun normal: var = 1/10 = 0.1
  , do srand 42
       v <- lecun normal 10 5
       checkClose "lecun normal 10 5" (-0.3365664532858412) v tol

  -- fixedRange ignores dimensions
  , do srand 42
       v <- fixedRange 2.0 10 10
       checkClose "fixedRange 2.0" (-0.18786127928139873) v tol
  ]
