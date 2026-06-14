module Test.Space

import Data.Vect

import Gym.Space
import Test.Harness

export
tests : List (IO Bool)
tests =
  [ check "spaceSize Discrete 5" (spaceSize (Discrete 5) == Just 5)
  , check "spaceSize Box" (spaceSize (Box [0.0] [1.0]) == Nothing)
  , check "spaceSize MultiBin 3" (spaceSize (MultiBin 3) == Just 8)
  , check "spaceSize MultiDisc" (spaceSize (MultiDisc [2,3,4]) == Nothing)

  , check "containsNat in range" (containsNat (Discrete 5) 3)
  , check "containsNat out of range" (not (containsNat (Discrete 5) 5))
  , check "containsNat on Box" (not (containsNat (Box [0.0] [1.0]) 0))

  , check "containsBox in range"
      (containsBox (Box [0.0, -1.0] [1.0, 1.0]) (the (Vect 2 Double) [0.5, 0.0]))
  , check "containsBox out of range"
      (not (containsBox (Box [0.0, -1.0] [1.0, 1.0]) (the (Vect 2 Double) [1.5, 0.0])))
  , check "containsBox wrong dim"
      (not (containsBox (Box [0.0, -1.0] [1.0, 1.0]) (the (Vect 1 Double) [0.5])))

  , check "spaceShape Discrete" (spaceShape (Discrete 5) == [])
  , check "spaceShape Box" (spaceShape (Box (the (Vect 2 Double) [0.0, 0.0]) [1.0, 1.0]) == [2])
  ]
