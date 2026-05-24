-- Test.Properties.Softmax — first Hedgehog property in the codebase.
module Test.Properties.Softmax

import Test.Property

%default total

softmax : List Double -> List Double
softmax xs =
  case xs of
    [] => []
    (x :: rest) =>
      let m  = foldl max x rest
          ys = map (\v => exp (v - m)) xs
          s  = sum ys
      in map (\y => y / s) ys

prop_softmax_sum_one : Property
prop_softmax_sum_one = property $ do
  xs <- forAll $ list (constant 1 100) (double $ linearFracFrom 0.0 (-50.0) 50.0)
  diff (abs (sum (softmax xs) - 1.0)) (<) 1.0e-9

export
tests : List (IO Bool)
tests =
  [ checkProperty "softmax_sum_one" prop_softmax_sum_one
  ]
