-- Test.Properties.Reshape — numel is preserved across valid reshapes.
-- Pure-Idris arithmetic invariant on Vect-of-Nat shape values. No FFI,
-- no tensor allocation.
module Test.Properties.Reshape

import Test.Property
import Data.Vect

%default total

-- A reshape is valid iff product source-dims == product target-dims.
-- Generate two shapes [a, b, c] and [a*b, c] (a, b, c >= 1) — covers
-- the canonical rank-3 -> rank-2 reshape that motivates the invariant.
prop_reshape_preserves_numel : Property
prop_reshape_preserves_numel = property $ do
  a <- forAll $ nat (constant 1 20)
  b <- forAll $ nat (constant 1 20)
  c <- forAll $ nat (constant 1 20)
  let shape3 : Vect 3 Nat = [a, b, c]
      shape2 : Vect 2 Nat = [a * b, c]
  product shape3 === product shape2

export
tests : List (IO Bool)
tests =
  [ checkProperty "reshape_preserves_numel" prop_reshape_preserves_numel
  ]
