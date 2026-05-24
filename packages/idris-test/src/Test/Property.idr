-- Test.Property — Hedgehog property-based test adapter.
--
-- Lifts a `Property` into the `IO Bool` shape `Test.Harness.runSuite`
-- consumes, so per-test files can mix unit assertions and properties:
--
--   tests : List (IO Bool)
--   tests =
--     [ check "shape preserved" (length xs == length ys)
--     , checkProperty "softmax_sum_one" prop_softmax_sum_one
--     ]
--
-- `checkProperty` runs the property at Hedgehog's default test count
-- (100); failures print the shrinking trace via Hedgehog's Report
-- module before returning `False`. PASS/FAIL formatting matches the
-- `check` helper so output stays uniform across suites.
--
-- For per-call test-count tuning use `checkPropertyN` (e.g. drop to
-- 10-25 when each generated case drives a slow FFI roundtrip, or
-- push to 500+ for extra confidence on critical invariants).
module Test.Property

import public Hedgehog
import Test.Harness

%default total

||| Run a single property at the default config. Returns True iff
||| Hedgehog reports OK across all generated cases.
export
checkProperty : String -> Property -> IO Bool
checkProperty name prop = do
  ok <- check prop
  Test.Harness.check name ok

||| Variant that accepts a custom Hedgehog test count.
export
checkPropertyN : String -> TestLimit -> Property -> IO Bool
checkPropertyN name n prop = do
  let prop' = withTests n prop
  ok <- check prop'
  Test.Harness.check name ok
