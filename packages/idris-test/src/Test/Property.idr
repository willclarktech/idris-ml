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
--
-- For *FFI-bound* invariants — properties whose body must construct
-- real tensors via `IO`-typed smart constructors (`tparam1dConst`
-- etc.) — use `checkPropertyIO`. The Idris-port of Hedgehog hardcodes
-- `PropertyT = TestT Gen` (no `MonadIO` hook in the test body), so a
-- standard `Property` can't host FFI calls without `unsafePerformIO`
-- — which the repo disallows. `checkPropertyIO` re-implements a small
-- check loop over Hedgehog's `Gen` and an `IO Bool` predicate; the
-- generator stays in Hedgehog (familiar combinators) while the test
-- body runs in `IO`. Cost: no integrated shrinking — failures print
-- the raw counterexample via `Show`. Justified for invariants whose
-- inputs are simple shapes / small scalar Vects where shrinking has
-- little to bite into.
module Test.Property

import Data.Cotree
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

----------------------------------------------------------------------
-- checkPropertyIO — IO-bound property runner
----------------------------------------------------------------------
--
-- Why this exists: the Idris-port `Property = MkProperty cfg PropertyT()`
-- wraps `PropertyT = TestT Gen` and the only `Gen` instance has no
-- IO embedding (see `Hedgehog/Internal/Gen.idr` — `Gen a` is
-- `Size -> StdGen -> Cotree a`, purely functional). Calls into
-- IO-typed FFI from inside `Property` would require `unsafePerformIO`.
-- The repo's hard rule (`CLAUDE.md`: "zero `unsafePerformIO`") makes
-- that path unavailable, so we provide a parallel test surface that
-- keeps Hedgehog's `Gen` as the value-generation language but runs
-- the predicate in `IO`.
--
-- Trade-off: no shrinking. The plumbing for that lives in
-- `Hedgehog/Internal/Runner.idr` (`takeSmallest`) and is bound to
-- `Cotree` walks over `TestRes`; reproducing it here in IO would
-- mean re-implementing meaningful runner logic. For the use cases
-- this unblocks (round-trip / norm-bounded invariants over small
-- generated inputs) the raw counterexample is sufficient signal.

||| Run an IO-bound property: sample `Gen a` many times and check
||| `predicate : a -> IO Bool` on each. Returns True iff every
||| generated case passes.
|||
||| Failures print the first counterexample via its `Show` instance.
||| There is NO integrated shrinking — the inputs you generate should
||| be small/structurally simple if you want the counterexample to be
||| useful. For complex inputs (large vectors, deep nested records),
||| extract the salient fields into the generated value type and
||| construct the rest inside the predicate.
|||
||| Default 100 generated cases. Use `checkPropertyIOn` to override.
export
checkPropertyIO : Show a => String -> Gen a -> (a -> IO Bool) -> IO Bool
checkPropertyIO name gen body = do
  seed0 <- initSeed
  ok <- loop 100 seed0
  Test.Harness.check name ok
  where
    -- Hedgehog default size; controls the magnitude of `nat` / `int` /
    -- `double` ranges with `linear`/`exponential` scaling. 30 is the
    -- Hedgehog default for "small but not tiny" inputs.
    sampleSize : Size
    sampleSize = 30

    loop : Nat -> StdGen -> IO Bool
    loop Z     _  = pure True
    loop (S k) se =
      let (s0, s1) = split se
          ct = runGen sampleSize s0 gen
          x  = ct.value
       in do
         pass <- body x
         if pass
            then loop k s1
            else do
              putStrLn $ "  ✗ counterexample: " ++ show x
              pure False

||| Variant of `checkPropertyIO` with a custom run count. Drop to
||| 10-25 when each case drives a slow FFI roundtrip; push to 500+
||| for extra confidence on critical invariants.
export
checkPropertyIOn : Show a => String -> Nat -> Gen a -> (a -> IO Bool) -> IO Bool
checkPropertyIOn name numRuns gen body = do
  seed0 <- initSeed
  ok <- loop numRuns seed0
  Test.Harness.check name ok
  where
    sampleSize : Size
    sampleSize = 30

    loop : Nat -> StdGen -> IO Bool
    loop Z     _  = pure True
    loop (S k) se =
      let (s0, s1) = split se
          ct = runGen sampleSize s0 gen
          x  = ct.value
       in do
         pass <- body x
         if pass
            then loop k s1
            else do
              putStrLn $ "  ✗ counterexample: " ++ show x
              pure False
