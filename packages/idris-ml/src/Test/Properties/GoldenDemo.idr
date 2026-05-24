-- Test.Properties.GoldenDemo — first checkGolden user.
--
-- Demonstrates the in-Idris golden-file primitive on a deterministic
-- output (the literal string "demo: 42"). Establishes the workflow:
--
-- 1. Write the test with `checkGolden "<name>" "fixtures/<file>" action`.
-- 2. Run once with `GOLDEN_UPDATE=1 make test-unit-idris-ml` to write
--    the initial fixture (in update mode the test always passes).
-- 3. Commit the fixture alongside the test.
-- 4. Subsequent runs compare the action's output against the fixture
--    and FAIL on drift.
--
-- Useful for: BENCHMARKS.md rendering, CLI help-text, schema dumps —
-- any byte-deterministic stdout where a verbatim comparison is the
-- right test surface (rather than a numerical-tolerance threshold).
-- The existing `.expect` harness handles RESULT-line threshold
-- checks; this is the complement.
module Test.Properties.GoldenDemo

import Test.Property.Golden

-- The "action" under test. In a real scenario this would invoke
-- something whose stdout we want to pin (renderBenchmarks, a CLI
-- --help-text generator, a schema dump). Here it's a fixed string
-- to keep the demo self-contained and fast.
demoAction : IO String
demoAction = pure "demo: 42\n"

export
tests : List (IO Bool)
tests =
  [ checkGolden "golden_demo_string"
                "packages/idris-ml/src/Test/Properties/fixtures/golden_demo.txt"
                demoAction
  ]
