module Test.Log

import Ml.Util.Log
import Test.Harness

-- Smoke checks for the `Util.Log` FFI surface — verifies that the
-- new `getLogLevel` binding to `idrisml_log_resolve_level` links
-- and returns a value in the documented range.
--
-- Note: `idrisml_log_resolve_level` reads the env once and caches,
-- so a single test process can't observe both INFO and TRACE.
-- These tests assert the FFI's contract (valid level returned,
-- constants distinct), not specific level values. The
-- "TRACE-on changes activation-dump behavior" gate lives in
-- Test.ActivationDump.

getLogLevelInRange : IO Bool
getLogLevelInRange = do
  lvl <- getLogLevel
  check ("getLogLevel returns valid level (" ++ show lvl ++ ")")
        (lvl >= levelSilent && lvl <= levelTrace)

levelConstantsDistinct : IO Bool
levelConstantsDistinct =
  check "level constants are strictly ordered"
        (  levelSilent < levelError
        && levelError  < levelWarn
        && levelWarn   < levelInfo
        && levelInfo   < levelDebug
        && levelDebug  < levelTrace)

export
tests : List (IO Bool)
tests =
  [ getLogLevelInRange
  , levelConstantsDistinct
  ]
