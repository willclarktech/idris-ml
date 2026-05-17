module Main

import Harness


-- Empty test suite until HfBert lands (Phase 5). The harness builds
-- + runs as a CI sanity check that the package wiring is intact.
main : IO ()
main = runAll
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ]
