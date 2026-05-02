module Main

import Harness
import Test.Array
import Test.GradMode
import Test.Math
import Test.Schedule
import Test.Init
import Test.Sampler
import Test.RL.Gae
import Test.RL.ReplayBuffer
import Test.Hpo.LrFinder
import Test.ManagedHandle
-- NOTE: Test.Transfer (UserDeviceTransfer / toDevice smoke) lives in
-- the source tree but isn't wired into this default `tests` list:
-- the test file is hardcoded to one backend (TapeDev), so linking
-- under any other BACKEND fails at FFI resolution (no
-- `tensor_to_device_tape` symbol in a torch/mlx-only dylib). The
-- other test buckets above route through unified-name C symbols
-- (aliased to the build's primary at link time) so they avoid this.
-- To run the Transfer smoke specifically, use a tape build.
-- Multi-backend cross-transfer tests remain a parked TODO.

main : IO ()
main = runAll
  [ ("Array",           Test.Array.tests)
  , ("GradMode",        Test.GradMode.tests)
  , ("Math",            Test.Math.tests)
  , ("Schedule",        Test.Schedule.tests)
  , ("Init",            Test.Init.tests)
  , ("Sampler",         Test.Sampler.tests)
  , ("RL.Gae",          Test.RL.Gae.tests)
  , ("RL.ReplayBuffer", Test.RL.ReplayBuffer.tests)
  , ("Hpo.LrFinder",    Test.Hpo.LrFinder.tests)
  , ("ManagedHandle",   Test.ManagedHandle.tests)
  ]
