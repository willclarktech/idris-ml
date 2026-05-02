||| Multi-backend test entry point.
|||
||| This binary's `Main.main` is everything in `Main.idr`'s default
||| `tests` list PLUS the `Test.Transfer` suite, which exercises
||| cross-backend `toDevice` hops and so needs tape, torch, *and*
||| mlx C symbols linked at runtime.
|||
||| Build via:
|||
|||     make BACKEND=torch,tape,mlx test-multi
|||
||| The `make test` target builds against `Main.idr` (single-backend
||| safe) and excludes Transfer — running this multi-backend Main on
||| a single-backend dylib crashes at FFI resolution on the first
||| cross-backend hop.
module MainMulti

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
import Test.Transfer

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
  , ("Transfer",        Test.Transfer.tests)
  ]
