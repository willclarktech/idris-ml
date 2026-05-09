module Main

import Harness
import Test.Array
import Test.Math
import Test.Schedule
import Test.Init
import Test.Sampler
import Test.RL.Gae
import Test.RL.ReplayBuffer
import Test.Hpo.LrFinder

main : IO ()
main = runAll
  [ ("Array",           Test.Array.tests)
  , ("Math",            Test.Math.tests)
  , ("Schedule",        Test.Schedule.tests)
  , ("Init",            Test.Init.tests)
  , ("Sampler",         Test.Sampler.tests)
  , ("RL.Gae",          Test.RL.Gae.tests)
  , ("RL.ReplayBuffer", Test.RL.ReplayBuffer.tests)
  , ("Hpo.LrFinder",    Test.Hpo.LrFinder.tests)
  ]
