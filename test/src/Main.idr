module Main

import Harness
import Test.Tensor
import Test.Math
import Test.Variable
import Test.Memory
import Test.Optimizer
import Test.Schedule
import Test.Init
import Test.Sampler

main : IO ()
main = runAll
  [ ("Tensor",    Test.Tensor.tests)
  , ("Math",      Test.Math.tests)
  , ("Variable",  Test.Variable.tests)
  , ("Memory",    Test.Memory.tests)
  , ("Optimizer", Test.Optimizer.tests)
  , ("Schedule",  Test.Schedule.tests)
  , ("Init",      Test.Init.tests)
  , ("Sampler",   Test.Sampler.tests)
  ]
