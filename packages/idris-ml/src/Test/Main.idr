module Test.Main

import Test.Harness
import Test.Array
import Test.GradMode
import Test.GradScaler
import Test.Math
import Test.Schedule
import Test.Init
import Test.Sampler
import Test.RL.Gae
import Test.RL.ReplayBuffer
import Test.Hpo.LrFinder
import Test.Lossless
import Test.ManagedHandle
import Test.BitLinear
import Test.BitNet
import Test.MixedLayerLike
import Test.Properties.GoldenDemo as Props.GoldenDemo
import Test.Properties.Reshape as Props.Reshape
import Test.Properties.Softmax as Props.Softmax
import Test.RmsNorm
import Test.RoPE
import Test.SwiGLU
-- NOTE: Test.Transfer (UserDeviceTransfer / toDevice smoke) lives in
-- the source tree but isn't wired into this default `tests` list:
-- it deliberately references TapeDev / TorchDev / MlxDev by name to
-- exercise cross-backend hops, so it crashes under any single-backend
-- build. The other buckets above use `{d=TestDevice}` (resolved at
-- build time from the active PRIMARY via the Makefile-generated
-- TestConfig.idr — same trick as BuildConfig for the examples), so
-- `make BACKEND=<b> test` works on every backend. Run the multi-
-- backend cross-transfer suite via `make test-multi`.

main : IO ()
main = runAll
  [ ("Array",           Test.Array.tests)
  , ("GradMode",        Test.GradMode.tests)
  , ("GradScaler",      Test.GradScaler.tests)
  , ("Math",            Test.Math.tests)
  , ("Schedule",        Test.Schedule.tests)
  , ("Init",            Test.Init.tests)
  , ("Sampler",         Test.Sampler.tests)
  , ("RL.Gae",          Test.RL.Gae.tests)
  , ("RL.ReplayBuffer", Test.RL.ReplayBuffer.tests)
  , ("Hpo.LrFinder",    Test.Hpo.LrFinder.tests)
  , ("Lossless",        Test.Lossless.tests)
  , ("ManagedHandle",   Test.ManagedHandle.tests)
  , ("BitLinear",       Test.BitLinear.tests)
  , ("BitNet",          Test.BitNet.tests)
  , ("MixedLayerLike",  Test.MixedLayerLike.tests)
  , ("Properties",      Props.Softmax.tests ++ Props.Reshape.tests ++ Props.GoldenDemo.tests)
  , ("RmsNorm",         Test.RmsNorm.tests)
  , ("RoPE",            Test.RoPE.tests)
  , ("SwiGLU",          Test.SwiGLU.tests)
  ]
