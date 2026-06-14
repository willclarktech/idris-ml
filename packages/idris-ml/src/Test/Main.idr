module Test.Main

import Test.Harness
import Test.Array
import Test.Backend
import Test.Construct
import Test.Data
import Test.Fit
import Test.GradMode
import Test.GradScaler
import Test.Math
import Test.Nn.Activation
import Test.Nn.BatchNorm
import Test.Nn.Conv
import Test.Nn.Dropout
import Test.Nn.Embedding
import Test.Nn.Equivalence
import Test.Nn.Freeze
import Test.Nn.Group
import Test.Nn.Init
import Test.Nn.LayerNorm
import Test.Nn.Linear
import Test.Nn.LoraLinear
import Test.Nn.Module
import Test.Nn.Residual
import Test.Nn.RmsNorm
import Test.Nn.Seq
import Test.Nn.SwiGLU
import Test.Schedule
import Test.Init
import Test.Sampler
import Test.TensorExpr
import Test.TrainEngine
import Test.RL.Gae
import Test.RL.ReplayBuffer
import Test.Hpo.LrFinder
import Test.ActivationDump
import Test.LoadOpts
import Test.Log
import Test.Lossless
import Test.LoraLinear
import Test.ManagedHandle
import Test.BitLinear
import Test.BitNet
import Test.CheckpointSubset
import Test.MixedLayerLike
import Test.Optimizer
import Test.Properties.F32GradParity as Props.F32GradParity
import Test.Properties.GoldenDemo as Props.GoldenDemo
import Test.Properties.Reshape as Props.Reshape
import Test.Properties.RmsNorm as Props.RmsNorm
import Test.Properties.RoPE as Props.RoPE
import Test.Properties.Softmax as Props.Softmax
import Test.RmsNorm
import Test.RoPE
import Test.SaveModelMatching
import Test.SwiGLU
-- NOTE: Test.Transfer (UserExecutorTransfer / toExecutor smoke) lives in
-- the source tree but isn't wired into this default `tests` list:
-- it deliberately references TapeExecutor / TorchExecutor / MlxExecutor by name to
-- exercise cross-backend hops, so it crashes under any single-backend
-- build. The other buckets above use `{ex=TestExecutor}` (resolved at
-- build time from the active PRIMARY via the Makefile-generated
-- TestConfig.idr — same trick as BuildConfig for the examples), so
-- `make BACKEND=<b> test` works on every backend. Run the multi-
-- backend cross-transfer suite via `make test-multi`.

main : IO ()
main = runAll
  [ ("Array",           Test.Array.tests)
  , ("Backend",         Test.Backend.tests)
  , ("Construct",       Test.Construct.tests)
  , ("Data",            Test.Data.tests)
  , ("Fit",             Test.Fit.tests)
  , ("GradMode",        Test.GradMode.tests)
  , ("GradScaler",      Test.GradScaler.tests)
  , ("Math",            Test.Math.tests)
  , ("Nn.Activation",   Test.Nn.Activation.tests)
  , ("Nn.BatchNorm",    Test.Nn.BatchNorm.tests)
  , ("Nn.Conv",         Test.Nn.Conv.tests)
  , ("Nn.Dropout",      Test.Nn.Dropout.tests)
  , ("Nn.Embedding",    Test.Nn.Embedding.tests)
  , ("Nn.Equivalence",  Test.Nn.Equivalence.tests)
  , ("Nn.Freeze",       Test.Nn.Freeze.tests)
  , ("Nn.Group",        Test.Nn.Group.tests)
  , ("Nn.Init",         Test.Nn.Init.tests)
  , ("Nn.LayerNorm",    Test.Nn.LayerNorm.tests)
  , ("Nn.Linear",       Test.Nn.Linear.tests)
  , ("Nn.LoraLinear",   Test.Nn.LoraLinear.tests)
  , ("Nn.Module",       Test.Nn.Module.tests)
  , ("Nn.Residual",     Test.Nn.Residual.tests)
  , ("Nn.RmsNorm",      Test.Nn.RmsNorm.tests)
  , ("Nn.Seq",          Test.Nn.Seq.tests)
  , ("Nn.SwiGLU",       Test.Nn.SwiGLU.tests)
  , ("Schedule",        Test.Schedule.tests)
  , ("TensorExpr",      Test.TensorExpr.tests)
  , ("TrainEngine",     Test.TrainEngine.tests)
  , ("Init",            Test.Init.tests)
  , ("Sampler",         Test.Sampler.tests)
  , ("RL.Gae",          Test.RL.Gae.tests)
  , ("RL.ReplayBuffer", Test.RL.ReplayBuffer.tests)
  , ("ActivationDump",  Test.ActivationDump.tests)
  , ("Hpo.LrFinder",    Test.Hpo.LrFinder.tests)
  , ("LoadOpts",        Test.LoadOpts.tests)
  , ("Log",             Test.Log.tests)
  , ("Lossless",        Test.Lossless.tests)
  , ("LoraLinear",      Test.LoraLinear.tests)
  , ("ManagedHandle",   Test.ManagedHandle.tests)
  , ("BitLinear",       Test.BitLinear.tests)
  , ("BitNet",          Test.BitNet.tests)
  , ("CheckpointSubset", Test.CheckpointSubset.tests)
  , ("MixedLayerLike",  Test.MixedLayerLike.tests)
  , ("Optimizer",       Test.Optimizer.tests)
  , ("Properties",      Props.Softmax.tests ++ Props.Reshape.tests ++ Props.GoldenDemo.tests
                                            ++ Props.RmsNorm.tests ++ Props.RoPE.tests
                                            ++ Props.F32GradParity.tests)
  , ("RmsNorm",         Test.RmsNorm.tests)
  , ("RoPE",            Test.RoPE.tests)
  , ("SaveModelMatching", Test.SaveModelMatching.tests)
  , ("SwiGLU",          Test.SwiGLU.tests)
  ]
