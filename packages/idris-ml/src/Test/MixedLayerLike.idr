module Test.MixedLayerLike

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Array
import Backprop
import DataPoint
import GradScaler
import Layer
import Layer.MixedCore
import Test.Config


-- ---------------------------------------------------------------
-- A0: LayerLikeMixed bridge + NetworkMixed
-- ---------------------------------------------------------------
--
-- These tests exercise the auto-conformance from `LayerLike l` to
-- `LayerLikeMixed AsMixed`: any existing single-dtype layer should
-- slot into `AnyLayerMixed` / `NetworkMixed` via `liftAnyLayer` /
-- `liftNetwork` without code changes elsewhere, and forward through
-- the bridged pipeline should produce the same numerics as the
-- direct `LayerLike` forward.


-- Build a [n] input tensor from a Vect of Doubles. Pattern lifted
-- from Test.RmsNorm / Test.SwiGLU.
mkInput : {n : Nat} -> Vect n Double -> Tensor [n] TestExecutor TestDType WithGrad
mkInput xs =
  let raw = bulkToTensor {ex=TestExecutor} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw


-- A2: LinearMixed constructs end-to-end and runs through the
-- mixed-precision pipeline. Uses paramDt = computeDt = TestDType so
-- the cast inside applyVarMixed is a no-op at the dtype level — the
-- test verifies the layer machinery composes, the lossy-cast variants
-- are covered by C-level cast_grad_propagation (A1).
linearMixedForwardTypechecks : IO Bool
linearMixedForwardTypechecks = do
  lin <- mixedLinearLayerAny {ex=TestExecutor} {paramDt=TestDType} {computeDt=TestDType}
                             {i=4} {o=3} "lin_mixed_test"
  let netM : NetworkMixed 4 [] 3 TestExecutor TestDType TestDType WithGrad
      netM = OutputLayerMixed lin
  let input = mkInput (the (Vect 4 Double) [0.5, -1.0, 0.0, 1.0])
  (_, _) <- forwardVarMixed netM input
  check "mixedLinearLayer + forwardVarMixed compose end-to-end" True


-- Parameter-free single-tanh wrapped via AsMixed lifts cleanly and
-- runs through `forwardVarMixed`. `tanhLayerAny` is a parameter-free
-- Activation layer so this test doesn't perturb global PRNG state
-- (matches the pattern in Test.GradMode.freezeUnfreezeRoundTrip).
bridgeForwardTypechecks : IO Bool
bridgeForwardTypechecks = do
  let net : Network 4 [] 4 TestExecutor TestDType WithGrad
      net = OutputLayer (the (AnyLayer 4 4 TestExecutor TestDType WithGrad) tanhLayerAny)
  let netM : NetworkMixed 4 [] 4 TestExecutor TestDType TestDType WithGrad
      netM = liftNetwork net
  let input = mkInput (the (Vect 4 Double) [0.5, -1.0, 0.0, 1.0])
  (_, _) <- forwardVarMixed netM input
  check "liftNetwork + forwardVarMixed compose end-to-end" True


-- Freeze/unfreeze round-trip on a NetworkMixed: walks the layer
-- chain calling each lifted layer's `freezeLayerMixed` /
-- `unfreezeLayerMixed`, which delegate to the underlying
-- `LayerLike` methods via the bridge.
bridgeFreezeUnfreezeRoundTrip : IO Bool
bridgeFreezeUnfreezeRoundTrip = do
  let net : Network 4 [] 4 TestExecutor TestDType WithGrad
      net = OutputLayer (the (AnyLayer 4 4 TestExecutor TestDType WithGrad) tanhLayerAny)
  let netM : NetworkMixed 4 [] 4 TestExecutor TestDType TestDType WithGrad
      netM = liftNetwork net
  frozen <- freezeNetworkMixed netM
  -- frozen : NetworkMixed 4 [] 4 TestExecutor TestDType TestDType NoGrad
  -- — compile-checked
  _ <- unfreezeNetworkMixed frozen
  -- back to WithGrad — compile-checked
  check "freezeNetworkMixed / unfreezeNetworkMixed round-trip" True


-- A4: epochVarMixed compiles, runs one epoch on a tiny LinearMixed
-- network + GradScaler, and the returned loss is finite. This is
-- the end-to-end smoke for the type-safe mixed-precision plan #410
-- — every prior piece (LayerLikeMixed, LinearMixed, autograd-aware
-- tcast, nativeTrainStepScaled, GradScaler) is exercised in series.
epochVarMixedSmoke : IO Bool
epochVarMixedSmoke = do
  lin <- mixedLinearLayerAny {ex=TestExecutor} {paramDt=TestDType} {computeDt=TestDType}
                             {i=2} {o=1} "epoch_mixed_smoke"
  let netM : NetworkMixed 2 [] 1 TestExecutor TestDType TestDType WithGrad
      netM = OutputLayerMixed lin
  gs <- defaultGradScaler {ex=TestExecutor} {dt=TestDType}
  opt <- pure $ nativeSgd {ex=TestExecutor} 0.01
  let dataPoints : Vect 2 (DataPoint 2 1 Double)
      dataPoints =
        [ MkDataPoint (VArray [1.0, 0.0]) (VArray [1.0])
        , MkDataPoint (VArray [0.0, 1.0]) (VArray [-1.0])
        ]
  (_, loss) <- epochVarMixed opt gs dataPoints tmseLoss netM
  let isFinite : Double -> Bool
      isFinite x = x == x && x /= 1.0/0.0 && x /= -1.0/0.0
  check ("epochVarMixed returns finite loss (got " ++ show loss ++ ")")
        (isFinite loss)


export
tests : List (IO Bool)
tests =
  [ bridgeForwardTypechecks
  , bridgeFreezeUnfreezeRoundTrip
  , linearMixedForwardTypechecks
  , epochVarMixedSmoke
  ]
