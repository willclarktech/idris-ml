module Test.MixedLayerLike

import Data.Vect

import Harness
import Device
import Tensor
import Array
import Layer
import Layer.MixedCore
import TestConfig


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
mkInput : {n : Nat} -> Vect n Double -> Tensor [n] TestDevice TestDType WithGrad
mkInput xs =
  let raw = bulkToTensor {d=TestDevice} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw


-- A2: LinearMixed constructs end-to-end and runs through the
-- mixed-precision pipeline. Uses paramDt = computeDt = TestDType so
-- the cast inside applyVarMixed is a no-op at the dtype level — the
-- test verifies the layer machinery composes, the lossy-cast variants
-- are covered by C-level cast_grad_propagation (A1).
linearMixedForwardTypechecks : IO Bool
linearMixedForwardTypechecks = do
  lin <- mixedLinearLayerAny {d=TestDevice} {paramDt=TestDType} {computeDt=TestDType}
                             {i=4} {o=3} "lin_mixed_test"
  let netM : NetworkMixed 4 [] 3 TestDevice TestDType TestDType WithGrad
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
  let net : Network 4 [] 4 TestDevice TestDType WithGrad
      net = OutputLayer (the (AnyLayer 4 4 TestDevice TestDType WithGrad) tanhLayerAny)
  let netM : NetworkMixed 4 [] 4 TestDevice TestDType TestDType WithGrad
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
  let net : Network 4 [] 4 TestDevice TestDType WithGrad
      net = OutputLayer (the (AnyLayer 4 4 TestDevice TestDType WithGrad) tanhLayerAny)
  let netM : NetworkMixed 4 [] 4 TestDevice TestDType TestDType WithGrad
      netM = liftNetwork net
  frozen <- freezeNetworkMixed netM
  -- frozen : NetworkMixed 4 [] 4 TestDevice TestDType TestDType NoGrad
  -- — compile-checked
  _ <- unfreezeNetworkMixed frozen
  -- back to WithGrad — compile-checked
  check "freezeNetworkMixed / unfreezeNetworkMixed round-trip" True


export
tests : List (IO Bool)
tests =
  [ bridgeForwardTypechecks
  , bridgeFreezeUnfreezeRoundTrip
  , linearMixedForwardTypechecks
  ]
