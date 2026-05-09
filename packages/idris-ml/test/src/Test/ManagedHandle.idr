module Test.ManagedHandle

import Harness
import Device
import Tensor
import TestConfig

-- These tests verify the Chez guardian + drain plumbing for the
-- wrapped-handle ABI (see docs/develop/tensor-lifecycle-plan.md).
-- Every Tensor-returning FFI wraps its result in a Chez vector +
-- registers it with the guardian + retains. Every Tensor-consuming FFI
-- extracts the raw pointer via vector-ref. The wrap is the Tensor's
-- runtime identity in Chez; it's not separable from the value.

-- Allocate Tensors and immediately discard the handle (no further use).
-- Idris's Chez codegen sees the binding `raw` as live only during the
-- primItem {d=TestDevice} read; after that the binding is dead and the wrap
-- is GC-eligible. Forced major GC will queue dead wraps with the guardian.
allocAndDropSum : Nat -> Double -> IO Double
allocAndDropSum Z acc = pure acc
allocAndDropSum (S k) acc = do
  let h = primCreateScalar {d=TestDevice} (cast k) 0
  let v = primItem {d=TestDevice} h
  allocAndDropSum k (acc + v)

allocAndDrop : Nat -> IO ()
allocAndDrop n = do
  s <- allocAndDropSum n 0.0
  -- Print s so the whole chain doesn't get dead-code-eliminated.
  putStrLn ("  (allocAndDrop accumulated sum: " ++ show s ++ ")")

-- initManagedHandles is self-init + idempotent. The return value
-- (1 on first call ever, 0 on subsequent) is unreliable when other
-- primitives self-init the guardian too — what we actually want to
-- verify is that two calls in a row both return 0 (the second call
-- can't be the very first invocation). Plus init must not throw.
initIsIdempotent : IO Bool
initIsIdempotent = do
  _ <- initManagedHandles
  second <- initManagedHandles
  check "init is idempotent (second call returns 0)" (second == 0)

drainCollectsAfterGc : IO Bool
drainCollectsAfterGc = do
  _ <- initManagedHandles
  -- Clear anything from prior tests
  _ <- drainManagedHandles
  forceMajorGc
  _ <- drainManagedHandles
  allocAndDrop 50
  -- Without forced GC, drain yields 0 (Chez doesn't auto-GC under
  -- foreign pressure — see docs/develop/tensor-lifecycle.md "The drain mechanism").
  -- The post-GC count is `>= 50` rather than `== 50` because some backends
  -- emit multiple wraps per `primCreateScalar` (torch wraps once for the
  -- raw create and again for the to-device migration, so 2 wraps per
  -- iteration → 100 wraps for 50 iters). The semantic invariant — "dead
  -- wraps drain after major GC, but not before" — is what the test
  -- guards; the precise multiplier is a backend implementation detail.
  preGc <- drainManagedHandles
  forceMajorGc
  postGc <- drainManagedHandles
  check ("drain post-GC >= 50, pre-GC = 0 (got pre=" ++ show preGc ++ " post=" ++ show postGc ++ ")")
        (preGc == 0 && postGc >= 50)

scalarRoundTrip : IO Bool
scalarRoundTrip = do
  _ <- initManagedHandles
  let h = primCreateScalar {d=TestDevice} 42.0 0
  let v = primItem {d=TestDevice} h
  check "create + item round-trips through wrapped ABI" (v == 42.0)

export
tests : List (IO Bool)
tests =
  [ initIsIdempotent
  , drainCollectsAfterGc
  , scalarRoundTrip
  ]
