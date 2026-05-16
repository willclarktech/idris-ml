module Test.ManagedHandle

import Harness
import Tensor

-- These tests verify the Chez guardian + drain plumbing introduced in
-- Phase 2.2 of the tensor-lifecycle refactor. They don't yet exercise
-- real Tensor lifetimes (that's Phase 2.3); they confirm the primitives
-- work end-to-end.

-- State Tensors are the only Tensors that go through the guardian: wrap
-- conditionally returns a Chez vector only when the C-side
-- `tensor_is_state` flag is set. `prim__createState1d` sets it; plain
-- scalar/intermediate allocations don't. Allocating a state Tensor +
-- dropping its wrapper + forcing GC should land it in the guardian's
-- dead queue.
allocAndDropSum : Nat -> Double -> IO Double
allocAndDropSum Z acc = pure acc
allocAndDropSum (S k) acc = do
  let buf = prim__allocDoubles 1
  let raw = prim__createState1d 1 buf
  let wrapped = prim__wrapHandle raw
  let v = prim__item (prim__unwrapHandle wrapped)
  allocAndDropSum k (acc + v)

allocAndDrop : Nat -> IO ()
allocAndDrop n = do
  s <- allocAndDropSum n 0.0
  -- Print s so the whole chain doesn't get dead-code-eliminated.
  putStrLn ("  (allocAndDrop accumulated sum: " ++ show s ++ ")")

initIsIdempotent : IO Bool
initIsIdempotent = do
  first  <- initManagedHandles
  second <- initManagedHandles
  check "init is idempotent" (first == 1 && second == 0)

drainCollectsAfterGc : IO Bool
drainCollectsAfterGc = do
  _ <- initManagedHandles
  -- Clear anything from prior tests
  _ <- drainManagedHandles
  forceMajorGc
  _ <- drainManagedHandles
  allocAndDrop 50
  -- Without forced GC, drain yields 0 (Chez doesn't auto-GC under
  -- foreign pressure — see tensor-lifecycle-spike.md).
  preGc <- drainManagedHandles
  forceMajorGc
  postGc <- drainManagedHandles
  check ("drain post-GC = 50, pre-GC = 0 (got pre=" ++ show preGc ++ " post=" ++ show postGc ++ ")")
        (preGc == 0 && postGc == 50)

-- Non-state Tensors get the raw pointer back from wrap (no guardian
-- registration). The unwrap is the identity in that case so existing
-- prim__item-style use keeps working.
unwrapRoundTrip : IO Bool
unwrapRoundTrip = do
  _ <- initManagedHandles
  let raw = prim__createScalar 42.0 0
  let wrapped = prim__wrapHandle raw
  let unwrapped = prim__unwrapHandle wrapped
  let v = prim__item unwrapped
  check "wrap + unwrap preserves identity" (v == 42.0)

export
tests : List (IO Bool)
tests =
  [ initIsIdempotent
  , drainCollectsAfterGc
  , unwrapRoundTrip
  ]
