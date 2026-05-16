module Test.ManagedHandle

import Harness
import Tensor

-- These tests verify the Chez guardian + drain plumbing introduced in
-- Phase 2.2 of the tensor-lifecycle refactor. They don't yet exercise
-- real Tensor lifetimes (that's Phase 2.3); they confirm the primitives
-- work end-to-end.

-- Wrap and *consume the wrapped value* so Idris doesn't elide the call.
-- We accumulate the scalar reads into a Double that's returned —
-- Idris's optimizer can't elide computation whose result flows to the
-- caller. In real Phase 2.3 usage, the wrapped result IS the return
-- value of every FFI wrapper, so this issue doesn't arise.
allocAndDropSum : Nat -> Double -> IO Double
allocAndDropSum Z acc = pure acc
allocAndDropSum (S k) acc = do
  let raw = prim__createScalar (cast k) 0
  let wrapped = prim__wrapHandle raw
  let v = prim__item (prim__unwrapHandle wrapped)
  allocAndDropSum k (acc + v)

allocAndDrop : Nat -> IO ()
allocAndDrop n = do
  s <- allocAndDropSum n 0.0
  -- Print or use s so the whole chain isn't dead-code-eliminated.
  -- (We don't care about the value itself; it's just to keep the
  -- compiler honest about evaluating the loop.)
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
  allocAndDrop 50
  -- Without forced GC, drain yields 0 (Chez doesn't auto-GC under
  -- foreign pressure — see tensor-lifecycle-spike.md).
  preGc <- drainManagedHandles
  forceMajorGc
  postGc <- drainManagedHandles
  check ("drain post-GC = 50, pre-GC = 0 (got pre=" ++ show preGc ++ " post=" ++ show postGc ++ ")")
        (preGc == 0 && postGc == 50)

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
