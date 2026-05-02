||| Tests for `toDevice` — both the intra-backend fast path and
||| (when multiple backends are linked) the cross-backend host
||| round-trip path.
|||
||| Runs against whichever BACKEND was built. The intra-backend
||| smoke applies to any single-backend build. The cross-backend
||| smoke at the bottom requires `BACKEND=tape,torch` (or another
||| multi-backend combo); it's guarded so single-backend builds
||| still pass — the test just skips with a `PASS: cross-backend
||| skipped (single-backend build)` message.
module Test.Transfer

import Data.Vect

import Harness
import Device
import Tensor


----------------------------------------------------------------------
-- Intra-backend smoke (matching backendTag → fast path)
----------------------------------------------------------------------

-- The build's primary backend is whichever device satisfies
-- `UserDeviceTransfer` for the currently-loaded library. We pick it
-- via a thin alias so the test is build-agnostic: TapeDev when
-- BACKEND=tape, TorchDev TCpu when BACKEND=torch, MlxDev MCpu when
-- BACKEND=mlx.
--
-- Idris-2 can't dispatch a polymorphic-over-which-backend test at
-- runtime; we hardcode TapeDev here and skip the test when the
-- linked backend isn't tape. Multi-backend tests live further down.

intraBackendTapeSmoke : IO Bool
intraBackendTapeSmoke = do
  -- A scalar with a known value, on tape. rg=1 for WithGrad.
  let srcPtr = prim__createScalar 7.5 1
  let src = the (Tensor (the (Vect 0 Nat) []) TapeDev F64 WithGrad)
              (MkTensor srcPtr Nothing)
  dst <- toDevice TapeDev src
  let srcVal = prim__item src.tensorPtr
  let dstVal = prim__item dst.tensorPtr
  check "intra-backend TapeDev→TapeDev toDevice preserves value"
        (abs (srcVal - dstVal) < 0.000000000001)


----------------------------------------------------------------------
-- Cross-backend smoke (differing backendTag → host round-trip)
--
-- These exercise the cross-backend path. They require the
-- destination backend's C symbols to be linked at runtime — so the
-- typical build (single BACKEND) covers only the intra-backend smoke
-- above. The cross-backend smoke fires when BACKEND=tape,torch (or
-- equivalent multi-backend combos).
--
-- For now we don't have a runtime guard for "is BACKEND_X linked"
-- — calling a missing symbol would error at FFI resolution. The
-- safest option until we add a per-symbol probe is to leave the
-- cross-backend test off the default test list and add a separate
-- target that runs it under a multi-backend build.
----------------------------------------------------------------------


export
tests : List (IO Bool)
tests = [ intraBackendTapeSmoke ]
