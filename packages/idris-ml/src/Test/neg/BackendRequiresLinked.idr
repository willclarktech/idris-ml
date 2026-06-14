||| Negative compile test: confirms the `Backend ex dt` bundle cannot
||| be resolved for an executor with no `Linked` instance. This file
||| MUST NOT type-check. Run via
||| `make test-integration-typegate-backend-linked`
||| (scripts/check-backend-bundle-gate.sh), which inverts the exit
||| code and asserts the error names the bundle.
|||
||| The load-bearing property: `Linked` is the per-build availability
||| gate (instances generated into HwConfig.idr from BACKEND). A
||| bundle that dropped it would let a tape-only build spell tensors
||| on unlinked backends. If this file ever starts to compile, that
||| gate has regressed.

module BackendRequiresLinked

import Data.Vect

import Executor
import Tensor

-- A fake executor with believe_me'd capability dictionaries for
-- everything EXCEPT Linked. (believe_me is fine here: this file never
-- compiles, by design — the dictionaries only need to satisfy the
-- searches that should succeed, isolating Linked as the one that
-- fails.)
data FakeExecutor : Type where

%hint
fakeTraining : UserExecutorTraining FakeExecutor
fakeTraining = believe_me ()

%hint
fakeCompat : Compatible FakeExecutor F64
fakeCompat = believe_me ()

-- Deliberately NO `Linked FakeExecutor` instance.

needsBundle : Backend FakeExecutor F64 => Double
needsBundle = 0.0

-- ^^^ EXPECTED COMPILE ERROR at the use below: the blanket
-- implementation requires (UserExecutorTraining, RuntimeDType,
-- Linked, Compatible); the first two and last are in scope, Linked
-- is not, so `Backend FakeExecutor F64` cannot be resolved.
broken : Double
broken = needsBundle
