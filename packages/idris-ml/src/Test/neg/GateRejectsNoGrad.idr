||| Negative compile test: confirms the GradMode gate rejects a
||| NoGrad loss being passed to `trainStep`. This file MUST
||| NOT type-check. Run via `make test-gradmode-gate` (or
||| `scripts/check-gradmode-gate.sh`), which inverts the exit code
||| and asserts the error mentions both `WithGrad` and `NoGrad`.
|||
||| If this file ever starts to compile, the gate has regressed —
||| an inference loss could silently be fed to training again.

module GateRejectsNoGrad

import Data.Vect

import Ml.Executor
import Ml.Tensor

-- A loss tensor materialised as NoGrad (e.g. via `weakenGrad` on
-- the output of a `withNoGrad`-wrapped forward). The C-side handle
-- is irrelevant for this compile test — the type itself triggers
-- the gate.
fakeNoGradLoss : Tensor (the (Vect 0 Nat) []) TapeExecutor F64 NoGrad
fakeNoGradLoss = believe_me ()

-- ^^^ EXPECTED COMPILE ERROR: trainStep requires its loss
-- argument to be `Tensor [] TapeExecutor F64 WithGrad`; we passed
-- `Tensor [] TapeExecutor F64 NoGrad`. Error must be the grad-mode
-- unification failure itself (a "Mismatch between: ..." line in
-- Idris 2 v0.8.0) — the gate script ignores source-echo lines, so
-- these comments can't satisfy its grep.
brokenStep : NativeOptimizer TapeExecutor -> IO Double
brokenStep opt = trainStep opt fakeNoGradLoss
