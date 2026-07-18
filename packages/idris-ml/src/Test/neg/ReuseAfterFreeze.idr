||| Negative compile test: confirms the linear-model gate rejects reusing a
||| model handle after it has been consumed by `eval` (the same class as
||| "freeze a model, then reuse the stale handle to train"). This file MUST
||| NOT type-check. Run via `make test-integration-typegate-linear-model`
||| (or `scripts/check-linear-model-gate.sh`), which inverts the exit code
||| and asserts the error is a *linearity* error (so an unrelated regression
||| doesn't pass the gate).
|||
||| If this file ever starts to compile, the gate has regressed — a stale
||| model alias could silently no-op training again.
module ReuseAfterFreeze

import Control.Linear.LIO
import Data.Linear

import Ml.Executor
import Ml.GradMode
import Ml.Nn.Linear
import Ml.Nn.Module

-- `m` is passed as a LINEAR resource (`1 _`). The first `eval m` CONSUMES
-- it; the second use is the bug. The compiler must reject it with a
-- linearity error ("There are 2 uses of linear name m" / "m is not
-- accessible in this context").
badReuse : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad) ->
           L IO {use=1} (Linear 2 3 TapeExecutor F64 NoGrad)
badReuse m = do
  m1 <- eval m
  m2 <- eval m   -- ^^^ EXPECTED LINEARITY ERROR: m already consumed above
  discard m1
  pure1 m2
