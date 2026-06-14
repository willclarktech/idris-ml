||| Positive compile test: the single-use counterpart of
||| `Test/neg/ReuseAfterFreeze.idr`. Each linear model handle is used exactly
||| once (consume with `evalL`, then `discardL` the result). This file MUST
||| compile — it proves the negative test fails for the *right* reason
||| (linearity), not because the whole `ModuleL`/`evalL` surface is broken.
module SingleUseCompiles

import Control.Linear.LIO
import Data.Linear

import Executor
import GradMode
import Nn.Linear
import Nn.Module
import Nn.SeqL

-- Leaf: consume the trainable model exactly once, return the inference one.
okSingle : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad) ->
           L IO {use=1} (Linear 2 3 TapeExecutor F64 NoGrad)
okSingle m = evalL m

-- Composite: the same through a linear `SeqL` (existential threading).
okSeq : (1 _ : SeqL 2 3 TapeExecutor F64 WithGrad) ->
        L IO {use=1} (SeqL 2 3 TapeExecutor F64 NoGrad)
okSeq s = evalL s
