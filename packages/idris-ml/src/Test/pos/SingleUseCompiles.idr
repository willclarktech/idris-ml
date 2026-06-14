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
import Nn.Activation
import Nn.Dropout
import Nn.Gru
import Nn.LayerNorm
import Nn.Linear
import Nn.Lstm
import Nn.Module
import Nn.Recurrent
import Nn.Residual
import Nn.SeqL
import Tensor

-- Leaf: consume the trainable model exactly once, return the inference one.
okSingle : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad) ->
           L IO {use=1} (Linear 2 3 TapeExecutor F64 NoGrad)
okSingle m = evalL m

-- Composite: the same through a linear `SeqL` (existential threading).
okSeq : (1 _ : SeqL 2 3 TapeExecutor F64 WithGrad) ->
        L IO {use=1} (SeqL 2 3 TapeExecutor F64 NoGrad)
okSeq s = evalL s

-- Stateless + param-bearing leaf layers also satisfy `ModuleL`/`evalL`.
okActivation : (1 _ : Activation 4 4 TapeExecutor F64 WithGrad) ->
               L IO {use=1} (Activation 4 4 TapeExecutor F64 NoGrad)
okActivation a = evalL a

okDropout : (1 _ : Dropout 4 4 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Dropout 4 4 TapeExecutor F64 NoGrad)
okDropout d = evalL d

okLayerNorm : (1 _ : LayerNorm 4 4 TapeExecutor F64 WithGrad) ->
              L IO {use=1} (LayerNorm 4 4 TapeExecutor F64 NoGrad)
okLayerNorm n = evalL n

-- Composite-with-sublayer: a linear residual block also satisfies `evalL`.
okResidual : (1 _ : ResidualL 4 4 TapeExecutor F64 WithGrad) ->
             L IO {use=1} (ResidualL 4 4 TapeExecutor F64 NoGrad)
okResidual r = evalL r

-- Recurrent: one linear timestep (consume cell, return banged output + cell).
-- `TVec` aliases dodge the `[2]`/`[3]` list-literal `(::)` ambiguity (SeqL /
-- Data.Linear `(::)` are in scope here).
okRnnStep : (1 _ : Rnn 2 3 TapeExecutor F64 WithGrad) -> TVec 2 TapeExecutor F64 WithGrad ->
            L IO {use=1} (LPair (!* (TVec 3 TapeExecutor F64 WithGrad))
                                (Rnn 2 3 TapeExecutor F64 WithGrad))
okRnnStep r x = recurStepL r x

-- Recurrent layers are `ParamsL`, so `evalL` works on them too.
okRnnEval : (1 _ : Rnn 2 3 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Rnn 2 3 TapeExecutor F64 NoGrad)
okRnnEval r = evalL r

okLstmEval : (1 _ : Lstm 2 3 TapeExecutor F64 WithGrad) ->
             L IO {use=1} (Lstm 2 3 TapeExecutor F64 NoGrad)
okLstmEval l = evalL l

okGruEval : (1 _ : Gru 2 3 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Gru 2 3 TapeExecutor F64 NoGrad)
okGruEval g = evalL g
