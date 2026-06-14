||| Positive compile test: the single-use counterpart of
||| `Test/neg/ReuseAfterFreeze.idr`. Each linear model handle is used exactly
||| once (consume with `eval`, then `discard` the result). This file MUST
||| compile — it proves the negative test fails for the *right* reason
||| (linearity), not because the whole `ModuleL`/`eval` surface is broken.
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
import Nn.Seq
import Tensor

-- Leaf: consume the trainable model exactly once, return the inference one.
okSingle : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad) ->
           L IO {use=1} (Linear 2 3 TapeExecutor F64 NoGrad)
okSingle m = eval m

-- Composite: the same through a linear `Seq` (existential threading).
okSeq : (1 _ : Seq 2 3 TapeExecutor F64 WithGrad) ->
        L IO {use=1} (Seq 2 3 TapeExecutor F64 NoGrad)
okSeq s = eval s

-- Stateless + param-bearing leaf layers also satisfy `ModuleL`/`eval`.
okActivation : (1 _ : Activation 4 4 TapeExecutor F64 WithGrad) ->
               L IO {use=1} (Activation 4 4 TapeExecutor F64 NoGrad)
okActivation a = eval a

okDropout : (1 _ : Dropout 4 4 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Dropout 4 4 TapeExecutor F64 NoGrad)
okDropout d = eval d

okLayerNorm : (1 _ : LayerNorm 4 4 TapeExecutor F64 WithGrad) ->
              L IO {use=1} (LayerNorm 4 4 TapeExecutor F64 NoGrad)
okLayerNorm n = eval n

-- Composite-with-sublayer: a linear residual block also satisfies `eval`.
okResidual : (1 _ : Residual 4 4 TapeExecutor F64 WithGrad) ->
             L IO {use=1} (Residual 4 4 TapeExecutor F64 NoGrad)
okResidual r = eval r

-- Recurrent: one linear timestep (consume cell, return banged output + cell).
-- `TVec` aliases dodge the `[2]`/`[3]` list-literal `(::)` ambiguity (Seq /
-- Data.Linear `(::)` are in scope here).
okRnnStep : (1 _ : Rnn 2 3 TapeExecutor F64 WithGrad) -> TVec 2 TapeExecutor F64 WithGrad ->
            L IO {use=1} (LPair (!* (TVec 3 TapeExecutor F64 WithGrad))
                                (Rnn 2 3 TapeExecutor F64 WithGrad))
okRnnStep r x = recurStep r x

-- Recurrent layers are `ParamsL`, so `eval` works on them too.
okRnnEval : (1 _ : Rnn 2 3 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Rnn 2 3 TapeExecutor F64 NoGrad)
okRnnEval r = eval r

okLstmEval : (1 _ : Lstm 2 3 TapeExecutor F64 WithGrad) ->
             L IO {use=1} (Lstm 2 3 TapeExecutor F64 NoGrad)
okLstmEval l = eval l

okGruEval : (1 _ : Gru 2 3 TapeExecutor F64 WithGrad) ->
            L IO {use=1} (Gru 2 3 TapeExecutor F64 NoGrad)
okGruEval g = eval g
