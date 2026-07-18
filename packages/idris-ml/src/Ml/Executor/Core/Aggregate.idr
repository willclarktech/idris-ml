||| The two aggregate constraints: UserExecutorTraining (full training
||| surface) and UserExecutorInference (inference-only minimum).
module Ml.Executor.Core.Aggregate

import Ml.Executor.Core.Compute
import Ml.Executor.Core.Kind
import Ml.Executor.Core.Training
import Ml.Executor.Core.Transfer

||| Legacy training aggregate. Holds the full pre-split surface for
||| backwards compatibility with all `UserExecutorTraining ex =>`
||| callsites. Per-backend instances are one-liner `UserExecutorTraining
||| FooExec where` declarations — the actual prim* assignments live
||| in the seven sub-instance blocks above (six sub-slices + the
||| `UserExecutorOptimizations` opt-in slice). Resolving this
||| constraint transitively brings in everything an existing layer
||| needs.
public export
interface (UserExecutorConv ex,
           UserExecutorOptimizations ex,
           UserExecutorSerialize ex,
           UserExecutorProfiling ex,
           UserExecutorDiagnostics ex,
           UserExecutorMemoryHygiene ex,
           UserExecutorStreamed ex,
           UserExecutorTensorCreate ex) =>
          UserExecutorTraining (0 ex : Executor) where

----------------------------------------------------------------------
-- UserExecutorInference — inference-only aggregate
----------------------------------------------------------------------

||| Inference-only aggregate. Documents the minimum surface a third-
||| party backend that ships only forward-pass + checkpoint-load (no
||| optimizer, no autograd) needs to implement: `Conv` (transitively
||| pulls in Core + Linear + NN), `Optimizations` (the fused ops
||| `primSdpa2d` / `primRmsNorm2d` / `primSwiGlu2d` /
||| `primSoftmaxXent2d` + fused param-init), `TensorCreate` (data
||| loading + dtype-streamed creators),
||| `Transfer` (cross-backend handles), and `Quant` (BitNet ternary
||| surface). Skipping Autograd / ParamRegistry / Optimizer /
||| Serialize is a real reduction — those four sub-slices together
||| hold 27 of the 57 legacy Training methods.
public export
interface (UserExecutorConv ex,
           UserExecutorOptimizations ex,
           UserExecutorMemoryHygiene ex,
           UserExecutorStreamed ex,
           UserExecutorTensorCreate ex,
           UserExecutorTransfer ex,
           UserExecutorQuant ex) =>
          UserExecutorInference (0 ex : Executor) where
