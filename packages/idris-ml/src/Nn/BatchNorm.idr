||| `BatchNorm` — per-channel batch normalisation via the fused
||| `primBatchNorm`. Params-only: two learnable params (gamma/beta,
||| `[channels]`) + two non-learnable running-stat buffers (mean/var); the
||| forward is 1-D over a `[channels*spatialDim]` sample. `params` lists
||| only gamma/beta (running stats are buffers, not optimizer params). The
||| `channels*spatialDim` product index means callers pin `{channels}` and
||| `{spatialDim}` at the forward (the legacy `applyBatchNorm` did too).
module Nn.BatchNorm

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| Batch norm over `channels` features, `spatialDim` positions each.
||| `i = o = channels * spatialDim`. gamma/beta learnable; mean/var buffers.
public export
data BatchNorm : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type where
  MkBatchNorm : {channels, spatialDim : Nat} ->
                TVec channels ex dt WithGrad ->   -- gamma
                TVec channels ex dt WithGrad ->   -- beta
                TVec channels ex dt NoGrad ->     -- running mean (buffer)
                TVec channels ex dt NoGrad ->     -- running var (buffer)
                (training : Bool) -> (momentum : Double) -> (eps : Double) ->
                BatchNorm (channels * spatialDim) (channels * spatialDim) ex dt

public export
Params BatchNorm where
  params (MkBatchNorm gamma beta _ _ _ _ _) = [toParam gamma, toParam beta]

||| 1-D batch-norm forward. Training mode uses batch stats + updates the
||| running buffers in place; eval mode uses the running buffers. Indexed by
||| the flattened size `i`; `channels`/`spatialDim` are recovered from the
||| constructor (the product index can't be factored from a signature).
export
batchNormForward : {0 ex : Executor} -> Backend ex dt => {0 g : GradMode} -> {i : Nat} ->
                   BatchNorm i i ex dt -> TVec i ex dt g -> IO (TVec i ex dt g)
batchNormForward (MkBatchNorm {channels} {spatialDim} gamma beta mean var training momentum eps) input = ioRerun (\_ =>
  let cI    = cast {to=Int} channels
      sI    = cast {to=Int} spatialDim
      tFlag = the Int (if training then 1 else 0)
  in MkTensor (primBatchNorm {ex} input.tensorPtr gamma.tensorPtr beta.tensorPtr
                             mean.tensorPtr var.tensorPtr cI sI tFlag momentum eps) Nothing)

||| Toggle training/eval mode.
export
setTraining : Bool -> BatchNorm i o ex dt -> BatchNorm i o ex dt
setTraining mode (MkBatchNorm g b m v _ mom eps) = MkBatchNorm g b m v mode mom eps

||| Raw handles of the running mean / variance buffers. Lets callers read
||| the buffer values back (via `primItem*`) without deconstructing the
||| product-indexed constructor, whose `channels` can't be recovered from
||| the `channels*spatialDim` index. Used by the save/load roundtrip test.
export
runningStatPtrs : BatchNorm i o ex dt -> (AnyPtr, AnyPtr)
runningStatPtrs (MkBatchNorm _ _ m v _ _ _) = (m.tensorPtr, v.tensorPtr)

||| Construct a `BatchNorm` inside an `Init` derivation. gamma=1, beta=0,
||| running mean=0, var=1; momentum 0.1, eps 1e-5; starts in training mode.
||| Registers `<scope>.batch_norm_<n>.weight` (gamma) / `.bias` (beta) as
||| learnable params, and `.running_mean` / `.running_var` as non-learnable
||| BUFFERS (PyTorch state_dict names) — the optimizer skips them, but
||| save/load persists the trained statistics. (PyTorch's
||| `num_batches_tracked` isn't tracked here; momentum is fixed.)
export partial
batchNorm : {0 ex : Executor} -> Backend ex dt => {channels, spatialDim : Nat} ->
            Init (BatchNorm (channels * spatialDim) (channels * spatialDim) ex dt)
batchNorm = do
  name  <- freshChild "batch_norm"
  gamma <- liftIO $ tparam1dConst  {ex} {dt} {n=channels} (name ++ ".weight")       1.0
  beta  <- liftIO $ tparam1dConst  {ex} {dt} {n=channels} (name ++ ".bias")         0.0
  mean  <- liftIO $ tbuffer1dConst {ex} {dt} {n=channels} (name ++ ".running_mean") 0.0
  var   <- liftIO $ tbuffer1dConst {ex} {dt} {n=channels} (name ++ ".running_var")  1.0
  pure (MkBatchNorm {channels} {spatialDim} gamma beta mean var True 0.1 1.0e-5)
