"""Optimizer construction + parameter tuning primitives."""

from .._entry import Entry


ENTRIES = {
    "native_train_step_scaled": Entry(args=('R', 'i', 'd', 'T', 'd', 'd'), ret='d', slice='UserExecutorOptimizer', idris_method='primNativeTrainStepScaled', mlx='direct'),
    "native_train_step": Entry(args=('R', 'i', 'd', 'T', 'd'), ret='d', slice='UserExecutorOptimizer', idris_method='primNativeTrainStep', mlx='direct'),
    "optimizer_create_adam_group": Entry(args=('d', 'd', 'd', 'd', 's'), ret='R', slice='UserExecutorOptimizer', idris_method='primOptimizerCreateAdamGroup', mlx='direct'),
    "optimizer_create_adam_w": Entry(args=('d', 'd', 'd', 'd', 'd'), ret='R', slice='UserExecutorOptimizer', idris_method='primOptimizerCreateAdamW', c_symbol='optimizer_create_adamw', mlx='direct'),
    "optimizer_create_adam": Entry(args=('d', 'd', 'd', 'd'), ret='R', slice='UserExecutorOptimizer', idris_method='primOptimizerCreateAdam', mlx='direct'),
    "optimizer_create_rmsprop": Entry(args=('d', 'd', 'd', 'd', 'd'), ret='R', slice='UserExecutorOptimizer', idris_method='primOptimizerCreateRmsprop', mlx='direct'),
    "optimizer_create_sgd": Entry(args=('d',), ret='R', slice='UserExecutorOptimizer', idris_method='primOptimizerCreateSgd', mlx='direct'),
    "optimizer_set_lr": Entry(args=('R', 'd'), ret='v', slice='UserExecutorOptimizer', idris_method='primOptimizerSetLr', mlx='direct'),
    "optimizer_set_param_lr": Entry(args=('R', 's', 'd'), ret='v', slice='UserExecutorOptimizer', idris_method='primOptimizerSetParamLr', mlx='direct'),
}
