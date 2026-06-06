"""Core elementwise + scalar arithmetic primitives."""

from .._entry import Entry


ENTRIES = {
    "tensor_abs": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primAbs'),
    "tensor_add_scalar": Entry(args=('T', 'd'), ret='T', slice='UserExecutorCore', idris_method='primAddScalar'),
    "tensor_add": Entry(args=('T', 'T'), ret='T', slice='UserExecutorCore', idris_method='primAdd'),
    "tensor_clamp_min": Entry(args=('T', 'd'), ret='T', slice='UserExecutorCore', idris_method='primClampMin'),
    "tensor_clamp": Entry(args=('T', 'd', 'd'), ret='T', slice='UserExecutorCore', idris_method='primClamp'),
    "tensor_clone": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primClone'),
    "tensor_create_scalar": Entry(args=('d', 'i'), ret='T', slice='UserExecutorCore', idris_method='primCreateScalar', torch='bespoke'),
    "tensor_div": Entry(args=('T', 'T'), ret='T', slice='UserExecutorCore', idris_method='primDiv'),
    "tensor_exp": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primExp'),
    "tensor_free": Entry(args=('T',), ret='v', slice='UserExecutorCore', idris_method='primFree'),
    "tensor_item_1d": Entry(args=('T', 'i'), ret='d', slice='UserExecutorCore', idris_method='primItem1d'),
    "tensor_item": Entry(args=('T',), ret='d', slice='UserExecutorCore', idris_method='primItem'),
    "tensor_log": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primLog'),
    "tensor_mul_scalar": Entry(args=('T', 'd'), ret='T', slice='UserExecutorCore', idris_method='primMulScalar'),
    "tensor_mul": Entry(args=('T', 'T'), ret='T', slice='UserExecutorCore', idris_method='primMul'),
    "tensor_neg": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primNeg'),
    "tensor_pow": Entry(args=('T', 'T'), ret='T', slice='UserExecutorCore', idris_method='primPow'),
    "tensor_round": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primRound'),
    "tensor_sigmoid": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primSigmoid'),
    "tensor_sqrt": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primSqrt'),
    "tensor_sub": Entry(args=('T', 'T'), ret='T', slice='UserExecutorCore', idris_method='primSub'),
    "tensor_tanh": Entry(args=('T',), ret='T', slice='UserExecutorCore', idris_method='primTanh'),
}
