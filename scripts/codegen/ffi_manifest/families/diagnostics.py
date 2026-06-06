"""Tensor introspection — live count, peak live, perf op count."""

from .._entry import Entry


ENTRIES = {
    "tensor_live_count": Entry(args=(), ret='i', slice='UserExecutorDiagnostics', idris_method='primLiveCount', mlx='direct'),
    "tensor_peak_live_count": Entry(args=(), ret='i', slice='UserExecutorDiagnostics', idris_method='primPeakLiveCount', mlx='direct'),
    "tensor_perf_op_count": Entry(args=(), ret='i', slice='UserExecutorDiagnostics', idris_method='primPerfOpCount', mlx='direct'),
}
