"""Op-timing counters — profile reset/report, perf reset."""

from .._entry import Entry


ENTRIES = {
    "tensor_perf_reset": Entry(args=(), ret='v', slice='UserExecutorProfiling', idris_method='primPerfReset', mlx='direct'),
    "tensor_profile_report": Entry(args=(), ret='v', slice='UserExecutorProfiling', idris_method='primProfileReport', c_symbol='backend_profile_report', mlx='direct'),
    "tensor_profile_reset": Entry(args=(), ret='v', slice='UserExecutorProfiling', idris_method='primProfileReset', c_symbol='backend_profile_reset', mlx='direct'),
}
