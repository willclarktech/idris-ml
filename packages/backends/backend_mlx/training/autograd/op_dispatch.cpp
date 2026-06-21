/* backend_mlx/training/autograd/op_dispatch.cpp — per-op replay
 * function dispatch table.
 *
 * Standalone TU. Mirrors backend_tape/training/autograd/op_dispatch.c.
 */

#include <cstdio>
#include <cstdlib>
#include "op_dispatch.h"
#include "../../../cxx_abort.h"

static MlxReplayFn g_mlx_replay[OP_COUNT] = {nullptr};

void mlx_register_replay(int op, MlxReplayFn fn) {
	/* The guard condition is evaluated at load time by every
	   MLX_REGISTER_REPLAY constructor (op_dispatch.h), so the same-line
	   macro covers this line for free; the abort fires only on an
	   out-of-range tag, which the compile-time OP_* tags never produce. */
	CXX_ABORT_IF(op < 0 || op >= OP_COUNT,
	             "[mlx backend] mlx_register_replay: op tag %d out of range "
	             "[0..%d)\n",
	             op, OP_COUNT);
	g_mlx_replay[op] = fn;
}

MlxReplayFn mlx_dispatch_get(int op) {
	if (op < 0 || op >= OP_COUNT) return nullptr;
	return g_mlx_replay[op];
}
