/* backend_mlx/training/autograd/op_dispatch.cpp — per-op replay
 * function dispatch table.
 *
 * Standalone TU. Mirrors backend_tape/training/autograd/op_dispatch.c.
 */

#include <cstdio>
#include <cstdlib>
#include "op_dispatch.h"

static MlxReplayFn g_mlx_replay[OP_COUNT] = {nullptr};

void mlx_register_replay(int op, MlxReplayFn fn) {
	/* GCOVR_EXCL_START — unreachable defensive guard: mlx_register_replay
	   is C++-linkage internal, called only by the load-time
	   MLX_REGISTER_REPLAY constructors (op_dispatch.h) with compile-time
	   OP_* tags that are always in range. No public FFI funnels an
	   arbitrary tag here, and a C Criterion test cannot call this C++
	   symbol (mangled name + std::vector<mx::array>/TapeEntry& signature)
	   to drive the abort. */
	if (op < 0 || op >= OP_COUNT) {
		std::fprintf(stderr,
		             "[mlx backend] mlx_register_replay: op tag %d out of range "
		             "[0..%d)\n",
		             op, OP_COUNT);
		std::abort();
	}
	/* GCOVR_EXCL_STOP */
	g_mlx_replay[op] = fn;
}

MlxReplayFn mlx_dispatch_get(int op) {
	if (op < 0 || op >= OP_COUNT) return nullptr;
	return g_mlx_replay[op];
}
