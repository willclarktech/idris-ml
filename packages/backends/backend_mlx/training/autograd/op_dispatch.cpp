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
    if (op < 0 || op >= OP_COUNT) {
        std::fprintf(stderr, "[mlx backend] mlx_register_replay: op tag %d out of range "
                     "[0..%d)\n", op, OP_COUNT);
        std::abort();
    }
    g_mlx_replay[op] = fn;
}

MlxReplayFn mlx_dispatch_get(int op) {
    if (op < 0 || op >= OP_COUNT) return nullptr;
    return g_mlx_replay[op];
}
