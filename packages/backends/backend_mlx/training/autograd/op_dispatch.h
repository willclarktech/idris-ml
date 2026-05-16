/* backend_mlx/training/autograd/op_dispatch.h — per-op replay
 * function dispatch table.
 *
 * mlx's backward is replay-based: tensor_backward (training/backward.cpp)
 * captures the live tape, builds an mx::array pool indexed by
 * Tensor::pool_idx, and passes a closure to mx::vjp. Inside the closure
 * a generic loop walks the tape and applies the matching forward
 * operation into a fresh pool slot per op. mx::vjp differentiates that
 * traced forward to produce param gradients.
 *
 * The dispatch table maps OP_* enum -> per-op replay function. Each
 * forward-kernel TU registers its replay alongside the forward via
 * MLX_REGISTER_REPLAY(OP_FOO, mlx_replay_foo) at file scope — pattern
 * mirrors the tape backend's TAPE_REGISTER_OP exactly. Constructor
 * order across TUs is unspecified, but the table is a BSS-zero global
 * so reads before any constructor fires return nullptr (the
 * well-defined unregistered state). All constructors complete before
 * Idris invokes tensor_backward.
 *
 * Per-op replay functions take (pool, e). They read inputs via
 * pool[e.arg1->pool_idx] / pool[e.arg2->pool_idx] (or e.argN->data
 * directly for non-differentiable index args — OP_GATHER, OP_SCATTER_ADD)
 * and write the result via pool[e.result->pool_idx] = ...;
 *
 * Helpers (scalar_like / zero_like / one_like / half_like) live in
 * precision.h; per-op TUs include it as needed.
 */

#ifndef IDRISML_BACKEND_MLX_OP_DISPATCH_H
#define IDRISML_BACKEND_MLX_OP_DISPATCH_H

#include <vector>
#include <mlx/array.h>
#include "../../tape.h"

namespace mx = mlx::core;

/* Replay function signature. Reads from `pool` via `e.argN->pool_idx`
   (or `e.argN->data` for index args) and `e.meta` / `e.scalar_arg`;
   writes the forward result into `pool[e.result->pool_idx]`. */
typedef void (*MlxReplayFn)(std::vector<mx::array>& pool, TapeEntry& e);

/* Register a replay function for op tag `op`. Idempotent: subsequent
   calls overwrite. Bounds-checked at registration; out-of-range `op`
   aborts loudly. */
void mlx_register_replay(int op, MlxReplayFn fn);

/* Look up the registered replay function for op tag `op`, or nullptr
   if none. */
MlxReplayFn mlx_dispatch_get(int op);

/* MLX_REGISTER_REPLAY(op, fn) — define a __attribute__((constructor))
   that fires at load time and registers `fn` as op's replay handler.
   Place one of these at file scope in each per-op TU:

       MLX_REGISTER_REPLAY(OP_ADD, mlx_replay_add)
*/
#define MLX_REGISTER_REPLAY(op, fn)                                      \
    __attribute__((constructor))                                         \
    static void _mlx_reg_##op##_##fn(void) { mlx_register_replay(op, fn); }

#endif /* IDRISML_BACKEND_MLX_OP_DISPATCH_H */
