/* core/lifecycle/create_scalar.c — Allocate a 0-rank tensor with a value.
 *
 * Phase 1a.1 (per /Users/admin/.claude/plans/modular-petting-minsky.md).
 * Always heap-allocated (persistent=1): scalars returned to Idris may
 * survive arena_reset (cached in Variables across epochs). The
 * per-epoch leak from training-data tensors (~15KB/epoch) is accepted.
 *
 * No backward — OP_CONST appears in the tape only as a marker entry.
 */

#include <stdlib.h>
#include "../../tape.h"
#include "../../tensor.h"
#include "../../../backend.h"

static int persistent_scalar_count = 0;

TensorHandle tensor_create_scalar(double value, int requires_grad) {
    persistent_scalar_count++;
    Tensor* t = calloc(1, sizeof(Tensor));
    t->data = malloc(sizeof(double));
    ((double*)t->data)[0] = value;
    t->rank = 0; t->numel = 1;
    t->requires_grad = requires_grad;
    t->tape_idx = -1;
    t->persistent = 1;
    if (requires_grad) tape_append(OP_CONST, t, NULL, NULL, 0);
    return t;
}
