/* core/lifecycle/create.c — Allocate an n-dim tensor from a host buffer.
 *
 * When requires_grad=1, heap-allocates a persistent tensor (matches
 * tensor_create_param_*'s lifecycle). Grad-tracked user tensors are typically
 * param_register'd and need to survive optimizer_step's arena_reset; if they
 * lived in the arena, a post-reset arena_alloc could reissue the tensor's
 * own struct or data-buffer address, leading to the chained-view corruption
 * exercised by `legacy_backend::tensor_view`.
 *
 * When requires_grad=0, keeps the lighter arena allocation — non-grad user
 * tensors are typically per-epoch inputs (MNIST batches etc.) and don't
 * outlive the next optimizer step.
 */

#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_create(double* data, int* shape, int rank, int requires_grad) {
    if (requires_grad) {
        int numel = 1;
        for (int i = 0; i < rank; i++) numel *= shape[i];
        Tensor* t = calloc(1, sizeof(Tensor));
        t->data = malloc(numel * sizeof(double));
        memcpy(t->data, data, numel * sizeof(double));
        t->shape = malloc(rank * sizeof(int));
        memcpy(t->shape, shape, rank * sizeof(int));
        t->rank = rank;
        t->numel = numel;
        t->requires_grad = 1;
        t->tape_idx = -1;
        t->grad = NULL;
        t->persistent = 1;
        t->dtype_tag = DT_F64;
        tape_append(OP_CONST, t, NULL, NULL, 0);
        return t;
    }
    return make_tensor(data, shape, rank, requires_grad);
}
