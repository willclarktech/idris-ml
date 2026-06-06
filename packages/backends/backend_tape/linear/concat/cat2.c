/* linear/concat/cat2.c — concatenate two tensors along axis 0
 * (rank-preserving). Forward: cat([a; b], axis=0). For row-major
 * storage on a contiguous a/b this is just a back-to-back memcpy of
 * the underlying buffers. Inputs must have matching rank, dtype, and
 * trailing-dim shapes; only shape[0] may differ.
 *
 * Matches torch::cat({a, b}, 0) / mx::concatenate({a, b}, 0). The
 * previous "1D-only, output rank=1 [a_numel + b_numel]" semantics
 * silently broke HfLlama's per-row applyRmsNorm2d fold which builds a
 * [seq, hidden] result by stacking [1, hidden] rows — every cat call
 * silently flattened the accumulator to rank-1, then downstream
 * narrow + RoPE broadcasts blew up on shape mismatch (#396).
 *
 * Stores a as arg1, b as arg2, scalar_arg = split point (a->shape[0]).
 * Backward splits along the row axis and scatters back.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_cat2(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_cat2");
    if (a->rank != b->rank) {
        fprintf(stderr, "tensor_cat2: rank mismatch (a->rank=%d, b->rank=%d)\n",
                a->rank, b->rank);
        abort();
    }
    for (int k = 1; k < a->rank; k++) {
        if (a->shape[k] != b->shape[k]) {
            fprintf(stderr, "tensor_cat2: trailing-dim mismatch at axis=%d "
                    "(a->shape[%d]=%d, b->shape[%d]=%d)\n",
                    k, k, a->shape[k], k, b->shape[k]);
            abort();
        }
    }
    int na = a->numel, nb = b->numel, total = na + nb;
    int rg = a->requires_grad || b->requires_grad;
    int* shape = arena_alloc((size_t)a->rank * sizeof(int));
    shape[0] = a->shape[0] + b->shape[0];
    for (int k = 1; k < a->rank; k++) shape[k] = a->shape[k];
    if (a->dtype_tag == DT_F32) {
        float* data = arena_alloc(total * sizeof(float));
        memcpy(data,      a->data, na * sizeof(float));
        memcpy(data + na, b->data, nb * sizeof(float));
        Tensor* r = make_tensor_arena_f32(data, total, shape, a->rank, rg);
        if (rg) tape_append(OP_CAT, r, a, b, (double)na);
        return r;
    }
    double* data = arena_alloc(total * sizeof(double));
    memcpy(data, a->data, na * sizeof(double));
    memcpy(data + na, b->data, nb * sizeof(double));
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = data; r->shape = shape; r->rank = a->rank;
    r->numel = total; r->requires_grad = rg;
    r->tape_idx = -1;
    if (rg) tape_append(OP_CAT, r, a, b, (double)na);
    return r;
}

static void tape_backward_cat(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    int split = (int)e->scalar_arg;
    ensure_grad(r);
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            tape_grad_add_d(a, j, tape_grad_load_d(r, j));
    }
    if (b) {
        ensure_grad(b);
        for (int j = 0; j < b->numel; j++)
            tape_grad_add_d(b, j, tape_grad_load_d(r, split + j));
    }
}

TAPE_REGISTER_OP(OP_CAT, tape_backward_cat)
