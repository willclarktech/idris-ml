/* linear/concat/cat2.c — concatenate two 1D tensors.
 *
 * Forward: [a] ++ [b] -> [a+b]. dtype-aware copy. Stores
 * a as arg1, b as arg2, scalar_arg = split point (na). Backward:
 * grad[0..na) -> a's grad; grad[na..) -> b's grad.
 */

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
    int na = a->numel, nb = b->numel, total = na + nb;
    int rg = a->requires_grad || b->requires_grad;
    int* shape = arena_alloc(sizeof(int));
    shape[0] = total;
    if (a->dtype_tag == DT_F32) {
        float* data = arena_alloc(total * sizeof(float));
        memcpy(data,      a->data, na * sizeof(float));
        memcpy(data + na, b->data, nb * sizeof(float));
        Tensor* r = make_tensor_arena_f32(data, total, shape, 1, rg);
        if (rg) tape_append(OP_CAT, r, a, b, (double)na);
        return r;
    }
    double* data = arena_alloc(total * sizeof(double));
    memcpy(data, a->data, na * sizeof(double));
    memcpy(data + na, b->data, nb * sizeof(double));
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = data; r->shape = shape; r->rank = 1;
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
            ((double*)a->grad)[j] += ((double*)r->grad)[j];
    }
    if (b) {
        ensure_grad(b);
        for (int j = 0; j < b->numel; j++)
            ((double*)b->grad)[j] += ((double*)r->grad)[split + j];
    }
}

TAPE_REGISTER_OP(OP_CAT, tape_backward_cat)
