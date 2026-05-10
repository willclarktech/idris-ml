/* linear/shape/reshape.c — tensor_reshape (forward + backward).
 *
 * Forward: same data buffer (shared, arena-allocated
 * header), new shape metadata. Propagates dtype_tag so an F32 reshape
 * stays F32-tagged. Backward: gradient passes through unchanged
 * (numel is identical between input and output).
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_reshape(TensorHandle h, int* shape, int rank) {
    Tensor* t = (Tensor*)h;
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = t->data;  /* shared */
    r->shape = arena_alloc(rank * sizeof(int));
    memcpy(r->shape, shape, rank * sizeof(int));
    r->rank = rank;
    r->numel = t->numel;
    r->requires_grad = t->requires_grad;
    r->tape_idx = -1;
    r->grad = NULL;
    r->dtype_tag = t->dtype_tag;
    if (r->requires_grad) tape_append(OP_RESHAPE, r, t, NULL, 0);
    return r;
}

static void tape_backward_reshape(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    if (a) {
        ensure_grad(a);
        ensure_grad(r);
        for (int j = 0; j < a->numel; j++)
            ((double*)a->grad)[j] += ((double*)r->grad)[j];
    }
}

TAPE_REGISTER_OP(OP_RESHAPE, tape_backward_reshape)
