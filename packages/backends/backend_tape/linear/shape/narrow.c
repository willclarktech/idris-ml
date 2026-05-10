/* linear/shape/narrow.c — view into a slice of a 1D tensor.
 *
 * Phase 1b.1.c. Forward: t[start..start+len) as a shared-storage view
 * with byte-correct offset (tape_elem_size honours dtype_tag).
 * Backward: scatter grad back to parent at offset.
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    (void)dim;
    Tensor* t = (Tensor*)h;
    Tensor* r = arena_alloc(sizeof(Tensor));
    memset(r, 0, sizeof(Tensor));
    r->data = (char*)t->data + (size_t)start * tape_elem_size(t->dtype_tag);
    r->shape = arena_alloc(sizeof(int));
    r->shape[0] = len;
    r->rank = 1; r->numel = len;
    r->requires_grad = t->requires_grad;
    r->tape_idx = -1;
    r->dtype_tag = t->dtype_tag;
    if (r->requires_grad) tape_append(OP_NARROW, r, t, NULL, (double)start);
    return r;
}

static void tape_backward_narrow(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    int start = (int)e->scalar_arg;
    ensure_grad(r);
    if (a) {
        ensure_grad(a);
        for (int j = 0; j < r->numel; j++)
            ((double*)a->grad)[start + j] += ((double*)r->grad)[j];
    }
}

TAPE_REGISTER_OP(OP_NARROW, tape_backward_narrow)
