/* linear/shape/select.c — tensor_select (forward + backward).
 *
 * Forward selects an element from a vector (rank-1 ->
 * scalar) or a row from a matrix (rank-2 dim=0 -> vector). Backward
 * scatters grad back into the parent at the selected index/row.
 *
 * Shares storage with parent via offset pointer (no copy). Element
 * stride honours parent's dtype_tag so F32 selects step 4 bytes,
 * F64 selects step 8.
 */

#include <string.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    (void)dim;
    Tensor* t = (Tensor*)h;
    /* Scalar: select(scalar, 0, 0) is identity — return the tensor directly
       to preserve tape connectivity (the scalar already has a tape entry). */
    if (t->rank == 0) return h;
    size_t es = tape_elem_size(t->dtype_tag);
    /* Snapshot parent fields before allocating the result tensor. After
       arena_reset (e.g. from optimizer_step), the bump pointer rewinds; the
       next arena_alloc(sizeof(Tensor)) may return t's own address when t is
       a user-created tensor (via tensor_create — which itself uses the arena
       since it doesn't set persistent=1). The subsequent memset would then
       zero t's struct mid-function, leaving t->data NULL and t->shape stale.
       Snapshot to locals before the alloc so the result is well-formed even
       when r aliases t. */
    void*  parent_data         = t->data;
    int    parent_requires_grad = t->requires_grad;
    int    parent_dtype_tag    = t->dtype_tag;
    if (t->rank == 1) {
        Tensor* v = arena_alloc(sizeof(Tensor));
        memset(v, 0, sizeof(Tensor));
        v->data = (char*)parent_data + (size_t)index * es;
        v->shape = NULL;
        v->rank = 0;
        v->numel = 1;
        v->requires_grad = parent_requires_grad;
        v->tape_idx = -1;
        v->grad = NULL;
        v->dtype_tag = parent_dtype_tag;
        if (v->requires_grad) tape_append(OP_SELECT, v, t, NULL, (double)index);
        return v;
    } else if (t->rank == 2 && dim == 0) {
        int cols = t->shape[1];
        Tensor* r = arena_alloc(sizeof(Tensor));
        memset(r, 0, sizeof(Tensor));
        r->data = (char*)parent_data + (size_t)(index * cols) * es;
        r->shape = arena_alloc(sizeof(int));
        r->shape[0] = cols;
        r->rank = 1;
        r->numel = cols;
        r->requires_grad = parent_requires_grad;
        r->tape_idx = -1;
        r->grad = NULL;
        r->dtype_tag = parent_dtype_tag;
        if (r->requires_grad) tape_append(OP_SELECT, r, t, NULL, (double)index);
        return r;
    }
    /* Fallback: high-rank select returns a fresh scalar with the correct dtype. */
    double v = tape_load_d(t, index);
    return (t->dtype_tag == DT_F32) ? make_scalar_f32(v, t->requires_grad)
                                    : make_scalar(v, t->requires_grad);
}

static void tape_backward_select(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    int sel_idx = (int)e->scalar_arg;
    if (a) {
        ensure_grad(a);
        ensure_grad(r);
        if (r->numel == 1) {
            /* Scalar select from vector */
            ((double*)a->grad)[sel_idx] += ((double*)r->grad)[0];
        } else {
            /* Row select from matrix */
            int cols = r->numel;
            for (int j = 0; j < cols; j++)
                ((double*)a->grad)[sel_idx * cols + j] += ((double*)r->grad)[j];
        }
    }
}

TAPE_REGISTER_OP(OP_SELECT, tape_backward_select)
