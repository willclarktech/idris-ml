/* linear/linalg/dot.c — vector dot product (forward + backward).
 *
 * r = sum_i a[i] * b[i] (scalar). d(r)/da = b, d(r)/db = a.
 * tape_load_d handles both F32 and F64 input storage.
 */

#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../../backend.h"

extern void tape_abort_mixed_dtype(const char* op) __attribute__((noreturn));

TensorHandle tensor_dot(TensorHandle ha, TensorHandle hb) {
    Tensor* a = (Tensor*)ha;
    Tensor* b = (Tensor*)hb;
    if (a->dtype_tag != b->dtype_tag) tape_abort_mixed_dtype("tensor_dot");
    double s = 0;
    for (int i = 0; i < a->numel; i++) s += tape_load_d(a, i) * tape_load_d(b, i);
    int rg = a->requires_grad || b->requires_grad;
    Tensor* r = (a->dtype_tag == DT_F32) ? make_scalar_f32(s, rg) : make_scalar(s, rg);
    if (r->requires_grad) tape_append(OP_DOT, r, a, b, 0);
    return r;
}

static void tape_backward_dot(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    if (a && a->numel > 1) {
        ensure_grad(a);
        for (int j = 0; j < a->numel; j++)
            tape_grad_add_d(a, j, tape_grad_load_d(r, 0) * tape_load_d(b, j));
    } else if (a) {
        ensure_grad(a);
        tape_grad_add_d(a, 0, tape_grad_load_d(r, 0) * tape_load_d(b, 0));
    }
    if (b && b->numel > 1) {
        ensure_grad(b);
        for (int j = 0; j < b->numel; j++)
            tape_grad_add_d(b, j, tape_grad_load_d(r, 0) * tape_load_d(a, j));
    } else if (b) {
        ensure_grad(b);
        tape_grad_add_d(b, 0, tape_grad_load_d(r, 0) * tape_load_d(a, 0));
    }
}

TAPE_REGISTER_OP(OP_DOT, tape_backward_dot)
