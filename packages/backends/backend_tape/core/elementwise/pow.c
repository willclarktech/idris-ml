/* core/elementwise/pow.c — element-wise power (forward + backward).
 *
 * Phase 1a.7. d(a^b)/da = b * a^(b-1); d(a^b)/db = log(a) * a^b = log(a) * r.
 * Floors a at 1e-20 inside backward to avoid log(0) / pow(0, b-1) blow-ups
 * (forward still produces 0 from pow(0, k) for k>0; the gradient just
 * approximates).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../broadcast.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_pow(double a, double b) { return pow(a, b); }
static float  fn_pow_f32(float a, float b) { return powf(a, b); }

TensorHandle tensor_pow(TensorHandle a, TensorHandle b) {
    Tensor* ta = (Tensor*)a;
    Tensor* tb = (Tensor*)b;
    if (ta->dtype_tag == DT_F32 || tb->dtype_tag == DT_F32) {
        if (ta->dtype_tag != tb->dtype_tag) tape_abort_mixed_dtype("tensor_pow");
        return binop_elementwise_f32_disp(a, b, OP_POW, fn_pow_f32);
    }
    return binop_elementwise(a, b, OP_POW, fn_pow);
}

static void tape_backward_pow(TapeEntry* e) {
    Tensor* r = e->result;
    Tensor* a = e->arg1;
    Tensor* b = e->arg2;
    int a_match = a && shapes_equal(a, r);
    int b_match = b && shapes_equal(b, r);
    if (a) ensure_grad(a);
    if (b) ensure_grad(b);
    ensure_grad(r);
    if (a_match && b_match) {
        for (int j = 0; j < r->numel; j++) {
            double av = fmax(tape_load_d(a, j), 1e-20);
            double bv = tape_load_d(b, j);
            ((double*)a->grad)[j] += ((double*)r->grad)[j] * bv * pow(av, bv - 1.0);
            ((double*)b->grad)[j] += ((double*)r->grad)[j] * tape_load_d(r, j) * log(av);
        }
    } else {
        int a_str[MAX_BCAST_RANK] = {0}, b_str[MAX_BCAST_RANK] = {0};
        int idx[MAX_BCAST_RANK] = {0};
        if (a) compute_bcast_strides(a, r->rank, r->shape, a_str);
        if (b) compute_bcast_strides(b, r->rank, r->shape, b_str);
        for (int i = 0; i < r->numel; i++) {
            int ai = 0, bi = 0;
            for (int k = 0; k < r->rank; k++) {
                ai += idx[k] * a_str[k];
                bi += idx[k] * b_str[k];
            }
            double av = fmax(tape_load_d(a, ai), 1e-20);
            double bv = tape_load_d(b, bi);
            if (a) ((double*)a->grad)[ai] += ((double*)r->grad)[i] * bv * pow(av, bv - 1.0);
            if (b) ((double*)b->grad)[bi] += ((double*)r->grad)[i] * tape_load_d(r, i) * log(av);
            for (int k = r->rank - 1; k >= 0; k--) {
                if (++idx[k] < r->shape[k]) break; idx[k] = 0;
            }
        }
    }
}

TAPE_REGISTER_OP(OP_POW, tape_backward_pow)
