/* nn/activation/gelu.c — GELU (tanh approximation) (forward + backward).
 *
 * Forward via unop_elementwise dispatch (kernel in monolith's
 * .inc file references fn_gelu_d / fn_gelu_f32). Forward gets a thin
 * wrapper here; backward computes the tanh-approx derivative.
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../core/elementwise/_helpers.h"
#include "../../../backend.h"

/* GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). */
static double fn_gelu_d(double x) {
	double c = 0.7978845608028654;
	double inner = c * (x + 0.044715 * x * x * x);
	return 0.5 * x * (1.0 + tanh(inner));
}
static float fn_gelu_f32(float x) {
	float c = 0.7978845608028654f;
	float inner = c * (x + 0.044715f * x * x * x);
	return 0.5f * x * (1.0f + tanhf(inner));
}

TensorHandle tensor_gelu(TensorHandle ha) {
	Tensor* a = (Tensor*)ha;
	if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_GELU, fn_gelu_f32);
	return unop_elementwise(ha, OP_GELU, fn_gelu_d);
}

static void tape_backward_gelu(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		double c = 0.7978845608028654;
		for (int j = 0; j < a->numel; j++) {
			double x = tape_load_d(a, j);
			double inner = c * (x + 0.044715 * x * x * x);
			double t = tanh(inner);
			double dtdx = (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x);
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) * (0.5 * (1.0 + t) + 0.5 * x * dtdx));
		}
	}
}

TAPE_REGISTER_OP(OP_GELU, tape_backward_gelu)
