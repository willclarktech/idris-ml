/* core/elementwise/softplus.c — softplus log(1+exp(x)) (forward + backward).
 *
 * Numerically-stable forward: x for x>30, exp(x) for x<-30,
 * else log(1+exp(x)). d softplus(x)/dx = sigmoid(x) = 1/(1+exp(-x)).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "_helpers.h"
#include "../../../backend.h"

static double fn_softplus_d(double x) {
	if (x > 30.0) return x;
	if (x < -30.0) return exp(x);
	return log(1.0 + exp(x));
}
static float fn_softplus_f32(float x) {
	if (x > 30.0f) return x;
	if (x < -30.0f) return expf(x);
	return logf(1.0f + expf(x));
}

TensorHandle tensor_softplus(TensorHandle ha) {
	Tensor* a = (Tensor*)ha;
	if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_SOFTPLUS, fn_softplus_f32);
	return unop_elementwise(ha, OP_SOFTPLUS, fn_softplus_d);
}

static void tape_backward_softplus(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		for (int j = 0; j < a->numel; j++) {
			double s = 1.0 / (1.0 + exp(-tape_load_d(a, j)));
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) * s);
		}
	}
}

TAPE_REGISTER_OP(OP_SOFTPLUS, tape_backward_softplus)
