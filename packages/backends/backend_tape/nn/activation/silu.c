/* nn/activation/silu.c — SiLU/Swish (forward + backward).
 *
 * Forward: x * sigmoid(x).
 * Backward: sigmoid(x) * (1 + x * (1 - sigmoid(x))).
 */

#include <math.h>
#include "../../tape.h"
#include "../../arena.h"
#include "../../tensor.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../core/elementwise/_helpers.h"
#include "../../../backend.h"

static double fn_silu_d(double x) {
	return x / (1.0 + exp(-x));
}
static float fn_silu_f32(float x) {
	return x / (1.0f + expf(-x));
}

TensorHandle tensor_silu(TensorHandle ha) {
	Tensor* a = (Tensor*)ha;
	if (a->dtype_tag == DT_F32) return unop_elementwise_f32_disp(ha, OP_SILU, fn_silu_f32);
	return unop_elementwise(ha, OP_SILU, fn_silu_d);
}

static void tape_backward_silu(TapeEntry* e) {
	Tensor* r = e->result;
	Tensor* a = e->arg1;
	if (a) {
		ensure_grad(a);
		for (int j = 0; j < a->numel; j++) {
			double x = tape_load_d(a, j);
			double s = 1.0 / (1.0 + exp(-x));
			tape_grad_add_d(a, j, tape_grad_load_d(r, j) * s * (1.0 + x * (1.0 - s)));
		}
	}
}

TAPE_REGISTER_OP(OP_SILU, tape_backward_silu)
