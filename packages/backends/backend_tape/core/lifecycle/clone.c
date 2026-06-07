/* core/lifecycle/clone.c — Deep-copy a tensor (new handle, same data).
 *
 * Dtype-aware: F32 clones honour the float storage,
 * non-F32 (DT_F64 + lingua-franca inference dtypes) copy double-wide.
 * The clone is arena-allocated + does NOT requires_grad (a fresh
 * autograd node would need its own tape_append; callers requesting
 * a grad-tracked clone go via tensor_create + their own forward op).
 */

#include <string.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_clone(TensorHandle h) {
	Tensor* t = (Tensor*)h;
	if (t->rank == 0) {
		double v = tape_load_d(t, 0);
		return (t->dtype_tag == DT_F32) ? make_scalar_f32(v, 0) : make_scalar(v, 0);
	}
	if (t->dtype_tag == DT_F32) {
		float* data = arena_alloc(t->numel * sizeof(float));
		memcpy(data, t->data, t->numel * sizeof(float));
		return make_tensor_arena_f32(data, t->numel, t->shape, t->rank, 0);
	}
	return make_tensor(t->data, t->shape, t->rank, 0);
}
