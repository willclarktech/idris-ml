/* nn/mask/expand_mask.c — broadcast a 2D mask across a batch dim.
 *
 * Non-differentiable; the result has
 * requires_grad=0 regardless of input.
 */

#include <stdlib.h>
#include <string.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_expand_mask(TensorHandle hmask, int B) {
	Tensor* mask = (Tensor*)hmask;
	int mn = mask->numel;
	int shape[] = {B, mask->shape[0], mask->shape[1]};
	if (mask->dtype_tag == DT_F32) {
		float* data = arena_alloc((size_t)B * mn * sizeof(float));
		for (int bi = 0; bi < B; bi++)
			memcpy(data + (size_t)bi * mn, mask->data, mn * sizeof(float));
		return make_tensor_arena_f32(data, B * mn, shape, 3, 0);
	}
	double* data = malloc((size_t)B * mn * sizeof(double));
	for (int bi = 0; bi < B; bi++)
		memcpy(data + (size_t)bi * mn, mask->data, mn * sizeof(double));
	Tensor* r = make_tensor(data, shape, 3, 0);
	free(data);
	return r;
}
