/* Batch / unbatch / one-hot — mlx.
 *
 *   - tensor_batch    stack along new leading dim ([..] × N → [N, ...]).
 *   - tensor_unbatch  inverse: tensor_select(dim=0, i) for each slice;
 *                     OP_SELECT is replayed at dim=0, so backward
 *                     reconstructs the same gathers.
 *   - tensor_one_hot  builds the 0/1 pattern in F32 or F64 (per dtag) —
 *                     mlx admits no other storage dtypes (Metal-F32,
 *                     CPU-F64). dtag 15 = F64, else F32. The Idris
 *                     `Compatible` gate prevents other dtags reaching
 *                     here; the F32 fallback is a defence-in-depth.
 */
#include "../../tensor.h"
#include "../../precision.h"
#include <cstdlib>
#include <vector>

extern "C" TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim);
extern "C" TensorHandle tensor_select(TensorHandle h, int dim, int index);

extern "C" TensorHandle tensor_batch(TensorHandle* handles, int count) {
	/* Batch [...] tensors -> [count, ...] = stack along new dim 0 */
	return tensor_stack(handles, count, 0);
}

extern "C" TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
	auto* t = (Tensor*)h;
	int const B = (int)t->data.shape(0);
	*out_count = B;
	auto* arr = (TensorHandle*)malloc(B * sizeof(TensorHandle));
	/* tensor_select picks dim=0 index=i and removes that dim — that is exactly
	   one slice of the unbatched output. OP_SELECT is already replayed at dim=0,
	   so backward replay reconstructs the same gathers. */
	for (int i = 0; i < B; i++) {
		arr[i] = tensor_select((TensorHandle)t, 0, i);
	}
	return arr;
}

extern "C" TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
	// Create one-hot encoded 1D tensor in the requested dtype so the result
	// honestly matches the Idris `dt` (0/1 is exact in every dtype). mlx
	// admits F32/F64/BF16/F16 per the Compatible table; under the kind-major
	// dtag layout dtag 13 = F16, dtag 14 = F32, dtag 15 = F64, dtag 17 = BF16.
	// Any other dtag would fail the Compatible gate Idris-side; this routes
	// to F32 as a sentinel so a stray call doesn't silently return F64.
	size_t const total = (size_t)n_tokens * vocab_size;
	std::vector<double> data(total, 0.0);
	for (int i = 0; i < n_tokens; i++) {
		int const tok = tokens[i];
		if (tok >= 0 && tok < vocab_size) data[(size_t)i * vocab_size + tok] = 1.0;
	}
	mx::Shape const sh = {(int)total};
	mx::Dtype dt = mx::float32;
	switch (dtag) {
	case 15:
		dt = mx::float64;
		break;
	case 17:
		dt = mx::bfloat16;
		break;
	case 13:
		dt = mx::float16;
		break;
	default:
		break;
	}
	auto* t = new Tensor(mx_array_from_doubles(data.data(), sh, dt), false);
	free(tokens);
	return (TensorHandle)t;
}
