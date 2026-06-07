/* tensor_stack + tensor_stack_from_array for the mlx backend.
 *
 * Both record OP_STACK with scalar_arg=dim and meta=[input pool indices].
 * Replay reads dim from scalar_arg so non-zero stack dims backprop
 * correctly. `_from_array` additionally owns the input handle array. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../training/autograd/op_dispatch.h"
#include "../../precision.h"

static TensorHandle stack_impl(TensorHandle* tensors, int count, int dim) {
	std::vector<mx::array> arrs;
	bool rg = false;
	for (int i = 0; i < count; i++) {
		auto t = (Tensor*)tensors[i];
		arrs.push_back(t->data);
		if (t->requires_grad) rg = true;
	}
	auto r = new Tensor(mx::stack(arrs, dim), rg);
	if (rg) {
		int idx = tape_append(OP_STACK, r, nullptr, nullptr, (double)dim);
		auto* indices = new std::vector<int>();
		for (int i = 0; i < count; i++)
			indices->push_back(((Tensor*)tensors[i])->pool_idx);
		tape[idx].meta = (void*)indices;
	}
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_stack_mlx_streamed(TensorHandle* tensors, int count, int dim,
                                                  int stream_tag) {
	WITH_STREAM(stream_tag);
	return stack_impl(tensors, count, dim);
}

extern "C" TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
	return tensor_stack_mlx_streamed(tensors, count, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
	auto r = stack_impl(arr, count, dim);
	free(arr);
	return r;
}

static void mlx_replay_stack(std::vector<mx::array>& pool, TapeEntry& e) {
	int out = e.result->pool_idx;
	[[maybe_unused]] auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();
	auto* indices = (std::vector<int>*)e.meta;
	if (indices) {
		std::vector<mx::array> arrs;
		for (int idx : *indices)
			arrs.push_back(pool[idx]);
		pool[out] = mx::stack(arrs, (int)e.scalar_arg);
	}
}
MLX_REGISTER_REPLAY(OP_STACK, mlx_replay_stack)
