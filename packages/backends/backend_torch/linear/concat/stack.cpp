/* tensor_stack + tensor_stack_from_array for the torch backend.
 *
 * `stack` is the unowned variant — caller keeps the handle array.
 * `stack_from_array` takes ownership (frees the input array). The
 * Idris-side `tstack` smart constructor allocates the handle buffer,
 * so `_from_array` is the binding for that surface. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_stack(TensorHandle* tensors, int count, int dim) {
	std::vector<at::Tensor> vec(count);
	for (int i = 0; i < count; i++)
		vec[i] = *to_tensor(tensors[i]);
	return from_tensor(torch::stack(vec, dim));
}

extern "C" TensorHandle tensor_stack_from_array(TensorHandle* arr, int count, int dim) {
	std::vector<at::Tensor> vec(count);
	for (int i = 0; i < count; i++)
		vec[i] = *to_tensor(arr[i]);
	free(arr);
	return from_tensor(torch::stack(vec, dim));
}
