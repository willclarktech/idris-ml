/* tensor_cat / tensor_cat_from_array / tensor_cat2 for the torch
 * backend. See stack.cpp for the unowned / owned variant distinction. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_cat(TensorHandle* tensors, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(tensors[i]);
    return from_tensor(torch::cat(vec, dim));
}

extern "C" TensorHandle tensor_cat_from_array(TensorHandle* arr, int count, int dim) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(arr[i]);
    free(arr);
    return from_tensor(torch::cat(vec, dim));
}

extern "C" TensorHandle tensor_cat2(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::cat({*to_tensor(a), *to_tensor(b)}, 0));
}
