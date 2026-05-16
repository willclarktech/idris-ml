/* tensor_bmm and tensor_bmm_3x3 for the torch backend.
 *
 * tensor_bmm: a = [B, m, n], b = [n, k] (shared weight) -> [B, m, k].
 * Implemented as a loop of torch::mm because libtorch's bmm expects
 * both inputs to be 3D — we widen by stacking per-batch results.
 *
 * tensor_bmm_3x3: a, b both [B, m, n] -> [B, m, k], which IS torch::bmm. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_bmm(TensorHandle a, TensorHandle b) {
    auto& ta = *to_tensor(a);
    auto& tb = *to_tensor(b);
    int B = ta.size(0);
    std::vector<at::Tensor> results;
    for (int i = 0; i < B; i++)
        results.push_back(torch::mm(ta[i], tb));
    return from_tensor(torch::stack(results));
}

extern "C" TensorHandle tensor_bmm_3x3(TensorHandle a, TensorHandle b) {
    return from_tensor(torch::bmm(*to_tensor(a), *to_tensor(b)));
}
