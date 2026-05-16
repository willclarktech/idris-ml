/* Batch / unbatch / one-hot — torch.
 *
 *   - tensor_batch    stack along new leading dim ([..] × N → [N, ...]).
 *   - tensor_unbatch  inverse: unbind along dim 0, contiguous() each slice.
 *   - tensor_one_hot  builds the 0/1 pattern in F64, then casts to the
 *                     requested dtype so the result honestly matches the
 *                     Idris `dt` (0/1 is exact in every dtype — float
 *                     or int — so the cast is lossless). An F32 model
 *                     gets a real F32 one-hot, an F64 model a real F64
 *                     one — no silent dtype divergence.
 */
#include "../../tensor.h"
#include "../../training/dtype_dispatch.h"
#include <torch/torch.h>
#include <cstdlib>
#include <vector>

extern c10::Device g_torch_target_device;

extern "C" TensorHandle tensor_one_hot(int* tokens, int n_tokens, int vocab_size, int dtag) {
    int total = n_tokens * vocab_size;
    // Build the 0/1 pattern in F64 on CPU (we need .accessor<> to write
    // the buffer cell-by-cell, which only works on CPU storage), then
    // cast + migrate in a single .to(opts) call. Without the migration
    // an F32-on-MPS build would see this tensor land on CPU and the
    // first downstream op (typically the loss's elementwise mul) would
    // abort with a "Expected all tensors to be on the same device" mismatch.
    auto t = torch::zeros({(int64_t)total}, torch::kFloat64);
    auto acc = t.accessor<double, 1>();
    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size)
            acc[i * vocab_size + tok] = 1.0;
    }
    /* Delegate to st_for_dtag for the kind-major dtag layout; invalid
       dtags abort there. */
    torch::ScalarType st = st_for_dtag(dtag);
    bool need_cast = st != torch::kFloat64;
    bool need_move = g_torch_target_device != at::kCPU;
    if (need_cast || need_move) {
        auto opts = torch::TensorOptions().dtype(st).device(g_torch_target_device);
        t = t.to(opts);
    }
    return from_tensor(std::move(t));
}

extern "C" TensorHandle tensor_batch(TensorHandle* handles, int count) {
    std::vector<at::Tensor> vec(count);
    for (int i = 0; i < count; i++) vec[i] = *to_tensor(handles[i]);
    return from_tensor(torch::stack(vec));
}

extern "C" TensorHandle* tensor_unbatch(TensorHandle h, int* out_count) {
    auto tensors = to_tensor(h)->unbind(0);
    *out_count = (int)tensors.size();
    auto* arr = (TensorHandle*)malloc(*out_count * sizeof(TensorHandle));
    for (int i = 0; i < *out_count; i++)
        arr[i] = from_tensor(tensors[i].contiguous());
    return arr;
}
