/* tensor_cosine_similarity for the mlx backend.
 *
 * memory=[n,m], key=[m] → result=[n]. The eps-stabilised norm pattern
 * is from the original NTM paper (Graves+Wayne+Danihelka 2014). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_cosine_similarity_mlx_streamed(TensorHandle hmemory, TensorHandle hkey, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)dim;
    auto mem = (Tensor*)hmemory; auto key = (Tensor*)hkey;
    auto eps = scalar_like(1.0e-8, mem->data);
    int n = (int)mem->data.shape(0);
    int m = (int)mem->data.shape(1);

    auto key_2d = mx::reshape(key->data, {1, m});
    auto dots = mx::sum(mx::multiply(mem->data, key_2d), std::vector<int>{1});
    auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(mem->data), std::vector<int>{1}), eps));
    auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(key->data)), eps));
    auto result = mx::divide(dots, mx::multiply(row_norms, key_norm));

    bool rg = mem->requires_grad || key->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_COSINE_SIM, r, mem, key, 0);
    (void)n;
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cosine_similarity(TensorHandle hmemory, TensorHandle hkey, int dim) {
    return tensor_cosine_similarity_mlx_streamed(hmemory, hkey, dim, default_stream_tag());
}
