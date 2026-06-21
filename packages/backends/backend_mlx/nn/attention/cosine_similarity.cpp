/* tensor_cosine_similarity for the mlx backend.
 *
 * memory=[n,m], key=[m] → result=[n]. The eps-stabilised norm pattern
 * is from the original NTM paper (Graves+Wayne+Danihelka 2014). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"
#include "../../training/autograd/op_dispatch.h"

extern "C" TensorHandle tensor_cosine_similarity_mlx_streamed(TensorHandle hmemory,
                                                              TensorHandle hkey, int dim,
                                                              int stream_tag) {
	WITH_STREAM(stream_tag);
	(void)dim;
	auto* mem = (Tensor*)hmemory;
	auto* key = (Tensor*)hkey;
	auto eps = scalar_like(1.0e-8, mem->data);
	int const n = (int)mem->data.shape(0);
	int const m = (int)mem->data.shape(1);

	auto key_2d = mx::reshape(key->data, {1, m});
	auto dots = mx::sum(mx::multiply(mem->data, key_2d), std::vector<int>{1});
	auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(mem->data), std::vector<int>{1}), eps));
	auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(key->data)), eps));
	auto result = mx::divide(dots, mx::multiply(row_norms, key_norm));

	bool const rg = mem->requires_grad || key->requires_grad;
	auto* r = new Tensor(result, rg);
	if (rg) tape_append(OP_COSINE_SIM, r, mem, key, 0);
	(void)n;
	return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cosine_similarity(TensorHandle hmemory, TensorHandle hkey, int dim) {
	return tensor_cosine_similarity_mlx_streamed(hmemory, hkey, dim, default_stream_tag());
}

static void mlx_replay_cosine_sim(std::vector<mx::array>& pool, TapeEntry& e) {
	int const out = e.result->pool_idx;
	[[maybe_unused]] auto a = (e.arg1 != nullptr) ? pool[e.arg1->pool_idx] : kF32_ZERO();
	[[maybe_unused]] auto b = (e.arg2 != nullptr) ? pool[e.arg2->pool_idx] : kF32_ZERO();
	// Inline cosine similarity forward
	int const m = (int)a.shape(1);
	auto key_2d = mx::reshape(b, {1, m});
	auto dots = mx::sum(mx::multiply(a, key_2d), std::vector<int>{1});
	auto eps = scalar_like(1.0e-8, a);
	auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(a), std::vector<int>{1}), eps));
	auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(b)), eps));
	pool[out] = mx::divide(dots, mx::multiply(row_norms, key_norm));
}
MLX_REGISTER_REPLAY(OP_COSINE_SIM, mlx_replay_cosine_sim)
