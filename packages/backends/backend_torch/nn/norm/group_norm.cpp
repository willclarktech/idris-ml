/* tensor_group_norm for the torch backend.
 *
 * Reshape to [1, C, spatial] for torch::group_norm (which expects
 * [N,C,...]), then flatten back. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_group_norm(TensorHandle hinput, TensorHandle hgamma,
                                          TensorHandle hbeta, int numGroups, int channels,
                                          int spatial, double eps) {
	auto& inp = *to_tensor(hinput);
	auto& gamma = *to_tensor(hgamma);
	auto& beta = *to_tensor(hbeta);
	auto inp_3d = inp.reshape({1, (int64_t)channels, (int64_t)spatial});
	auto out = torch::group_norm(inp_3d, numGroups, gamma, beta, eps);
	return from_tensor(out.reshape({-1}));
}
