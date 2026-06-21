/* tensor_batch_norm for the torch backend.
 *
 * Reshape to [1, C, spatial] for torch::batch_norm (which expects
 * [N,C,...]), then flatten back. cudnn is disabled so the build does
 * not depend on libcudnn being installed. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_batch_norm(TensorHandle hinput, TensorHandle hgamma,
                                          TensorHandle hbeta, TensorHandle hrunning_mean,
                                          TensorHandle hrunning_var, int channels, int spatial,
                                          int training, double momentum, double eps) {
	auto& inp = *to_tensor(hinput);
	auto& gamma = *to_tensor(hgamma);
	auto& beta = *to_tensor(hbeta);
	auto& rm = *to_tensor(hrunning_mean);
	auto& rv = *to_tensor(hrunning_var);

	auto inp_3d = inp.reshape({1, (int64_t)channels, (int64_t)spatial});
	auto out = torch::batch_norm(inp_3d, gamma, beta, rm, rv,
	                             /*training=*/training != 0, momentum, eps,
	                             /*cudnn_enabled=*/false);
	return from_tensor(out.reshape({-1}));
}
