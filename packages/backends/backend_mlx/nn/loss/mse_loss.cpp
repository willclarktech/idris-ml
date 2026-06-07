/* tensor_mse_loss for the mlx backend.
 *
 * MSE = mean((input - target)^2). Decomposed via existing primitives. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sub(TensorHandle ha, TensorHandle hb);
extern "C" TensorHandle tensor_mul(TensorHandle ha, TensorHandle hb);
extern "C" TensorHandle tensor_mean(TensorHandle h);

extern "C" TensorHandle tensor_mse_loss(TensorHandle hinput, TensorHandle htarget) {
	TensorHandle diff = tensor_sub(hinput, htarget);
	TensorHandle sq = tensor_mul(diff, diff);
	return tensor_mean(sq);
}
