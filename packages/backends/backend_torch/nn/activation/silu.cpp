/* tensor_silu (Swish) for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_silu(TensorHandle h) {
	return from_tensor(torch::silu(*to_tensor(h)));
}
