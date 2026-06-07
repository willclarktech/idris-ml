/* tensor_leaky_relu for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_leaky_relu(TensorHandle h, double alpha) {
	return from_tensor(torch::leaky_relu(*to_tensor(h), alpha));
}
