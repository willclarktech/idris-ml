/* tensor_log_softmax + 2d variant for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_log_softmax(TensorHandle h, int dim) {
	return from_tensor(torch::log_softmax(*to_tensor(h), dim));
}

extern "C" TensorHandle tensor_log_softmax_2d(TensorHandle h) {
	return from_tensor(torch::log_softmax(*to_tensor(h), -1));
}
