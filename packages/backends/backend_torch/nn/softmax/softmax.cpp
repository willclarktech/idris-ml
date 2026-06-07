/* tensor_softmax + fixed-rank variants for the torch backend.
 *
 * The _2d / _3d aliases are bound to type-safe Idris surfaces that
 * fix the reduction axis to the last dim. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_softmax(TensorHandle h, int dim) {
	return from_tensor(torch::softmax(*to_tensor(h), dim));
}

extern "C" TensorHandle tensor_softmax_2d(TensorHandle h) {
	return from_tensor(torch::softmax(*to_tensor(h), -1));
}

extern "C" TensorHandle tensor_softmax_3d(TensorHandle h) {
	return from_tensor(torch::softmax(*to_tensor(h), -1));
}
