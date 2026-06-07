/* tensor_squeeze / tensor_unsqueeze for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_squeeze(TensorHandle h, int dim) {
	return from_tensor(to_tensor(h)->squeeze(dim));
}

extern "C" TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
	return from_tensor(to_tensor(h)->unsqueeze(dim));
}
