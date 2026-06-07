/* tensor_sigmoid for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sigmoid(TensorHandle h) {
	return from_tensor(torch::sigmoid(*to_tensor(h)));
}
