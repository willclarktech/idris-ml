/* tensor_exp for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_exp(TensorHandle h) {
	return from_tensor(torch::exp(*to_tensor(h)));
}
