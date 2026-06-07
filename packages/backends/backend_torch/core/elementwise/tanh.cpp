/* tensor_tanh for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_tanh(TensorHandle h) {
	return from_tensor(torch::tanh(*to_tensor(h)));
}
