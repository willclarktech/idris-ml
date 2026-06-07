/* tensor_sqrt for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_sqrt(TensorHandle h) {
	return from_tensor(torch::sqrt(*to_tensor(h)));
}
