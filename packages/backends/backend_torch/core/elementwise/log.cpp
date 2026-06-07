/* tensor_log for the torch backend. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_log(TensorHandle h) {
	return from_tensor(torch::log(*to_tensor(h)));
}
