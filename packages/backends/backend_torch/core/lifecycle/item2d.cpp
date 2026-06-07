/* tensor_item_2d for the torch backend — `t[row, col]` readback. */
#include "../../tensor.h"

extern "C" double tensor_item_2d(TensorHandle h, int row, int col) {
	return to_tensor(h)->index({row, col}).cpu().item<double>();
}
