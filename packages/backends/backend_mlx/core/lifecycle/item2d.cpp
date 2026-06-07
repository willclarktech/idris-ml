/* tensor_item_2d for the mlx backend. mx::flatten produces a
 * contiguous view so the row*cols+col offset is valid even on
 * non-contiguous source matrices (e.g. transposed). */
#include "../../tensor.h"
#include "../../precision.h"

extern "C" double tensor_item_2d(TensorHandle mat, int row, int col) {
	auto t = (Tensor*)mat;
	auto flat = mx::flatten(t->data, mx::StreamOrDevice{});
	mx::eval(flat);
	int cols = t->data.shape(1);
	return mx_read_double(flat, (long)row * cols + col);
}
