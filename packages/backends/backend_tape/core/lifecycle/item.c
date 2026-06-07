/* core/lifecycle/item.c — Read a 0-rank tensor's scalar value.
 *
 * Dtype-aware via tape_load_d (handles F32 + F64;
 * lingua-franca inference dtypes pass through their stored double).
 */

#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

double tensor_item(TensorHandle h) {
	return tape_load_d((Tensor*)h, 0);
}
