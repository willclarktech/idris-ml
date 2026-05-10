/* core/lifecycle/item1d.c — read element [idx] of a 1D tensor as Double.
 *
 * Phase 1a.10. Dtype-aware via tape_load_d (handles F32 + F64 +
 * lingua-franca inference dtypes). No backward — pure host read.
 */

#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

double tensor_item_1d(TensorHandle h, int idx) {
    return tape_load_d((Tensor*)h, idx);
}
