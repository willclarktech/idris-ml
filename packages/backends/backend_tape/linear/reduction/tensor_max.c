/* linear/reduction/tensor_max.c — maximum element reducing to a scalar.
 *
 * Non-differentiable (see tensor_min for rationale).
 */

#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_max(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double m = tape_load_d(t, 0);
    for (int i = 1; i < t->numel; i++) {
        double v = tape_load_d(t, i);
        if (v > m) m = v;
    }
    return (t->dtype_tag == DT_F32) ? make_scalar_f32(m, 0) : make_scalar(m, 0);
}
