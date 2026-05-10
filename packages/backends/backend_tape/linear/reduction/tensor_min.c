/* linear/reduction/tensor_min.c — minimum element reducing to a scalar.
 *
 * Phase 1b.3. Non-differentiable: result has requires_grad=0, no
 * tape_append. (A differentiable version would put grad at the
 * argmin position; idris-ml today doesn't backprop through min/max.)
 */

#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_min(TensorHandle h) {
    Tensor* t = (Tensor*)h;
    double m = tape_load_d(t, 0);
    for (int i = 1; i < t->numel; i++) {
        double v = tape_load_d(t, i);
        if (v < m) m = v;
    }
    return (t->dtype_tag == DT_F32) ? make_scalar_f32(m, 0) : make_scalar(m, 0);
}
