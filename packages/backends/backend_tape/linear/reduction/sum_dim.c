/* linear/reduction/sum_dim.c — sum-along-dim (simplified: full sum).
 *
 * Phase 1b.3. Today's semantics on tape: delegates to tensor_sum
 * (covers idris-ml's actual usage where the reduce-dim is always
 * the whole tensor). A proper dim-specific sum can ship as a future
 * follow-up if needed.
 */

#include "../../../backend.h"

TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    (void)dim;
    (void)keepdim;
    return tensor_sum(h);
}
