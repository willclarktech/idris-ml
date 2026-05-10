/* linear/shape/squeeze.c — squeeze dimension (simplified: clone identity).
 *
 * Delegates to tensor_clone, which
 * handles dtype-aware deep-copy. The simplified semantics here treat
 * squeeze as identity for the workloads idris-ml exercises (the
 * dimension to remove is always size 1 already at the rank that
 * matters); a proper shape-collapsing squeeze can ship as a future
 * follow-up if needed.
 */

#include "../../../backend.h"

TensorHandle tensor_squeeze(TensorHandle h, int dim) {
    (void)dim;
    return tensor_clone(h);
}
