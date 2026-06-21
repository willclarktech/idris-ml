/* In-place scalar subtract for the mlx backend.
 *
 * Used by SAC's target-network polyak blend. mlx doesn't have a true
 * in-place subtract (its arrays are immutable), so this just rewrites
 * `h->data` to the new array — the surface still looks in-place to
 * Idris callers (the handle is unchanged). */
#include "../tensor.h"
#include "../precision.h"

extern "C" TensorHandle tensor_subtract_scalar_inplace(TensorHandle h, double val) {
	auto* t = (Tensor*)h;
	t->data = mx::subtract(t->data, scalar_like(val, t->data));
	return h;
}
