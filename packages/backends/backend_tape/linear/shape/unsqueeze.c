/* linear/shape/unsqueeze.c — insert a size-1 dimension.
 *
 * Delegates to tensor_reshape (shares storage + emits an OP_RESHAPE
 * tape entry so backward flows via reshape's grad-passthrough). The
 * input's rank-N tensor becomes rank-(N+1) with a size-1 axis
 * inserted at `dim`.
 *
 * Pre-2026-06-04 this implementation only handled rank-1 input
 * (rank-1 → rank-2 with a size-1 leading axis) and silently
 * fell through to `tensor_clone(h)` for other ranks. That broke
 * DNC's `primCat2 (primUnsqueeze onesScalar 0) slicedT` chain on
 * tape — `onesScalar` is rank-0, the unsqueeze cloned it as rank-0,
 * cat2's rank-match check rejected `(0, 1)`. Now properly handles
 * rank-0 (scalar → size-1 vector) and arbitrary rank-N (insert a
 * size-1 axis at `dim`).
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../tensor.h"
#include "../../arena.h"
#include "../../../backend.h"

TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
	Tensor* t = (Tensor*)h;
	int old_rank = t->rank;
	int new_rank = old_rank + 1;
	if (dim < 0 || dim > old_rank) {
		// GCOVR_EXCL_START — out-of-range guard; abort() skips the gcov flush so
		// the forked child can't register these lines. Behavior is asserted by
		// test_unsqueeze.c::out_of_range_dim_aborts (a .signal=SIGABRT death test).
		fprintf(stderr,
		        "tensor_unsqueeze: dim=%d out of range for rank=%d "
		        "(valid: 0..%d)\n",
		        dim, old_rank, old_rank);
		// NOLINTNEXTLINE(misc-include-cleaner): macOS SDK: abort via _abort.h umbrella
		abort();
		// GCOVR_EXCL_STOP
	}
	int* new_shape = arena_alloc((size_t)new_rank * sizeof(int));
	int j = 0;
	for (int i = 0; i < new_rank; i++) {
		if (i == dim) {
			new_shape[i] = 1;
		} else {
			new_shape[i] = (old_rank == 0) ? 1 : t->shape[j++];
		}
	}
	return tensor_reshape(h, new_shape, new_rank);
}
