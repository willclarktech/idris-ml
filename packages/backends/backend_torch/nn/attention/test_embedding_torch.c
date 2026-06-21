/* torch embedding — the already-int64 / same-device fast arm.
 * The agnostic embedding path feeds non-kLong indices (so the .to(kLong)
 * cast arm runs); passing an int64 index on the same device exercises the
 * `? indices` fast arm of tensor_embedding + tensor_embedding_2d. */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

#define DTAG_F64 15
#define DTAG_I64 11

Test(embedding_torch, klong_index_fast_path) {
	/* weight [3,2] = rows {10,11},{20,21},{30,31}; indices int64 {0,2}. */
	double w[] = {10.0, 11.0, 20.0, 21.0, 30.0, 31.0};
	double idx[] = {0.0, 2.0};
	TensorHandle weight = tensor_create_2d_streamed(3, 2, hcopy(w, 6), 0, 0, DTAG_F64);
	TensorHandle indices = tensor_create_1d_streamed(2, hcopy(idx, 2), 0, 0, DTAG_I64);
	TensorHandle out =
	    tensor_embedding_2d(weight, indices, 2, 2); /* kLong same-device -> fast arm */
	double o[4];
	tensor_to_doubles(out, o);
	cr_assert_float_eq(o[0], 10.0, 1e-9, "row0 col0 (got %.6f)", o[0]);
	cr_assert_float_eq(o[2], 30.0, 1e-9, "row2 col0 (got %.6f)", o[2]);

	/* tensor_embedding (1d-returning) — same fast arm. */
	TensorHandle idx2 = tensor_create_1d_streamed(2, hcopy(idx, 2), 0, 0, DTAG_I64);
	TensorHandle out1 = tensor_embedding(weight, idx2, 2, 2);
	double o1[4];
	tensor_to_doubles(out1, o1);
	cr_assert_float_eq(o1[0], 10.0, 1e-9, "1d row0 col0 (got %.6f)", o1[0]);
}

#endif /* BACKEND_TORCH */
