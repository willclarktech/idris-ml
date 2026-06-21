/* mlx conv1d replay-meta teardown (tape.cpp OP_CONV1D free-arm).
 * A grad-tracked conv1d records OP_CONV1D + Conv1DReplayMeta; backend_reset_for_eval
 * -> tape_reset frees that meta. */
#include <criterion/criterion.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

#define DTAG_F64 15

Test(conv1d_mlx, reset_frees_replay_meta) {
	double in[3] = {1.0, 2.0, 3.0};
	double k[2] = {1.0, 1.0};
	int ksh[3] = {1, 1, 2}; /* [outC, inC, kL] */
	TensorHandle input = tensor_create_2d_streamed(1, 3, hcopy(in, 3), /*rg=*/1, 0, DTAG_F64);
	TensorHandle kernel = tensor_create_streamed(hcopy(k, 2), ksh, 3, /*rg=*/0, 0, DTAG_F64);
	TensorHandle out = tensor_conv1d(input, kernel, (TensorHandle)0, /*pad=*/0, /*stride=*/1);
	cr_assert_eq(tensor_numel(out), 2, "conv1d [1,3] k[1,1,2] -> [1,2]");
	backend_reset_for_eval(); /* tape_reset -> frees the OP_CONV1D replay meta */
	cr_assert(1, "no crash on teardown");
}

#endif /* BACKEND_MLX */
