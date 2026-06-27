/* mlx-only Criterion suite for tensor_embedding / tensor_embedding_2d
 * BACKWARD path.
 *
 * The common (cross-backend) embedding suite drives the forward shape +
 * gather of both the flat (tensor_embedding) and 2D (tensor_embedding_2d)
 * variants, but never registers the weight as a param with requires_grad,
 * so on mlx the `if (weight->requires_grad) { tape_append(...) }` branch
 * (embedding.cpp lines 23-26 flat, 48-51 2D) and the OP_EMBEDDING /
 * OP_EMBEDDING_2D mlx replay closures (lines 60-78) stay uncovered.
 *
 * These tests register the weight as a learnable param and run a full
 * backward, exercising the grad tape-append AND the mlx replay closures.
 * The gradient of sum(embedding(W, idx)) w.r.t. W is a count matrix:
 * dL/dW[r, :] = (number of times row r appears in idx).
 *
 * mlx-only because the replay closures (mlx_replay_embedding* via
 * MLX_REGISTER_REPLAY) are the mlx lazy-graph re-execution path; tape and
 * torch carry their own distinct embedding backward.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* 2D variant backward: covers embedding.cpp lines 48-51 (requires_grad
 * tape_append in tensor_embedding_2d) + lines 71-78 (mlx_replay_embedding_2d).
 *
 * W is [4, 3]; idx = [2, 0, 2]. Row 2 appears twice, row 0 once, rows 1
 * and 3 never. loss = sum(embedding_2d(W, idx)) so dL/dW[r, j] = count(r):
 *   row0 -> 1, row1 -> 0, row2 -> 2, row3 -> 0  (for every column j). */
Test(mlx_nn_attention_embedding, backward_2d_grad_is_row_counts) {
	param_clear();
	double w_d[12] = {
	    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
	};
	double idx_d[3] = {2.0, 0.0, 2.0};
	TensorHandle w = tensor_create_param_2d_f64(4, 3, hcopy(w_d, 12));
	param_register("emb.weight", w);
	TensorHandle idx = tensor_create_1d_f64(3, hcopy(idx_d, 3), 0);

	TensorHandle r = tensor_embedding_2d(w, idx, 3, 3);
	cr_assert_eq(tensor_dim(r), 2, "embedding_2d output should be rank 2");
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);

	/* expected grad row counts, broadcast over 3 columns */
	double expected[12] = {
	    1.0, 1.0, 1.0, /* row0: appears once */
	    0.0, 0.0, 0.0, /* row1: never */
	    2.0, 2.0, 2.0, /* row2: twice */
	    0.0, 0.0, 0.0, /* row3: never */
	};
	for (int k = 0; k < 12; k++) {
		cr_assert_float_eq(param_grad_item_at(0, k), expected[k], TEST_TOL_RELAXED,
		                   "embedding_2d grad[%d] expected %.1f got %.6f", k, expected[k],
		                   param_grad_item_at(0, k));
	}
}

/* Flat variant backward: covers embedding.cpp lines 22-26 (requires_grad
 * tape_append in tensor_embedding) + lines 60-69 (mlx_replay_embedding,
 * the flatten replay). Same fixture / same expected grad — the flatten
 * doesn't change which rows are gathered. */
Test(mlx_nn_attention_embedding, backward_flat_grad_is_row_counts) {
	param_clear();
	double w_d[12] = {
	    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
	};
	double idx_d[3] = {2.0, 0.0, 2.0};
	TensorHandle w = tensor_create_param_2d_f64(4, 3, hcopy(w_d, 12));
	param_register("emb.weight", w);
	TensorHandle idx = tensor_create_1d_f64(3, hcopy(idx_d, 3), 0);

	TensorHandle r = tensor_embedding(w, idx, 3, 3);
	cr_assert_eq(tensor_dim(r), 1, "flat embedding output should be rank 1");
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);

	double expected[12] = {
	    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
	};
	for (int k = 0; k < 12; k++) {
		cr_assert_float_eq(param_grad_item_at(0, k), expected[k], TEST_TOL_RELAXED,
		                   "embedding flat grad[%d] expected %.1f got %.6f", k, expected[k],
		                   param_grad_item_at(0, k));
	}
}

#endif /* BACKEND_MLX */
