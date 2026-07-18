/* Dtype-scaffolding Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"

#if defined(BACKEND_TAPE)
Test(dtype_scaffolding, tape_dtype_storage) {
	param_clear();

	/* F32 create + dtype + readback (value-exact: these all fit f32). */
	double fv[] = {1.5, -2.25, 3.0};
	TensorHandle f32t = tensor_create_1d_streamed(3, hcopy(fv, 3), 0, 0, 14);
	ASSERT_TRUE("tape F32 dtype is F32", strcmp(tensor_dtype_name(f32t), "F32") == 0);
	double fout[3];
	tensor_to_doubles(f32t, fout);
	ASSERT_NEAR("tape F32 readback[1]", fout[1], -2.25, 1e-6);

	/* I32 create + dtype + readback (integer-valued). */
	double iv[] = {1.0, 2.0, 3.0};
	TensorHandle i32t = tensor_create_1d_streamed(3, hcopy(iv, 3), 0, 0, 10);
	ASSERT_TRUE("tape I32 dtype is I32", strcmp(tensor_dtype_name(i32t), "I32") == 0);
	double iout[3];
	tensor_to_doubles(i32t, iout);
	ASSERT_NEAR("tape I32 readback[2]", iout[2], 3.0, 1e-10);

	/* BF16 create + dtype + readback (bf16 tolerance). */
	double bv[] = {1.5, 2.25, -0.5};
	TensorHandle bf = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 17);
	ASSERT_TRUE("tape BF16 dtype is BF16", strcmp(tensor_dtype_name(bf), "BF16") == 0);
	double bout[3];
	tensor_to_doubles(bf, bout);
	ASSERT_NEAR("tape BF16 readback[0]", bout[0], 1.5, 1e-2);

	/* Cast F64 -> F32 -> F64 round-trip (value-exact for these). */
	double dv[] = {1.5, 2.25};
	TensorHandle d0 = tensor_create_1d_streamed(2, hcopy(dv, 2), 0, 0, 15);
	TensorHandle to_f32 = tensor_cast_dtype_streamed(d0, 0, 14);
	ASSERT_TRUE("tape cast F64->F32", strcmp(tensor_dtype_name(to_f32), "F32") == 0);
	TensorHandle back = tensor_cast_dtype_streamed(to_f32, 0, 15);
	ASSERT_TRUE("tape cast F32->F64", strcmp(tensor_dtype_name(back), "F64") == 0);
	double rt[2];
	tensor_to_doubles(back, rt);
	ASSERT_NEAR("tape F32 roundtrip[0]", rt[0], 1.5, 1e-6);

	/* Cast F64 -> I32 -> F64 round-trip (integer-valued). */
	double ev[] = {4.0, 5.0, 6.0};
	TensorHandle e0 = tensor_create_1d_streamed(3, hcopy(ev, 3), 0, 0, 15);
	TensorHandle to_i32 = tensor_cast_dtype_streamed(e0, 0, 10);
	ASSERT_TRUE("tape cast F64->I32", strcmp(tensor_dtype_name(to_i32), "I32") == 0);
	double iback[3];
	tensor_to_doubles(to_i32, iback);
	ASSERT_NEAR("tape I32 cast readback[1]", iback[1], 5.0, 1e-10);

	param_clear();
}
#endif

#if defined(BACKEND_TAPE)
Test(dtype_scaffolding, tape_f32_gradcheck_oracle) {
	/* Rung 1: scalar + elementwise. y = (w + x) * (w - x); L = sum(y).
	   Analytic: dL/dw = 2*w. Chains through add/sub/mul + sum. */
#ifndef TAPE_F32_SKIP_ELEMENTWISE
	{
		double wv[] = {1.5, -0.25, 0.5};
		double xv[] = {0.5, 0.75, -1.0};
		double y_f64[3], y_f32[3];
		double g_f64[3], g_f32[3];

		/* F64 reference */
		param_clear();
		TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
		param_register("w", w64);
		TensorHandle x64 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 15);
		TensorHandle add64 = tensor_add(w64, x64);
		TensorHandle sub64 = tensor_sub(w64, x64);
		TensorHandle y64 = tensor_mul(add64, sub64);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 3; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		/* F32 path: same numeric chain, F32-tagged inputs. */
		TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
		param_register("w", w32);
		TensorHandle x32 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 14);
		TensorHandle add32 = tensor_add(w32, x32);
		TensorHandle sub32 = tensor_sub(w32, x32);
		TensorHandle y32 = tensor_mul(add32, sub32);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 3; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("elementwise: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 3; i++) {
			char m[64];
			snprintf(m, sizeof m, "elementwise: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
			snprintf(m, sizeof m, "elementwise: w.grad_f32[%d] ~ w.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: elementwise (TAPE_F32_SKIP_ELEMENTWISE, Phase 3 step 7)\n");
#endif

	/* Rung 2: matmul / linear / reductions. y = W @ x; L = sum(y).
	   Analytic: dL/dW[i,j] = x[j], dL/dx[j] = sum_i W[i,j]. */
#ifndef TAPE_F32_SKIP_MATMUL
	{
		double Wv[] = {1.0, 0.5, -0.25, 0.75, -0.5, 0.25}; /* [2,3] */
		double xv[] = {0.5, -1.0, 0.25};                   /* [3]   */
		double y_f64[2], y_f32[2];
		double gW_f64[6], gW_f32[6];

		param_clear();
		TensorHandle W64 = tensor_create_param_2d_streamed(2, 3, hcopy(Wv, 6), 0, 15);
		param_register("W", W64);
		TensorHandle x64 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 15);
		TensorHandle y64 = tensor_mv(W64, x64);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 6; i++)
			gW_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle W32 = tensor_create_param_2d_streamed(2, 3, hcopy(Wv, 6), 0, 14);
		param_register("W", W32);
		TensorHandle x32 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 14);
		TensorHandle y32 = tensor_mv(W32, x32);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 6; i++)
			gW_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("matmul: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 2; i++) {
			char m[64];
			snprintf(m, sizeof m, "matmul: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
		}
		for (int i = 0; i < 6; i++) {
			char m[64];
			snprintf(m, sizeof m, "matmul: W.grad_f32[%d] ~ W.grad_f64", i);
			ASSERT_NEAR(m, gW_f32[i], gW_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: matmul (TAPE_F32_SKIP_MATMUL, Phase 3 step 8)\n");
#endif

	/* Rung 3: softmax / norm / rnn / conv. softmax forward+backward
	   on a 1D logits vector; L = sum(softmax(w)). Analytic for
	   softmax-then-sum: dL/dw = 0 (sum of softmax is 1, derivative
	   is 0), but we exercise the chain numerically — F32 must match
	   F64 within tol and propagate the tag. */
#ifndef TAPE_F32_SKIP_NORM
	{
		double wv[] = {0.25, -0.5, 1.0, 0.75};
		double y_f64[4], y_f32[4];
		double g_f64[4], g_f32[4];

		param_clear();
		TensorHandle w64 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 15);
		param_register("w", w64);
		TensorHandle y64 = tensor_softmax(w64, 0);
		tensor_to_doubles(y64, y_f64);
		/* Use a non-trivial loss so grad isn't analytically zero:
		   L = sum(softmax(w) * c) for fixed c = [1, 2, 3, 4]. */
		double cv[] = {1.0, 2.0, 3.0, 4.0};
		TensorHandle c64 = tensor_create_1d_streamed(4, hcopy(cv, 4), 0, 0, 15);
		TensorHandle wt64 = tensor_mul(y64, c64);
		tensor_backward(tensor_sum(wt64));
		for (int i = 0; i < 4; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle w32 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 14);
		param_register("w", w32);
		TensorHandle y32 = tensor_softmax(w32, 0);
		tensor_to_doubles(y32, y_f32);
		TensorHandle c32 = tensor_create_1d_streamed(4, hcopy(cv, 4), 0, 0, 14);
		TensorHandle wt32 = tensor_mul(y32, c32);
		tensor_backward(tensor_sum(wt32));
		for (int i = 0; i < 4; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("softmax: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 4; i++) {
			char m[64];
			snprintf(m, sizeof m, "softmax: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
			snprintf(m, sizeof m, "softmax: w.grad_f32[%d] ~ w.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: softmax/norm (TAPE_F32_SKIP_NORM, Phase 3 step 9)\n");
#endif

	/* Rung 4: optimizer step. One SGD step on an F32 param vs F64
	   param with identical lr and (post-backward) grad. The F32
	   param's data must (a) keep its F32 tag, and (b) round to
	   F32 precision after the step (assert data[i] is bit-exact
	   under (double)(float)data[i] cast). */
#ifndef TAPE_F32_SKIP_OPTIMIZER
	{
		double wv[] = {0.5, 1.5, -0.25};
		double xv[] = {1.0 / 3.0, -2.0 / 7.0, 5.0 / 11.0}; /* irrational in F32 */
		double w_f64_after[3], w_f32_after[3];

		param_clear();
		TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
		param_register("w", w64);
		TensorHandle x64 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 15);
		TensorHandle dot64 = tensor_dot(w64, x64); /* L = w·x */
		tensor_backward(dot64);
		OptimizerHandle opt64 = optimizer_create_sgd(0.01);
		optimizer_step(opt64);
		tensor_to_doubles(w64, w_f64_after);
		optimizer_free(opt64);
		param_clear();

		TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
		param_register("w", w32);
		TensorHandle x32 = tensor_create_1d_streamed(3, hcopy(xv, 3), 0, 0, 14);
		TensorHandle dot32 = tensor_dot(w32, x32);
		tensor_backward(dot32);
		OptimizerHandle opt32 = optimizer_create_sgd(0.01);
		optimizer_step(opt32);
		tensor_to_doubles(w32, w_f32_after);
		optimizer_free(opt32);

		ASSERT_TRUE("optimizer: F32 param keeps F32 tag after step",
		            strcmp(tensor_dtype_name(w32), "F32") == 0);
		for (int i = 0; i < 3; i++) {
			char m[64];
			snprintf(m, sizeof m, "optimizer: w_f32[%d] ~ w_f64 after step", i);
			ASSERT_NEAR(m, w_f32_after[i], w_f64_after[i], 1e-5);
			/* F32-exact: under real F32 storage, the updated value
			   is representable as float — round-trip through float
			   is bit-identical. Today's lingua-franca writes a raw
			   F64 result back, so this fires RED. */
			snprintf(m, sizeof m, "optimizer: w_f32[%d] is F32-exact after step", i);
			ASSERT_TRUE(m, w_f32_after[i] == (double)(float)w_f32_after[i]);
		}
		param_clear();
	}
#else
	printf("rung skipped: optimizer (TAPE_F32_SKIP_OPTIMIZER, Phase 3 step 10)\n");
#endif
}
#endif

#if defined(BACKEND_TAPE)
Test(dtype_scaffolding, tape_f32_non_elementwise_coverage) {
	/* Batch 1 Group A: scalar ops. y = op(w, s); L = sum(y).
	      add_scalar:  dL/dw[i] = 1
	      mul_scalar:  dL/dw[i] = s
	      clamp_min:   dL/dw[i] = (w[i] > min ? 1 : 0)
	   tape_load_d in OP_CLAMP_MIN backward picks up F32 input data. */
	{
		double wv[] = {1.5, -0.25, 0.5};
		double y_f64[3], y_f32[3];
		double g_f64[3], g_f32[3];
		double s_add = 0.75, s_mul = -1.5, s_clamp = 0.0;

		/* add_scalar */
		{
			/* F64 reference */
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle y64 = tensor_add_scalar(w64, s_add);
			tensor_to_doubles(y64, y_f64);
			tensor_backward(tensor_sum(y64));
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			/* F32 path */
			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle y32 = tensor_add_scalar(w32, s_add);
			tensor_to_doubles(y32, y_f32);
			tensor_backward(tensor_sum(y32));
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("add_scalar: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(y32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "add_scalar: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
				snprintf(m, sizeof m, "add_scalar: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		/* mul_scalar */
		{
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle y64 = tensor_mul_scalar(w64, s_mul);
			tensor_to_doubles(y64, y_f64);
			tensor_backward(tensor_sum(y64));
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle y32 = tensor_mul_scalar(w32, s_mul);
			tensor_to_doubles(y32, y_f32);
			tensor_backward(tensor_sum(y32));
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("mul_scalar: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(y32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "mul_scalar: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
				snprintf(m, sizeof m, "mul_scalar: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		/* clamp_min — backward reads input data (sign check), so this
		   also exercises OP_CLAMP_MIN's `tape_load_d` swap. */
		{
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle y64 = tensor_clamp_min(w64, s_clamp);
			tensor_to_doubles(y64, y_f64);
			tensor_backward(tensor_sum(y64));
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle y32 = tensor_clamp_min(w32, s_clamp);
			tensor_to_doubles(y32, y_f32);
			tensor_backward(tensor_sum(y32));
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("clamp_min: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(y32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "clamp_min: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
				snprintf(m, sizeof m, "clamp_min: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}
	}

	/* Batch 1 Group B: extra unary activations. y = act(w); L = sum(y).
	   Exercises tape_load_d in OP_LEAKY_RELU / OP_SILU backward (both
	   read a->data for the derivative). tensor_silu and tensor_softplus
	   previously single-dispatched through unop_elementwise; both move
	   onto the TAPE_UNOP_DISPATCH macro with paired fn_*_f32 helpers. */
	{
		double wv[] = {0.5, -1.0, 1.5};
		double y_f64[3], y_f32[3];
		double g_f64[3], g_f32[3];

#define RUN_UNARY_F32_VS_F64(label, opcall)                                                        \
	do {                                                                                           \
		param_clear();                                                                             \
		TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);                \
		param_register("w", w64);                                                                  \
		TensorHandle y64 = opcall(w64);                                                            \
		tensor_to_doubles(y64, y_f64);                                                             \
		tensor_backward(tensor_sum(y64));                                                          \
		for (int i = 0; i < 3; i++)                                                                \
			g_f64[i] = param_grad_item_at(0, i);                                                   \
		param_clear();                                                                             \
		TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);                \
		param_register("w", w32);                                                                  \
		TensorHandle y32 = opcall(w32);                                                            \
		tensor_to_doubles(y32, y_f32);                                                             \
		tensor_backward(tensor_sum(y32));                                                          \
		for (int i = 0; i < 3; i++)                                                                \
			g_f32[i] = param_grad_item_at(0, i);                                                   \
		ASSERT_TRUE(label ": F32 output propagates F32 tag",                                       \
		            strcmp(tensor_dtype_name(y32), "F32") == 0);                                   \
		for (int i = 0; i < 3; i++) {                                                              \
			char m[64];                                                                            \
			snprintf(m, sizeof m, label ": y_f32[%d] ~ y_f64", i);                                 \
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);                                              \
			snprintf(m, sizeof m, label ": w.grad_f32[%d] ~ w.grad_f64", i);                       \
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);                                              \
		}                                                                                          \
		param_clear();                                                                             \
	} while (0)

		/* leaky_relu — has its own alpha-arg dispatcher (not unop_elementwise). */
		{
			double alpha = 0.1;
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle y64 = tensor_leaky_relu(w64, alpha);
			tensor_to_doubles(y64, y_f64);
			tensor_backward(tensor_sum(y64));
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle y32 = tensor_leaky_relu(w32, alpha);
			tensor_to_doubles(y32, y_f32);
			tensor_backward(tensor_sum(y32));
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("leaky_relu: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(y32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "leaky_relu: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
				snprintf(m, sizeof m, "leaky_relu: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		RUN_UNARY_F32_VS_F64("silu", tensor_silu);
		RUN_UNARY_F32_VS_F64("softplus", tensor_softplus);

#undef RUN_UNARY_F32_VS_F64
	}

	/* Batch 1 Group C: tensor_log_softmax. Same shape as softmax (rung 3)
	   but the backward case reads r->data for the d/dx exp(output) factor —
	   OP_LOG_SOFTMAX gets a tape_load_d swap. Loss is sum(log_softmax(w))
	   directly (no auxiliary tensor) so the chain stays within one dtype. */
	{
		double wv[] = {0.25, -0.5, 1.0, 0.75};
		double y_f64[4], y_f32[4], g_f64[4], g_f32[4];

		param_clear();
		TensorHandle w64 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 15);
		param_register("w", w64);
		TensorHandle y64 = tensor_log_softmax(w64, 0);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 4; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle w32 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 14);
		param_register("w", w32);
		TensorHandle y32 = tensor_log_softmax(w32, 0);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 4; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("log_softmax: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 4; i++) {
			char m[64];
			snprintf(m, sizeof m, "log_softmax: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
			snprintf(m, sizeof m, "log_softmax: w.grad_f32[%d] ~ w.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}

	/* 2026-07-27: fused softmax cross-entropy (OP_SOFTMAX_XENT_2D).
	   Forward + input-grad F32-vs-F64 paired contract at 1e-5, plus tag
	   propagation — the T29 rung the RMSNorm/SwiGLU landings skipped
	   (perf-changes.md follow-up notes). Rank-1 input rides the [1, n]
	   acceptance; one-hot target so d_in = scale * (softmax - target). */
	{
		double wv[] = {0.25, -0.5, 1.0, 0.75};
		double tv[] = {0.0, 1.0, 0.0, 0.0};
		double g_f64[4], g_f32[4];

		param_clear();
		TensorHandle w64 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 15);
		TensorHandle t64 = tensor_create_param_1d_streamed(4, hcopy(tv, 4), 0, 15);
		param_register("w", w64);
		TensorHandle y64 = tensor_softmax_xent_2d(w64, t64, 0.25);
		double loss64 = tensor_item(y64);
		tensor_backward(y64);
		for (int i = 0; i < 4; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle w32 = tensor_create_param_1d_streamed(4, hcopy(wv, 4), 0, 14);
		TensorHandle t32 = tensor_create_param_1d_streamed(4, hcopy(tv, 4), 0, 14);
		param_register("w", w32);
		TensorHandle y32 = tensor_softmax_xent_2d(w32, t32, 0.25);
		double loss32 = tensor_item(y32);
		tensor_backward(y32);
		for (int i = 0; i < 4; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("softmax_xent: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		ASSERT_NEAR("softmax_xent: loss_f32 ~ loss_f64", loss32, loss64, 1e-5);
		for (int i = 0; i < 4; i++) {
			char m[64];
			snprintf(m, sizeof m, "softmax_xent: w.grad_f32[%d] ~ w.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}

	/* Batch 1 Group D: reductions. tensor_sum already routes via
	   tape_load_d (Phase 3); rest get F32 scalar outputs + tape_load_d
	   for input reads. tensor_min / tensor_max are non-differentiable
	   so only forward + tag are checked. */
	{
		double wv[] = {1.5, -0.25, 0.5};
		double v64, v32, g_f64[3], g_f32[3];

		/* tensor_sum: forward + grad. */
		{
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle r64 = tensor_sum(w64);
			v64 = tensor_item(r64);
			tensor_backward(r64);
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle r32 = tensor_sum(w32);
			v32 = tensor_item(r32);
			tensor_backward(r32);
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("sum: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("sum: v_f32 ~ v_f64", v32, v64, 1e-5);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "sum: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_mean: forward + grad. */
		{
			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 15);
			param_register("w", w64);
			TensorHandle r64 = tensor_mean(w64);
			v64 = tensor_item(r64);
			tensor_backward(r64);
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(3, hcopy(wv, 3), 0, 14);
			param_register("w", w32);
			TensorHandle r32 = tensor_mean(w32);
			v32 = tensor_item(r32);
			tensor_backward(r32);
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("mean: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("mean: v_f32 ~ v_f64", v32, v64, 1e-5);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "mean: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_min / tensor_max — non-differentiable; check forward
		   + tag only. param_clear keeps them out of registry. */
		{
			TensorHandle w64 = tensor_create_1d_streamed(3, hcopy(wv, 3), 0, 0, 15);
			TensorHandle w32 = tensor_create_1d_streamed(3, hcopy(wv, 3), 0, 0, 14);
			ASSERT_NEAR("min: v_f32 ~ v_f64", tensor_item(tensor_min(w32)),
			            tensor_item(tensor_min(w64)), 1e-5);
			ASSERT_NEAR("max: v_f32 ~ v_f64", tensor_item(tensor_max(w32)),
			            tensor_item(tensor_max(w64)), 1e-5);
			ASSERT_TRUE("min: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(tensor_min(w32)), "F32") == 0);
			ASSERT_TRUE("max: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(tensor_max(w32)), "F32") == 0);
		}
	}

	/* Batch 2 Group E: view ops (reshape, narrow, select).
	   These share storage with the parent so the F32 dispatch is mostly
	   tag propagation + correct pointer stride (char* + tape_elem_size
	   rather than implicit double* arithmetic). Backward writes parent
	   grad (F64 by Phase 3 design) so the OP_<X> cases stay unchanged. */
	{
		double wv[] = {1.5, -0.25, 0.5, 2.0, -1.0};

		/* tensor_reshape: 1D [6] → 2D [2,3], values preserved. */
		{
			double rv[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
			double out_f64[6], out_f32[6];
			int new_shape[] = {2, 3};

			TensorHandle w64 = tensor_create_1d_streamed(6, hcopy(rv, 6), 0, 0, 15);
			TensorHandle r64 = tensor_reshape(w64, new_shape, 2);
			tensor_to_doubles(r64, out_f64);

			TensorHandle w32 = tensor_create_1d_streamed(6, hcopy(rv, 6), 0, 0, 14);
			TensorHandle r32 = tensor_reshape(w32, new_shape, 2);
			tensor_to_doubles(r32, out_f32);

			ASSERT_TRUE("reshape: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < 6; i++) {
				char m[64];
				snprintf(m, sizeof m, "reshape: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, out_f32[i], out_f64[i], 1e-5);
			}
		}

		/* tensor_narrow: [5] → [3] at start=1. Backward scatters grad
		   back to parent[start..start+len]. */
		{
			double out_f64[3], out_f32[3], g_f64[5], g_f32[5];

			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(5, hcopy(wv, 5), 0, 15);
			param_register("w", w64);
			TensorHandle r64 = tensor_narrow(w64, 0, 1, 3);
			tensor_to_doubles(r64, out_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < 5; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(5, hcopy(wv, 5), 0, 14);
			param_register("w", w32);
			TensorHandle r32 = tensor_narrow(w32, 0, 1, 3);
			tensor_to_doubles(r32, out_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < 5; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("narrow: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "narrow: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, out_f32[i], out_f64[i], 1e-5);
			}
			for (int i = 0; i < 5; i++) {
				char m[64];
				snprintf(m, sizeof m, "narrow: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_select: rank-1 [5] → scalar at index 2. Backward
		   adds grad to parent[index]. */
		{
			double g_f64[5], g_f32[5];

			param_clear();
			TensorHandle w64 = tensor_create_param_1d_streamed(5, hcopy(wv, 5), 0, 15);
			param_register("w", w64);
			TensorHandle r64 = tensor_select(w64, 0, 2);
			double v64 = tensor_item(r64);
			tensor_backward(r64);
			for (int i = 0; i < 5; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle w32 = tensor_create_param_1d_streamed(5, hcopy(wv, 5), 0, 14);
			param_register("w", w32);
			TensorHandle r32 = tensor_select(w32, 0, 2);
			double v32 = tensor_item(r32);
			tensor_backward(r32);
			for (int i = 0; i < 5; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("select: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("select: v_f32 ~ v_f64", v32, v64, 1e-5);
			for (int i = 0; i < 5; i++) {
				char m[64];
				snprintf(m, sizeof m, "select: w.grad_f32[%d] ~ w.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}
	}

	/* Batch 2 Group F: concat / cat / stack. memcpy widths get
	   tape_elem_size-aware; tag propagated to results. Backward
	   cases (OP_CAT, OP_CONCAT_2D_AXIS1, OP_STACK) write parent
	   grads (F64) and don't read data, so they're unchanged. */
	{
		/* tensor_cat2: [3] ++ [2] → [5], backward splits grad. */
		{
			double av[] = {1.5, -0.25, 0.5};
			double bv[] = {2.0, -1.0};
			double out_f64[5], out_f32[5], gA_f64[3], gA_f32[3], gB_f64[2], gB_f32[2];

			param_clear();
			TensorHandle a64 = tensor_create_param_1d_streamed(3, hcopy(av, 3), 0, 15);
			param_register("a", a64);
			TensorHandle b64 = tensor_create_param_1d_streamed(2, hcopy(bv, 2), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_cat2(a64, b64);
			tensor_to_doubles(r64, out_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < 3; i++)
				gA_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < 2; i++)
				gB_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 = tensor_create_param_1d_streamed(3, hcopy(av, 3), 0, 14);
			param_register("a", a32);
			TensorHandle b32 = tensor_create_param_1d_streamed(2, hcopy(bv, 2), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_cat2(a32, b32);
			tensor_to_doubles(r32, out_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < 3; i++)
				gA_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < 2; i++)
				gB_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("cat2: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < 5; i++) {
				char m[64];
				snprintf(m, sizeof m, "cat2: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, out_f32[i], out_f64[i], 1e-5);
			}
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "cat2: a.grad_f32[%d] ~ a.grad_f64", i);
				ASSERT_NEAR(m, gA_f32[i], gA_f64[i], 1e-5);
			}
			for (int i = 0; i < 2; i++) {
				char m[64];
				snprintf(m, sizeof m, "cat2: b.grad_f32[%d] ~ b.grad_f64", i);
				ASSERT_NEAR(m, gB_f32[i], gB_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_concat_2d_axis1: [2,2] ++ [2,3] along axis 1 → [2,5] */
		{
			double av[] = {1.0, 2.0, 3.0, 4.0};            /* [2,2] */
			double bv[] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0}; /* [2,3] */
			double out_f64[10], out_f32[10];

			TensorHandle a64 = tensor_create_2d_streamed(2, 2, hcopy(av, 4), 0, 0, 15);
			TensorHandle b64 = tensor_create_2d_streamed(2, 3, hcopy(bv, 6), 0, 0, 15);
			TensorHandle r64 = tensor_concat_2d_axis1(a64, b64);
			tensor_to_doubles(r64, out_f64);

			TensorHandle a32 = tensor_create_2d_streamed(2, 2, hcopy(av, 4), 0, 0, 14);
			TensorHandle b32 = tensor_create_2d_streamed(2, 3, hcopy(bv, 6), 0, 0, 14);
			TensorHandle r32 = tensor_concat_2d_axis1(a32, b32);
			tensor_to_doubles(r32, out_f32);

			ASSERT_TRUE("concat_2d_axis1: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < 10; i++) {
				char m[64];
				snprintf(m, sizeof m, "concat_2d_axis1: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, out_f32[i], out_f64[i], 1e-5);
			}
		}

		/* tensor_stack: stack 3 F32 scalars → F32 [3]. */
		{
			double s0v = 1.5, s1v = -0.25, s2v = 0.5;
			double out_f64[3], out_f32[3];

			TensorHandle s0_64 = tensor_create_scalar_streamed(s0v, 0, 0, 15);
			TensorHandle s1_64 = tensor_create_scalar_streamed(s1v, 0, 0, 15);
			TensorHandle s2_64 = tensor_create_scalar_streamed(s2v, 0, 0, 15);
			TensorHandle inputs64[3] = {s0_64, s1_64, s2_64};
			TensorHandle r64 = tensor_stack(inputs64, 3, 0);
			tensor_to_doubles(r64, out_f64);

			TensorHandle s0_32 = tensor_create_scalar_streamed(s0v, 0, 0, 14);
			TensorHandle s1_32 = tensor_create_scalar_streamed(s1v, 0, 0, 14);
			TensorHandle s2_32 = tensor_create_scalar_streamed(s2v, 0, 0, 14);
			TensorHandle inputs32[3] = {s0_32, s1_32, s2_32};
			TensorHandle r32 = tensor_stack(inputs32, 3, 0);
			tensor_to_doubles(r32, out_f32);

			ASSERT_TRUE("stack: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "stack: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(m, out_f32[i], out_f64[i], 1e-5);
			}
		}
	}

	/* Batch 3: losses. tensor_mse_loss + tensor_cross_entropy are non-
	   differentiable (just scalar output, no tape entry). BCE-with-
	   logits records OP_BCE_WITH_LOGITS so backward gets data reads
	   swapped to tape_load_d. */
	{
		double pv[] = {0.5, -1.0, 1.5}; /* logits / input */
		double tv[] = {1.0, 0.0, 1.0};  /* targets */

		/* MSE — non-differentiable check (forward + tag). */
		{
			TensorHandle p64 = tensor_create_1d_streamed(3, hcopy(pv, 3), 0, 0, 15);
			TensorHandle t64 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 15);
			TensorHandle p32 = tensor_create_1d_streamed(3, hcopy(pv, 3), 0, 0, 14);
			TensorHandle t32 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 14);
			TensorHandle r64 = tensor_mse_loss(p64, t64);
			TensorHandle r32 = tensor_mse_loss(p32, t32);
			ASSERT_TRUE("mse: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("mse: v_f32 ~ v_f64", tensor_item(r32), tensor_item(r64), 1e-5);
		}

		/* cross_entropy — non-differentiable check; depends on log_softmax. */
		{
			TensorHandle p64 = tensor_create_1d_streamed(3, hcopy(pv, 3), 0, 0, 15);
			TensorHandle t64 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 15);
			TensorHandle p32 = tensor_create_1d_streamed(3, hcopy(pv, 3), 0, 0, 14);
			TensorHandle t32 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 14);
			TensorHandle r64 = tensor_cross_entropy(p64, t64);
			TensorHandle r32 = tensor_cross_entropy(p32, t32);
			ASSERT_TRUE("cross_entropy: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("cross_entropy: v_f32 ~ v_f64", tensor_item(r32), tensor_item(r64), 1e-5);
		}

		/* bce_with_logits — differentiable; forward + grad both checked. */
		{
			double g_f64[3], g_f32[3];

			param_clear();
			TensorHandle p64 = tensor_create_param_1d_streamed(3, hcopy(pv, 3), 0, 15);
			param_register("p", p64);
			TensorHandle t64 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 15);
			TensorHandle r64 = tensor_bce_with_logits(p64, t64);
			double v64 = tensor_item(r64);
			tensor_backward(r64);
			for (int i = 0; i < 3; i++)
				g_f64[i] = param_grad_item_at(0, i);
			param_clear();

			TensorHandle p32 = tensor_create_param_1d_streamed(3, hcopy(pv, 3), 0, 14);
			param_register("p", p32);
			TensorHandle t32 = tensor_create_1d_streamed(3, hcopy(tv, 3), 0, 0, 14);
			TensorHandle r32 = tensor_bce_with_logits(p32, t32);
			double v32 = tensor_item(r32);
			tensor_backward(r32);
			for (int i = 0; i < 3; i++)
				g_f32[i] = param_grad_item_at(0, i);

			ASSERT_TRUE("bce_with_logits: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			ASSERT_NEAR("bce_with_logits: v_f32 ~ v_f64", v32, v64, 1e-5);
			for (int i = 0; i < 3; i++) {
				char m[64];
				snprintf(m, sizeof m, "bce_with_logits: p.grad_f32[%d] ~ p.grad_f64", i);
				ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
			}
			param_clear();
		}
	}

	/* Batch 4: BLAS-heavy linalg. Each kernel uses cblas_s* on F32 and
	   the LinearMeta-style double* x_vals cache (convert on store) so
	   the existing backward case body works for both dtypes. Where
	   BLAS would need a double* matrix, the F32 backward falls back to
	   plain loops via tape_load_d. */
	{
		/* tensor_linear: W [m,n] @ x [n] + b [m] -> [m] */
		{
			int m = 2, n = 3;
			double Wv[] = {1.0, 0.5, -0.25, 0.75, -0.5, 0.25};
			double xv[] = {0.5, -1.0, 0.25};
			double bv[] = {0.125, -0.25};
			double y_f64[2], y_f32[2], gW_f64[6], gW_f32[6], gx_f64[3], gx_f32[3], gb_f64[2],
			    gb_f32[2];

			param_clear();
			TensorHandle W64 = tensor_create_param_2d_streamed(m, n, hcopy(Wv, m * n), 0, 15);
			param_register("W", W64);
			TensorHandle x64 = tensor_create_param_1d_streamed(n, hcopy(xv, n), 0, 15);
			param_register("x", x64);
			TensorHandle b64 = tensor_create_param_1d_streamed(m, hcopy(bv, m), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_linear(W64, x64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < m * n; i++)
				gW_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n; i++)
				gx_f64[i] = param_grad_item_at(1, i);
			for (int i = 0; i < m; i++)
				gb_f64[i] = param_grad_item_at(2, i);
			param_clear();

			TensorHandle W32 = tensor_create_param_2d_streamed(m, n, hcopy(Wv, m * n), 0, 14);
			param_register("W", W32);
			TensorHandle x32 = tensor_create_param_1d_streamed(n, hcopy(xv, n), 0, 14);
			param_register("x", x32);
			TensorHandle b32 = tensor_create_param_1d_streamed(m, hcopy(bv, m), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_linear(W32, x32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < m * n; i++)
				gW_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n; i++)
				gx_f32[i] = param_grad_item_at(1, i);
			for (int i = 0; i < m; i++)
				gb_f32[i] = param_grad_item_at(2, i);

			ASSERT_TRUE("linear: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < m; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < m * n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear: W.grad_f32[%d] ~ W.grad_f64", i);
				ASSERT_NEAR(buf, gW_f32[i], gW_f64[i], 1e-5);
			}
			for (int i = 0; i < n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear: x.grad_f32[%d] ~ x.grad_f64", i);
				ASSERT_NEAR(buf, gx_f32[i], gx_f64[i], 1e-5);
			}
			for (int i = 0; i < m; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear: b.grad_f32[%d] ~ b.grad_f64", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_matmul (1D × 2D = OP_VECMAT) */
		{
			int n = 3, m = 2;
			double av[] = {1.5, -0.25, 0.5};
			double Bv[] = {1.0, -1.0, 0.5, 2.0, -0.5, 0.25}; /* [3, 2] */
			double y_f64[2], y_f32[2], ga_f64[3], ga_f32[3], gB_f64[6], gB_f32[6];

			param_clear();
			TensorHandle a64 = tensor_create_param_1d_streamed(n, hcopy(av, n), 0, 15);
			param_register("a", a64);
			TensorHandle B64 = tensor_create_param_2d_streamed(n, m, hcopy(Bv, n * m), 0, 15);
			param_register("B", B64);
			TensorHandle r64 = tensor_matmul(a64, B64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < n; i++)
				ga_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n * m; i++)
				gB_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 = tensor_create_param_1d_streamed(n, hcopy(av, n), 0, 14);
			param_register("a", a32);
			TensorHandle B32 = tensor_create_param_2d_streamed(n, m, hcopy(Bv, n * m), 0, 14);
			param_register("B", B32);
			TensorHandle r32 = tensor_matmul(a32, B32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < n; i++)
				ga_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n * m; i++)
				gB_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("matmul1D2D: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < m; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "matmul1D2D: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "matmul1D2D: a.grad_f32[%d] ~ a.grad_f64", i);
				ASSERT_NEAR(buf, ga_f32[i], ga_f64[i], 1e-5);
			}
			for (int i = 0; i < n * m; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "matmul1D2D: B.grad_f32[%d] ~ B.grad_f64", i);
				ASSERT_NEAR(buf, gB_f32[i], gB_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_outer */
		{
			int m = 3, n = 2;
			double av[] = {1.5, -0.25, 0.5};
			double bv[] = {2.0, -1.0};
			double y_f64[6], y_f32[6], ga_f64[3], ga_f32[3], gb_f64[2], gb_f32[2];

			param_clear();
			TensorHandle a64 = tensor_create_param_1d_streamed(m, hcopy(av, m), 0, 15);
			param_register("a", a64);
			TensorHandle b64 = tensor_create_param_1d_streamed(n, hcopy(bv, n), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_outer(a64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < m; i++)
				ga_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n; i++)
				gb_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 = tensor_create_param_1d_streamed(m, hcopy(av, m), 0, 14);
			param_register("a", a32);
			TensorHandle b32 = tensor_create_param_1d_streamed(n, hcopy(bv, n), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_outer(a32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < m; i++)
				ga_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n; i++)
				gb_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("outer: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < m * n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "outer: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < m; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "outer: a.grad_f32[%d] ~ a.grad_f64", i);
				ASSERT_NEAR(buf, ga_f32[i], ga_f64[i], 1e-5);
			}
			for (int i = 0; i < n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "outer: b.grad_f32[%d] ~ b.grad_f64", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_mm: [2,3] @ [3,2] -> [2,2] */
		{
			int M = 2, N = 3, K = 2;
			double av[] = {1.0, 0.5, -0.25, 0.75, -0.5, 0.25}; /* [2,3] */
			double bv[] = {0.5, -1.0, 0.25, 2.0, -0.5, 0.5};   /* [3,2] */
			double y_f64[4], y_f32[4], ga_f64[6], ga_f32[6], gb_f64[6], gb_f32[6];

			param_clear();
			TensorHandle a64 = tensor_create_param_2d_streamed(M, N, hcopy(av, M * N), 0, 15);
			param_register("a", a64);
			TensorHandle b64 = tensor_create_param_2d_streamed(N, K, hcopy(bv, N * K), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_mm(a64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < M * N; i++)
				ga_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < N * K; i++)
				gb_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 = tensor_create_param_2d_streamed(M, N, hcopy(av, M * N), 0, 14);
			param_register("a", a32);
			TensorHandle b32 = tensor_create_param_2d_streamed(N, K, hcopy(bv, N * K), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_mm(a32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < M * N; i++)
				ga_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < N * K; i++)
				gb_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("mm: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < M * K; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "mm: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < M * N; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "mm: a.grad_f32[%d] ~ a.grad_f64", i);
				ASSERT_NEAR(buf, ga_f32[i], ga_f64[i], 1e-5);
			}
			for (int i = 0; i < N * K; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "mm: b.grad_f32[%d] ~ b.grad_f64", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_linear_2d: Y[B,o] = X[B,i] @ W[o,i]^T + bias[o] */
		{
			int B = 2, ii = 3, oo = 2;
			double Xv[] = {0.5, -1.0, 0.25, 0.75, -0.5, 1.0};  /* [2,3] */
			double Wv[] = {1.0, 0.5, -0.25, 0.75, -0.5, 0.25}; /* [2,3] */
			double bv[] = {0.125, -0.25};                      /* [2]   */
			double y_f64[4], y_f32[4], gW_f64[6], gW_f32[6], gX_f64[6], gX_f32[6], gb_f64[2],
			    gb_f32[2];

			param_clear();
			TensorHandle W64 = tensor_create_param_2d_streamed(oo, ii, hcopy(Wv, oo * ii), 0, 15);
			param_register("W", W64);
			TensorHandle X64 = tensor_create_param_2d_streamed(B, ii, hcopy(Xv, B * ii), 0, 15);
			param_register("X", X64);
			TensorHandle b64 = tensor_create_param_1d_streamed(oo, hcopy(bv, oo), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_linear_2d(W64, X64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < oo * ii; i++)
				gW_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < B * ii; i++)
				gX_f64[i] = param_grad_item_at(1, i);
			for (int i = 0; i < oo; i++)
				gb_f64[i] = param_grad_item_at(2, i);
			param_clear();

			TensorHandle W32 = tensor_create_param_2d_streamed(oo, ii, hcopy(Wv, oo * ii), 0, 14);
			param_register("W", W32);
			TensorHandle X32 = tensor_create_param_2d_streamed(B, ii, hcopy(Xv, B * ii), 0, 14);
			param_register("X", X32);
			TensorHandle b32 = tensor_create_param_1d_streamed(oo, hcopy(bv, oo), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_linear_2d(W32, X32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < oo * ii; i++)
				gW_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < B * ii; i++)
				gX_f32[i] = param_grad_item_at(1, i);
			for (int i = 0; i < oo; i++)
				gb_f32[i] = param_grad_item_at(2, i);

			ASSERT_TRUE("linear_2d: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < B * oo; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear_2d: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < oo * ii; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear_2d: W.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gW_f32[i], gW_f64[i], 1e-5);
			}
			for (int i = 0; i < B * ii; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear_2d: X.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gX_f32[i], gX_f64[i], 1e-5);
			}
			for (int i = 0; i < oo; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "linear_2d: b.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_bmm: [B,m,n] x [n,k] -> [B,m,k], b shared across batch */
		{
			int B = 2, m = 2, n = 3, k = 2;
			double av[12] = {1.0, 0.5,  -0.25, 0.75, -0.5, 0.25,
			                 0.5, -1.0, 1.0,   0.25, 0.5,  -0.5}; /* [2,2,3] */
			double bv[6] = {0.5, -1.0, 0.25, 2.0, -0.5, 0.5};     /* [3,2]   */
			double y_f64[8], y_f32[8], ga_f64[12], ga_f32[12], gb_f64[6], gb_f32[6];

			param_clear();
			TensorHandle a64 =
			    tensor_create_param_3d_streamed(B, m, n, hcopy(av, B * m * n), 0, 15);
			param_register("a", a64);
			TensorHandle b64 = tensor_create_param_2d_streamed(n, k, hcopy(bv, n * k), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_bmm(a64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < B * m * n; i++)
				ga_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n * k; i++)
				gb_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 =
			    tensor_create_param_3d_streamed(B, m, n, hcopy(av, B * m * n), 0, 14);
			param_register("a", a32);
			TensorHandle b32 = tensor_create_param_2d_streamed(n, k, hcopy(bv, n * k), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_bmm(a32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < B * m * n; i++)
				ga_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < n * k; i++)
				gb_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("bmm: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < B * m * k; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm: y_f32[%d]", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < B * m * n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm: a.grad_f32[%d]", i);
				ASSERT_NEAR(buf, ga_f32[i], ga_f64[i], 1e-5);
			}
			for (int i = 0; i < n * k; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm: b.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_batch_norm: channels-first, training mode. Tests
		   that running_mean/running_var updates and forward output
		   match between F32 and F64 (within F32 tol). */
		{
			int C = 2, sp = 3;
			int n = C * sp;
			double xv[] = {1.5, -0.25, 0.5, 0.25, 1.0, -0.75};
			double gv[] = {1.0, 0.5};
			double bv[] = {0.1, -0.2};
			double rmv[] = {0.0, 0.0};
			double rvv[] = {1.0, 1.0};
			double y_f64[6], y_f32[6];

			/* F64 reference (no grad — exercise forward + running stats only). */
			TensorHandle x64 = tensor_create_1d_streamed(n, hcopy(xv, n), 0, 0, 15);
			TensorHandle g64 = tensor_create_1d_streamed(C, hcopy(gv, C), 0, 0, 15);
			TensorHandle b64 = tensor_create_1d_streamed(C, hcopy(bv, C), 0, 0, 15);
			TensorHandle rm64 = tensor_create_state_1d_streamed(C, hcopy(rmv, C), 0, 15);
			TensorHandle rv64 = tensor_create_state_1d_streamed(C, hcopy(rvv, C), 0, 15);
			TensorHandle r64 = tensor_batch_norm(x64, g64, b64, rm64, rv64, C, sp, 1, 0.1, 1e-5);
			tensor_to_doubles(r64, y_f64);

			TensorHandle x32 = tensor_create_1d_streamed(n, hcopy(xv, n), 0, 0, 14);
			TensorHandle g32 = tensor_create_1d_streamed(C, hcopy(gv, C), 0, 0, 14);
			TensorHandle b32 = tensor_create_1d_streamed(C, hcopy(bv, C), 0, 0, 14);
			TensorHandle rm32 = tensor_create_state_1d_streamed(C, hcopy(rmv, C), 0, 14);
			TensorHandle rv32 = tensor_create_state_1d_streamed(C, hcopy(rvv, C), 0, 14);
			TensorHandle r32 = tensor_batch_norm(x32, g32, b32, rm32, rv32, C, sp, 1, 0.1, 1e-5);
			tensor_to_doubles(r32, y_f32);

			ASSERT_TRUE("batch_norm: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "batch_norm: y_f32[%d] ~ y_f64", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			/* Running stats should also be updated. */
			double rm_f64[2], rv_f64[2], rm_f32[2], rv_f32[2];
			tensor_to_doubles(rm64, rm_f64);
			tensor_to_doubles(rv64, rv_f64);
			tensor_to_doubles(rm32, rm_f32);
			tensor_to_doubles(rv32, rv_f32);
			for (int i = 0; i < C; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "batch_norm: running_mean_f32[%d] ~ f64", i);
				ASSERT_NEAR(buf, rm_f32[i], rm_f64[i], 1e-5);
				snprintf(buf, sizeof buf, "batch_norm: running_var_f32[%d] ~ f64", i);
				ASSERT_NEAR(buf, rv_f32[i], rv_f64[i], 1e-5);
			}
		}

		/* tensor_layer_norm_2d: row-wise LN with gamma/bias affine. */
		{
			int M = 2, N = 3;
			double xv[] = {1.5, -0.25, 0.5, 0.25, 1.0, -0.75};
			double gv[] = {1.0, 0.5, 0.25};
			double bv[] = {0.1, -0.2, 0.3};
			double y_f64[6], y_f32[6], gx_f64[6], gx_f32[6], gg_f64[3], gg_f32[3], gb_f64[3],
			    gb_f32[3];

			param_clear();
			TensorHandle x64 = tensor_create_param_2d_streamed(M, N, hcopy(xv, M * N), 0, 15);
			param_register("x", x64);
			TensorHandle g64 = tensor_create_param_1d_streamed(N, hcopy(gv, N), 0, 15);
			param_register("g", g64);
			TensorHandle b64 = tensor_create_param_1d_streamed(N, hcopy(bv, N), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_layer_norm_2d(x64, g64, b64, 1e-5);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < M * N; i++)
				gx_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < N; i++)
				gg_f64[i] = param_grad_item_at(1, i);
			for (int i = 0; i < N; i++)
				gb_f64[i] = param_grad_item_at(2, i);
			param_clear();

			TensorHandle x32 = tensor_create_param_2d_streamed(M, N, hcopy(xv, M * N), 0, 14);
			param_register("x", x32);
			TensorHandle g32 = tensor_create_param_1d_streamed(N, hcopy(gv, N), 0, 14);
			param_register("g", g32);
			TensorHandle b32 = tensor_create_param_1d_streamed(N, hcopy(bv, N), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_layer_norm_2d(x32, g32, b32, 1e-5);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < M * N; i++)
				gx_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < N; i++)
				gg_f32[i] = param_grad_item_at(1, i);
			for (int i = 0; i < N; i++)
				gb_f32[i] = param_grad_item_at(2, i);

			ASSERT_TRUE("layer_norm_2d: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < M * N; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "layer_norm_2d: y_f32[%d]", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < M * N; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "layer_norm_2d: x.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gx_f32[i], gx_f64[i], 1e-5);
			}
			for (int i = 0; i < N; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "layer_norm_2d: g.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gg_f32[i], gg_f64[i], 1e-5);
			}
			for (int i = 0; i < N; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "layer_norm_2d: b.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}

		/* tensor_bmm_3x3: [B,m,n] x [B,n,k] -> [B,m,k], per-batch b */
		{
			int B = 2, m = 2, n = 2, k = 2;
			double av[8] = {1.0, 0.5, -0.25, 0.75, -0.5, 0.25, 0.5, -1.0}; /* [2,2,2] */
			double bv[8] = {0.5, -1.0, 0.25, 2.0, -0.5, 0.5, 1.0, -0.25};  /* [2,2,2] */
			double y_f64[8], y_f32[8], ga_f64[8], ga_f32[8], gb_f64[8], gb_f32[8];

			param_clear();
			TensorHandle a64 =
			    tensor_create_param_3d_streamed(B, m, n, hcopy(av, B * m * n), 0, 15);
			param_register("a", a64);
			TensorHandle b64 =
			    tensor_create_param_3d_streamed(B, n, k, hcopy(bv, B * n * k), 0, 15);
			param_register("b", b64);
			TensorHandle r64 = tensor_bmm_3x3(a64, b64);
			tensor_to_doubles(r64, y_f64);
			tensor_backward(tensor_sum(r64));
			for (int i = 0; i < B * m * n; i++)
				ga_f64[i] = param_grad_item_at(0, i);
			for (int i = 0; i < B * n * k; i++)
				gb_f64[i] = param_grad_item_at(1, i);
			param_clear();

			TensorHandle a32 =
			    tensor_create_param_3d_streamed(B, m, n, hcopy(av, B * m * n), 0, 14);
			param_register("a", a32);
			TensorHandle b32 =
			    tensor_create_param_3d_streamed(B, n, k, hcopy(bv, B * n * k), 0, 14);
			param_register("b", b32);
			TensorHandle r32 = tensor_bmm_3x3(a32, b32);
			tensor_to_doubles(r32, y_f32);
			tensor_backward(tensor_sum(r32));
			for (int i = 0; i < B * m * n; i++)
				ga_f32[i] = param_grad_item_at(0, i);
			for (int i = 0; i < B * n * k; i++)
				gb_f32[i] = param_grad_item_at(1, i);

			ASSERT_TRUE("bmm_3x3: F32 output propagates F32 tag",
			            strcmp(tensor_dtype_name(r32), "F32") == 0);
			for (int i = 0; i < B * m * k; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm_3x3: y_f32[%d]", i);
				ASSERT_NEAR(buf, y_f32[i], y_f64[i], 1e-5);
			}
			for (int i = 0; i < B * m * n; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm_3x3: a.grad_f32[%d]", i);
				ASSERT_NEAR(buf, ga_f32[i], ga_f64[i], 1e-5);
			}
			for (int i = 0; i < B * n * k; i++) {
				char buf[64];
				snprintf(buf, sizeof buf, "bmm_3x3: b.grad_f32[%d]", i);
				ASSERT_NEAR(buf, gb_f32[i], gb_f64[i], 1e-5);
			}
			param_clear();
		}
	}
}
#endif

#if defined(BACKEND_TAPE)
/* W5 follow-on: F32 paired oracle for kernels NOT covered by the
 * existing tape_f32_gradcheck_oracle rungs 1-4 or the
 * tape_f32_non_elementwise_coverage batches A-H.
 *
 * Each "Rung" block targets one high-impact op family; the pattern is
 * unchanged from the rest of this file: run identical computation on
 * F32-tagged and F64-tagged inputs via the streamed entry points
 * (dtag 14 = F32, dtag 15 = F64), assert the F32 result propagates the
 * F32 dtype tag and matches the F64 reference within 1e-5 (forward and
 * gradient).
 *
 * Coverage gain (rungs added here):
 *   Rung 5  conv1d                — TAPE_F32_SKIP_CONV
 *   Rung 6  avg_pool1d            — TAPE_F32_SKIP_POOL
 *   Rung 8  gru_cell              — TAPE_F32_SKIP_RNN
 *   Rung 10 embedding             — TAPE_F32_SKIP_INDEX
 */
Test(dtype_scaffolding, tape_f32_rnn_conv_coverage) {
	/* Rung 5: conv1d. Small 1-in-1-out kernel.
	 *   input  [inC=1, L=4] = [1, 2, 3, 4]
	 *   kernel [outC=1, inC=1, kL=2] = [0.5, -1.0]
	 *   bias   [outC=1] = [0.1]
	 *   pad=0, stride=1 -> output [1, 3]
	 *   loss = sum(output). */
#ifndef TAPE_F32_SKIP_CONV
	{
		double inv[] = {1.0, 2.0, 3.0, 4.0};
		double kv[] = {0.5, -1.0};
		double bv[] = {0.1};
		double y_f64[3], y_f32[3];
		double gk_f64[2], gk_f32[2];

		/* F64 path */
		param_clear();
		TensorHandle in64 = tensor_create_2d_streamed(1, 4, hcopy(inv, 4), 0, 0, 15);
		TensorHandle k64 = tensor_create_param_3d_streamed(1, 1, 2, hcopy(kv, 2), 0, 15);
		param_register("k", k64);
		TensorHandle b64 = tensor_create_1d_streamed(1, hcopy(bv, 1), 0, 0, 15);
		TensorHandle y64 = tensor_conv1d(in64, k64, b64, 0, 1);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 2; i++)
			gk_f64[i] = param_grad_item_at(0, i);
		param_clear();

		/* F32 path */
		TensorHandle in32 = tensor_create_2d_streamed(1, 4, hcopy(inv, 4), 0, 0, 14);
		TensorHandle k32 = tensor_create_param_3d_streamed(1, 1, 2, hcopy(kv, 2), 0, 14);
		param_register("k", k32);
		TensorHandle b32 = tensor_create_1d_streamed(1, hcopy(bv, 1), 0, 0, 14);
		TensorHandle y32 = tensor_conv1d(in32, k32, b32, 0, 1);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 2; i++)
			gk_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("conv1d: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 3; i++) {
			char m[64];
			snprintf(m, sizeof m, "conv1d: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
		}
		for (int i = 0; i < 2; i++) {
			char m[64];
			snprintf(m, sizeof m, "conv1d: k.grad_f32[%d] ~ k.grad_f64", i);
			ASSERT_NEAR(m, gk_f32[i], gk_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: conv1d (TAPE_F32_SKIP_CONV)\n");
#endif

	/* Rung 6: avg_pool1d. input [1, 4] = [1,2,3,4], kL=2, stride=2
	 *   -> output [1, 2] = [1.5, 3.5]. */
#ifndef TAPE_F32_SKIP_POOL
	{
		double inv[] = {1.0, 2.0, 3.0, 4.0};
		double y_f64[2], y_f32[2];
		double g_f64[4], g_f32[4];

		param_clear();
		TensorHandle in64 = tensor_create_param_2d_streamed(1, 4, hcopy(inv, 4), 0, 15);
		param_register("in", in64);
		TensorHandle y64 = tensor_avg_pool1d(in64, 2, 2);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 4; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle in32 = tensor_create_param_2d_streamed(1, 4, hcopy(inv, 4), 0, 14);
		param_register("in", in32);
		TensorHandle y32 = tensor_avg_pool1d(in32, 2, 2);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 4; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("avg_pool1d: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		for (int i = 0; i < 2; i++) {
			char m[64];
			snprintf(m, sizeof m, "avg_pool1d: y_f32[%d] ~ y_f64", i);
			ASSERT_NEAR(m, y_f32[i], y_f64[i], 1e-5);
		}
		for (int i = 0; i < 4; i++) {
			char m[64];
			snprintf(m, sizeof m, "avg_pool1d: in.grad_f32[%d] ~ in.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: avg_pool1d (TAPE_F32_SKIP_POOL)\n");
#endif

	/* Rung 8: gru_cell. Tiny 1-d hidden (o=1).
	 *   ih shape [3] (gate-stacked: r, z, n)
	 *   hh shape [3]
	 *   prev_hidden shape [1]
	 * Returns [1]. */
#ifndef TAPE_F32_SKIP_RNN
	{
		double ihv[] = {0.5, -0.25, 0.75};
		double hhv[] = {0.1, 0.2, -0.3};
		double prevv[] = {0.2};
		double y_f64[1], y_f32[1];
		double g_f64[3], g_f32[3];

		param_clear();
		TensorHandle ih64 = tensor_create_param_1d_streamed(3, hcopy(ihv, 3), 0, 15);
		param_register("ih", ih64);
		TensorHandle hh64 = tensor_create_1d_streamed(3, hcopy(hhv, 3), 0, 0, 15);
		TensorHandle prev64 = tensor_create_1d_streamed(1, hcopy(prevv, 1), 0, 0, 15);
		TensorHandle y64 = tensor_gru_cell(ih64, hh64, prev64, 1);
		tensor_to_doubles(y64, y_f64);
		tensor_backward(tensor_sum(y64));
		for (int i = 0; i < 3; i++)
			g_f64[i] = param_grad_item_at(0, i);
		param_clear();

		TensorHandle ih32 = tensor_create_param_1d_streamed(3, hcopy(ihv, 3), 0, 14);
		param_register("ih", ih32);
		TensorHandle hh32 = tensor_create_1d_streamed(3, hcopy(hhv, 3), 0, 0, 14);
		TensorHandle prev32 = tensor_create_1d_streamed(1, hcopy(prevv, 1), 0, 0, 14);
		TensorHandle y32 = tensor_gru_cell(ih32, hh32, prev32, 1);
		tensor_to_doubles(y32, y_f32);
		tensor_backward(tensor_sum(y32));
		for (int i = 0; i < 3; i++)
			g_f32[i] = param_grad_item_at(0, i);

		ASSERT_TRUE("gru_cell: F32 output propagates F32 tag",
		            strcmp(tensor_dtype_name(y32), "F32") == 0);
		{
			char m[64];
			snprintf(m, sizeof m, "gru_cell: y_f32[0] ~ y_f64");
			ASSERT_NEAR(m, y_f32[0], y_f64[0], 1e-5);
		}
		for (int i = 0; i < 3; i++) {
			char m[64];
			snprintf(m, sizeof m, "gru_cell: ih.grad_f32[%d] ~ ih.grad_f64", i);
			ASSERT_NEAR(m, g_f32[i], g_f64[i], 1e-5);
		}
		param_clear();
	}
#else
	printf("rung skipped: gru_cell (TAPE_F32_SKIP_RNN)\n");
#endif
}
#endif

#if defined(BACKEND_TAPE)
Test(dtype_scaffolding, tape_inference_dtype_matrix) {
	/* Half-precision rung.
	   bf16 nearest representable for 0.1: 0x3DCD -> ~0.10009765625
	   f16  nearest representable for 0.1: 0x2E66 -> ~0.0999755859375 */
	{
		double bv[] = {1.5, 0.1, -2.0}; /* 1.5 + -2.0 exact in both; 0.1 rounds */
		TensorHandle bf = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 17); /* dtag 17 = BF16 */
		ASSERT_TRUE("BF16 dtype name", strcmp(tensor_dtype_name(bf), "BF16") == 0);
		double bout[3];
		tensor_to_doubles(bf, bout);
		ASSERT_NEAR("BF16 exact: 1.5", bout[0], 1.5, 1e-12);
		ASSERT_NEAR("BF16 round: 0.1 -> 0.10009765625", bout[1], 0.10009765625, 1e-7);
		ASSERT_NEAR("BF16 exact: -2.0", bout[2], -2.0, 1e-12);

		TensorHandle hf = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 13); /* dtag 13 = F16 */
		ASSERT_TRUE("F16 dtype name", strcmp(tensor_dtype_name(hf), "F16") == 0);
		double hout[3];
		tensor_to_doubles(hf, hout);
		ASSERT_NEAR("F16 exact: 1.5", hout[0], 1.5, 1e-12);
		ASSERT_NEAR("F16 round: 0.1 -> 0.0999755859375", hout[1], 0.0999755859375, 1e-7);
		ASSERT_NEAR("F16 exact: -2.0", hout[2], -2.0, 1e-12);
	}

	/* Integer dtypes — exact round-trip within range via Phase 2 rounding. */
	{
		/* I8: -128..127 */
		double v8[] = {-128.0, -1.0, 0.0, 1.0, 127.0};
		TensorHandle i8 = tensor_create_1d_streamed(5, hcopy(v8, 5), 0, 0, 8);
		ASSERT_TRUE("I8 dtype name", strcmp(tensor_dtype_name(i8), "I8") == 0);
		double out8[5];
		tensor_to_doubles(i8, out8);
		for (int i = 0; i < 5; i++) {
			char m[48];
			snprintf(m, sizeof m, "I8 in-range[%d]", i);
			ASSERT_NEAR(m, out8[i], v8[i], 1e-12);
		}

		/* I16: -32768..32767 */
		double v16[] = {-32768.0, 32767.0, 0.0};
		TensorHandle i16 = tensor_create_1d_streamed(3, hcopy(v16, 3), 0, 0, 9);
		ASSERT_TRUE("I16 dtype name", strcmp(tensor_dtype_name(i16), "I16") == 0);
		double out16[3];
		tensor_to_doubles(i16, out16);
		for (int i = 0; i < 3; i++) {
			char m[48];
			snprintf(m, sizeof m, "I16 in-range[%d]", i);
			ASSERT_NEAR(m, out16[i], v16[i], 1e-12);
		}

		/* I32: full 32-bit range */
		double v32[] = {-2147483648.0, 0.0, 2147483647.0};
		TensorHandle i32 = tensor_create_1d_streamed(3, hcopy(v32, 3), 0, 0, 10);
		ASSERT_TRUE("I32 dtype name", strcmp(tensor_dtype_name(i32), "I32") == 0);
		double out32[3];
		tensor_to_doubles(i32, out32);
		for (int i = 0; i < 3; i++) {
			char m[48];
			snprintf(m, sizeof m, "I32 in-range[%d]", i);
			ASSERT_NEAR(m, out32[i], v32[i], 1e-12);
		}

		/* I64: within 2^53 (documented caveat — above loses precision via double). */
		double v64[] = {-1e15, 0.0, 1e15};
		TensorHandle i64 = tensor_create_1d_streamed(3, hcopy(v64, 3), 0, 0, 11);
		ASSERT_TRUE("I64 dtype name", strcmp(tensor_dtype_name(i64), "I64") == 0);
		double out64[3];
		tensor_to_doubles(i64, out64);
		for (int i = 0; i < 3; i++) {
			char m[48];
			snprintf(m, sizeof m, "I64 within 2^53[%d]", i);
			ASSERT_NEAR(m, out64[i], v64[i], 1e-12);
		}
	}

	/* U8 + Bool — exact via Phase 2 rounding. */
	{
		double vu[] = {0.0, 1.0, 128.0, 255.0};
		TensorHandle u8 = tensor_create_1d_streamed(4, hcopy(vu, 4), 0, 0, 4);
		ASSERT_TRUE("U8 dtype name", strcmp(tensor_dtype_name(u8), "U8") == 0);
		double outu[4];
		tensor_to_doubles(u8, outu);
		for (int i = 0; i < 4; i++) {
			char m[48];
			snprintf(m, sizeof m, "U8 in-range[%d]", i);
			ASSERT_NEAR(m, outu[i], vu[i], 1e-12);
		}

		/* Bool: 0 -> 0, anything-nonzero -> 1. */
		double vb[] = {0.0, 1.0, 0.5, -3.0, 0.0};
		double xb[] = {0.0, 1.0, 1.0, 1.0, 0.0};
		TensorHandle bo = tensor_create_1d_streamed(5, hcopy(vb, 5), 0, 0, 1);
		ASSERT_TRUE("Bool dtype name", strcmp(tensor_dtype_name(bo), "BOOL") == 0);
		double outb[5];
		tensor_to_doubles(bo, outb);
		for (int i = 0; i < 5; i++) {
			char m[48];
			snprintf(m, sizeof m, "Bool 0/1 normalize[%d]", i);
			ASSERT_NEAR(m, outb[i], xb[i], 1e-12);
		}
	}

	/* Cast paths: F64 -> <dtype> -> F64 via tensor_cast_dtype_streamed. */
	{
		/* F64 -> U8 -> F64. -3 wraps to 253 via unsigned-char cast (documented). */
		double cv[] = {10.0, -3.0, 7.0};
		TensorHandle d0 = tensor_create_1d_streamed(3, hcopy(cv, 3), 0, 0, 15);
		TensorHandle to_u8 = tensor_cast_dtype_streamed(d0, 0, 4);
		ASSERT_TRUE("cast F64->U8 dtype", strcmp(tensor_dtype_name(to_u8), "U8") == 0);
		TensorHandle back = tensor_cast_dtype_streamed(to_u8, 0, 15);
		ASSERT_TRUE("cast U8->F64 dtype", strcmp(tensor_dtype_name(back), "F64") == 0);
		double rb[3];
		tensor_to_doubles(back, rb);
		ASSERT_NEAR("U8 roundtrip 10", rb[0], 10.0, 1e-12);
		ASSERT_NEAR("U8 roundtrip -3->253", rb[1], 253.0, 1e-12);
		ASSERT_NEAR("U8 roundtrip 7", rb[2], 7.0, 1e-12);

		/* F64 -> Bool -> F64 */
		double bv[] = {0.0, 5.0, -2.0};
		TensorHandle s0 = tensor_create_1d_streamed(3, hcopy(bv, 3), 0, 0, 15);
		TensorHandle to_bool = tensor_cast_dtype_streamed(s0, 0, 1);
		ASSERT_TRUE("cast F64->Bool dtype", strcmp(tensor_dtype_name(to_bool), "BOOL") == 0);
		TensorHandle bback = tensor_cast_dtype_streamed(to_bool, 0, 15);
		double brt[3];
		tensor_to_doubles(bback, brt);
		ASSERT_NEAR("Bool roundtrip 0", brt[0], 0.0, 1e-12);
		ASSERT_NEAR("Bool roundtrip 5", brt[1], 1.0, 1e-12);
		ASSERT_NEAR("Bool roundtrip -2", brt[2], 1.0, 1e-12);
	}
	param_clear();
}
#endif

#if defined(BACKEND_TAPE)
Test(dtype_scaffolding, tape_f32_cast_readout_agreement) {
	/* π and √2: F32 nearest values differ from F64 source past the
	   7th decimal, so the float-vs-double misread shows up clearly. */
	double pv[] = {3.14159265358979, 1.4142135623730951};
	TensorHandle f64src = tensor_create_1d_streamed(2, hcopy(pv, 2), 0, 0, 15); /* dtag 15 = F64 */
	TensorHandle f32cast = tensor_cast_dtype_streamed(f64src, 0, 14);           /* dtag 14 = F32 */
	ASSERT_TRUE("F32 cast dtype name", strcmp(tensor_dtype_name(f32cast), "F32") == 0);

	/* Reader paths must agree: tensor_to_doubles, tensor_item_1d, and a
	   direct (float*) read all see the same F32-narrowed values. */
	double via_to_doubles[2];
	tensor_to_doubles(f32cast, via_to_doubles);
	ASSERT_NEAR("F32 cast→to_doubles[0]", via_to_doubles[0], 3.1415927410125732, 1e-12);
	ASSERT_NEAR("F32 cast→to_doubles[1]", via_to_doubles[1], 1.4142135381698608, 1e-12);
	ASSERT_NEAR("F32 cast→item_1d[0]", tensor_item_1d(f32cast, 0), 3.1415927410125732, 1e-12);
	ASSERT_NEAR("F32 cast→item_1d[1]", tensor_item_1d(f32cast, 1), 1.4142135381698608, 1e-12);

	/* F32 → F64 round-trip via cast: widened values match the
	   F32-narrowed readout exactly (no further precision loss). */
	TensorHandle f64back = tensor_cast_dtype_streamed(f32cast, 0, 15);
	ASSERT_TRUE("F32→F64 widened dtype name", strcmp(tensor_dtype_name(f64back), "F64") == 0);
	double widened[2];
	tensor_to_doubles(f64back, widened);
	ASSERT_NEAR("F32→F64 widened[0]", widened[0], 3.1415927410125732, 1e-12);
	ASSERT_NEAR("F32→F64 widened[1]", widened[1], 1.4142135381698608, 1e-12);
	param_clear();
}
#endif
