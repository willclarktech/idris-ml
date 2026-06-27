/* Regression guard for the MNIST "invalid memory reference" crash that
 * fired on the very first training step (campaign 2026-06-17, every seed,
 * 4-6 s after build). The crasher is the LeNet conv/pool backward at
 * realistic h>1 dims with overlapping windows — untested before this:
 * test_conv2d_batched backward only ran [B=1,inC=1,H=2,W=2]; the one
 * realistic shape there was forward-only, and test_max_pool2d_batched
 * backward was [B=2,C=1,H=2,W=2]. seq-classify (Conv1D/MaxPool1D, unit
 * height) exercised the same prims and passed, which is why the height-axis
 * backward slipped through.
 *
 * This test replays Example/Mnist.idr's exact chain at its exact dims:
 *   in[64,1,28,28] -conv2d(16,k5)-> [64,16,24,24] -relu-> -pool(2,2)->
 *   [64,16,12,12] -conv2d(32,k5)-> [64,32,8,8] -relu-> -pool(2,2)->
 *   [64,32,4,4] -reshape-> [64,512] -linear(512->10)-> [64,10] -sum-> loss
 * then backward + an optimizer step. Under `-fsanitize=address` (the
 * test-unit-c-asan-tape lane) the OOB pins to its source line.
 *
 * RED before the fix: ASan heap-buffer-overflow / SIGSEGV in the conv or
 * pool backward; or a plain crash in the non-ASan run.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include "backend.h"

/* relu = leaky_relu with alpha 0 (no dedicated relu FFI). */
static TensorHandle relu(TensorHandle t) {
	return tensor_leaky_relu(t, 0.0);
}

Test(conv_mnist_lenet_step, full_forward_backward_step_at_real_dims) {
	param_clear();

	enum { B = 64, IMG = 28, INC = 1, OC1 = 16, OC2 = 32, K = 5 };
	enum { C1 = IMG - K + 1 };      /* 24 */
	enum { P1 = (C1 - 2) / 2 + 1 }; /* 12 */
	enum { C2 = P1 - K + 1 };       /* 8 */
	enum { P2 = (C2 - 2) / 2 + 1 }; /* 4 */
	enum { FLAT = OC2 * P2 * P2 };  /* 512 */
	enum { CLS = 10 };

	/* Input: data tensor (requires_grad=0, like the MNIST mini-batch). */
	int in_numel = B * INC * IMG * IMG;
	double* in_data = (double*)malloc((size_t)in_numel * sizeof(double));
	for (int i = 0; i < in_numel; i++)
		in_data[i] = ((i % 17) - 8) * 0.0625; /* small, F32-exact range */
	int sh_in[4] = {B, INC, IMG, IMG};
	TensorHandle in = tensor_create(in_data, sh_in, 4, 0);

	/* conv1 kernel [16,1,5,5] + bias [16], both params. */
	int k1_numel = OC1 * INC * K * K;
	double* k1 = (double*)malloc((size_t)k1_numel * sizeof(double));
	for (int i = 0; i < k1_numel; i++)
		k1[i] = ((i % 13) - 6) * 0.03125;
	double* b1 = (double*)calloc(OC1, sizeof(double));
	TensorHandle K1 = tensor_create_param_4d_f64(OC1, INC, K, K, k1);
	TensorHandle B1 = tensor_create_param_1d_f64(OC1, b1);
	param_register("conv1.weight", K1);
	param_register("conv1.bias", B1);

	/* conv2 kernel [32,16,5,5] + bias [32]. */
	int k2_numel = OC2 * OC1 * K * K;
	double* k2 = (double*)malloc((size_t)k2_numel * sizeof(double));
	for (int i = 0; i < k2_numel; i++)
		k2[i] = ((i % 13) - 6) * 0.0078125;
	double* b2 = (double*)calloc(OC2, sizeof(double));
	TensorHandle K2 = tensor_create_param_4d_f64(OC2, OC1, K, K, k2);
	TensorHandle B2 = tensor_create_param_1d_f64(OC2, b2);
	param_register("conv2.weight", K2);
	param_register("conv2.bias", B2);

	/* linear [10,512] + bias [10]. */
	double* wl = (double*)malloc((size_t)CLS * FLAT * sizeof(double));
	for (int i = 0; i < CLS * FLAT; i++)
		wl[i] = ((i % 19) - 9) * 0.001953125;
	double* bl = (double*)calloc(CLS, sizeof(double));
	TensorHandle WL = tensor_create_param_2d_f64(CLS, FLAT, wl);
	TensorHandle BL = tensor_create_param_1d_f64(CLS, bl);
	param_register("fc.weight", WL);
	param_register("fc.bias", BL);

	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);

	/* Forward — mirrors Example/Mnist.idr's Seq exactly. */
	optimizer_zero_grad(opt);
	TensorHandle c1 = tensor_conv2d_batched(in, K1, B1, 0, 0, 1, 1); /* [64,16,24,24] */
	TensorHandle r1 = relu(c1);
	TensorHandle p1 = tensor_max_pool2d_batched(r1, 2, 2, 2, 2);     /* [64,16,12,12] */
	TensorHandle c2 = tensor_conv2d_batched(p1, K2, B2, 0, 0, 1, 1); /* [64,32,8,8] */
	TensorHandle r2 = relu(c2);
	TensorHandle p2 = tensor_max_pool2d_batched(r2, 2, 2, 2, 2); /* [64,32,4,4] */
	TensorHandle flat = tensor_reshape_2d(p2, B, FLAT);          /* [64,512] */
	TensorHandle y = tensor_linear_2d(WL, flat, BL);             /* [64,10] */
	TensorHandle loss = tensor_sum(y);

	/* Backward + step — this is where the campaign crash fired. */
	tensor_backward(loss);
	optimizer_step(opt);

	/* If we got here without an ASan trap / SIGSEGV, the path is sound.
	   Sanity-check a gradient actually flowed to the first conv kernel. */
	double g = param_grad_item_at(0, 0);
	cr_assert(g == g, "conv1.weight grad[0] is NaN"); /* NaN != NaN */

	optimizer_free(opt);
	free(in_data); /* tensor_create copies; the rest are param buffers the
	                  tensor_create_param_* contract already freed. */
	param_clear();
}

/* Higher-fidelity replay of Example/Mnist.idr: adds the pieces the kernel-
 * only test above omits — dropout(0.5), the real tnllLossMean loss
 * (log_softmax_2d * one-hot, summed, negated, scaled by 1/(b*n)), and the
 * fused native_train_step with NormClip(1.0) (clip_mode=2) that the example's
 * `adam {clip := NormClip 1.0}` drives. Narrows whether the campaign crash
 * lives in the C path at all, or purely in the Idris linear-surface wiring. */
Test(conv_mnist_lenet_step, faithful_loss_dropout_normclip_step) {
	param_clear();

	enum { B = 64, IMG = 28, INC = 1, OC1 = 16, OC2 = 32, K = 5 };
	enum { C1 = IMG - K + 1, P1 = (C1 - 2) / 2 + 1, C2 = P1 - K + 1, P2 = (C2 - 2) / 2 + 1 };
	enum { FLAT = OC2 * P2 * P2, CLS = 10 };

	int in_numel = B * INC * IMG * IMG;
	double* in_data = (double*)malloc((size_t)in_numel * sizeof(double));
	for (int i = 0; i < in_numel; i++)
		in_data[i] = ((i % 17) - 8) * 0.0625;
	int sh_in[4] = {B, INC, IMG, IMG};

	int k1_numel = OC1 * INC * K * K;
	double* k1 = (double*)malloc((size_t)k1_numel * sizeof(double));
	for (int i = 0; i < k1_numel; i++)
		k1[i] = ((i % 13) - 6) * 0.03125;
	TensorHandle K1 = tensor_create_param_4d_f64(OC1, INC, K, K, k1);
	TensorHandle B1 = tensor_create_param_1d_f64(OC1, (double*)calloc(OC1, sizeof(double)));
	param_register("conv1.weight", K1);
	param_register("conv1.bias", B1);

	int k2_numel = OC2 * OC1 * K * K;
	double* k2 = (double*)malloc((size_t)k2_numel * sizeof(double));
	for (int i = 0; i < k2_numel; i++)
		k2[i] = ((i % 13) - 6) * 0.0078125;
	TensorHandle K2 = tensor_create_param_4d_f64(OC2, OC1, K, K, k2);
	TensorHandle B2 = tensor_create_param_1d_f64(OC2, (double*)calloc(OC2, sizeof(double)));
	param_register("conv2.weight", K2);
	param_register("conv2.bias", B2);

	double* wl = (double*)malloc((size_t)CLS * FLAT * sizeof(double));
	for (int i = 0; i < CLS * FLAT; i++)
		wl[i] = ((i % 19) - 9) * 0.001953125;
	TensorHandle WL = tensor_create_param_2d_f64(CLS, FLAT, wl);
	TensorHandle BL = tensor_create_param_1d_f64(CLS, (double*)calloc(CLS, sizeof(double)));
	param_register("fc.weight", WL);
	param_register("fc.bias", BL);

	/* One-hot target [64,10] (class i%10 for row i). */
	double* tgt_data = (double*)calloc((size_t)B * CLS, sizeof(double));
	for (int r = 0; r < B; r++)
		tgt_data[r * CLS + (r % CLS)] = 1.0;
	int sh_tgt[2] = {B, CLS};

	OptimizerHandle opt = optimizer_create_adam(0.001, 0.9, 0.999, 1e-8);

	/* Loop several steps. The arena bug needs a prior step's arena_reset to
	   leave a recyclable undersized chunk in the chain; conv1's 4.5MB output
	   then overruns it (RED: ASan heap-buffer-overflow at arena.c make_tensor).
	   The batch tensors are arena-allocated and invalidated by each step's
	   reset, so they're recreated per step — exactly as the fit loop pulls a
	   fresh batch each iteration. */
	for (int step = 0; step < 6; step++) {
		/* Fresh batch each step (arena-backed; tensor_create copies in_data). */
		TensorHandle in = tensor_create(in_data, sh_in, 4, 0);
		TensorHandle tgt = tensor_create(tgt_data, sh_tgt, 2, 0);

		optimizer_zero_grad(opt);
		TensorHandle c1 = tensor_conv2d_batched(in, K1, B1, 0, 0, 1, 1);
		TensorHandle r1 = relu(c1);
		TensorHandle p1 = tensor_max_pool2d_batched(r1, 2, 2, 2, 2);
		TensorHandle c2 = tensor_conv2d_batched(p1, K2, B2, 0, 0, 1, 1);
		TensorHandle r2 = relu(c2);
		TensorHandle p2 = tensor_max_pool2d_batched(r2, 2, 2, 2, 2);
		TensorHandle flat = tensor_reshape_2d(p2, B, FLAT);
		TensorHandle drop = tensor_dropout(flat, 0.5, 1, 12345u + (unsigned)step);
		TensorHandle y = tensor_linear_2d(WL, drop, BL); /* [64,10] logits */

		/* tnllLossMean: -(tgt * log_softmax(y)).sum() / (b*n) */
		TensorHandle logp = tensor_log_softmax_2d(y);
		TensorHandle prod = tensor_mul(logp, tgt);
		TensorHandle loss =
		    tensor_mul_scalar(tensor_neg(tensor_sum(prod)), 1.0 / (double)(B * CLS));

		/* Fused zero+bwd+clip(norm,1.0)+step — clip_mode=2 is NormClip. */
		double lv = tensor_item(loss);
		native_train_step(opt, 2, 1.0, loss, lv);

		double g = param_grad_item_at(0, 0);
		cr_assert(g == g, "step %d: conv1.weight grad[0] is NaN", step);
	}

	optimizer_free(opt);
	free(in_data);
	free(tgt_data);
	param_clear();
}
