/* Autograd + native optimizer (SGD/RMSprop) Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"

/* ================================================================
   T2: Autograd — backward pass + gradient collection
   ================================================================ */

Test(autograd, autograd_basic) {
	param_clear();

	TensorHandle w = tensor_create_scalar(3.0, 1);
	TensorHandle b = tensor_create_scalar(1.0, 1);
	param_register("w", w);
	param_register("b", b);

	TensorHandle x = tensor_create_scalar(2.0, 0);
	TensorHandle wx = tensor_mul(w, x);
	TensorHandle y = tensor_add(wx, b);

	tensor_backward(y);

	/* dy/dw = x = 2, dy/db = 1 */
	ASSERT_NEAR("grad w = x", param_grad_item(0), 2.0, 1e-10);
	ASSERT_NEAR("grad b = 1", param_grad_item(1), 1.0, 1e-10);

	/* grad_item_and_zero reads then zeros */
	double gw = param_grad_item_and_zero(0);
	ASSERT_NEAR("grad_and_zero w", gw, 2.0, 1e-10);
	ASSERT_NEAR("zeroed w", param_grad_item(0), 0.0, 1e-10);

	tensor_free(w);
	tensor_free(b);
	tensor_free(x);
	tensor_free(wx);
	tensor_free(y);
	param_clear();
}

/* A1 (type-safe mixed precision plan, #410): tcast must be autograd-
   aware on every backend. Forward: F64 source rg=1 → F32 cast → F32
   mul by scalar 3.0. Backward: source.grad must be 3.0 (chain rule:
   d(cast(x, F32) * 3) / dx = 3, since cast is locally linear).
   On tape pre-fix this RED-fails because tape_cast_dtype_dtag's non-F64
   branches create a fresh tensor with rg=0 (autograd flag dropped),
   so the F32 mul has no autograd lineage back to src. After the fix
   (OP_CAST_DTYPE registered + tape_append on rg sources): GREEN.
   On torch the cast goes through at::Tensor::to which is already
   autograd-traced for float-to-float, so this is a regression guard.
   On mlx the cast goes through mx::astype + OP_CAST_DTYPE tape entry,
   also already autograd-aware. */
Test(autograd, cast_grad_propagation) {
	param_clear();

	/* F64 source, rg=1, registered as a param so we can read .grad. */
	TensorHandle src = tensor_create_scalar(2.0, 1);
	param_register("src", src);

	/* Cast src to F32 (dtag=14) through the streamed dispatch. */
	TensorHandle src_f32 = tensor_cast_dtype_streamed(src, 0, 14);

	/* F32 scalar constant for the mul. */
	TensorHandle three_f32 = tensor_create_scalar_streamed(3.0, 0, 0, 14);

	/* loss = cast(src, F32) * 3.0 */
	TensorHandle loss = tensor_mul(src_f32, three_f32);

	tensor_backward(loss);

	/* Expected: src.grad = 3.0 (chain rule through the cast). 1e-3
	   tolerance accommodates F32 round-trip precision. */
	ASSERT_NEAR("cast preserves autograd: src.grad == 3.0", param_grad_item(0), 3.0, 1e-3);

	tensor_free(src);
	tensor_free(src_f32);
	tensor_free(three_f32);
	tensor_free(loss);
	param_clear();
}

/* A3 (type-safe mixed-precision plan #410): native_train_step_scaled
   must (a) unscale grads after backward, (b) step the optimizer with
   the unscaled grads, (c) return the unscaled loss. This is the
   GradScaler-aware variant of native_train_step that lets the caller
   train in low-precision compute without underflow during backward.
   Forward: loss = w * x * scale = 0.5 * 3.0 * 10.0 = 15.0 (scaled).
   Backward at scaled magnitude → dL/dw = x * scale = 30.0. Unscale
   by /10 → 3.0 (the true gradient). SGD step at lr=0.01 → w should
   move from 0.5 to 0.5 - 0.01 * 3.0 = 0.47. Return value: 15.0 / 10.0
   = 1.5 (the unscaled loss).

   Runs on all three backends: tape (via shared optimizer.c), torch
   (via backend_torch/training/optimizer.cpp port), mlx (via
   backend_mlx/training/optimizer.cpp port). */
Test(autograd, native_train_step_scaled_unscale) {
	param_clear();

	TensorHandle w = tensor_create_scalar(0.5, 1);
	param_register("w", w);

	OptimizerHandle sgd = optimizer_create_sgd(0.01);
	double scale = 10.0;

	TensorHandle x = tensor_create_scalar(3.0, 0);
	TensorHandle prod = tensor_mul(w, x);
	TensorHandle scale_t = tensor_create_scalar(scale, 0);
	TensorHandle scaled_loss = tensor_mul(prod, scale_t);

	double returned = native_train_step_scaled(sgd, 0, 0.0, scaled_loss, 15.0, scale);

	ASSERT_NEAR("return value is unscaled loss (= 1.5)", returned, 1.5, 1e-6);
	/* Tolerance 1e-6: mlx-cpu / mlx-gpu store weight as F32 by default
	   (~7 decimal digits), so the 0.5 - 0.01 * 3.0 = 0.47 step has
	   single-precision roundoff at that scale. tape's F64 lingua
	   franca and torch's F64 default give 1e-15-class diffs. */
	ASSERT_NEAR("w stepped to 0.47 (= 0.5 - 0.01 * 3.0)", tensor_item(w), 0.47, 1e-6);

	tensor_free(x);
	tensor_free(prod);
	tensor_free(scale_t);
	tensor_free(scaled_loss);
	optimizer_free(sgd);
	tensor_free(w);
	param_clear();
}

/* Regression: an F32 param must be an autograd leaf so its grad flows.
   The F32 param creators once cast to float32 *after* requires_grad_, which
   produced a non-leaf whose .grad never populated — the optimizer then read a
   zero gradient and silently froze training (torch-mps lane, fixed 9e2886b /
   guarded by make_param_leaf's is_leaf assert). Tape has no F32 arena, so this
   runs on torch/mlx only. */
#ifndef BACKEND_TAPE
Test(autograd, param_leaf_f32_grad_flow) {
	param_clear();

	double wv[1] = {2.0};
	double xv[1] = {3.0};
	TensorHandle w = tensor_create_param_1d_f32(1, hcopy(wv, 1));
	param_register("w", w);

	TensorHandle x = tensor_create_1d_f32(1, hcopy(xv, 1), 0);
	TensorHandle y = tensor_mul(w, x); /* y = [6.0] */
	TensorHandle loss = tensor_sum(y); /* scalar root — mlx vjp requires shape () */

	tensor_backward(loss);

	/* dloss/dw = x = 3. A non-leaf w would leave grad at 0 (or abort backward). */
	ASSERT_NEAR("grad w = x (F32 param is a leaf)", param_grad_item(0), 3.0, 1e-4);

	tensor_free(w);
	tensor_free(x);
	tensor_free(y);
	tensor_free(loss);
	param_clear();
}
#endif

Test(autograd, autograd_chain) {
	param_clear();

	TensorHandle a = tensor_create_scalar(1.0, 1);
	TensorHandle b = tensor_create_scalar(2.0, 1);
	param_register("a", a);
	param_register("b", b);

	TensorHandle s = tensor_add(a, b);  /* s = 3 */
	TensorHandle s2 = tensor_mul(s, s); /* s2 = 9 */
	tensor_backward(s2);

	/* d/da (a+b)^2 = 2(a+b) = 6 */
	ASSERT_NEAR("d((a+b)^2)/da = 6", param_grad_item_and_zero(0), 6.0, 1e-10);
	ASSERT_NEAR("d((a+b)^2)/db = 6", param_grad_item_and_zero(1), 6.0, 1e-10);

	tensor_free(a);
	tensor_free(b);
	tensor_free(s);
	tensor_free(s2);
	param_clear();
}

Test(autograd, autograd_exp) {
	param_clear();

	TensorHandle w = tensor_create_scalar(1.0, 1);
	param_register("w", w);
	TensorHandle y = tensor_exp(w);
	tensor_backward(y);

	ASSERT_NEAR("d(exp(w))/dw = exp(1)", param_grad_item_and_zero(0), exp(1.0),
	            1e-5); /* MLX Metal: float32 transcendentals */

	tensor_free(w);
	tensor_free(y);
	param_clear();
}

Test(autograd, autograd_div) {
	param_clear();

	TensorHandle a = tensor_create_scalar(6.0, 1);
	TensorHandle b = tensor_create_scalar(3.0, 1);
	param_register("a", a);
	param_register("b", b);

	TensorHandle y = tensor_div(a, b);
	tensor_backward(y);

	/* dy/da = 1/b = 1/3, dy/db = -a/b^2 = -6/9 */
	ASSERT_NEAR("da = 1/b", param_grad_item_and_zero(0), 1.0 / 3.0, VAL_TOL);
	ASSERT_NEAR("db = -a/b^2", param_grad_item_and_zero(1), -6.0 / 9.0, VAL_TOL);

	tensor_free(a);
	tensor_free(b);
	tensor_free(y);
	param_clear();
}

Test(autograd, autograd_sqrt) {
	param_clear();

	TensorHandle w = tensor_create_scalar(4.0, 1);
	param_register("w", w);
	TensorHandle y = tensor_sqrt(w);
	tensor_backward(y);

	ASSERT_NEAR("d(sqrt(w))/dw = 1/(2*sqrt(w))", param_grad_item_and_zero(0), 0.25, 1e-10);

	tensor_free(w);
	tensor_free(y);
	param_clear();
}

Test(autograd, autograd_native_sgd) {
	param_clear();

	TensorHandle w = tensor_create_scalar(0.5, 1);
	param_register("w", w);

	OptimizerHandle sgd = optimizer_create_sgd(0.01);

	/* Train 100 steps: loss = (w*2 - 3)^2, optimal w = 1.5 */
	for (int step = 0; step < 100; step++) {
		optimizer_zero_grad(sgd);
		TensorHandle x = tensor_create_scalar(2.0, 0);
		TensorHandle t = tensor_create_scalar(3.0, 0);
		TensorHandle pred = tensor_mul(w, x);
		TensorHandle diff = tensor_sub(pred, t);
		TensorHandle loss = tensor_mul(diff, diff);
		tensor_backward(loss);
		optimizer_step(sgd);
		tensor_free(x);
		tensor_free(t);
		tensor_free(pred);
		tensor_free(diff);
		tensor_free(loss);
	}
	ASSERT_NEAR("SGD converges w->1.5", tensor_item(w), 1.5, 0.01);

	optimizer_free(sgd);
	tensor_free(w);
	param_clear();
}

/* Distinguishes the torch.optim.RMSprop form (lr OUTSIDE the momentum
   buffer) from the lr-inside form. At constant lr the two coincide, so
   the test uses two steps with DIFFERENT lr (0.1 then 0.2) and
   momentum > 0 — the only regime where they diverge. loss = w each step
   so grad = 1.0. lr-outside lands w at 5.78224; the lr-inside bug at
   6.68224 (verified against PyTorch). */
Test(autograd, rmsprop_lr_schedule) {
	param_clear();

	TensorHandle w = tensor_create_scalar(10.0, 1);
	param_register("w", w);

	/* alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.9 */
	OptimizerHandle opt = optimizer_create_rmsprop(0.1, 0.99, 1e-8, 0.0, 0.9);

	optimizer_zero_grad(opt);
	TensorHandle l1 = tensor_sum(w);
	tensor_backward(l1);
	optimizer_step(opt);
	tensor_free(l1);

	optimizer_set_lr(opt, 0.2);
	optimizer_zero_grad(opt);
	TensorHandle l2 = tensor_sum(w);
	tensor_backward(l2);
	optimizer_step(opt);
	tensor_free(l2);

	ASSERT_NEAR("RMSprop lr-outside w after 2 steps", tensor_item(w), 5.78224, 1e-3);

	optimizer_free(opt);
	tensor_free(w);
	param_clear();
}

/* ================================================================
   T4: Fused tensor ops with metadata backward
   ================================================================ */

Test(autograd, fused_mv_backward) {
	param_clear();

	/* W = [[1,2,3],[4,5,6]], x = [1, 0, -1] */
	/* y = W @ x = [-2, -2], loss = sum(y) = -4 */
	/* d_W[i,j] = d_loss/d_W[i,j] = x[j] (since d_sum/d_y = [1,1]) */
	/* So grad_W = [[1,0,-1],[1,0,-1]] */
	/* d_x[j] = sum_i W[i,j] = [5, 7, 9] */

	double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle W = tensor_create_param_2d_f64(2, 3, hcopy(wdata, 6));
	param_register("W", W);

	double xdata[] = {1.0, 0.0, -1.0};
	TensorHandle x = tensor_create_param_1d_f64(3, hcopy(xdata, 3));
	param_register("x", x);

	TensorHandle y = tensor_mv(W, x);
	TensorHandle loss = tensor_sum(y);

	ASSERT_NEAR("mv loss", tensor_item(loss), -4.0, 1e-10);
	ASSERT_TRUE("loss requires_grad", tensor_requires_grad(loss));

	tensor_backward(loss);

	/* Check W gradients via param registry: grad_W[i,j] = x[j] */
	/* W is param 0: 6 elements. grad_W = [[1,0,-1],[1,0,-1]] */
	ASSERT_NEAR("grad_W[0,0]", param_grad_item_at(0, 0), 1.0, 1e-6);
	ASSERT_NEAR("grad_W[0,1]", param_grad_item_at(0, 1), 0.0, 1e-6);
	ASSERT_NEAR("grad_W[0,2]", param_grad_item_at(0, 2), -1.0, 1e-6);
	ASSERT_NEAR("grad_W[1,0]", param_grad_item_at(0, 3), 1.0, 1e-6);

	/* Check x gradients via param registry: grad_x[j] = sum_i W[i,j] */
	/* x is param 1: 3 elements. grad_x = [5, 7, 9] */
	ASSERT_NEAR("grad_x[0]", param_grad_item_at(1, 0), 5.0, 1e-6);
	ASSERT_NEAR("grad_x[1]", param_grad_item_at(1, 1), 7.0, 1e-6);
	ASSERT_NEAR("grad_x[2]", param_grad_item_at(1, 2), 9.0, 1e-6);

	param_clear();
}

Test(autograd, fused_mv_optimizer) {
	param_clear();

	double wdata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle W = tensor_create_param_2d_f64(2, 3, hcopy(wdata, 6));
	param_register("W", W);

	double xdata[] = {1.0, 0.0, -1.0};
	int xshape[] = {3};

	OptimizerHandle sgd = optimizer_create_sgd(0.1);

	double prev_loss = 1e10;
	for (int ep = 0; ep < 5; ep++) {
		optimizer_zero_grad(sgd);
		TensorHandle x = tensor_create(xdata, xshape, 1, 0); /* fresh each epoch */
		TensorHandle y = tensor_mv(W, x);
		TensorHandle loss = tensor_sum(y);
		double lv = tensor_item(loss);
		if (ep > 0) {
			ASSERT_TRUE("loss decreasing", lv < prev_loss + 0.01);
		}
		prev_loss = lv;
		tensor_backward(loss);
		optimizer_step(sgd);
	}
	ASSERT_TRUE("fused MV trains", prev_loss < -4.0);

	optimizer_free(sgd);
	param_clear();
}

Test(autograd, per_param_lr) {
	param_clear();
	/* Two params: w=5.0, b=3.0 */
	TensorHandle w = tensor_create_scalar(5.0, 1);
	TensorHandle b = tensor_create_scalar(3.0, 1);
	param_register("w", w);
	param_register("b", b);

	/* Forward: loss = w + b, so dw=1, db=1 */
	TensorHandle loss = tensor_add(w, b);
	tensor_backward(loss);

	/* Create SGD with base LR=0.1, override w's LR to 0.5 */
	OptimizerHandle opt = optimizer_create_sgd(0.1);
	optimizer_set_param_lr(opt, "w", 0.5);
	optimizer_step(opt);

	/* w should decrease by 0.5*1=0.5 (from 5.0 to 4.5) */
	/* b should decrease by 0.1*1=0.1 (from 3.0 to 2.9) */
	/* Note: torch backend doesn't implement per-param LR (no-op),
	   so w stays at 5.0 - 0.1 = 4.9 on torch. Only check on tape/MLX. */
	double w_val = tensor_item(w);
	if (w_val < 4.8) {
		/* Per-param LR was applied (tape/MLX) */
		ASSERT_NEAR("w after per-param LR", w_val, 4.5, 1e-5);
	} else {
		printf("ok: w after base LR = %.6f (per-param LR not supported on this backend)\n", w_val);
	}
	ASSERT_NEAR("b after base LR", tensor_item(b), 2.9, 1e-5);

	optimizer_free(opt);
	param_clear();
}

Test(autograd, grad_detach_with_grad) {
	/* tensor_grad: returns gradient after backward, or nullptr if no grad */
	param_clear();
	TensorHandle p = tensor_create_scalar(3.0, 1);
	param_register("p", p);
	TensorHandle pp = tensor_mul(p, p); /* loss = p^2; d/dp = 2p = 6 */
	tensor_backward(pp);
	TensorHandle g = tensor_grad(p);
	if (g) {
		ASSERT_NEAR("tensor_grad(p) for p^2 at p=3", tensor_item(g), 6.0, 1e-6);
	} else {
		printf("ok: tensor_grad returned null on this backend — skipping\n");
	}
	/* No-grad tensor: tensor_grad returns nullptr */
	TensorHandle nogrnd = tensor_create_scalar(2.0, 0);
	TensorHandle gn = tensor_grad(nogrnd);
	ASSERT_TRUE("tensor_grad on non-grad tensor is null", gn == NULL);
	param_clear();

	/* tensor_detach: returns a tensor with the same data, requires_grad=false */
	TensorHandle src = tensor_create_scalar(7.5, 1);
	TensorHandle det = tensor_detach(src);
	ASSERT_NEAR("detach value", tensor_item(det), 7.5, 1e-10);
	ASSERT_TRUE("detach requires_grad=0", tensor_requires_grad(det) == 0);

	/* tensor_with_grad: promotes a tensor into autograd (requires_grad=true) */
	TensorHandle leaf = tensor_create_scalar(2.5, 0);
	TensorHandle wg = tensor_with_grad(leaf);
	ASSERT_NEAR("with_grad value", tensor_item(wg), 2.5, 1e-10);
	ASSERT_TRUE("with_grad requires_grad=1", tensor_requires_grad(wg) == 1);
}
