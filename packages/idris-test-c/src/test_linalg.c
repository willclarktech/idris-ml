/* Linear-algebra + structural tensor-op Criterion suite.
 *
 * Split out of the original bulk backend port. Shared assertion shims +
 * tolerances live in port_assert.h.
 */
#include "port_assert.h"


Test(linalg, scalar_creation) {
    TensorHandle a = tensor_create_scalar(3.0, 0);
    ASSERT_NEAR("scalar item", tensor_item(a), 3.0, 1e-10);
    ASSERT_NEAR("numel", (double)tensor_numel(a), 1.0, 1e-10);
    ASSERT_NEAR("dim", (double)tensor_dim(a), 0.0, 1e-10);
    ASSERT_TRUE("requires_grad=0", tensor_requires_grad(a) == 0);

    TensorHandle b = tensor_create_scalar(4.0, 1);
    ASSERT_TRUE("requires_grad=1", tensor_requires_grad(b) == 1);

    tensor_free(a);
    tensor_free(b);
}

Test(linalg, arithmetic) {
    TensorHandle a = tensor_create_scalar(3.0, 0);
    TensorHandle b = tensor_create_scalar(4.0, 0);

    TensorHandle c;
    c = tensor_add(a, b);
    ASSERT_NEAR("3+4", tensor_item(c), 7.0, 1e-10);
    tensor_free(c);

    c = tensor_sub(a, b);
    ASSERT_NEAR("3-4", tensor_item(c), -1.0, 1e-10);
    tensor_free(c);

    c = tensor_mul(a, b);
    ASSERT_NEAR("3*4", tensor_item(c), 12.0, 1e-10);
    tensor_free(c);

    c = tensor_div(a, b);
    ASSERT_NEAR("3/4", tensor_item(c), 0.75, 1e-10);
    tensor_free(c);

    c = tensor_neg(a);
    ASSERT_NEAR("neg(3)", tensor_item(c), -3.0, 1e-10);
    tensor_free(c);

    c = tensor_abs(tensor_create_scalar(-5.0, 0));
    ASSERT_NEAR("abs(-5)", tensor_item(c), 5.0, 1e-10);
    tensor_free(c);

    c = tensor_exp(a);
    ASSERT_NEAR("exp(3)", tensor_item(c), exp(3.0), 1e-5);  /* MLX Metal: float32 transcendentals */
    tensor_free(c);

    c = tensor_log(b);
    ASSERT_NEAR("log(4)", tensor_item(c), log(4.0), 1e-5);  /* MLX Metal: float32 transcendentals */
    tensor_free(c);

    c = tensor_sqrt(b);
    ASSERT_NEAR("sqrt(4)", tensor_item(c), 2.0, 1e-10);
    tensor_free(c);

    c = tensor_pow(a, tensor_create_scalar(2.0, 0));
    ASSERT_NEAR("3^2", tensor_item(c), 9.0, 1e-10);
    tensor_free(c);

    c = tensor_sigmoid(a);
    ASSERT_NEAR("sigmoid(3)", tensor_item(c), 1.0/(1.0+exp(-3.0)), 1e-5);  /* MLX Metal: float32 transcendentals */
    tensor_free(c);

    c = tensor_tanh(a);
    ASSERT_NEAR("tanh(3)", tensor_item(c), tanh(3.0), 1e-5);  /* MLX Metal: float32 transcendentals */
    tensor_free(c);

    c = tensor_add_scalar(a, 10.0);
    ASSERT_NEAR("3+10", tensor_item(c), 13.0, 1e-10);
    tensor_free(c);

    c = tensor_mul_scalar(a, 2.5);
    ASSERT_NEAR("3*2.5", tensor_item(c), 7.5, 1e-10);
    tensor_free(c);

    tensor_free(a);
    tensor_free(b);
}

/* ================================================================
   T3: Multi-dimensional tensors + linalg
   ================================================================ */

Test(linalg, multidim) {
    double mat_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int mat_shape[] = {2, 3};
    TensorHandle mat = tensor_create(mat_data, mat_shape, 2, 0);
    ASSERT_NEAR("mat dim", (double)tensor_dim(mat), 2.0, 1e-10);
    ASSERT_NEAR("mat size(0)", (double)tensor_size(mat, 0), 2.0, 1e-10);
    ASSERT_NEAR("mat size(1)", (double)tensor_size(mat, 1), 3.0, 1e-10);
    ASSERT_NEAR("mat numel", (double)tensor_numel(mat), 6.0, 1e-10);

    double vec_data[] = {1.0, 0.0, -1.0};
    int vec_shape[] = {3};
    TensorHandle vec = tensor_create(vec_data, vec_shape, 1, 0);

    /* mv: [2,3] x [3] = [2]: [1*1+2*0+3*(-1), 4*1+5*0+6*(-1)] = [-2, -2] */
    TensorHandle mv = tensor_mv(mat, vec);
    ASSERT_NEAR("mv[0]", tensor_item(tensor_select(mv, 0, 0)), -2.0, 1e-10);
    ASSERT_NEAR("mv[1]", tensor_item(tensor_select(mv, 0, 1)), -2.0, 1e-10);

    tensor_free(mat); tensor_free(vec); tensor_free(mv);
}

/* ================================================================
   T7: Transformer ops (matrix multiply, transpose, softmax_2d)
   ================================================================ */

Test(linalg, mm_and_transpose) {
    param_clear();

    /* A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2) */
    double a_data[] = {1,2,3, 4,5,6};
    double b_data[] = {7,8, 9,10, 11,12};
    int a_shape[] = {2, 3};
    int b_shape[] = {3, 2};
    TensorHandle a = tensor_create(a_data, a_shape, 2, 0);
    TensorHandle b = tensor_create(b_data, b_shape, 2, 0);

    /* C = A @ B = [[58,64],[139,154]] */
    TensorHandle c = tensor_mm(a, b);
    ASSERT_NEAR("mm[0,0]", tensor_item_2d(c, 0, 0), 58.0, 1e-10);
    ASSERT_NEAR("mm[0,1]", tensor_item_2d(c, 0, 1), 64.0, 1e-10);
    ASSERT_NEAR("mm[1,0]", tensor_item_2d(c, 1, 0), 139.0, 1e-10);
    ASSERT_NEAR("mm[1,1]", tensor_item_2d(c, 1, 1), 154.0, 1e-10);

    /* Transpose: A^T should be [[1,4],[2,5],[3,6]] (3x2) */
    TensorHandle at = tensor_transpose_2d(a);
    ASSERT_NEAR("transpose[0,0]", tensor_item_2d(at, 0, 0), 1.0, 1e-10);
    ASSERT_NEAR("transpose[0,1]", tensor_item_2d(at, 0, 1), 4.0, 1e-10);
    ASSERT_NEAR("transpose[2,1]", tensor_item_2d(at, 2, 1), 6.0, 1e-10);
}

Test(linalg, concat_2d_axis1_matches_per_sample) {
    /* For each row, prim__cat2 of [a-row] and [b-row] should equal the
       corresponding row of tensor_concat_2d_axis1(A, B). */
    double a_data[] = {1.0, 2.0, 3.0, -1.0, 0.5, 0.25};
    double b_data[] = {0.7, -0.8};
    int a_shape[] = {2, 3};
    int b_shape[] = {2, 1};
    TensorHandle A = tensor_create(a_data, a_shape, 2, 0);
    TensorHandle B = tensor_create(b_data, b_shape, 2, 0);
    TensorHandle Y = tensor_concat_2d_axis1(A, B);

    /* Per-sample */
    double a0[] = {1.0, 2.0, 3.0}; int a_row_shape[] = {3};
    double a1[] = {-1.0, 0.5, 0.25};
    double b0[] = {0.7}; double b1[] = {-0.8}; int b_row_shape[] = {1};
    TensorHandle A0 = tensor_create(a0, a_row_shape, 1, 0);
    TensorHandle B0 = tensor_create(b0, b_row_shape, 1, 0);
    TensorHandle A1 = tensor_create(a1, a_row_shape, 1, 0);
    TensorHandle B1 = tensor_create(b1, b_row_shape, 1, 0);
    TensorHandle row0 = tensor_cat2(A0, B0);
    TensorHandle row1 = tensor_cat2(A1, B1);

    for (int j = 0; j < 4; j++) {
        char msg[32];
        snprintf(msg, 32, "row0[%d]==Y[0,%d]", j, j);
        ASSERT_NEAR(msg, tensor_item_2d(Y, 0, j), tensor_item_1d(row0, j), 1e-9);
    }
    for (int j = 0; j < 4; j++) {
        char msg[32];
        snprintf(msg, 32, "row1[%d]==Y[1,%d]", j, j);
        ASSERT_NEAR(msg, tensor_item_2d(Y, 1, j), tensor_item_1d(row1, j), 1e-9);
    }
}

/* Test: narrow → mm → cat → sum backward gradient check.
   This mimics the batched transformer forward pattern:
   - Input [2*3, 2] (2 sequences of 3 positions, dModel=2)
   - Narrow to get two [3,2] slices
   - mm each slice with a shared weight [2,2]
   - Cat results back to [2*3, 2]
   - Sum → scalar loss → backward
   Compare weight gradient against finite difference. */
Test(linalg, narrow_cat_gradient) {
    param_clear();

    /* Input: [6,2] = 2 sequences of [3,2] */
    double x_data[] = {1,2, 3,4, 5,6,   7,8, 9,10, 11,12};
    int x_shape[] = {6, 2};
    /* Weight: [2,2] (shared across both sequences) */
    double w_data[] = {0.1, 0.2, 0.3, 0.4};
    int w_shape[] = {2, 2};

    /* Analytical path: narrow→mm→cat→sum→backward */
    TensorHandle x = tensor_create(x_data, x_shape, 2, 0);
    TensorHandle w = tensor_create(w_data, w_shape, 2, 1);
    param_register("w", w);

    /* Flatten x to 1D for narrow */
    int x_flat_shape[] = {12};
    TensorHandle x_flat = tensor_reshape(x, x_flat_shape, 1);
    /* Slice: seq0 = x[0:6], seq1 = x[6:12] */
    TensorHandle s0_flat = tensor_narrow(x_flat, 0, 0, 6);
    TensorHandle s1_flat = tensor_narrow(x_flat, 0, 6, 6);
    /* Reshape to [3,2] */
    int seq_shape[] = {3, 2};
    TensorHandle s0 = tensor_reshape(s0_flat, seq_shape, 2);
    TensorHandle s1 = tensor_reshape(s1_flat, seq_shape, 2);
    /* MM with shared weight: s0 @ w^T, s1 @ w^T */
    TensorHandle wt = tensor_transpose_2d(w);
    TensorHandle o0 = tensor_mm(s0, wt);
    TensorHandle o1 = tensor_mm(s1, wt);
    /* Flatten outputs and cat */
    int o_flat_shape[] = {6};
    TensorHandle o0_flat = tensor_reshape(o0, o_flat_shape, 1);
    TensorHandle o1_flat = tensor_reshape(o1, o_flat_shape, 1);
    TensorHandle catted = tensor_cat2(o0_flat, o1_flat);
    /* Sum → loss */
    TensorHandle loss = tensor_sum(catted);
    tensor_backward(loss);

    double grad_w00 = param_grad_item_at(0, 0);
    printf("  Analytical w[0,0] grad = %f\n", grad_w00);

    /* Reference: just do mm on full [6,2] without narrow/cat */
    param_clear();
    TensorHandle x2 = tensor_create(x_data, x_shape, 2, 0);
    TensorHandle w2 = tensor_create(w_data, w_shape, 2, 1);
    param_register("w2", w2);
    TensorHandle wt2 = tensor_transpose_2d(w2);
    TensorHandle out_full = tensor_mm(x2, wt2);
    TensorHandle loss_full = tensor_sum(out_full);
    tensor_backward(loss_full);
    double grad_w00_ref = param_grad_item_at(0, 0);
    printf("  Reference w[0,0] grad  = %f\n", grad_w00_ref);

    ASSERT_NEAR("narrow-cat grad matches direct", grad_w00, grad_w00_ref, 1e-6);

    /* Test with requires_grad input (like layer norm output) */
    param_clear();
    TensorHandle x_rg = tensor_create(x_data, x_shape, 2, 1);
    param_register("x_rg", x_rg);
    TensorHandle w_rg = tensor_create(w_data, w_shape, 2, 1);
    param_register("w_rg", w_rg);
    {
        int xf_shape[] = {12};
        TensorHandle xf = tensor_reshape(x_rg, xf_shape, 1);
        TensorHandle s0f = tensor_narrow(xf, 0, 0, 6);
        TensorHandle s1f = tensor_narrow(xf, 0, 6, 6);
        int ss[] = {3, 2};
        TensorHandle s0r = tensor_reshape(s0f, ss, 2);
        TensorHandle s1r = tensor_reshape(s1f, ss, 2);
        TensorHandle wtr = tensor_transpose_2d(w_rg);
        TensorHandle o0r = tensor_mm(s0r, wtr);
        TensorHandle o1r = tensor_mm(s1r, wtr);
        int of_shape[] = {6};
        TensorHandle o0f = tensor_reshape(o0r, of_shape, 1);
        TensorHandle o1f = tensor_reshape(o1r, of_shape, 1);
        TensorHandle catr = tensor_cat2(o0f, o1f);
        TensorHandle lossr = tensor_sum(catr);
        tensor_backward(lossr);

        double grad_x00 = param_grad_item_at(0, 0);
        double grad_w00_rg = param_grad_item_at(1, 0);
        printf("  rg: x[0,0] grad = %f, w[0,0] grad = %f\n", grad_x00, grad_w00_rg);
        ASSERT_NEAR("rg x grad", grad_x00, 0.1 + 0.3, 1e-6);  /* sum of w col 0 */
        ASSERT_NEAR("rg w grad (same)", grad_w00_rg, grad_w00_ref, 1e-6);
    }
    param_clear();

    /* Also finite diff check */
    double eps = 1e-5;
    {
        double w_copy[4]; memcpy(w_copy, w_data, sizeof(w_data));
        param_clear();
        w_copy[0] = w_data[0] + eps;
        TensorHandle xp = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle wp = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle wtp = tensor_transpose_2d(wp);
        double f_plus = tensor_item(tensor_sum(tensor_mm(xp, wtp)));
        w_copy[0] = w_data[0] - eps;
        TensorHandle xm = tensor_create(x_data, x_shape, 2, 0);
        TensorHandle wm = tensor_create(w_copy, w_shape, 2, 0);
        TensorHandle wtm = tensor_transpose_2d(wm);
        double f_minus = tensor_item(tensor_sum(tensor_mm(xm, wtm)));
        double fd = (f_plus - f_minus) / (2 * eps);
        printf("  Finite diff w[0,0]     = %f\n", fd);
        ASSERT_NEAR("narrow-cat grad vs finite diff", grad_w00, fd, FD_TOL);
    }
    param_clear();
}

Test(linalg, gather_scatter) {
    double data[] = {10, 20, 30, 40, 50};
    int ds[] = {5};
    TensorHandle t = tensor_create(data, ds, 1, 0);
    double idx_d[] = {2, 0, 4};
    int is[] = {3};
    TensorHandle idx = tensor_create(idx_d, is, 1, 0);
    TensorHandle g = tensor_gather(t, idx, 3);
    double gr[3];
    tensor_to_doubles(g, gr);
    ASSERT_NEAR("gather[0]", gr[0], 30.0, 1e-10);
    ASSERT_NEAR("gather[1]", gr[1], 10.0, 1e-10);
    ASSERT_NEAR("gather[2]", gr[2], 50.0, 1e-10);

    double src_d[] = {1, 2, 3};
    TensorHandle src = tensor_create(src_d, is, 1, 0);
    TensorHandle s = tensor_scatter_add(idx, src, 5);
    double sr[5];
    tensor_to_doubles(s, sr);
    ASSERT_NEAR("scatter[0]", sr[0], 2.0, 1e-10);
    ASSERT_NEAR("scatter[2]", sr[2], 1.0, 1e-10);
    ASSERT_NEAR("scatter[4]", sr[4], 3.0, 1e-10);
    ASSERT_NEAR("scatter[1]", sr[1], 0.0, 1e-10);
}

Test(linalg, argsort_cumprod) {
    param_clear();

    /* Argsort ascending */
    double data[] = {0.9, 0.1, 0.5, 0.3};
    int ds[] = {4};
    TensorHandle t = tensor_create(data, ds, 1, 0);
    TensorHandle sorted_idx = tensor_argsort(t, 0, 0); /* ascending */
    double idx_out[4];
    tensor_to_doubles(sorted_idx, idx_out);
    /* 0.1(idx1), 0.3(idx3), 0.5(idx2), 0.9(idx0) */
    ASSERT_NEAR("argsort[0]", idx_out[0], 1.0, 1e-10);
    ASSERT_NEAR("argsort[1]", idx_out[1], 3.0, 1e-10);
    ASSERT_NEAR("argsort[2]", idx_out[2], 2.0, 1e-10);
    ASSERT_NEAR("argsort[3]", idx_out[3], 0.0, 1e-10);
#if defined(BACKEND_TORCH)
    /* Type-safety guard: argsort must materialize *integer* indices, not
       a float dtype. The typed `targsort` Idris surface returns I64; this
       pins the C contract it rests on. tape/mlx store F64 by design (no
       integer Compatible instance), so this is torch-gated. */
    ASSERT_TRUE("argsort result is integral (I64)",
                strcmp(tensor_dtype_name(sorted_idx), "I64") == 0);
#endif

    /* Argsort descending */
    TensorHandle sorted_desc = tensor_argsort(t, 0, 1); /* descending */
    double desc_out[4];
    tensor_to_doubles(sorted_desc, desc_out);
    /* 0.9(idx0), 0.5(idx2), 0.3(idx3), 0.1(idx1) */
    ASSERT_NEAR("argsort_desc[0]", desc_out[0], 0.0, 1e-10);
    ASSERT_NEAR("argsort_desc[1]", desc_out[1], 2.0, 1e-10);

    /* Cumprod forward */
    double cp_data[] = {2.0, 3.0, 4.0};
    int cp_s[] = {3};
    TensorHandle cp_in = tensor_create(cp_data, cp_s, 1, 1);
    param_register("cp_in", cp_in);
    TensorHandle cp_out = tensor_cumprod(cp_in, 0);
    double cp_result[3];
    tensor_to_doubles(cp_out, cp_result);
    ASSERT_NEAR("cumprod[0]", cp_result[0], 2.0, 1e-10);
    ASSERT_NEAR("cumprod[1]", cp_result[1], 6.0, 1e-10);
    ASSERT_NEAR("cumprod[2]", cp_result[2], 24.0, 1e-10);

    /* Cumprod backward */
    TensorHandle cp_loss = tensor_sum(cp_out);
    tensor_backward(cp_loss);
    /* d_in[0] = d_out[0]*1 + d_out[1]*3 + d_out[2]*12 = 1 + 3 + 12 = 16 */
    /* d_in[1] = d_out[1]*2 + d_out[2]*8 = 2 + 8 = 10 */
    /* d_in[2] = d_out[2]*6 = 6 */
    ASSERT_NEAR("d_cumprod[0]", param_grad_item_at(0, 0), 16.0, 1e-6);
    ASSERT_NEAR("d_cumprod[1]", param_grad_item_at(0, 1), 10.0, 1e-6);
    ASSERT_NEAR("d_cumprod[2]", param_grad_item_at(0, 2), 6.0, 1e-6);

    param_clear();
}

Test(linalg, min_max_reductions) {
    double data[] = {3.0, -1.0, 7.0, 2.0, -5.0};
    int ds[] = {5};
    TensorHandle t = tensor_create(data, ds, 1, 0);
    TensorHandle mn = tensor_min(t);
    TensorHandle mx = tensor_max(t);
    ASSERT_NEAR("min([3,-1,7,2,-5])", tensor_item(mn), -5.0, 1e-10);
    ASSERT_NEAR("max([3,-1,7,2,-5])", tensor_item(mx), 7.0, 1e-10);
}

Test(linalg, batch_convenience) {
    double a[] = {1, 2}, b[] = {3, 4};
    int s[] = {2};
    TensorHandle ta = tensor_create(a, s, 1, 0);
    TensorHandle tb = tensor_create(b, s, 1, 0);
    TensorHandle in[] = {ta, tb};
    TensorHandle bt = tensor_batch(in, 2);
    ASSERT_NEAR("batch rank", (double)tensor_dim(bt), 2.0, 1e-10);
    ASSERT_NEAR("batch sz0", (double)tensor_size(bt, 0), 2.0, 1e-10);
    ASSERT_NEAR("batch sz1", (double)tensor_size(bt, 1), 2.0, 1e-10);
    double bout[4];
    tensor_to_doubles(bt, bout);
    ASSERT_NEAR("batch[0,0]", bout[0], 1.0, 1e-10);
    ASSERT_NEAR("batch[1,1]", bout[3], 4.0, 1e-10);
}

Test(linalg, unbatch) {
    param_clear();
    double d[] = {1, 2, 3, 4, 5, 6};
    int s[] = {3, 2};
    TensorHandle t = tensor_create(d, s, 2, 0);

    int n = 0;
    TensorHandle* parts = tensor_unbatch(t, &n);
    ASSERT_NEAR("unbatch count", (double)n, 3.0, 1e-10);
    double p0[2], p1[2], p2[2];
    tensor_to_doubles(parts[0], p0);
    tensor_to_doubles(parts[1], p1);
    tensor_to_doubles(parts[2], p2);
    ASSERT_NEAR("unbatch[0][0]", p0[0], 1.0, 1e-10);
    ASSERT_NEAR("unbatch[0][1]", p0[1], 2.0, 1e-10);
    ASSERT_NEAR("unbatch[1][0]", p1[0], 3.0, 1e-10);
    ASSERT_NEAR("unbatch[2][1]", p2[1], 6.0, 1e-10);
    free(parts);
    param_clear();
}
