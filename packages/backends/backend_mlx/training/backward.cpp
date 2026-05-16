/* Replay-based backward — mlx.
 *
 * mlx's autograd is replay-based: forward ops push entries to a Wengert
 * tape (op_code, result_ptr, arg1_ptr, arg2_ptr, scalar_arg, meta_ptr)
 * — see tape.h. `tensor_backward` walks the tape inside mx::vjp,
 * building a pool of mx::array values indexed by `Tensor::pool_idx`
 * (params are inputs to the vjp, every other Tensor on the live tape is
 * a constant input), then applies the per-op forward rule into the pool
 * to reach the loss. mx::vjp differentiates that path automatically and
 * returns one gradient per param, which we write back into each param
 * Tensor's `grad` slot.
 *
 * The closure passed to mx::vjp signs the explicit-inputs contract that
 * mx::compile (controlled by `IDRIS_MLX_OPT_COMPILE`) can eventually
 * trace + cache — per-batch constant values aren't baked into the graph
 * at trace time because they're passed in as named arguments.
 *
 * Optional NaN trap (`DEBUG_NAN_TRAP=1`) walks param grads after the
 * vjp and names the offending param at first NaN/Inf; useful to localise
 * gradient blow-up at the peaked-attention working point in NTM/DNC.
 */
#include "../tensor.h"
#include "../tape.h"
#include "../precision.h"
#include "profiling.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

/* Param registry surface — defined in shared/training/param_registry.c. */
extern "C" int param_count(void);
extern "C" void* param_tensor(int i);
extern "C" const char* param_name(int i);

/* mx::compile invocation counter — incremented at the cached compile-path
   trace point. Non-static so the optimizer TU can read it. Definition
   lives in backend_mlx.cpp. */
extern int g_compile_invocations;

void tensor_backward(TensorHandle h) {
    double t0_bwd = _wall_ms_mlx();
    Tensor* loss = (Tensor*)h;
    if (loss->tape_idx < 0) { prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd; return; }

    // Collect param pool indices and arrays
    std::vector<int> param_pool_indices;
    std::vector<mx::array> param_arrays;
    for (int i_ = 0; i_ < param_count(); i_++) {
        auto* tensor = (Tensor*)param_tensor(i_);
        param_pool_indices.push_back(tensor->pool_idx);
        param_arrays.push_back(tensor->data);
    }
    if (param_arrays.empty()) return;

    // Build constant pool from tape (O(tape_size), not O(all_tensors)).
    // Index/mask args (OP_GATHER.arg2, OP_SCATTER_ADD.arg2) are discrete and
    // have no derivative — keep them out of the vjp inputs entirely.
    // Replay reads them via closure-captured `e.argN->data` (see below).
    std::vector<std::pair<int, mx::array>> constants;
    std::unordered_set<int> seen;
    for (auto& idx : param_pool_indices) seen.insert(idx);
    auto add_const = [&](Tensor* t) {
        if (t && !seen.count(t->pool_idx)) {
            seen.insert(t->pool_idx);
            constants.emplace_back(t->pool_idx, t->data);
        }
    };
    auto arg2_is_index = [](int op) {
        return op == OP_GATHER || op == OP_SCATTER_ADD;
    };
    for (int i = 0; i <= loss->tape_idx; i++) {
        auto& e = tape[i];
        add_const(e.result);
        add_const(e.arg1);
        if (!arg2_is_index(e.op)) add_const(e.arg2);
    }

    // Capture tape state for the closure
    int loss_pool_idx = loss->pool_idx;
    int loss_tape_idx = loss->tape_idx;
    auto tape_ref = &tape;

    // Job 3 Phase B — explicit-inputs forward. The closure takes
    // [params..., constants...] so that mx::compile (if enabled) does
    // NOT bake per-batch constant values into the compiled graph at
    // trace time. The eager path uses the same closure to keep both
    // paths in lockstep; vjp returns grads for all inputs, but only
    // the leading n_params are written back to param tensors.
    int n_params = (int)param_arrays.size();
    int n_consts = (int)constants.size();
    std::vector<int> constants_pool_indices;
    constants_pool_indices.reserve(n_consts);
    for (auto& [idx, arr] : constants) constants_pool_indices.push_back(idx);

    // Replay forward pass inside mlx::vjp
    int pool_size = next_pool_idx;
    auto forward_fn = [&](const std::vector<mx::array>& xs) -> mx::array {
        // xs[0..n_params) = params, xs[n_params..n_params+n_consts) = constants
        std::vector<mx::array> pool(pool_size, kF32_ZERO());
        for (int i = 0; i < n_params; i++)
            pool[param_pool_indices[i]] = xs[i];
        for (int i = 0; i < n_consts; i++)
            pool[constants_pool_indices[i]] = xs[n_params + i];

        for (int i = 0; i <= loss_tape_idx; i++) {
            auto& e = (*tape_ref)[i];
            int out = e.result->pool_idx;
            auto a = e.arg1 ? pool[e.arg1->pool_idx] : kF32_ZERO();
            auto b = e.arg2 ? pool[e.arg2->pool_idx] : kF32_ZERO();

            switch (e.op) {
            case OP_CONST: break;
            case OP_ADD: pool[out] = mx::add(a, b); break;
            case OP_SUB: pool[out] = mx::subtract(a, b); break;
            case OP_MUL: pool[out] = mx::multiply(a, b); break;
            case OP_DIV: pool[out] = mx::divide(a, b); break;
            case OP_NEG: pool[out] = mx::negative(a); break;
            case OP_ABS: pool[out] = mx::abs(a); break;
            case OP_EXP: pool[out] = mx::exp(a); break;
            case OP_LOG: pool[out] = mx::log(a); break;
            case OP_SQRT: pool[out] = mx::sqrt(a); break;
            case OP_POW: pool[out] = mx::power(a, b); break;
            case OP_SIGMOID: pool[out] = mx::sigmoid(a); break;
            case OP_TANH: pool[out] = mx::tanh(a); break;
            case OP_GELU: {
                auto kGeluC  = scalar_like(0.7978845608028654, a);
                auto kGeluC3 = scalar_like(0.044715,           a);
                auto kThree  = scalar_like(3.0,                a);
                auto inner = mx::multiply(kGeluC, mx::add(a, mx::multiply(kGeluC3, mx::power(a, kThree))));
                pool[out] = mx::multiply(mx::multiply(half_like(a), a), mx::add(one_like(a), mx::tanh(inner)));
                break;
            }
            case OP_LEAKY_RELU: {
                auto alpha = scalar_like(e.scalar_arg, a);
                pool[out] = mx::maximum(mx::multiply(alpha, a), a);
                break;
            }
            case OP_SILU: pool[out] = mx::multiply(a, mx::sigmoid(a)); break;
            case OP_SOFTPLUS: {
                pool[out] = mx::add(mx::maximum(a, zero_like(a)),
                                    mx::log(mx::add(one_like(a), mx::exp(mx::negative(mx::abs(a))))));
                break;
            }
            case OP_TILE_2D: {
                int* reps = (int*)e.meta;
                pool[out] = mx::tile(a, {reps[0], reps[1]});
                break;
            }
            case OP_CAST_DTYPE: {
                mx::Dtype target = (e.scalar_arg == 0.0 ? mx::float32 : mx::float64);
                pool[out] = mx::astype(a, target);
                break;
            }
            case OP_ADD_SCALAR: pool[out] = mx::add(a, scalar_like(e.scalar_arg, a)); break;
            case OP_MUL_SCALAR: pool[out] = mx::multiply(a, scalar_like(e.scalar_arg, a)); break;
            case OP_CLAMP_MIN: pool[out] = mx::maximum(a, scalar_like(e.scalar_arg, a)); break;
            case OP_SUM: pool[out] = mx::sum(a); break;
            case OP_MEAN: pool[out] = mx::mean(a); break;
            case OP_SUM_DIM: {
                auto* sm = (SumDimReplayMeta*)e.meta;
                pool[out] = mx::sum(a, std::vector<int>{sm->dim}, sm->keepdim != 0);
                break;
            }
            case OP_MM: case OP_BMM: case OP_BMM_3X3: pool[out] = mx::matmul(a, b); break;
            case OP_SOFTMAX_3D: pool[out] = mx::softmax(a, -1); break;
            case OP_TRANSPOSE_LAST2: pool[out] = mx::transpose(a, {0, 2, 1}); break;
            case OP_MV: {
                auto col = mx::reshape(b, {(int)b.size(), 1});
                pool[out] = mx::reshape(mx::matmul(a, col), {(int)a.shape(0)});
                break;
            }
            case OP_OUTER: pool[out] = mx::outer(a, b); break;
            case OP_TRANSPOSE_2D: pool[out] = mx::transpose(a, {1, 0}); break;
            case OP_SOFTMAX_2D: pool[out] = mx::softmax(a, -1); break;
            case OP_LOG_SOFTMAX_2D: {
                int dim = (int)e.scalar_arg;  // stored by forward (0 for 1D, -1 for 2D)
                auto maxv = mx::max(a, dim, true);
                auto shifted = mx::subtract(a, maxv);
                auto lse = mx::add(mx::log(mx::sum(mx::exp(shifted), dim, true)), maxv);
                pool[out] = mx::subtract(a, lse);
                break;
            }
            case OP_MASKED_FILL: {
                /* mask `b` may be bool; the fill value must match `a`'s
                   dtype so `mx::where` doesn't force a promotion. */
                auto kNegInfMask = scalar_like(-1e9, a);
                pool[out] = mx::where(b, kNegInfMask, a);
                break;
            }
            case OP_RESHAPE: pool[out] = mx::reshape(a, e.result->data.shape()); break;
            case OP_SELECT: pool[out] = mx::take(a, mx::array((int)e.scalar_arg), 0); break;
            case OP_NARROW: {
                int start = (int)e.scalar_arg;
                int len = (int)e.result->data.size();
                pool[out] = mx::slice(mx::flatten(a), {start}, {start + len});
                break;
            }
            case OP_CAT: pool[out] = mx::concatenate({a, b}, 0); break;
            case OP_STACK: {
                auto* indices = (std::vector<int>*)e.meta;
                if (indices) {
                    std::vector<mx::array> arrs;
                    for (int idx : *indices) arrs.push_back(pool[idx]);
                    pool[out] = mx::stack(arrs, (int)e.scalar_arg);
                }
                break;
            }
            case OP_CAT_MULTI: {
                auto* indices = (std::vector<int>*)e.meta;
                if (indices) {
                    std::vector<mx::array> arrs;
                    for (int idx : *indices) arrs.push_back(pool[idx]);
                    pool[out] = mx::concatenate(arrs, (int)e.scalar_arg);
                }
                break;
            }
            case OP_COSINE_SIM: {
                // Inline cosine similarity forward
                int n = (int)a.shape(0), m = (int)a.shape(1);
                auto key_2d = mx::reshape(b, {1, m});
                auto dots = mx::sum(mx::multiply(a, key_2d), std::vector<int>{1});
                auto eps = scalar_like(1.0e-8, a);
                auto row_norms = mx::sqrt(mx::add(mx::sum(mx::square(a), std::vector<int>{1}), eps));
                auto key_norm = mx::sqrt(mx::add(mx::sum(mx::square(b)), eps));
                pool[out] = mx::divide(dots, mx::multiply(row_norms, key_norm));
                break;
            }
            case OP_CONV1D_CIRC: {
                // Inline circular convolution forward
                int n = (int)a.size(), k = (int)b.size();
                int half_k = k / 2;
                auto result = mx::zeros({n}, a.dtype());
                for (int j = 0; j < k; j++) {
                    auto shifted = mx::roll(a, half_k - j);
                    auto kern_j = mx::take(b, mx::array(j));
                    result = mx::add(result, mx::multiply(shifted, kern_j));
                }
                pool[out] = result;
                break;
            }
            case OP_LAYER_NORM_2D: {
                auto meta = (LayerNormReplayMeta*)e.meta;
                auto gamma = pool[meta->gamma_pool_idx];
                auto bias = pool[meta->bias_pool_idx];
                auto mean = mx::mean(a, -1, true);
                auto centered = mx::subtract(a, mean);
                auto var = mx::mean(mx::square(centered), -1, true);
                auto rstd = mx::rsqrt(mx::add(var, scalar_like(meta->eps, var)));
                auto x_hat = mx::multiply(centered, rstd);
                pool[out] = mx::add(mx::multiply(gamma, x_hat), bias);
                break;
            }
            case OP_LINEAR_2D: {
                /* a = X [B,i], b = W [o,i]. Y = X @ W^T + bias */
                auto meta = (LinearReplayMeta*)e.meta;
                auto WT = mx::transpose(b, {1, 0});
                auto y = mx::matmul(a, WT);
                if (meta && meta->bias_pool_idx >= 0)
                    y = mx::add(y, pool[meta->bias_pool_idx]);
                pool[out] = y;
                break;
            }
            case OP_CONCAT_2D_AXIS1: {
                /* a = A [m,n], b = B [m,k]. Result = concat along axis 1 -> [m,n+k] */
                pool[out] = mx::concatenate({a, b}, 1);
                break;
            }
            case OP_GRU_CELL: {
                /* nn.GRU: a=ih, b=hh, prev via meta->prev_pool_idx.
                     z = sigmoid(ih_z + hh_z), r = sigmoid(ih_r + hh_r)
                     n = tanh(ih_n + r * hh_n)
                     h' = (1-z)*n + z*prev                                 */
                auto meta = (GruCellReplayMeta*)e.meta;
                int oo = meta->o;
                auto prev = pool[meta->prev_pool_idx];
                auto ih_z = mx::slice(a, {0}, {oo});
                auto ih_r = mx::slice(a, {oo}, {2*oo});
                auto ih_n = mx::slice(a, {2*oo}, {3*oo});
                auto hh_z = mx::slice(b, {0}, {oo});
                auto hh_r = mx::slice(b, {oo}, {2*oo});
                auto hh_n = mx::slice(b, {2*oo}, {3*oo});
                auto z = mx::sigmoid(mx::add(ih_z, hh_z));
                auto r_gate = mx::sigmoid(mx::add(ih_r, hh_r));
                auto n = mx::tanh(mx::add(ih_n, mx::multiply(r_gate, hh_n)));
                pool[out] = mx::add(mx::multiply(mx::subtract(one_like(z), z), n),
                                    mx::multiply(z, prev));
                break;
            }
            case OP_EMBEDDING: {
                // a = weight, b = indices (int32), scalar_arg = embedDim
                auto idx_int = mx::astype(b, mx::int32);
                auto rows = mx::take(a, idx_int, 0);
                pool[out] = mx::flatten(rows);
                break;
            }
            case OP_BATCH_NORM: {
                auto* bm = (BatchNormReplayMeta*)e.meta;
                auto x = mx::reshape(a, {bm->C, bm->spatial});
                auto mean = mx::mean(x, std::vector<int>{1}, true);
                auto var = mx::var(x, std::vector<int>{1}, true);
                auto rstd = mx::rsqrt(mx::add(var, scalar_like(bm->eps, var)));
                auto x_hat = mx::multiply(mx::subtract(x, mean), rstd);
                auto g = mx::reshape(pool[bm->gamma_pool_idx], {bm->C, 1});
                auto bt = mx::reshape(pool[bm->beta_pool_idx], {bm->C, 1});
                pool[out] = mx::flatten(mx::add(mx::multiply(g, x_hat), bt));
                break;
            }
            case OP_DROPOUT: {
                // b holds the stored mask tensor; just multiply
                pool[out] = mx::multiply(a, b);
                break;
            }
            case OP_AVG_POOL1D: {
                // scalar_arg encodes kL + stride*0.001
                int kL = (int)e.scalar_arg;
                int stride = (int)((e.scalar_arg - kL) * 1000 + 0.5);
                if (stride == 0) stride = kL;
                int oL = ((int)a.shape(1) - kL) / stride + 1;
                mx::array res = mx::zeros({(int)a.shape(0), oL}, a.dtype());
                for (int kl = 0; kl < kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {(int)a.shape(0), kl + oL*stride}, {1, stride});
                    res = mx::add(res, sliced);
                }
                pool[out] = mx::divide(res, scalar_like((double)kL, a));
                break;
            }
            case OP_AVG_POOL2D: {
                // For simplicity, re-derive dims from input shape. Only k=2 s=2 common case tested.
                int CC = (int)a.shape(0), HH = (int)a.shape(1), WW = (int)a.shape(2);
                // Default: k=2, stride=2 (most common usage)
                int kH = 2, kW = 2, sH = 2, sW = 2;
                int oH = (HH - kH)/sH + 1, oW = (WW - kW)/sW + 1;
                mx::array res = mx::zeros({CC, oH, oW}, a.dtype());
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        auto sl = mx::slice(a, {0,kh,kw}, {CC,kh+oH*sH,kw+oW*sW}, {1,sH,sW});
                        res = mx::add(res, sl);
                    }
                pool[out] = mx::divide(res, scalar_like((double)(kH*kW), a));
                break;
            }
            case OP_CONV1D: {
                auto* cm = (Conv1DReplayMeta*)e.meta;
                int inC = cm->inC, LL = cm->L;
                auto inp_lc = mx::transpose(a, {1, 0});
                auto inp_nlc = mx::reshape(inp_lc, {1, LL, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 1});
                auto cv = mx::conv1d(inp_nlc, ker_mlx, cm->stride, cm->pad);
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {1, 0});
                if (cm->bias_pool_idx >= 0)
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1}));
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL1D: {
                auto* pm = (MaxPool1DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oL}, -1e30, a.dtype());
                for (int kl = 0; kl < pm->kL; kl++) {
                    auto sliced = mx::slice(a, {0, kl}, {pm->C, kl + pm->oL * pm->stride}, {1, pm->stride});
                    res = mx::maximum(res, sliced);
                }
                pool[out] = res;
                break;
            }
            case OP_CONV2D: {
                auto* cm = (Conv2DReplayMeta*)e.meta;
                int inC = cm->inC, HH = cm->H, WW = cm->W;
                auto inp_hwc = mx::transpose(a, {1, 2, 0});
                auto inp_nhwc = mx::reshape(inp_hwc, {1, HH, WW, inC});
                auto ker_mlx = mx::transpose(b, {0, 2, 3, 1});
                auto cv = mx::conv2d(inp_nhwc, ker_mlx,
                                     {cm->strH, cm->strW}, {cm->padH, cm->padW});
                auto cv_sq = mx::squeeze(cv, 0);
                auto cv_out = mx::transpose(cv_sq, {2, 0, 1});
                if (cm->bias_pool_idx >= 0) {
                    cv_out = mx::add(cv_out, mx::reshape(pool[cm->bias_pool_idx], {-1, 1, 1}));
                }
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL2D: {
                auto* pm = (MaxPool2DReplayMeta*)e.meta;
                mx::array res = mx::full({pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
                for (int kh = 0; kh < pm->kH; kh++) {
                    for (int kw = 0; kw < pm->kW; kw++) {
                        auto sliced = mx::slice(a,
                            {0, kh, kw},
                            {pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
                            {1, pm->strH, pm->strW});
                        res = mx::maximum(res, sliced);
                    }
                }
                pool[out] = res;
                break;
            }
            case OP_CONV2D_BATCHED: {
                auto* cm = (Conv2DBatchedReplayMeta*)e.meta;
                int B = cm->B, inC = cm->inC, HH = cm->H, WW = cm->W;
                (void)inC; (void)HH; (void)WW;  // dimensions inferred from shape
                auto inp_nhwc = mx::transpose(a, {0, 2, 3, 1});
                auto ker_mlx  = mx::transpose(b, {0, 2, 3, 1});
                auto cv = mx::conv2d(inp_nhwc, ker_mlx,
                                     {cm->strH, cm->strW}, {cm->padH, cm->padW});
                auto cv_out = mx::transpose(cv, {0, 3, 1, 2});
                if (cm->bias_pool_idx >= 0) {
                    cv_out = mx::add(cv_out,
                                     mx::reshape(pool[cm->bias_pool_idx], {1, -1, 1, 1}));
                }
                (void)B;
                pool[out] = cv_out;
                break;
            }
            case OP_MAX_POOL2D_BATCHED: {
                auto* pm = (MaxPool2DBatchedReplayMeta*)e.meta;
                mx::array res = mx::full({pm->B, pm->C, pm->oH, pm->oW}, -1e30, a.dtype());
                for (int kh = 0; kh < pm->kH; kh++) {
                    for (int kw = 0; kw < pm->kW; kw++) {
                        auto sliced = mx::slice(a,
                            {0, 0, kh, kw},
                            {pm->B, pm->C, kh + pm->oH * pm->strH, kw + pm->oW * pm->strW},
                            {1, 1, pm->strH, pm->strW});
                        res = mx::maximum(res, sliced);
                    }
                }
                pool[out] = res;
                break;
            }
            case OP_CUMPROD: {
                pool[out] = mx::cumprod(a, 0);
                break;
            }
            case OP_GATHER: {
                // Indices are discrete and non-differentiable — read directly
                // from the tape entry's tensor (closure-captured, not via
                // pool). The constants-collection above intentionally
                // excludes arg2 for this op so mlx::vjp never sees it as a
                // differentiable input. See `arg2_is_index` above.
                auto idx_int = mx::astype(e.arg2->data, mx::int32);
                pool[out] = mx::take(a, idx_int, 0);
                break;
            }
            case OP_SCATTER_ADD: {
                int out_size = (int)e.scalar_arg;
                auto idx_int = mx::astype(e.arg2->data, mx::int32);
                auto base = mx::zeros({out_size}, a.dtype());
                auto updates_2d = mx::reshape(a, {(int)a.size(), 1});
                pool[out] = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
                break;
            }
            default: break;
            }
        }
        return pool[loss_pool_idx];
    };

    // Compute gradients via MLX native autograd (vjp with unit cotangent)
    auto forward_vec = [&](const std::vector<mx::array>& xs) -> std::vector<mx::array> {
        return {forward_fn(xs)};
    };

    // Build the [params..., constants...] inputs vector
    std::vector<mx::array> all_inputs;
    all_inputs.reserve(n_params + n_consts);
    for (auto& p : param_arrays) all_inputs.push_back(p);
    for (auto& [idx, arr] : constants) all_inputs.push_back(arr);

    // Job 3 Phase B — compile-enabled path. Stage 4 wires mx::compile
    // for real. The compile call is the public C++ overload; until we
    // add caching (Stage 5+), it recompiles every backward.
    std::pair<std::vector<mx::array>, std::vector<mx::array>> vjp_result;
    if (tensor_mlx_compile_enabled()) {
        g_compile_invocations++;
        auto compiled = mx::compile(forward_vec);
        vjp_result = mx::vjp(compiled, all_inputs, {mx::array(1.0f)});
    } else {
        vjp_result = mx::vjp(forward_vec, all_inputs, {mx::array(1.0f)});
    }
    // vjp returned grads for [params..., constants...]; truncate to params.
    // mx::array has no default ctor, so erase the tail rather than resize.
    auto& grads = vjp_result.second;
    if ((int)grads.size() > n_params)
        grads.erase(grads.begin() + n_params, grads.end());

    // Distribute gradients to parameter tensors
    for (int i = 0; i < param_count(); i++) {
        auto* tensor = (Tensor*)param_tensor(i);
        tensor->grad = grads[i];
        tensor->has_grad = true;
    }

    // Optional NaN trap — fires only when DEBUG_NAN_TRAP=1 in the env.
    // Walks every param grad on first appearance of NaN/Inf and logs the
    // offending param name. Useful to localise gradient blow-up at the
    // peaked-attention working point in NTM/DNC training.
    {
        static int reported = 0;
        const char* env = getenv("DEBUG_NAN_TRAP");
        if (env && env[0] == '1' && !reported) {
            int any_nan = 0;
            for (int i = 0; i < param_count(); i++) {
                const char* p_name = param_name(i);
                auto* p_tensor = (Tensor*)param_tensor(i);
                auto contig = mx::contiguous(p_tensor->grad);
                mx::eval(contig);
                long n = (long)contig.size();
                std::vector<double> buf((size_t)n);
                mx_to_doubles(contig, buf.data());
                const double* gp = buf.data();
                int nan_count = 0, inf_count = 0;
                double maxabs = 0.0;
                for (long j = 0; j < n; j++) {
                    double v = gp[j];
                    if (v != v) nan_count++;
                    else if (v > 1e30 || v < -1e30) inf_count++;
                    else { double a = v < 0 ? -v : v; if (a > maxabs) maxabs = a; }
                }
                if (nan_count || inf_count) {
                    fprintf(stderr, "[NAN_TRAP] param[%d]=%s NaN=%d Inf=%d maxabs=%.3e (n=%ld)\n",
                            i, p_name, nan_count, inf_count, maxabs, n);
                    any_nan = 1;
                }
            }
            // If any param grad is bad, walk the forward tape and find the
            // first NaN-producing op. result->data already holds the actual
            // forward value, so we just check those in tape order.
            if (any_nan) {
                static const char* OP_NAMES[] = {
                    "CONST", "ADD", "SUB", "MUL", "DIV", "NEG", "EXP", "LOG", "SQRT",
                    "SIGMOID", "TANH", "ADD_SCALAR", "MUL_SCALAR", "CLAMP_MIN",
                    "SUM", "MEAN", "MM", "BMM", "TRANSPOSE_2D", "SOFTMAX_2D",
                    "LOG_SOFTMAX_2D", "MASKED_FILL", "LAYER_NORM_2D", "RESHAPE",
                    "NARROW", "CAT", "POW", "ABS", "STACK", "OUTER", "COSINE_SIM",
                    "CONV1D_CIRC", "MV", "SELECT", "BMM_3X3", "SOFTMAX_3D",
                    "TRANSPOSE_LAST2", "GELU", "GRU_CELL", "EMBEDDING", "BATCH_NORM",
                    "DROPOUT", "AVG_POOL1D", "AVG_POOL2D", "CONV1D", "MAX_POOL1D",
                    "CONV2D", "MAX_POOL2D", "CUMPROD", "GATHER", "SCATTER_ADD",
                    "LEAKY_RELU", "SILU", "SUM_DIM", "CAT_MULTI", "LINEAR_2D",
                    "CONCAT_2D_AXIS1", "SOFTPLUS",
                };
                int n_names = sizeof(OP_NAMES) / sizeof(OP_NAMES[0]);
                fprintf(stderr, "[NAN_TRAP] scanning forward tape (size=%d) for first NaN op...\n",
                        (int)tape.size());
                for (int i = 0; i < (int)tape.size(); i++) {
                    auto& e = tape[i];
                    if (!e.result) continue;
                    auto contig = mx::contiguous(e.result->data);
                    mx::eval(contig);
                    long n = (long)contig.size();
                    if (n == 0) continue;
                    std::vector<double> r_buf((size_t)n);
                    mx_to_doubles(contig, r_buf.data());
                    const double* dp = r_buf.data();
                    int nan_count = 0;
                    for (long j = 0; j < n; j++) {
                        double v = dp[j];
                        if (v != v) { nan_count++; }
                    }
                    if (nan_count) {
                        const char* opn = (e.op >= 0 && e.op < n_names)
                            ? OP_NAMES[e.op] : "UNKNOWN";
                        fprintf(stderr, "[NAN_TRAP] first NaN at tape[%d] op=%s (id=%d) result.size=%ld nan_count=%d arg1.op=%d arg2.op=%d\n",
                                i, opn, e.op, n, nan_count,
                                e.arg1 ? (int)tape[e.arg1->tape_idx].op : -1,
                                e.arg2 ? (int)tape[e.arg2->tape_idx].op : -1);
                        // Sample arg1/arg2 values to spot inputs that are
                        // already large/small.
                        if (e.arg1) {
                            auto a = mx::contiguous(e.arg1->data);
                            mx::eval(a);
                            std::vector<double> a_buf((size_t)a.size());
                            mx_to_doubles(a, a_buf.data());
                            const double* ap = a_buf.data();
                            double amin = ap[0], amax = ap[0];
                            int anan = 0;
                            for (long j = 0; j < (long)a.size(); j++) {
                                double v = ap[j];
                                if (v != v) anan++;
                                else { if (v < amin) amin = v; if (v > amax) amax = v; }
                            }
                            fprintf(stderr, "[NAN_TRAP]   arg1 size=%ld nan=%d range=[%.3e, %.3e]\n",
                                    (long)a.size(), anan, amin, amax);
                        }
                        if (e.arg2) {
                            auto b = mx::contiguous(e.arg2->data);
                            mx::eval(b);
                            std::vector<double> b_buf((size_t)b.size());
                            mx_to_doubles(b, b_buf.data());
                            const double* bp = b_buf.data();
                            double bmin = bp[0], bmax = bp[0];
                            int bnan = 0;
                            for (long j = 0; j < (long)b.size(); j++) {
                                double v = bp[j];
                                if (v != v) bnan++;
                                else { if (v < bmin) bmin = v; if (v > bmax) bmax = v; }
                            }
                            fprintf(stderr, "[NAN_TRAP]   arg2 size=%ld nan=%d range=[%.3e, %.3e]\n",
                                    (long)b.size(), bnan, bmin, bmax);
                        }
                        reported = 1;
                        break;
                    }
                }
            }
            if (reported) fflush(stderr);
        }
    }

    prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd;
}

