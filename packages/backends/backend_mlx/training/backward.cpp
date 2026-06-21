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
#include "autograd/op_dispatch.h"
#include "profiling.h"
#include <algorithm>
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
	double const t0_bwd = _wall_ms_mlx();
	Tensor const* loss = (Tensor*)h;
	if (loss->tape_idx < 0) {
		prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd;
		return;
	}

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
	for (auto& idx : param_pool_indices)
		seen.insert(idx);
	auto add_const = [&](Tensor* t) {
		if (t && !seen.contains(t->pool_idx)) {
			seen.insert(t->pool_idx);
			constants.emplace_back(t->pool_idx, t->data);
		}
	};
	auto arg2_is_index = [](int op) {
		return op == OP_GATHER || op == OP_SCATTER_ADD || op == OP_GATHER_ROWS;
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
	auto* tape_ref = &tape;

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
	for (auto& [idx, arr] : constants)
		constants_pool_indices.push_back(idx);

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
			auto fn = mlx_dispatch_get(e.op);
			if (fn) fn(pool, e);
		}
		return pool[loss_pool_idx];
	};

	// Compute gradients via MLX native autograd (vjp with unit cotangent)
	auto forward_vec = [&](const std::vector<mx::array>& xs) -> std::vector<mx::array> {
		// GCOVR_EXCL_LINE — runs inside mx::vjp's trace; gcov mis-attributes the braced return
		return {forward_fn(xs)};
	};

	// Build the [params..., constants...] inputs vector
	std::vector<mx::array> all_inputs;
	all_inputs.reserve(n_params + n_consts);
	for (auto& p : param_arrays)
		all_inputs.push_back(p);
	for (auto& [idx, arr] : constants)
		all_inputs.push_back(arr);

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
	if ((int)grads.size() > n_params) grads.erase(grads.begin() + n_params, grads.end());

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
		if ((env != nullptr) && env[0] == '1' && (reported == 0)) {
			// GCOVR_EXCL_START — env-gated NaN-locating diagnostic; body runs
			// only on DEBUG_NAN_TRAP=1 + an actual NaN/Inf in a param grad.
			int any_nan = 0;
			for (int i = 0; i < param_count(); i++) {
				const char* p_name = param_name(i);
				auto* p_tensor = (Tensor*)param_tensor(i);
				auto contig = mx::contiguous(p_tensor->grad);
				mx::eval(contig);
				long const n = (long)contig.size();
				std::vector<double> buf((size_t)n);
				mx_to_doubles(contig, buf.data());
				const double* gp = buf.data();
				int nan_count = 0, inf_count = 0;
				double maxabs = 0.0;
				for (long j = 0; j < n; j++) {
					double const v = gp[j];
					if (v != v)
						nan_count++;
					else if (v > 1e30 || v < -1e30)
						inf_count++;
					else {
						double const a = v < 0 ? -v : v;
						maxabs = std::max(a, maxabs);
					}
				}
				if ((nan_count != 0) || (inf_count != 0)) {
					fprintf(stderr, "[NAN_TRAP] param[%d]=%s NaN=%d Inf=%d maxabs=%.3e (n=%ld)\n",
					        i, p_name, nan_count, inf_count, maxabs, n);
					any_nan = 1;
				}
			}
			// If any param grad is bad, walk the forward tape and find the
			// first NaN-producing op. result->data already holds the actual
			// forward value, so we just check those in tape order.
			if (any_nan != 0) {
				static const char* const OP_NAMES[] = {
				    "CONST",
				    "ADD",
				    "SUB",
				    "MUL",
				    "DIV",
				    "NEG",
				    "EXP",
				    "LOG",
				    "SQRT",
				    "SIGMOID",
				    "TANH",
				    "ADD_SCALAR",
				    "MUL_SCALAR",
				    "CLAMP_MIN",
				    "SUM",
				    "MEAN",
				    "MM",
				    "BMM",
				    "TRANSPOSE_2D",
				    "SOFTMAX_2D",
				    "LOG_SOFTMAX_2D",
				    "MASKED_FILL",
				    "LAYER_NORM_2D",
				    "RESHAPE",
				    "NARROW",
				    "CAT",
				    "POW",
				    "ABS",
				    "STACK",
				    "OUTER",
				    "COSINE_SIM",
				    "CONV1D_CIRC",
				    "MV",
				    "SELECT",
				    "BMM_3X3",
				    "SOFTMAX_3D",
				    "TRANSPOSE_LAST2",
				    "GELU",
				    "GRU_CELL",
				    "EMBEDDING",
				    "BATCH_NORM",
				    "DROPOUT",
				    "AVG_POOL1D",
				    "AVG_POOL2D",
				    "CONV1D",
				    "MAX_POOL1D",
				    "CONV2D",
				    "MAX_POOL2D",
				    "CUMPROD",
				    "GATHER",
				    "SCATTER_ADD",
				    "LEAKY_RELU",
				    "SILU",
				    "SUM_DIM",
				    "CAT_MULTI",
				    "LINEAR_2D",
				    "CONCAT_2D_AXIS1",
				    "SOFTPLUS",
				};
				int const n_names = sizeof(OP_NAMES) / sizeof(OP_NAMES[0]);
				fprintf(stderr, "[NAN_TRAP] scanning forward tape (size=%d) for first NaN op...\n",
				        (int)tape.size());
				for (int i = 0; i < (int)tape.size(); i++) {
					auto& e = tape[i];
					if (e.result == nullptr) continue;
					auto contig = mx::contiguous(e.result->data);
					mx::eval(contig);
					long const n = (long)contig.size();
					if (n == 0) continue;
					std::vector<double> r_buf((size_t)n);
					mx_to_doubles(contig, r_buf.data());
					const double* dp = r_buf.data();
					int nan_count = 0;
					for (long j = 0; j < n; j++) {
						double const v = dp[j];
						if (v != v) {
							nan_count++;
						}
					}
					if (nan_count != 0) {
						const char* opn =
						    (e.op >= 0 && e.op < n_names) ? OP_NAMES[e.op] : "UNKNOWN";
						fprintf(stderr,
						        "[NAN_TRAP] first NaN at tape[%d] op=%s (id=%d) result.size=%ld "
						        "nan_count=%d arg1.op=%d arg2.op=%d\n",
						        i, opn, e.op, n, nan_count,
						        ((e.arg1 != nullptr) && e.arg1->tape_idx >= 0)
						            ? (int)tape[e.arg1->tape_idx].op
						            : -1,
						        ((e.arg2 != nullptr) && e.arg2->tape_idx >= 0)
						            ? (int)tape[e.arg2->tape_idx].op
						            : -1);
						// Sample arg1/arg2 values to spot inputs that are
						// already large/small.
						if (e.arg1 != nullptr) {
							auto a = mx::contiguous(e.arg1->data);
							mx::eval(a);
							std::vector<double> a_buf((size_t)a.size());
							mx_to_doubles(a, a_buf.data());
							const double* ap = a_buf.data();
							double amin = ap[0], amax = ap[0];
							int anan = 0;
							for (long j = 0; j < (long)a.size(); j++) {
								double const v = ap[j];
								if (v != v)
									anan++;
								else {
									amin = std::min(v, amin);
									amax = std::max(v, amax);
								}
							}
							fprintf(stderr,
							        "[NAN_TRAP]   arg1 size=%ld nan=%d range=[%.3e, %.3e]\n",
							        (long)a.size(), anan, amin, amax);
						}
						if (e.arg2 != nullptr) {
							auto b = mx::contiguous(e.arg2->data);
							mx::eval(b);
							std::vector<double> b_buf((size_t)b.size());
							mx_to_doubles(b, b_buf.data());
							const double* bp = b_buf.data();
							double bmin = bp[0], bmax = bp[0];
							int bnan = 0;
							for (long j = 0; j < (long)b.size(); j++) {
								double const v = bp[j];
								if (v != v)
									bnan++;
								else {
									bmin = std::min(v, bmin);
									bmax = std::max(v, bmax);
								}
							}
							fprintf(stderr,
							        "[NAN_TRAP]   arg2 size=%ld nan=%d range=[%.3e, %.3e]\n",
							        (long)b.size(), bnan, bmin, bmax);
						}
						reported = 1;
						break;
					}
				}
			}
			if (reported != 0) fflush(stderr);
			// GCOVR_EXCL_STOP
		}
	}

	prof_backward_ms_mlx += _wall_ms_mlx() - t0_bwd;
}
