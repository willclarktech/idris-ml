/* Tape mechanics — mlx.
 *
 * mlx's autograd is replay-based: forward ops push entries to a Wengert
 * tape (op_code, result_ptr, arg1_ptr, arg2_ptr, scalar_arg, meta_ptr);
 * backward replays the tape inside mx::vjp to compute gradients —
 * tensor_backward in training/backward.cpp owns the replay.
 *
 * This TU owns the tape itself + the two mutators:
 *
 *   - tape_append   pushes a forward op; gated by no_grad_depth_mlx
 *                   (externed via tape.h, defined in training/autograd.cpp).
 *                   Retains args + result so the tape entry's pointers
 *                   stay valid until tape_reset.
 *   - tape_reset    evals pending lazy graphs, releases the retains, frees
 *                   per-op meta blobs, sweeps refcount=0 Tensors, reassigns
 *                   pool indices, frees TensorPair structs, and calls
 *                   mx::clear_cache() so the MetalAllocator returns
 *                   buffers to the OS each epoch (the GH Actions VM hits
 *                   the Metal-derived cache ceiling within a few epochs
 *                   otherwise — locally on M-series the call is cheap).
 *
 * The `tape` vector + `prof_tape_appends_mlx` counter are externed via
 * tape.h; defined here for symbol uniqueness.
 */
#include "tensor.h"
#include "tape.h"
#include <vector>

extern "C" void free(void*);

/* Tape state — externed via tape.h. */
std::vector<TapeEntry> tape;
long prof_tape_appends_mlx = 0;

/* CONTRACT: returns -1 while no_grad_depth_mlx > 0 (params keep
 * requires_grad=true inside withNoGrad, so meta-carrying call sites
 * still reach their `if (rg)` block under no-grad). Every
 * `tape[idx].meta = ...` MUST be guarded with `if (idx >= 0)` —
 * an unguarded write lands at tape[-1], the word before the tape
 * vector's heap block (heap underrun; or a near-NULL write when the
 * tape is empty). That underrun was the layout-dependent SIGABRT /
 * "invalid memory reference" that killed the mlx RL examples in CI
 * run 27373449876. Gated by the mlx_no_grad_meta criterion suite. */
int tape_append(int op, Tensor* result, Tensor* arg1, Tensor* arg2, double scalar_arg) {
	if (no_grad_depth_mlx > 0) {
		if (result) {
			result->requires_grad = false;
			result->tape_idx = -1;
		}
		return -1;
	}
	int idx = (int)tape.size();
	tape.push_back({op, result, arg1, arg2, scalar_arg, nullptr});
	result->tape_idx = idx;
	// The tape holds args until tape_reset; retain them while it does.
	// Also retain the result — the FFI wrapper's wrap-and-retain holds
	// refcount=1, but the tape entry holding `result` is a second
	// long-term holder that must be reflected in the count, or backward
	// replay can see a freed Tensor when the Idris wrap dies + drain
	// releases before tape_reset.
	tensor_retain_internal(result);
	tensor_retain_internal(arg1);
	tensor_retain_internal(arg2);
	prof_tape_appends_mlx++;
	return idx;
}

void tape_reset() {
	// Force evaluation of all pending lazy ops first. Survivors may have
	// mx::array graphs that reference soon-to-be-freed state Tensors;
	// materializing those graphs now means the freed mx::array's
	// refcounted impl gets dropped cleanly via mlx's internal accounting
	// rather than dangling.
	{
		std::vector<mx::array> to_eval;
		for (auto* t : all_tensors) {
			to_eval.push_back(t->data);
			if (t->has_grad) to_eval.push_back(t->grad);
		}
		if (!to_eval.empty()) mx::eval(to_eval);
	}
	// Release the args + result we retained in tape_append. Must
	// happen before tape.clear() — once entries are gone we can't
	// find which Tensors we retained.
	for (auto& e : tape) {
		tensor_release_internal(e.result);
		tensor_release_internal(e.arg1);
		tensor_release_internal(e.arg2);
	}
	// Free op metadata
	for (auto& e : tape) {
		if (e.op == OP_LAYER_NORM_2D && e.meta) {
			delete (LayerNormReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_RMS_NORM_2D && e.meta) {
			delete (RmsNormReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_GRU_CELL && e.meta) {
			delete (GruCellReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_STACK && e.meta) {
			delete (std::vector<int>*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_CAT_MULTI && e.meta) {
			delete (std::vector<int>*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_TILE_2D && e.meta) {
			std::free(e.meta);
			e.meta = nullptr;
		}
		if (e.op == OP_BATCH_NORM && e.meta) {
			delete (BatchNormReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_CONV1D && e.meta) {
			delete (Conv1DReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_MAX_POOL1D && e.meta) {
			delete (MaxPool1DReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_CONV2D && e.meta) {
			delete (Conv2DReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_CONV2D_BATCHED && e.meta) {
			delete (Conv2DBatchedReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_MAX_POOL2D && e.meta) {
			delete (MaxPool2DReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_AVG_POOL2D && e.meta) {
			delete (AvgPool2DReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_MAX_POOL2D_BATCHED && e.meta) {
			delete (MaxPool2DBatchedReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_SUM_DIM && e.meta) {
			delete (SumDimReplayMeta*)e.meta;
			e.meta = nullptr;
		}
		if (e.op == OP_LINEAR_2D && e.meta) {
			delete (LinearReplayMeta*)e.meta;
			e.meta = nullptr;
		}
	}
	tape.clear();
	// Refcount-driven sweep: delete Tensors whose count is 0 (no
	// long-term holder left — no Idris wrap, no tape entry, no
	// param_registry entry). Everything else stays.
	std::vector<Tensor*> survivors;
	for (auto* t : all_tensors) {
		if (t->refcount > 0)
			survivors.push_back(t);
		else
			delete t;
	}
	all_tensors = std::move(survivors);
	// Reassign pool indices to be contiguous (keeps pool vector compact)
	next_pool_idx = 0;
	for (auto* t : all_tensors)
		t->pool_idx = next_pool_idx++;
	// Free TensorPair structs
	for (auto* p : all_pairs)
		free(p);
	all_pairs.clear();
	// Hand cached buffers back to the OS each epoch. Without this, mlx's
	// cache holds onto buffers from the just-collected non-persistent
	// tensors, and on GH Actions macOS-latest VMs the cache hits its
	// (Metal-derived) limit fast enough to abort small allocations like
	// `[malloc] Unable to allocate 4 bytes`. Locally on M-series the
	// cache is fine; the call is cheap either way.
	mx::clear_cache();
}
