/* Autograd surface — mlx.
 *
 * Holds the small surface that isn't replay (tensor_backward lives in
 * backward.cpp) but is autograd-shaped: grad accessor, zero/detach/
 * with_grad/set_requires_grad, plus the no_grad scope + per-epoch
 * generation-scoped sweep that bounds the live-Tensor / MTLBuffer count
 * under the paravirt-Metal ceiling.
 *
 *   - tensor_grad             read the (forced-contiguous) .grad
 *   - tensor_zero_grad        zero .grad in-place
 *   - tensor_requires_grad    bool predicate
 *   - tensor_detach           clone with requires_grad=false, no tape entry
 *   - tensor_with_grad        clone with requires_grad=true, push OP_CONST
 *   - tensor_set_requires_grad
 *   - tensor_no_grad_begin / tensor_no_grad_end
 *   - tensor_epoch_begin / tensor_epoch_end
 *   - mlx_sweep_generation    (internal; shared by no_grad_end + epoch_end)
 *
 * `no_grad_depth_mlx` is defined here (non-static) and externed by
 * tape_append in the monolith — same nesting-counter the gating fires on.
 */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"
#include <vector>

/* No-grad nesting counter. tape_append in backend_mlx.cpp reads it via
   extern decl in tape.h; this TU's no_grad_begin/end are the only
   mutators. */
int no_grad_depth_mlx = 0;

TensorHandle tensor_grad(TensorHandle h) {
    auto t = (Tensor*)h;
    if (!t->has_grad) return nullptr;
    /* mx::vjp may return non-contiguous grads (broadcast strides). Force
       a contiguous copy so the returned tensor has the expected layout. */
    auto contig = mx::contiguous(t->grad);
    mx::eval(contig);
    return (TensorHandle)new Tensor(contig, false);
}

void tensor_zero_grad(TensorHandle h) {
    auto t = (Tensor*)h;
    if (t->has_grad) {
        t->grad = mx::zeros(t->data.shape(), t->data.dtype());
    }
}

int tensor_requires_grad(TensorHandle h) { return ((Tensor*)h)->requires_grad ? 1 : 0; }

extern "C" TensorHandle tensor_detach_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    /* Detach: clone data, requires_grad=false, no tape entry. The result is
       a leaf with no autograd linkage to the source tensor. */
    auto t = (Tensor*)h;
    return (TensorHandle)new Tensor(mx::array(t->data), false);
}
TensorHandle tensor_detach(TensorHandle h) {
    return tensor_detach_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_with_grad_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    /* Promote a tensor into the autograd graph: clone with requires_grad=true,
       record an OP_CONST tape entry so the constant pool picks up its data
       during backward replay. Note: for the result's gradient to actually be
       computed, the caller still needs to register it via param_register. */
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::array(t->data), true);
    tape_append(OP_CONST, r, nullptr, nullptr, 0);
    return (TensorHandle)r;
}
TensorHandle tensor_with_grad(TensorHandle h) {
    return tensor_with_grad_mlx_streamed(h, default_stream_tag());
}

void tensor_set_requires_grad(TensorHandle h, int rg) {
    ((Tensor*)h)->requires_grad = (rg != 0);
}

// Generation-scoped sweep: eval pending lazy graphs, then delete every
// wrap-only (rc==1) Tensor created at or after `block_start`. Those are
// block/epoch-local intermediates whose results have been extracted to
// scalars or retained (rc>=2) by KeepAlive. Registry params (rc>1) and
// pre-generation state (lower create_id) are spared. Bounds the live
// handle / Metal-buffer count instead of letting it accumulate past the
// paravirt-Metal ceiling. Shared by no_grad_end and epoch_end.
static void mlx_sweep_generation(long block_start) {
    std::vector<mx::array> to_eval;
    for (auto* t : all_tensors) to_eval.push_back(t->data);
    if (!to_eval.empty()) {
        try { mx::eval(to_eval); } catch (...) { /* best-effort */ }
    }
    std::vector<Tensor*> survivors;
    survivors.reserve(all_tensors.size());
    for (auto* t : all_tensors) {
        if (t->refcount == 1 && t->create_id >= block_start) {
            // Wrap-only block-local intermediate. We must NOT `delete` it:
            // rc==1 here is the Idris guardian wrap's own retain, and that
            // wrap is still registered (un-drained — a drained wrap would
            // have dropped this to rc==0). Its eventual drain calls
            // tensor_release_handle on this exact pointer; freeing the
            // object now makes that an unguarded `refcount--` on freed (and,
            // under allocation churn, recycled) memory → malloc-freelist
            // corruption (the intermittent F32 mlx-gpu SIGTRAP). Instead
            // release the heavy mx::array buffers now — that reclaims the
            // Metal MTLBuffer, which is the only thing the live-handle
            // ceiling actually cares about — and keep the lightweight husk
            // alive (its address pinned) until the wrap drains it to rc==0,
            // when the branch below frees it safely.
            //
            // Assign a single process-wide empty scalar rather than a fresh
            // `mx::array(0.0f)` per husk: mx::array is copy-on-write (a
            // shared_ptr to its buffer), so this is a refcount bump, not an
            // allocation, yet it still drops the husk's heavy buffer. A fresh
            // per-husk scalar *does* allocate, and on Apple Silicon every
            // buffer — even 4–8 bytes — routes through MetalAllocator; under
            // the paravirt-Metal MTLBuffer ceiling (Tart/GHA VMs) those
            // per-sweep allocations throw `[malloc] Unable to allocate N
            // bytes` mid-training (regression from 8482788, NTM/DNC/mnist/RL).
            static const mx::array g_husk_empty = mx::array(0.0f);
            t->data = g_husk_empty;
            t->grad = g_husk_empty;
            t->has_grad = false;
            survivors.push_back(t);
            continue;
        }
        if (t->refcount > 0) survivors.push_back(t);
        else delete t;
    }
    all_tensors = std::move(survivors);
    try { mx::clear_cache(); } catch (...) { /* best-effort */ }
}

static long g_nograd_block_start = 0;  // create_id at outermost no_grad_begin
void tensor_no_grad_begin(void) {
    if (no_grad_depth_mlx == 0) g_nograd_block_start = g_mlx_create_calls_global;
    no_grad_depth_mlx++;
}
void tensor_no_grad_end(void) {
    if (no_grad_depth_mlx > 0) no_grad_depth_mlx--;
    if (no_grad_depth_mlx > 0) return;  // only sweep on outermost end
    mlx_sweep_generation(g_nograd_block_start);
}

// Generation-scoped free for grad-mode training, nestable via a marker
// stack: the per-epoch bracket (runTrainingIO) is the outer frame and a
// per-step `withGenFree` bracket is an inner frame. begin pushes the
// current create_id; end pops it and frees wrap-only handles created since.
static std::vector<long> g_gen_stack;
void tensor_epoch_begin(void) { g_gen_stack.push_back(g_mlx_create_calls_global); }
void tensor_epoch_end(void) {
    if (g_gen_stack.empty()) return;
    long start = g_gen_stack.back();
    g_gen_stack.pop_back();
    mlx_sweep_generation(start);
}
