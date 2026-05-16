/* tensor_free for the mlx backend.
 *
 * Refcount-driven world: forcing `delete t` here would leave dangling
 * Tensor* pointers in tape entries that still reference this
 * result/arg, and the next tape_reset crashes when it walks the tape
 * to release retains. Instead drop the caller's implicit hold; the
 * tape's own retains (set by tape_append on result/arg1/arg2) keep
 * the Tensor alive until tape_reset releases them and sweeps
 * refcount=0.
 *
 * Defends against the caller passing a handle that was already swept
 * by a prior tape_reset (common when optimizer_step ran between the
 * user's create and their free): touching `t` would be use-after-free.
 * Probe all_tensors first; skip if absent. Also skip registered params
 * — they're managed by param_clear. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" void tensor_free_mlx_streamed(TensorHandle h, int stream_tag) {
    (void)stream_tag;  /* no kernel — pure C-side bookkeeping */
    if (!h) return;
    auto t = (Tensor*)h;
    for (int i_ = 0; i_ < param_count(); i_++) {
        if ((Tensor*)param_tensor(i_) == t) return;
    }
    for (auto* alive : all_tensors) {
        if (alive == t) { tensor_release_internal(t); return; }
    }
}

extern "C" void tensor_free(TensorHandle h) {
    tensor_free_mlx_streamed(h, default_stream_tag());
}
