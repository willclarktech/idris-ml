/* MPS eager-init — torch.
 *
 * libtorch lazily initializes its MPS allocator + Metal command queue on
 * the first tensor that touches MPS. In multi-backend builds where cross-
 * backend tensor transfers run early (test-multi's Transfer suite), that
 * lazy init races macOS work-queue threads (MTLDevice setup, MPS
 * allocator pool ramp-up) and sporadically aborts the process with
 * SIGSEGV inside libtorch's `at::native::mps::*` paths. We saw this
 * empirically: re-ordering tests so an intra-torch CPU→MPS migration
 * runs FIRST dropped the crash rate from 100% to ~3%. Forcing the
 * init at dylib-load time (here) closes the window entirely — by the
 * time any Idris-side code calls `tensor_to_device_torch(h, "mps")`,
 * the MPS subsystem is already warm.
 *
 * Cost: one MPS tensor alloc+dealloc at process start (microseconds).
 * Skipped if torch wasn't built with MPS support or if no MPS device
 * is available on this host (Linux CI, non-Apple-Silicon Mac).
 */
#include <torch/torch.h>
#include <ATen/ATen.h>

__attribute__((constructor))
static void torch_mps_eager_init(void) {
    if (!at::hasMPS()) return;
    try {
        auto opts = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(at::DeviceType::MPS));
        auto warm = torch::zeros({1}, opts);
        // Touch the data once to force the Metal command buffer to drain
        // (synchronize). The `.cpu()` round-trip is what
        // tensor_to_doubles_torch does at runtime — same code path.
        (void)warm.cpu();
        // `warm` falls out of scope; its storage refcount hits zero
        // and libtorch returns the MPS buffer to the allocator pool.
    } catch (...) {
        // First-touch failures (paravirt-MPS quirks on Tart VMs etc.)
        // shouldn't prevent dylib load. Subsequent Idris-side MPS use
        // will surface the real error.
    }
}
