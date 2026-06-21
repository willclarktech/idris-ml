/* MPS eager-init — torch + process-wide target-device pin.
 *
 * Two co-located dylib-load surfaces:
 *
 * 1. MPS eager-init. libtorch lazily initializes its MPS allocator +
 *    Metal command queue on the first tensor that touches MPS. In
 *    multi-backend builds where cross-backend tensor transfers run
 *    early (test-multi's Transfer suite), that lazy init races macOS
 *    work-queue threads (MTLDevice setup, MPS allocator pool ramp-up)
 *    and sporadically aborts the process with SIGSEGV inside
 *    libtorch's `at::native::mps::*` paths. Forcing the init at
 *    dylib-load time closes the window entirely. Cost: one MPS tensor
 *    alloc+dealloc at process start (microseconds). Skipped if torch
 *    wasn't built with MPS support or if no MPS device is available.
 *
 * 2. Process-wide target-device pin (`g_torch_target_device`). The
 *    streamed creation path (`tensor_create_*_streamed` → dtag
 *    dispatchers → `tensor_create_*_{f32,f64}` / `make_param_leaf`)
 *    has no per-call device argument; before this constructor landed,
 *    every streamed-created tensor stayed on CPU regardless of
 *    `TORCH_DEVICE`, silently degrading `BACKEND=torch TORCH_DEVICE=mps`
 *    to CPU-resident layer params + inputs. We observe `TORCH_DEVICE`
 *    once at dylib load and stash the resolved `c10::Device` in a
 *    process-wide variable; the creators consult it when staging the
 *    cast+move pair. Mirrors the legacy `prim__toDeviceTorch` wrap on
 *    `UserExecutorCore.primCreate*` and the `UserExecutorTraining`
 *    non-streamed `primCreate{Param,State}*` methods, but in C so the
 *    streamed path gets the same treatment without an FFI signature
 *    change.
 *
 *    Resolution order: env `TORCH_DEVICE` (string: "cpu" / "mps" /
 *    "cuda" / "cuda:N") → if unavailable on this host (e.g. mps with
 *    no MPS, cuda:N out of range), libtorch's `.to()` throws at the
 *    first create — the Idris EAFP gate (toDeviceChecked /
 *    builtinExecutors probe) surfaces that as `Left ExecutorError`. If
 *    `TORCH_DEVICE` is unset we leave the default at `at::kCPU`.
 */
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cstdlib>
#include <cstring>
#include <string>

/* Process-wide pin observed by the streamed-path creators in
   dtype_dispatch.cpp. Set once at dylib load from `TORCH_DEVICE`; never
   mutated afterwards. Default `at::kCPU` keeps the no-env path identical
   to the pre-existing behaviour. The c10::Device(DeviceType) ctor is
   constexpr + noexcept, so this load-time init cannot throw. */
c10::Device g_torch_target_device = at::kCPU; // NOLINT(cert-err58-cpp)

__attribute__((constructor)) static void torch_mps_eager_init(void) {
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
	} catch (...) { // NOLINT(bugprone-empty-catch)
		            // Deliberate first-touch swallow: a load-time MPS warm-up failure
		            // (paravirt-MPS quirks on Tart VMs etc.) must not abort dylib load.
		            // The real error resurfaces on the first Idris-side MPS use.
	}
}

__attribute__((constructor)) static void torch_target_device_init(void) {
	const char* env = std::getenv("TORCH_DEVICE");
	if (env == nullptr || *env == '\0') return; // leave at kCPU
	if (std::strcmp(env, "cpu") == 0) {
		g_torch_target_device = at::kCPU;
	} else if (std::strcmp(env, "mps") == 0) {
		// Normalize to indexed (Device(MPS, 0)) to match what
		// `tensor.device()` returns after a .to() lands on MPS — bare
		// `Device(MPS, -1)` would never compare equal to a tensor's
		// device, defeating the "skip .to() when already on target"
		// no-op check in torch_migrate_to_target / the inline opt-out
		// branches. There's only one Metal device per Mac, so index 0
		// is the canonical form.
		g_torch_target_device = at::Device(at::DeviceType::MPS, 0);
	} else if (std::strncmp(env, "cuda", 4) == 0) {
		// accepts "cuda" or "cuda:N"; bare "cuda" parses to
		// Device(CUDA, -1) which libtorch then resolves to the current
		// CUDA device on first use, so the same .device()-mismatch
		// concern applies. Normalize unindexed to 0 explicitly.
		const std::string s(env);
		g_torch_target_device = (s == "cuda") ? at::Device(at::DeviceType::CUDA, 0) : at::Device(s);
	}
	// Unknown strings fall through silently — the first .to() will throw
	// and the EAFP gate surfaces it.
}
