/* Backend init — mlx.
 *
 * Two distinct surfaces, co-located because both run at dylib load:
 *
 *   1. Device selection. Default to CPU stream. mlx 0.31 GPU (Metal)
 *      on GH Actions macOS runners hits "Unable to allocate N bytes"
 *      for tiny allocations under sustained load (NTM/DNC scalar-heavy
 *      backward; CI run 25457289084 on commit 1b8feff). Local Apple
 *      Silicon machines handle GPU fine, so users opt in via
 *      `MLX_DEVICE=gpu`. See TODO.md "MLX backend: support CPU+f64
 *      mode + dependent-types demo" for the proper device-aware Tensor
 *      parameterization.
 *
 *   2. std::terminate gate. Apple Virtualization VMs (Tart, GHA macOS)
 *      hit `std::runtime_error: [malloc] Unable to allocate N bytes`
 *      (tiny N, 4–512) during process *shutdown* on scalar-heavy
 *      workloads (NTM/DNC, SAC actor). Training completes cleanly and
 *      the profile report prints; the failure is in mlx-internal
 *      static destructors racing against the Metal device teardown.
 *      Synchronising + clearing caches before destructors fire (via
 *      atexit) doesn't help — the throwing destructor is inside mlx
 *      itself. Fix: gate std::terminate to swallow post-main
 *      exceptions. A flag set by atexit distinguishes "after main
 *      returned" from "during training" — real exceptions during
 *      training still abort normally.
 */
#include "tensor.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <unistd.h> // _exit
#include <signal.h>
#include <execinfo.h>
#include <cstdio>

static bool g_mlx_past_main = false;
static std::terminate_handler g_prev_terminate_handler = nullptr;

// GCOVR_EXCL_START — atexit hook; fires only during process teardown, not reachable from a unit
// test
static void mlx_set_past_main(void) {
	g_mlx_past_main = true;
}
// GCOVR_EXCL_STOP

// GCOVR_EXCL_START — std::terminate handler; runs only on an uncaught exception (crash path), not
// unit-testable without aborting the runner
static void mlx_terminate_handler(void) {
	if (g_mlx_past_main) {
		// Process already exited cleanly; this is a destructor-order
		// crash we can't fix without a libmlx-upstream change. Exit 0.
		_exit(0);
	}
	if (g_prev_terminate_handler != nullptr) g_prev_terminate_handler();
	std::abort();
}
// GCOVR_EXCL_STOP

__attribute__((constructor)) static void mlx_backend_init(void) {
	const char* env = std::getenv("MLX_DEVICE");
	if ((env != nullptr) && (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "metal") == 0)) {
		// GCOVR_EXCL_START — mlx GPU-only branch; CI runs the CPU stream (MLX_DEVICE unset)
		mx::set_default_device(mx::Device(mx::Device::gpu));
		// GCOVR_EXCL_STOP
	} else {
		mx::set_default_device(mx::Device(mx::Device::cpu));
	}
	// Leave memory_limit / cache_limit at mlx's defaults. The
	// "[malloc] Unable to allocate N bytes" failure on Apple
	// Virtualization VMs (Tart, GHA macOS) is *not* hit because of an
	// mlx limit — it's MetalAllocator throwing when paravirtualized
	// Metal refuses a new MTLBuffer (per-process resource limit, not
	// bytes). Stack trace confirms: throw originates in
	// MetalAllocator::malloc even when MLX_DEVICE=cpu, because on
	// Apple Silicon mlx routes all buffer allocations through Metal
	// (unified memory). The real fix is keeping live MTLBuffer count
	// low; see the refcount-driven Tensor lifecycle work.
	g_prev_terminate_handler = std::set_terminate(mlx_terminate_handler);
	std::atexit(mlx_set_past_main);

	// Crash-trace install — opt-in via MLX_CRASH_TRACE=1. On SIGSEGV/SIGILL/
	// SIGBUS, write a host-side backtrace to stderr then re-raise the
	// signal with default disposition (which kills the process). Chez's
	// signal handler normally swallows these with "invalid memory
	// reference" — installing ours after the constructor leaves Chez's
	// later sigaction call to overwrite us, so we re-install on the
	// first FFI entry too. For diagnosis only.
	// GCOVR_EXCL_START — opt-in diagnostic (MLX_CRASH_TRACE=1); the body is a SIGSEGV/SIGILL/SIGBUS
	// handler that re-raises and kills the process — not reachable from a unit test without
	// crashing the runner
	if (std::getenv("MLX_CRASH_TRACE") != nullptr) {
		struct sigaction sa;
		std::memset(&sa, 0, sizeof(sa));
		sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
		sa.sa_sigaction = [](int signo, siginfo_t* info, void*) {
			const char* sname = "SIG?";
			switch (signo) {
			case SIGSEGV:
				sname = "SIGSEGV";
				break;
			case SIGILL:
				sname = "SIGILL";
				break;
			case SIGBUS:
				sname = "SIGBUS";
				break;
			default:
				break;
			}
			std::fprintf(stderr, "\n=== mlx crash-trace: %s at addr=%p ===\n", sname,
			             info ? info->si_addr : nullptr);
			void* frames[64];
			int const n = backtrace(frames, 64);
			backtrace_symbols_fd(frames, n, 2);
			std::fflush(stderr);
			// Re-raise (SA_RESETHAND restored default disposition)
			raise(signo);
		};
		sigaction(SIGSEGV, &sa, nullptr);
		sigaction(SIGILL, &sa, nullptr);
		sigaction(SIGBUS, &sa, nullptr);
	}
	// GCOVR_EXCL_STOP
}
