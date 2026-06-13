/* Device surface — torch.
 *
 * EAFP availability gate: a device-pin to absent/invalid hardware
 * (e.g. "cuda:1" on a 1-GPU box, or MPS on a non-Apple host) makes
 * libtorch's `.to()` throw a c10::Error. Unguarded, that exception
 * crosses the C->Chez FFI boundary and becomes std::terminate/SIGABRT.
 * Catch it here and return a NULL handle; the Idris side lifts NULL ->
 * Left ExecutorError. This is the one source of truth for availability —
 * no separate is_available probe to drift. All torch device-pinning
 * (primCreateFromHost, primIntraMigrate, primCreate's post-create
 * migration) routes through here, so this single guard covers them.
 */
#include "tensor.h"
#include <torch/torch.h>
#include <cstdio>
#include <string>

extern "C" TensorHandle tensor_to_device(TensorHandle h, const char* device) {
	try {
		return from_tensor(to_tensor(h)->to(std::string(device)));
	} catch (const std::exception& e) {
		fprintf(stderr, "[torch] tensor_to_device(%s) failed: %s\n", device, e.what());
		return nullptr;
	} catch (...) {
		fprintf(stderr, "[torch] tensor_to_device(%s) failed: unknown\n", device);
		return nullptr;
	}
}

/* Param-lifetime migration: same semantics as tensor_to_device but the
   result is NOT pushed onto the intermediates vector, so it survives
   optimizer_step's free_intermediates(). The create-then-migrate
   primCreateScalar / primCreateFromHost / primIntraMigrate overrides
   route through this — their results are params or user-held tensors;
   the tracked variant made every such tensor a use-after-free once the
   first step's cleanup ran (Idris suite crash + Hpo.LrFinder SIGABRT,
   root-caused 2026-06-12). */
extern "C" TensorHandle tensor_to_device_persistent(TensorHandle h, const char* device) {
	try {
		return from_tensor_persistent(to_tensor(h)->to(std::string(device)));
	} catch (const std::exception& e) {
		fprintf(stderr, "[torch] tensor_to_device_persistent(%s) failed: %s\n", device, e.what());
		return nullptr;
	} catch (...) {
		fprintf(stderr, "[torch] tensor_to_device_persistent(%s) failed: unknown\n", device);
		return nullptr;
	}
}

extern "C" const char* tensor_device(TensorHandle h) {
	static thread_local std::string device_str;
	auto d = to_tensor(h)->device();
	device_str = d.str();
	return device_str.c_str();
}
