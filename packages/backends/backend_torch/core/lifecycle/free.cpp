/* tensor_free for the torch backend.
 *
 * Torch tensors participate in autograd graphs — explicit deletion can
 * corrupt torch's internal bookkeeping. Let free_intermediates (called
 * by optimizer_step) handle bulk cleanup of computation intermediates.
 * Persistent user-created tensors leak slightly — acceptable for tests.
 *
 * The matching `freed_by_cleanup` set stays in backend_torch.cpp where
 * free_intermediates lives. */
#include "../../tensor.h"

extern "C" void tensor_free(TensorHandle h) {
	(void)h;
}
