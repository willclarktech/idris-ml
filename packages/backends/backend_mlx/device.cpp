/* Device surface — mlx.
 *
 * mlx has no torch-style intra-device migration: tensors live on the
 * stream they were created on, and inter-stream movement is handled
 * by mlx's stream selection (see stream.h). So `tensor_to_device` is
 * a no-op identity, and `tensor_device` returns "gpu" as a placeholder
 * (Idris-side rendering doesn't depend on the exact string).
 */
#include "tensor.h"

extern "C" TensorHandle tensor_to_device(TensorHandle t, const char* device) {
	(void)device;
	return t;
}

extern "C" const char* tensor_device(TensorHandle t) {
	(void)t;
	return "gpu";
}
