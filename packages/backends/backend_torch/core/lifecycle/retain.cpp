/* tensor_retain_handle / tensor_release_handle for the torch backend.
 *
 * Currently no-ops; the intermediates-vector cleanup driven by
 * optimizer_step is sufficient. Stubs exist so the multi-link build
 * resolves these symbols across all three backends consistently. */
#include "../../tensor.h"

extern "C" void tensor_retain_handle(TensorHandle h) { (void)h; }
extern "C" void tensor_release_handle(TensorHandle h) { (void)h; }
