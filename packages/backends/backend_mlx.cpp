/* backend_mlx.cpp — primary-name handle for the mlx backend.
 *
 * Uses Apple's MLX framework for GPU-accelerated tensor operations on
 * Apple Silicon via Metal. Forward ops record to a Wengert tape; backward
 * replays the tape inside mx::vjp for native autograd — zero hand-written
 * backward rules.
 *
 * The implementation lives at packages/backends/backend_mlx/ — per-op
 * .cpp files mirroring the tape + torch trees. This file is the build's
 * primary-name compile unit (the Makefile's backend rule expects
 * `backend_<b>.cpp` to exist); it holds nothing but a pointer index back
 * into the modular tree.
 *
 * Build: make BACKEND=mlx backend  (or BACKEND=mlx MLX_DEVICE=gpu install)
 *
 * Tree layout (all paths under packages/backends/backend_mlx/):
 *   tensor.{h,cpp}                Tensor struct + tracking globals.
 *   tape.{h,cpp}                  OP_* enum + ReplayMeta structs +
 *                                 tape vector + tape_append + tape_reset.
 *   precision.h                   F32↔F64 bridge (mx::array ↔ host
 *                                 doubles, scalar_like / zero_like / etc).
 *   stream.h                      cpu_stream / gpu_stream / WITH_STREAM
 *                                 macro / default_stream_tag.
 *   init.cpp                      mlx_backend_init ctor + std::terminate
 *                                 gate for the Apple Virtualization VM
 *                                 shutdown crash.
 *   core/lifecycle/               create / create_scalar / cast / clone /
 *                                 free / item{,1d,2d} / accessors /
 *                                 lifecycle_core (Tensor ctor + refcount +
 *                                 live-handle accessors) /
 *                                 create_param_state (1d/2d/param/state
 *                                 creators × F32/F64/streamed) / batch /
 *                                 view.
 *   core/elementwise/             add / sub / mul / div / neg / abs / exp /
 *                                 log / sqrt / pow / sigmoid / tanh /
 *                                 softplus.
 *   core/scalar/                  add_scalar / mul_scalar / clamp_min.
 *   core/backend_meta.cpp         backend_name / reset_for_eval / print /
 *                                 mlx_compile_* + g_compile_invocations.
 *   linear/linalg/                matmul / mv / mm / linear / dot / outer /
 *                                 bmm / transpose / tile.
 *   linear/reduction/             sum / mean / min / max.
 *   linear/shape/                 reshape / squeeze / narrow / select /
 *                                 expand_mask.
 *   linear/concat/                cat / stack / concat_2d_axis1.
 *   linear/index/                 gather / scatter_add.
 *   linear/sort/                  argsort / cumprod.
 *   nn/softmax/                   softmax / log_softmax.
 *   nn/mask/                      masked_fill.
 *   nn/norm/                      layer_norm / batch_norm / group_norm /
 *                                 dropout.
 *   nn/attention/                 cross_attention / embedding /
 *                                 cosine_similarity.
 *   nn/loss/                      bce_with_logits / cross_entropy / mse_loss.
 *   nn/recurrent/                 lstm_cell / lstm_gates_pair / gru_cell /
 *                                 pair_helpers.
 *   nn/activation/                gelu / leaky_relu / silu.
 *   conv/                         conv1d{,_circular} / conv2d{,_batched} /
 *                                 conv_transpose / conv_grouped /
 *                                 max_pool1d / max_pool2d{,_batched} /
 *                                 avg_pool1d / avg_pool2d.
 *   training/optimizer.cpp        Optimizer struct + Adam (eager + cached
 *                                 mx::compile) / AdamW / RMSprop / SGD +
 *                                 clip / native_train_step.
 *   training/adapter.cpp          BackendPort + mlx_port_* shims.
 *   training/dtype_dispatch.cpp   tensor_create_*_streamed dtag routers.
 *   training/backward.cpp         tensor_backward (replay-based, ~570 LOC).
 *   training/autograd.cpp         grad / detach / no_grad_begin/end +
 *                                 generation-scoped sweep.
 *   training/diagnostics.cpp      DEBUG_LSTM_TRAJ + DEBUG_PARAM_GRADS_MLX.
 *   training/ntm_specific.cpp     tensor_subtract_scalar_inplace.
 *   training/profiling.cpp        prof_*_ms counters + _wall_ms_mlx.
 *   device.cpp                    tensor_to_device (no-op) + tensor_device.
 *
 * Shared (compiled per backend via SHARED_BACKENDS_<tu> lists):
 *   shared/training/param_registry.c
 *   shared/training/ffi_shims.c
 */
