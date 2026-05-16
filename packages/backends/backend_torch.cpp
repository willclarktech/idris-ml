/* backend_torch.cpp — primary-name handle for the torch backend.
 *
 * The torch backend's implementation lives at packages/backends/
 * backend_torch/ — per-op .cpp files mirroring the tape backend's tree.
 * This file is the build's primary-name compile unit (the Makefile's
 * backend rule expects `backend_<b>.cpp` to exist); it holds nothing
 * but a pointer index back into the modular tree.
 *
 * Tree layout (all paths under packages/backends/backend_torch/):
 *   tensor.{h,cpp}                Helpers (to_tensor inline; from_tensor /
 *                                 from_tensor_persistent in
 *                                 training/intermediates.cpp).
 *   core/lifecycle/               create / create_scalar / cast / clone /
 *                                 free / item{,1d,2d} / retain / accessors
 *                                 / create_param_state / batch.
 *   core/elementwise/             add / sub / mul / div / neg / abs / exp /
 *                                 log / sqrt / pow / sigmoid / tanh /
 *                                 softplus.
 *   core/scalar/                  add_scalar / mul_scalar / clamp_min.
 *   core/backend_meta.cpp         backend_name / reset_for_eval / print /
 *                                 mlx_compile stubs.
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
 *   conv/                         conv1d{,_circular} / conv2d / conv_transpose
 *                                 / conv_grouped / max_pool1d / max_pool2d
 *                                 / avg_pool1d / avg_pool2d.
 *   training/optimizer.cpp        OptWrapper + Adam/RMSprop/AdamW/SGD steps +
 *                                 clip / native_train_step.
 *   training/adapter.cpp          BackendPort + torch_port_* shims.
 *   training/dtype_dispatch.cpp   make_param_leaf + st_for_dtag +
 *                                 torch_create_*_dtag (+ F32/F64 wrappers).
 *   training/autograd.cpp         tensor_backward + grad / detach /
 *                                 no_grad_begin/end.
 *   training/intermediates.cpp    from_tensor + free_intermediates +
 *                                 intermediates_torch / all_pairs_torch.
 *   training/profiling.cpp        prof_*_ms counters + _wall_ms_torch.
 *   training/diagnostics.cpp      DEBUG_LSTM_TRAJ + DEBUG_PARAM_GRADS.
 *   training/ntm_specific.cpp     tensor_lstm_gates + subtract_scalar_inplace.
 *   device.cpp                    tensor_to_device + tensor_device.
 *   mps_init.cpp                  MPS allocator eager-init constructor.
 *
 * Shared (compiled per backend via SHARED_BACKENDS_<tu> lists):
 *   shared/training/param_registry.c
 *   shared/training/dtype_streamed.c
 *   shared/training/ffi_shims.c
 */
