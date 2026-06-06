"""Shared manifest + helpers for FFI wrap-template tooling and instance
generation.

This module is the single source of truth for two related concerns:
1. Which Idris-side `%foreign` declarations are Tensor-touching (and
   therefore must use the wrap-on-return Scheme template) — read by
   `ffi-convert-to-scheme.py` and `check-ffi-wrap-template.py`.
2. Which typeclass methods on `Executor/{Tape,Torch,Mlx}.idr` are
   generated from which FFI — read by `gen-executor-instances.py` and
   `check-executor-method-drift.py`.

Each entry is an `Entry` dataclass:
- `args` / `ret`: classifier tuple for the FFI's C signature.
- `slice`: typeclass sub-interface the method lives in (None if the FFI
  is internal-only and not bound to any instance method).
- `idris_method`: name of the typeclass method that calls this FFI.
- `c_symbol`: canonical C function name (defaults to the manifest key;
  override for aliased FFIs where one C symbol backs multiple methods).
- `tape` / `torch` / `mlx`: per-backend generation flavor.
"""

import re
from dataclasses import dataclass

# Type abbreviations for arg/return classification.
# T  = wrapped Tensor handle (Idris AnyPtr, vector-ref to unwrap)
# R  = raw AnyPtr (pass through; not a wrapped Tensor)
# i  = Int
# d  = Double
# s  = String
# v  = void / unit (return only)


@dataclass(frozen=True)
class Entry:
    """One FFI primitive's full description.

    Manifest is keyed by base C function name (no _tape/_torch/_mlx
    backend suffix; those are stripped before lookup). Multiple instance
    methods can share a C symbol via the `c_symbol` field — the manifest
    key stays distinct (one per Idris-level method), but `c_symbol` points
    to the actual C function called by the wrap-template.
    """
    args: tuple              # arg-class tuple ("T", "i", "d", "s", "R", or "v")
    ret: str                 # ret-class
    slice: str = None        # None when the FFI is not bound to any instance method
    idris_method: str = None # typeclass method name (`primX`); set iff slice is set
    c_symbol: str = None     # canonical C name; None = use the manifest key
    tape: str = "direct"     # direct | bespoke
    torch: str = "direct"    # direct | bespoke
    mlx: str = "streamed"    # streamed | direct | bespoke

MANIFEST = {
    "idrisml_seq": Entry(args=("R", "R"), ret="R"),
    "native_train_step_scaled": Entry(args=("R", "i", "d", "T", "d", "d"), ret="d", slice="UserExecutorOptimizer", idris_method="primNativeTrainStepScaled", mlx="direct"),
    "native_train_step": Entry(args=("R", "i", "d", "T", "d"), ret="d", slice="UserExecutorOptimizer", idris_method="primNativeTrainStep", mlx="direct"),
    "optimizer_create_adam_group": Entry(args=("d", "d", "d", "d", "s"), ret="R", slice="UserExecutorOptimizer", idris_method="primOptimizerCreateAdamGroup", mlx="direct"),
    "optimizer_create_adam_w": Entry(args=("d", "d", "d", "d", "d"), ret="R", slice="UserExecutorOptimizer", idris_method="primOptimizerCreateAdamW", c_symbol="optimizer_create_adamw", mlx="direct"),
    "optimizer_create_adam": Entry(args=("d", "d", "d", "d"), ret="R", slice="UserExecutorOptimizer", idris_method="primOptimizerCreateAdam", mlx="direct"),
    "optimizer_create_rmsprop": Entry(args=("d", "d", "d", "d", "d"), ret="R", slice="UserExecutorOptimizer", idris_method="primOptimizerCreateRmsprop", mlx="direct"),
    "optimizer_create_sgd": Entry(args=("d",), ret="R", slice="UserExecutorOptimizer", idris_method="primOptimizerCreateSgd", mlx="direct"),
    "optimizer_load": Entry(args=("R", "s"), ret="i", slice="UserExecutorSerialize", idris_method="primOptimizerLoad", mlx="direct"),
    "optimizer_save": Entry(args=("R", "s"), ret="i", slice="UserExecutorSerialize", idris_method="primOptimizerSave", mlx="direct"),
    "optimizer_set_lr": Entry(args=("R", "d"), ret="v", slice="UserExecutorOptimizer", idris_method="primOptimizerSetLr", mlx="direct"),
    "optimizer_set_param_lr": Entry(args=("R", "s", "d"), ret="v", slice="UserExecutorOptimizer", idris_method="primOptimizerSetParamLr", mlx="direct"),
    "param_count": Entry(args=(), ret="i", slice="UserExecutorParamRegistry", idris_method="primParamCount", mlx="direct"),
    "param_grad_item_at": Entry(args=("i", "i"), ret="d", slice="UserExecutorParamRegistry", idris_method="primParamGradItemAt", mlx="direct"),
    "param_load_with_policy": Entry(args=("s", "i"), ret="i", slice="UserExecutorSerialize", idris_method="primParamLoadWithPolicy", mlx="direct"),
    "param_load_with_prefix": Entry(args=("s", "i", "s"), ret="i", slice="UserExecutorSerialize", idris_method="primParamLoadWithPrefix", mlx="direct"),
    "param_load": Entry(args=("s",), ret="i", slice="UserExecutorSerialize", idris_method="primParamLoad", mlx="direct"),
    "param_name": Entry(args=("i",), ret="s", slice="UserExecutorParamRegistry", idris_method="primParamName", mlx="direct"),
    "param_register_return": Entry(args=("s", "T"), ret="T"),
    "param_register": Entry(args=("s", "T"), ret="T", slice="UserExecutorParamRegistry", idris_method="primParamRegister", c_symbol="param_register_return", mlx="direct"),
    "param_save": Entry(args=("s",), ret="i", slice="UserExecutorSerialize", idris_method="primParamSave", mlx="direct"),
    "param_save_by_name": Entry(args=("s", "s", "i"), ret="i", slice="UserExecutorSerialize", idris_method="primParamSaveByName", mlx="direct"),
    "param_save_by_name_renamed": Entry(args=("s", "s", "s", "i"), ret="i", slice="UserExecutorSerialize", idris_method="primParamSaveByNameRenamed", mlx="direct"),
    "param_tensor": Entry(args=("i",), ret="T"),
    "param_zero_all": Entry(args=(), ret="v", slice="UserExecutorParamRegistry", idris_method="primParamZeroAll", c_symbol="param_zero_all_grads", mlx="direct"),
    "param_erase_by_prefix": Entry(args=("s",), ret="v", slice="UserExecutorParamRegistry", idris_method="primParamEraseByPrefix", mlx="direct"),
    "polyak_blend": Entry(args=("d", "s", "s"), ret="i", slice="UserExecutorParamRegistry", idris_method="primPolyakBlend", mlx="direct"),
    "tensor_abs": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primAbs"),
    "tensor_absmean_per_row_2d": Entry(args=("T",), ret="T", slice="UserExecutorQuant", idris_method="primAbsmeanPerRow2d", mlx="bespoke"),
    "tensor_add_scalar": Entry(args=("T", "d"), ret="T", slice="UserExecutorCore", idris_method="primAddScalar"),
    "tensor_add": Entry(args=("T", "T"), ret="T", slice="UserExecutorCore", idris_method="primAdd"),
    "tensor_alloc_host": Entry(args=("i",), ret="T", slice="UserExecutorTransfer", idris_method="primAllocHost", c_symbol="tensor_alloc_doubles", mlx="direct"),
    "tensor_alloc_int_host": Entry(args=("i",), ret="T", slice="UserExecutorTransfer", idris_method="primAllocIntHost", c_symbol="tensor_alloc_ints", mlx="direct"),
    "tensor_argsort": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primArgsort"),
    "tensor_avg_pool1d": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primAvgPool1d"),
    "tensor_avg_pool2d": Entry(args=("T", "i", "i", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primAvgPool2d"),
    "tensor_backward_conditional": Entry(args=("T",), ret="i"),
    "tensor_backward_return_loss": Entry(args=("T", "d"), ret="d"),
    "tensor_backward_return": Entry(args=("T",), ret="T"),
    "tensor_backward": Entry(args=("T",), ret="v", slice="UserExecutorAutograd", idris_method="primBackward", mlx="direct"),
    "tensor_batch_norm": Entry(args=("T", "T", "T", "T", "T", "i", "i", "i", "d", "d"), ret="T", slice="UserExecutorNN", idris_method="primBatchNorm"),
    "tensor_batch": Entry(args=("R", "i"), ret="T"),
    "tensor_bce_with_logits": Entry(args=("T", "T"), ret="T", slice="UserExecutorNN", idris_method="primBceWithLogits"),
    "tensor_bitlinear_fwd_hf_quant": Entry(args=("T", "d", "T", "T", "i", "T", "d"), ret="T", slice="UserExecutorQuant", idris_method="primBitlinearFwdHfQuant", mlx="bespoke"),
    "tensor_bitlinear_fwd": Entry(args=("T", "T", "T", "T"), ret="T", slice="UserExecutorQuant", idris_method="primBitlinearFwd", mlx="bespoke"),
    "tensor_bmm_3x3": Entry(args=("T", "T"), ret="T"),
    "tensor_bmm": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primBmm"),
    "tensor_cast_dtype_streamed": Entry(args=("T", "i", "i"), ret="T"),
    "tensor_cast_streamed": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCastStreamed", c_symbol="tensor_cast_dtype_streamed", mlx="direct"),
    "tensor_cat_from_array": Entry(args=("R", "i", "i"), ret="T"),
    "tensor_cat": Entry(args=("R", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primCat"),
    "tensor_cat2": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primCat2"),
    "tensor_causal_mask": Entry(args=("i",), ret="T"),
    "tensor_clamp_min": Entry(args=("T", "d"), ret="T", slice="UserExecutorCore", idris_method="primClampMin"),
    "tensor_clamp": Entry(args=("T", "d", "d"), ret="T", slice="UserExecutorCore", idris_method="primClamp"),
    "tensor_clone": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primClone"),
    "tensor_concat_2d_axis1": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primConcat2dAxis1"),
    "tensor_conv_transpose1d": Entry(args=("T", "T", "T", "i", "i"), ret="T"),
    "tensor_conv_transpose2d": Entry(args=("T", "T", "T", "i", "i", "i", "i"), ret="T"),
    "tensor_conv1d_circular": Entry(args=("T", "T"), ret="T", slice="UserExecutorConv", idris_method="primConv1dCircular"),
    "tensor_conv1d_grouped": Entry(args=("T", "T", "T", "i", "i", "i"), ret="T"),
    "tensor_conv1d": Entry(args=("T", "T", "T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primConv1d"),
    "tensor_conv2d_batched": Entry(args=("T", "T", "T", "i", "i", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primConv2dBatched"),
    "tensor_conv2d_grouped": Entry(args=("T", "T", "T", "i", "i", "i", "i", "i"), ret="T"),
    "tensor_conv2d": Entry(args=("T", "T", "T", "i", "i", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primConv2d"),
    "tensor_cosine_similarity": Entry(args=("T", "T", "i"), ret="T", slice="UserExecutorNN", idris_method="primCosineSimilarity"),
    "tensor_create_1d_streamed": Entry(args=("i", "R", "i", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreate1dStreamed", mlx="direct"),
    "tensor_create_1d": Entry(args=("i", "R", "i"), ret="T"),
    "tensor_create_2d_streamed": Entry(args=("i", "i", "R", "i", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreate2dStreamed", mlx="direct"),
    "tensor_create_2d": Entry(args=("i", "i", "R", "i"), ret="T"),
    "tensor_create_from_host": Entry(args=("R", "R", "i", "i"), ret="T", slice="UserExecutorTransfer", idris_method="primCreateFromHost", c_symbol="tensor_create", torch="bespoke", mlx="direct"),
    "tensor_create_param_1d_const_streamed": Entry(args=("i", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam1dConstStreamed", mlx="direct"),
    "tensor_create_param_1d_normal_streamed": Entry(args=("i", "d", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam1dNormalStreamed", mlx="direct"),
    "tensor_create_param_1d_streamed": Entry(args=("i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam1dStreamed", mlx="direct"),
    "tensor_create_param_1d": Entry(args=("i", "R"), ret="T"),
    "tensor_create_param_2d_const_streamed": Entry(args=("i", "i", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam2dConstStreamed", mlx="direct"),
    "tensor_create_param_2d_normal_streamed": Entry(args=("i", "i", "d", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam2dNormalStreamed", mlx="direct"),
    "tensor_create_param_2d_streamed": Entry(args=("i", "i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam2dStreamed", mlx="direct"),
    "tensor_create_param_2d": Entry(args=("i", "i", "R"), ret="T"),
    "tensor_create_param_3d_const_streamed": Entry(args=("i", "i", "i", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam3dConstStreamed", mlx="direct"),
    "tensor_create_param_3d_normal_streamed": Entry(args=("i", "i", "i", "d", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam3dNormalStreamed", mlx="direct"),
    "tensor_create_param_3d_streamed": Entry(args=("i", "i", "i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam3dStreamed", mlx="direct"),
    "tensor_create_param_3d": Entry(args=("i", "i", "i", "R"), ret="T"),
    "tensor_create_param_4d_const_streamed": Entry(args=("i", "i", "i", "i", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam4dConstStreamed", mlx="direct"),
    "tensor_create_param_4d_normal_streamed": Entry(args=("i", "i", "i", "i", "d", "d", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam4dNormalStreamed", mlx="direct"),
    "tensor_create_param_4d_streamed": Entry(args=("i", "i", "i", "i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateParam4dStreamed", mlx="direct"),
    "tensor_create_param_4d": Entry(args=("i", "i", "i", "i", "R"), ret="T"),
    "tensor_create_scalar_streamed": Entry(args=("d", "i", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateScalarStreamed", mlx="direct"),
    "tensor_create_scalar": Entry(args=("d", "i"), ret="T", slice="UserExecutorCore", idris_method="primCreateScalar", torch="bespoke"),
    "tensor_create_state_1d_streamed": Entry(args=("i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateState1dStreamed", mlx="direct"),
    "tensor_create_state_1d": Entry(args=("i", "R"), ret="T"),
    "tensor_create_state_2d_streamed": Entry(args=("i", "i", "R", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateState2dStreamed", mlx="direct"),
    "tensor_create_state_2d": Entry(args=("i", "i", "R"), ret="T"),
    "tensor_create_streamed": Entry(args=("R", "R", "i", "i", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primCreateStreamed", mlx="direct"),
    "tensor_create_ternary_from_hf_packed_2d": Entry(args=("R", "i", "i"), ret="T", slice="UserExecutorQuant", idris_method="primCreateTernaryFromHfPacked2d", mlx="bespoke"),
    "tensor_create_ternary_packed_2d": Entry(args=("R", "i", "i", "i", "i"), ret="T", slice="UserExecutorQuant", idris_method="primCreateTernaryPacked2d", mlx="bespoke"),
    "tensor_cross_attention": Entry(args=("T", "T", "T", "T", "d"), ret="T", slice="UserExecutorNN", idris_method="primCrossAttention"),
    "tensor_cross_entropy": Entry(args=("T", "T"), ret="T"),
    "tensor_cumprod": Entry(args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primCumprod"),
    "tensor_detach": Entry(args=("T",), ret="T", slice="UserExecutorAutograd", idris_method="primDetach"),
    "tensor_device": Entry(args=("T",), ret="s"),
    "tensor_dim": Entry(args=("T",), ret="i"),
    "tensor_div": Entry(args=("T", "T"), ret="T", slice="UserExecutorCore", idris_method="primDiv"),
    "tensor_dot": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primDot"),
    "tensor_dropout": Entry(args=("T", "d", "i", "i"), ret="T", slice="UserExecutorNN", idris_method="primDropout"),
    "tensor_embedding_2d": Entry(args=("T", "T", "i", "i"), ret="T", slice="UserExecutorNN", idris_method="primEmbedding2d"),
    "tensor_embedding": Entry(args=("T", "T", "i", "i"), ret="T", slice="UserExecutorNN", idris_method="primEmbedding"),
    "tensor_epoch_begin": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primEpochBegin", mlx="direct"),
    "tensor_epoch_end": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primEpochEnd", mlx="direct"),
    "tensor_exp": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primExp"),
    "tensor_expand_mask": Entry(args=("T", "i"), ret="T", slice="UserExecutorNN", idris_method="primExpandMask"),
    "tensor_free_host": Entry(args=("T",), ret="v", slice="UserExecutorTransfer", idris_method="primFreeHost", c_symbol="tensor_free_doubles", mlx="direct"),
    "tensor_free_int_host": Entry(args=("T",), ret="v", slice="UserExecutorTransfer", idris_method="primFreeIntHost", c_symbol="tensor_free_ints", mlx="direct"),
    "tensor_free": Entry(args=("T",), ret="v", slice="UserExecutorCore", idris_method="primFree"),
    "tensor_gather": Entry(args=("T", "T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primGather"),
    "tensor_gelu": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primGelu"),
    "tensor_grad": Entry(args=("T",), ret="T"),
    "tensor_group_norm": Entry(args=("T", "T", "T", "i", "i", "i", "d"), ret="T"),
    "tensor_gru_cell": Entry(args=("T", "T", "T", "i"), ret="T", slice="UserExecutorNN", idris_method="primGruCell"),
    "tensor_intra_migrate": Entry(args=("T", "s"), ret="T", slice="UserExecutorTransfer", idris_method="primIntraMigrate", c_symbol="tensor_to_device", torch="bespoke", mlx="direct"),
    "tensor_item_1d": Entry(args=("T", "i"), ret="d", slice="UserExecutorCore", idris_method="primItem1d"),
    "tensor_item_2d": Entry(args=("T", "i", "i"), ret="d", slice="UserExecutorTensorCreate", idris_method="primItem2d", mlx="direct"),
    "tensor_item": Entry(args=("T",), ret="d", slice="UserExecutorCore", idris_method="primItem"),
    "tensor_layer_norm_2d": Entry(args=("T", "T", "T", "d"), ret="T", slice="UserExecutorNN", idris_method="primLayerNorm2d"),
    "tensor_leaky_relu": Entry(args=("T", "d"), ret="T", slice="UserExecutorNN", idris_method="primLeakyRelu"),
    "tensor_linear_2d": Entry(args=("T", "T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primLinear2d"),
    "tensor_linear": Entry(args=("T", "T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primLinear"),
    "tensor_live_count": Entry(args=("i",), ret="i", slice="UserExecutorProfiling", idris_method="primLiveCount", mlx="direct"),
    "tensor_log_softmax_2d": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primLogSoftmax2d"),
    "tensor_log_softmax": Entry(args=("T", "i"), ret="T", slice="UserExecutorNN", idris_method="primLogSoftmax"),
    "tensor_log": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primLog"),
    "tensor_lstm_cell": Entry(args=("T", "T", "T", "T", "T", "T", "T", "R", "R"), ret="v"),
    "tensor_lstm_gates_pair": Entry(args=("T", "T", "i"), ret="R", slice="UserExecutorNN", idris_method="primLstmGatesPair"),
    "tensor_lstm_gates": Entry(args=("T", "T", "i", "R", "R"), ret="v"),
    "tensor_masked_fill": Entry(args=("T", "T", "d"), ret="T", slice="UserExecutorNN", idris_method="primMaskedFill"),
    "tensor_matmul": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMatmul"),
    "tensor_max_pool1d": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primMaxPool1d"),
    "tensor_max_pool2d_batched": Entry(args=("T", "i", "i", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primMaxPool2dBatched"),
    "tensor_max_pool2d": Entry(args=("T", "i", "i", "i", "i"), ret="T", slice="UserExecutorConv", idris_method="primMaxPool2d"),
    "tensor_max": Entry(args=("T",), ret="T"),
    "tensor_mean": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primMean"),
    "tensor_min": Entry(args=("T",), ret="T"),
    "tensor_mm": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMm"),
    "tensor_mse_loss": Entry(args=("T", "T"), ret="T"),
    "tensor_mul_scalar": Entry(args=("T", "d"), ret="T", slice="UserExecutorCore", idris_method="primMulScalar"),
    "tensor_mul": Entry(args=("T", "T"), ret="T", slice="UserExecutorCore", idris_method="primMul"),
    "tensor_mv": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primMv"),
    "tensor_narrow": Entry(args=("T", "i", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primNarrow"),
    "tensor_neg": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primNeg"),
    "tensor_no_grad_begin": Entry(args=(), ret="v", slice="UserExecutorAutograd", idris_method="primNoGradBegin", mlx="direct"),
    "tensor_no_grad_end": Entry(args=(), ret="v", slice="UserExecutorAutograd", idris_method="primNoGradEnd", mlx="direct"),
    "tensor_numel": Entry(args=("T",), ret="i"),
    "tensor_one_hot": Entry(args=("R", "i", "i", "i"), ret="T", slice="UserExecutorTensorCreate", idris_method="primOneHot", mlx="direct"),
    "tensor_outer": Entry(args=("T", "T"), ret="T", slice="UserExecutorLinear", idris_method="primOuter"),
    "tensor_pair_first": Entry(args=("R",), ret="T", slice="UserExecutorNN", idris_method="primPairFirst"),
    "tensor_pair_free": Entry(args=("R",), ret="v"),
    "tensor_pair_second": Entry(args=("R",), ret="T", slice="UserExecutorNN", idris_method="primPairSecond"),
    "tensor_peak_live_count": Entry(args=("i",), ret="i", slice="UserExecutorProfiling", idris_method="primPeakLiveCount", mlx="direct"),
    "tensor_perf_op_count": Entry(args=(), ret="i", slice="UserExecutorProfiling", idris_method="primPerfOpCount", mlx="direct"),
    "tensor_perf_reset": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primPerfReset", mlx="direct"),
    "tensor_pow": Entry(args=("T", "T"), ret="T", slice="UserExecutorCore", idris_method="primPow"),
    "tensor_print": Entry(args=("T",), ret="v"),
    "tensor_profile_report": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primProfileReport", c_symbol="backend_profile_report", mlx="direct"),
    "tensor_profile_reset": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primProfileReset", c_symbol="backend_profile_reset", mlx="direct"),
    "tensor_ptr_array_set_return": Entry(args=("R", "i", "T"), ret="R"),
    "tensor_ptr_array_set": Entry(args=("R", "i", "T"), ret="v"),
    "tensor_release_all_persistent": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primReleaseAllPersistent", c_symbol="backend_release_all_persistent", mlx="direct"),
    "tensor_requires_grad": Entry(args=("T",), ret="i", slice="UserExecutorAutograd", idris_method="primRequiresGrad", mlx="direct"),
    "tensor_reset_for_eval": Entry(args=(), ret="v", slice="UserExecutorProfiling", idris_method="primResetForEval", c_symbol="backend_reset_for_eval", mlx="direct"),
    "tensor_reshape_1d": Entry(args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape1d"),
    "tensor_reshape_2d": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape2d"),
    "tensor_reshape_3d": Entry(args=("T", "i", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape3d"),
    "tensor_reshape_4d": Entry(args=("T", "i", "i", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primReshape4d"),
    "tensor_reshape": Entry(args=("T", "R", "i"), ret="T"),
    "tensor_rms_norm_2d": Entry(args=("T", "T", "d"), ret="T", slice="UserExecutorOptimizations", idris_method="primRmsNorm2d", mlx="direct"),
    "tensor_round": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primRound"),
    "tensor_scatter_add": Entry(args=("T", "T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primScatterAdd"),
    "tensor_sdpa_2d": Entry(args=("T", "T", "T", "i", "i", "i", "i"), ret="T", slice="UserExecutorOptimizations", idris_method="primSdpa2d", mlx="direct"),
    "tensor_select": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSelect"),
    "tensor_set_init_seed_streamed": Entry(args=("i", "i"), ret="v", slice="UserExecutorTensorCreate", idris_method="primSetInitSeedStreamed", mlx="direct"),
    "tensor_set_int_host": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorTransfer", idris_method="primSetIntHost", c_symbol="tensor_write_int_return", mlx="direct"),
    "tensor_set_requires_grad": Entry(args=("T", "i"), ret="v", slice="UserExecutorAutograd", idris_method="primSetRequiresGrad", mlx="direct"),
    "tensor_sigmoid": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primSigmoid"),
    "tensor_silu": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primSilu"),
    "tensor_size": Entry(args=("T", "i"), ret="i"),
    "tensor_softmax_2d": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primSoftmax2d"),
    "tensor_softmax_3d": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primSoftmax3d"),
    "tensor_softmax": Entry(args=("T", "i"), ret="T", slice="UserExecutorNN", idris_method="primSoftmax"),
    "tensor_softplus": Entry(args=("T",), ret="T", slice="UserExecutorNN", idris_method="primSoftplus"),
    "tensor_sqrt": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primSqrt"),
    "tensor_squeeze": Entry(args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSqueeze"),
    "tensor_stack_from_array": Entry(args=("R", "i", "i"), ret="T"),
    "tensor_stack": Entry(args=("R", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primStack"),
    "tensor_sub": Entry(args=("T", "T"), ret="T", slice="UserExecutorCore", idris_method="primSub"),
    "tensor_subtract_scalar_inplace": Entry(args=("T", "d"), ret="T"),
    "tensor_sum_dim": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primSumDim"),
    "tensor_sum": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primSum"),
    "tensor_swi_glu_2d": Entry(args=("T", "T"), ret="T", slice="UserExecutorOptimizations", idris_method="primSwiGlu2d", c_symbol="tensor_swiglu_2d", mlx="direct"),
    "tensor_swiglu_2d": Entry(args=("T", "T"), ret="T"),
    "tensor_tanh": Entry(args=("T",), ret="T", slice="UserExecutorCore", idris_method="primTanh"),
    "tensor_tensor_dim": Entry(args=("T",), ret="i", slice="UserExecutorTensorCreate", idris_method="primTensorDim", c_symbol="tensor_dim", mlx="direct"),
    "tensor_tensor_max": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTensorMax", c_symbol="tensor_max"),
    "tensor_tensor_min": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTensorMin", c_symbol="tensor_min"),
    "tensor_tensor_size_at": Entry(args=("T", "i"), ret="i", slice="UserExecutorTensorCreate", idris_method="primTensorSizeAt", c_symbol="tensor_size", mlx="direct"),
    "tensor_ternary_quant_with_scale_2d": Entry(args=("T", "T"), ret="T", slice="UserExecutorQuant", idris_method="primTernaryQuantWithScale2d", mlx="bespoke"),
    "tensor_tile_2d": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primTile2d"),
    "tensor_to_device": Entry(args=("T", "s"), ret="T"),
    "tensor_to_doubles": Entry(args=("T", "R"), ret="v"),
    "tensor_to_host": Entry(args=("T", "T"), ret="T", slice="UserExecutorTransfer", idris_method="primToHost", c_symbol="tensor_to_doubles", mlx="direct"),
    "tensor_transpose_2d": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTranspose2d"),
    "tensor_transpose_last2": Entry(args=("T",), ret="T", slice="UserExecutorLinear", idris_method="primTransposeLast2"),
    "tensor_unbatch": Entry(args=("T", "R"), ret="R"),
    "tensor_unsqueeze": Entry(args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primUnsqueeze"),
    "tensor_view_1d": Entry(args=("T", "i"), ret="T", slice="UserExecutorLinear", idris_method="primView1d"),
    "tensor_view_2d": Entry(args=("T", "i", "i"), ret="T", slice="UserExecutorLinear", idris_method="primView2d"),
    "tensor_with_grad": Entry(args=("T",), ret="T", slice="UserExecutorAutograd", idris_method="primWithGrad"),
    "tensor_zero_grad": Entry(args=("T",), ret="v"),
}

# C function names to LEAVE AS-IS (don't convert).
# Reasons:
# - Take no Tensor args and don't return Tensors
# - OR are part of the refcount/lifecycle machinery itself (would recurse)
# - OR are too special to mechanically convert (refcount, retain/release).
SKIP = {
    "tensor_retain_handle",
    "tensor_release_handle",
}

# Note: every literal " inside the Scheme body must be emitted as \" so
# Idris's surrounding `%foreign "scheme:..."` string literal stays intact.
#
# We *do not* inline a libidrisml/guardian init check in every wrapper —
# that runs on every FFI call (~3-5M times in a real training run) and
# costs measurable wall time. Instead, we rely on:
#   1. Idris-2's chez codegen calls `load-shared-object "libidrisml.dylib"`
#      at module load time for every `%foreign "C:..."` declaration. Some
#      of those still exist (mnist_*, optimizer_*, no_grad_*, etc.), so
#      libidrisml is loaded before any Scheme wrapper executes.
#   2. `initManagedHandles` (called by MkTensor / withNoGrad / first Tensor
#      creation) sets up the guardian; if not already done, the very first
#      Tensor-creating Scheme wrapper (e.g. tensor_create_scalar) lazy-
#      inits it via its own (already-existing) init check.
#
# So new wrappers can assume `idris-tensor-guardian` is bound; for the
# create-scalar / create-{state,param}_*d wrappers that *might* be the
# first to run, we add a one-shot guardian lazy-init.
# The guardian itself — created once if absent.
GUARDIAN_ONLY_INIT = (
    "(when (not (top-level-bound? 'idris-tensor-guardian))"
    " (set-top-level-value! 'idris-tensor-guardian (make-guardian)))"
)

# Prime the guardian *drain* helper at the same point the guardian is
# created. `idris-drain-once` pops one dead wrap, reads the backend tag at
# slot 1 + raw pointer at slot 2, and calls the matching
# `tensor_release_handle_<tag>` (caching the foreign-procedure per tag).
# This is the EXACT logic of `prim__installDrainHelperC` in Tensor.idr —
# keep the two in sync (that one stays for the test harness). Self-guarded
# on `idris-drain-once` so it installs once and is a cheap bound-check on
# every later create call. Without this the drain epilogues in
# `native_train_step_*` and `withNoGrad` are dormant (their
# `(top-level-bound? 'idris-drain-once)` guard is false), so mlx husks
# never reach rc==0 and leak on long grad-mode runs. See
# docs/develop/gotchas.md "The mlx generation sweep must never delete…".
DRAIN_ONCE_INSTALL = (
    "(when (not (top-level-bound? 'idris-drain-once))"
    " (when (not (top-level-bound? 'idris-release-cache))"
    " (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?)))"
    " (set-top-level-value! 'idris-drain-once (lambda ()"
    " (when (not (top-level-bound? 'idris-tensor-guardian))"
    " (set-top-level-value! 'idris-tensor-guardian (make-guardian)))"
    " (let ((d ((top-level-value 'idris-tensor-guardian))))"
    " (if (not d) #f"
    " (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache)))"
    " (let ((rel (or (hashtable-ref cache tag #f)"
    " (let ((sym (if (string=? tag \\\"primary\\\") \\\"tensor_release_handle\\\""
    " (string-append \\\"tensor_release_handle_\\\" tag))))"
    " (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp)))))"
    " (rel raw) #t)))))))"
)

# Both run on the first Tensor-creating Scheme wrapper (the create-scalar
# / create-{state,param}_*d ones), so by the time any training/eval drain
# point fires the guardian + drain helper are both bound.
GUARDIAN_LAZY_INIT = GUARDIAN_ONLY_INIT + " " + DRAIN_ONCE_INSTALL

# C function names whose Scheme wrapper carries the guardian-lazy-init
# (they're the ones that can be the very first Tensor-creating call).
INIT_FFI = {
    "tensor_create_scalar",
    "tensor_create",
    "tensor_create_1d",
    "tensor_create_2d",
    "tensor_create_param_1d",
    "tensor_create_param_2d",
    "tensor_create_param_3d",
    "tensor_create_param_4d",
    "tensor_create_state_1d",
    "tensor_create_state_2d",
    "tensor_one_hot",
    # Unified dtag create/cast wrappers — each can be the first
    # Tensor-creating call, so they carry the guardian lazy-init.
    "tensor_create_scalar_streamed",
    "tensor_create_streamed",
    "tensor_create_1d_streamed",
    "tensor_create_2d_streamed",
    "tensor_create_param_1d_streamed",
    "tensor_create_param_2d_streamed",
    "tensor_create_param_3d_streamed",
    "tensor_create_param_4d_streamed",
    "tensor_create_state_1d_streamed",
    "tensor_create_state_2d_streamed",
    "tensor_cast_dtype_streamed",
    # Fused-init creators (added 2026-05-28) — same rule: each can be
    # the first Tensor-creating call in a program (HfBert's
    # makeBertLinear is now the first FFI on the BERT path, etc.).
    "tensor_create_param_1d_normal_streamed",
    "tensor_create_param_2d_normal_streamed",
    "tensor_create_param_3d_normal_streamed",
    "tensor_create_param_4d_normal_streamed",
    "tensor_create_param_1d_const_streamed",
    "tensor_create_param_2d_const_streamed",
    "tensor_create_param_3d_const_streamed",
    "tensor_create_param_4d_const_streamed",
    # Quantization. `tensor_create_ternary_packed_2d` is a tensor-
    # creating wrapper that may be the first FFI on a BitNet inference
    # path (load weights → forward), so it needs the guardian
    # lazy-init. `tensor_bitlinear_fwd` takes existing handles only.
    # `tensor_create_ternary_from_hf_packed_2d` is the HF-format variant —
    # same first-FFI rationale.
    "tensor_create_ternary_packed_2d",
    "tensor_create_ternary_from_hf_packed_2d",
}


def strip_suffix(cname):
    """Strip _mlx / _tape / _torch suffix from a C name to get the base.

    Note: `_<backend>_streamed` compound names (mlx-specific, e.g.
    `tensor_mv_mlx_streamed`) are NOT stripped — they have an extra
    stream-arg vs their base manifest entry (`tensor_mv` is 2-arg, the
    streamed variant is 3-arg), so the manifest classifiers don't
    apply. Those variants are managed by hand outside the manifest
    pipeline."""
    for suf in ("_mlx", "_tape", "_torch"):
        if cname.endswith(suf):
            return cname[: -len(suf)]
    return cname


def parse_args(idris_sig):
    """Parse 'export prim__foo : T1 -> T2 -> ... -> Tn' into (name, [T1..T_{n-1}], Tn)."""
    s = idris_sig.strip()
    if s.startswith("export"):
        s = s[len("export"):].strip()
    name, _, rest = s.partition(":")
    name = name.strip()
    parts = [p.strip() for p in rest.split("->")]
    args = parts[:-1]
    ret = parts[-1]
    return name, args, ret


def idris_type_to_class(t, manifest_class):
    """Map Idris type → classifier. manifest_class disambiguates AnyPtr."""
    t = t.strip()
    if t == "AnyPtr":
        return manifest_class
    if t == "Int":
        return "i"
    if t == "Double":
        return "d"
    if t == "String":
        return "s"
    if t in ("()", "PrimIO ()"):
        return "v"
    if t == "PrimIO Int":
        return "i"
    if t == "PrimIO Double":
        return "d"
    if t == "PrimIO String":
        return "s"
    if t == "PrimIO AnyPtr":
        return manifest_class
    # Default: assume raw pointer
    return manifest_class


def scheme_type(cls):
    """Classifier → foreign-procedure type."""
    if cls == "T" or cls == "R":
        return "void*"
    if cls == "i":
        return "int"
    if cls == "d":
        return "double"
    if cls == "s":
        return "string"
    if cls == "v":
        return "void"
    raise ValueError(f"Unknown classifier {cls!r}")


def cache_var(c_symbol):
    """Per-FFI Chez top-level binding name for the cached foreign-procedure.

    Maps `tensor_add_torch` → `idris-ffi-tensor-add-torch`. The
    `idris-ffi-` prefix scopes the cache to this binding family and avoids
    collisions with the existing `idris-tensor-guardian` /
    `idris-release-cache` / `idris-drain-once` top-level symbols.

    Globally unique because the C symbols themselves are globally unique
    within libidrisml.dylib (each is suffixed `_tape` / `_torch` / `_mlx`
    unless it's a primary-backend unified alias).
    """
    return "idris-ffi-" + c_symbol.replace("_", "-")


def backend_tag_of(cname):
    """Derive the backend tag for a wrap from the C function name.

    `tensor_add_tape`  → "tape"
    `tensor_add_torch` → "torch"
    `tensor_add_mlx`   → "mlx"
    `tensor_add`       → "primary" (unified name, link-time aliased to
                          primary backend; the drain dispatches "primary"
                          to the unified `tensor_release_handle` so the
                          same alias still routes correctly).

    `*_mlx_streamed` and similar variants strip the `_streamed` infix
    first via `strip_suffix` to find the backend suffix.
    """
    # Streamed variants like `tensor_add_mlx_streamed` — strip the trailing
    # `_streamed` infix first (only mlx ever carries it) so the backend
    # suffix is at the tail of the name.
    base = cname[:-len("_streamed")] if cname.endswith("_streamed") else cname
    for suf, tag in (("_tape", "tape"), ("_torch", "torch"), ("_mlx", "mlx")):
        if base.endswith(suf):
            return tag
    return "primary"


def gen_scheme_wrapper(cname, arg_classes, ret_class):
    """Generate the canonical Scheme lambda body for one FFI.

    The output is the literal string that would appear inside the
    surrounding `%foreign "scheme:..."` declaration — i.e. `"` is
    already escaped as `\\"`.

    Wrap layout (v2): a Tensor-returning wrap returns a 3-slot vector
        `(vector 'tensor-handle-v2 <tag-string> raw_r)`
    where tag is one of "tape", "torch", "mlx", or "primary" (for
    unsuffixed C names that link-alias to the build's primary).

    The drain function in Tensor.idr reads the tag at slot 1 and the
    raw pointer at slot 2, then dispatches to the matching
    `tensor_release_handle_<tag>` (or unified `tensor_release_handle`
    for "primary"). Retain is symmetric — each wrap calls the
    suffixed retain so refcounts land on the right backend.

    **FFI symbol caching (added 2026-05-27):** each `foreign-procedure`
    is lazy-cached at first call via a per-FFI Chez top-level binding
    (`idris-ffi-<c-symbol-with-dashes>`). Without the cache every call
    re-evaluates `(foreign-procedure …)` → fresh dlsym → walks every
    loaded library's symbol table; on a Llama-3.2-1B forward pass that
    dominated 100% of CPU wall (see sample at
    `/tmp/scheme_2026-05-27_180602_BTJW.sample.txt`). The lazy-init
    block uses the same `(when (not (top-level-bound? 'name))
    (set-top-level-value! 'name …))` idiom the codebase already uses
    for `idris-tensor-guardian`, extended from one shared symbol to
    one per `%foreign`. First call to each FFI still pays dlsym;
    subsequent warm calls pay only a hashtable lookup.
    """
    n_args = len(arg_classes)
    arg_names = [f"a{i}" for i in range(n_args)]
    fp_arg_types = " ".join(scheme_type(c) for c in arg_classes)
    fp_ret_type = scheme_type(ret_class)
    call_args = []
    for nm, cls in zip(arg_names, arg_classes):
        if cls == "T":
            # v2 layout: raw pointer lives at slot 2 (slot 0 = sentinel,
            # slot 1 = backend tag string).
            call_args.append(f"(vector-ref {nm} 2)")
        else:
            call_args.append(nm)
    call_args_str = " ".join(call_args)

    # Lazy-init for the main FFI symbol. Constructs the foreign-procedure
    # once on first call, then reuses the cached top-level binding.
    main_var = cache_var(cname)
    init_main = (
        f" (when (not (top-level-bound? '{main_var}))"
        f" (set-top-level-value! '{main_var}"
        f" (foreign-procedure \\\"{cname}\\\" ({fp_arg_types}) {fp_ret_type})))"
    )
    call_main = f"((top-level-value '{main_var}) {call_args_str})"

    if ret_class == "T":
        tag = backend_tag_of(cname)
        retain_sym = (
            "tensor_retain_handle"
            if tag == "primary"
            else f"tensor_retain_handle_{tag}"
        )
        # Lazy-init for the per-backend retain symbol (mirrors the main
        # FFI cache; one top-level binding per distinct retain symbol).
        retain_var = cache_var(retain_sym)
        init_retain = (
            f" (when (not (top-level-bound? '{retain_var}))"
            f" (set-top-level-value! '{retain_var}"
            f" (foreign-procedure \\\"{retain_sym}\\\" (void*) void)))"
        )
        call_retain = f"((top-level-value '{retain_var}) raw_r)"
        ffi_init = init_main + init_retain
        body = (
            f" (let ((raw_r {call_main}))"
            f" (let ((wr (vector 'tensor-handle-v2 \\\"{tag}\\\" raw_r)))"
            f" ((top-level-value 'idris-tensor-guardian) wr)"
            f" {call_retain}"
            f" wr))"
        )
    else:
        ffi_init = init_main
        body = f" {call_main}"

    # `GUARDIAN_LAZY_INIT` is conditional on this being an INIT_FFI
    # function; it installs the guardian/drain-once. Independent of the
    # per-FFI cache above. Order: guardian first (existing convention),
    # then per-FFI cache, then body — though they're commutative.
    guardian_init = GUARDIAN_LAZY_INIT if strip_suffix(cname) in INIT_FFI else ""
    return f"(lambda ({' '.join(arg_names)}) {guardian_init}{ffi_init}{body})"


# Files in the wrap-handle FFI set — the linter and converter both
# operate on these.
WRAP_HANDLE_FILES = [
    "packages/idris-ml/src/Tensor.idr",
    "packages/idris-ml/src/Executor/Mlx.idr",
    "packages/idris-ml/src/Executor/Tape.idr",
    "packages/idris-ml/src/Executor/Torch.idr",
]


# Matches a `%foreign "C:cname,libidrisml"` declaration + its
# immediately-following Idris signature line.
C_FFI_RE = re.compile(
    r'(%foreign\s+"C:([a-zA-Z_0-9]+),libidrisml"\s*\n)'
    r'((?:[ \t]*export[ \t]*\n)?)'
    r'([ \t]*(?:export[ \t]+)?'
    r'[a-zA-Z_][a-zA-Z_0-9\']*'
    r'\s*:\s*[^\n]+\n)',
    re.MULTILINE,
)

# Matches any `%foreign "..."` declaration + its signature.
ANY_FFI_RE = re.compile(
    r'(%foreign\s+"(C|scheme):([^"]*(?:\\"[^"]*)*)"\s*\n)'
    r'((?:[ \t]*export[ \t]*\n)?)'
    r'([ \t]*(?:export[ \t]+)?'
    r'[a-zA-Z_][a-zA-Z_0-9\']*'
    r'\s*:\s*[^\n]+\n)',
    re.MULTILINE,
)
