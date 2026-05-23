"""Shared manifest + helpers for FFI wrap-template tooling.

This module is the single source of truth for which Idris-side `%foreign`
declarations are Tensor-touching (and therefore must use the
wrap-on-return Scheme template) vs raw FFIs. Both
`ffi-convert-to-scheme.py` (the converter) and `check-ffi-wrap-template.py`
(the linter) import from here.
"""

import re

# Type abbreviations for arg/return classification.
# T  = wrapped Tensor handle (Idris AnyPtr, vector-ref to unwrap)
# R  = raw AnyPtr (pass through; not a wrapped Tensor)
# i  = Int
# d  = Double
# s  = String
# v  = void / unit (return only)
#
# Each manifest entry is (args, ret) tuple.
# Manifest is keyed by base C function name (no _tape/_torch/_mlx/_unified
# suffix — those are stripped before lookup).

MANIFEST = {
    # Lifecycle
    "tensor_create_scalar":             (("d", "i"), "T"),
    "tensor_create":                    (("R", "R", "i", "i"), "T"),
    "tensor_clone":                     (("T",), "T"),
    "tensor_free":                      (("T",), "v"),

    # Accessors
    "tensor_item":                      (("T",), "d"),
    "tensor_numel":                     (("T",), "i"),
    "tensor_dim":                       (("T",), "i"),
    "tensor_size":                      (("T", "i"), "i"),
    "tensor_to_doubles":                (("T", "R"), "v"),

    # Arithmetic
    "tensor_add":                       (("T", "T"), "T"),
    "tensor_sub":                       (("T", "T"), "T"),
    "tensor_mul":                       (("T", "T"), "T"),
    "tensor_div":                       (("T", "T"), "T"),
    "tensor_neg":                       (("T",), "T"),
    "tensor_abs":                       (("T",), "T"),
    "tensor_exp":                       (("T",), "T"),
    "tensor_log":                       (("T",), "T"),
    "tensor_sqrt":                      (("T",), "T"),
    "tensor_pow":                       (("T", "T"), "T"),
    "tensor_sigmoid":                   (("T",), "T"),
    "tensor_tanh":                      (("T",), "T"),
    "tensor_gelu":                      (("T",), "T"),
    "tensor_leaky_relu":                (("T", "d"), "T"),
    "tensor_silu":                      (("T",), "T"),
    "tensor_softplus":                  (("T",), "T"),
    "tensor_add_scalar":                (("T", "d"), "T"),
    "tensor_mul_scalar":                (("T", "d"), "T"),
    "tensor_clamp_min":                 (("T", "d"), "T"),
    "tensor_subtract_scalar_inplace":   (("T", "d"), "T"),

    # Reduction
    "tensor_sum":                       (("T",), "T"),
    "tensor_sum_dim":                   (("T", "i", "i"), "T"),
    "tensor_mean":                      (("T",), "T"),
    "tensor_min":                       (("T",), "T"),
    "tensor_max":                       (("T",), "T"),

    # Linear algebra
    "tensor_matmul":                    (("T", "T"), "T"),
    "tensor_mv":                        (("T", "T"), "T"),
    "tensor_linear":                    (("T", "T", "T"), "T"),
    "tensor_linear_2d":                 (("T", "T", "T"), "T"),
    "tensor_concat_2d_axis1":           (("T", "T"), "T"),
    "tensor_dot":                       (("T", "T"), "T"),
    "tensor_outer":                     (("T", "T"), "T"),

    # Activation / normalization
    "tensor_softmax":                   (("T", "i"), "T"),
    "tensor_log_softmax":               (("T", "i"), "T"),
    "tensor_softmax_2d":                (("T",), "T"),
    "tensor_log_softmax_2d":            (("T",), "T"),
    "tensor_softmax_3d":                (("T",), "T"),

    # Loss
    "tensor_bce_with_logits":           (("T", "T"), "T"),
    "tensor_cross_entropy":             (("T", "T"), "T"),
    "tensor_mse_loss":                  (("T", "T"), "T"),

    # Norm + dropout
    "tensor_batch_norm":                (("T", "T", "T", "T", "T", "i", "i", "i", "d", "d"), "T"),
    "tensor_group_norm":                (("T", "T", "T", "i", "i", "i", "d"), "T"),
    "tensor_layer_norm_2d":             (("T", "T", "T", "d"), "T"),
    "tensor_dropout":                   (("T", "d", "i", "i"), "T"),

    # Conv + pool
    "tensor_conv1d":                    (("T", "T", "T", "i", "i"), "T"),
    "tensor_conv1d_grouped":            (("T", "T", "T", "i", "i", "i"), "T"),
    "tensor_conv1d_circular":           (("T", "T"), "T"),
    "tensor_conv2d":                    (("T", "T", "T", "i", "i", "i", "i"), "T"),
    "tensor_conv2d_batched":            (("T", "T", "T", "i", "i", "i", "i"), "T"),
    "tensor_conv2d_grouped":            (("T", "T", "T", "i", "i", "i", "i", "i"), "T"),
    "tensor_conv_transpose1d":          (("T", "T", "T", "i", "i"), "T"),
    "tensor_conv_transpose2d":          (("T", "T", "T", "i", "i", "i", "i"), "T"),
    "tensor_avg_pool1d":                (("T", "i", "i"), "T"),
    "tensor_avg_pool2d":                (("T", "i", "i", "i", "i"), "T"),
    "tensor_max_pool1d":                (("T", "i", "i"), "T"),
    "tensor_max_pool2d":                (("T", "i", "i", "i", "i"), "T"),
    "tensor_max_pool2d_batched":        (("T", "i", "i", "i", "i"), "T"),

    # NTM
    "tensor_cosine_similarity":         (("T", "T", "i"), "T"),

    # Shape
    "tensor_reshape":                   (("T", "R", "i"), "T"),
    "tensor_reshape_2d":                (("T", "i", "i"), "T"),
    "tensor_reshape_3d":                (("T", "i", "i", "i"), "T"),
    "tensor_reshape_4d":                (("T", "i", "i", "i", "i"), "T"),
    "tensor_unsqueeze":                 (("T", "i"), "T"),
    "tensor_squeeze":                   (("T", "i"), "T"),
    "tensor_select":                    (("T", "i", "i"), "T"),
    "tensor_stack":                     (("R", "i", "i"), "T"),
    "tensor_cat":                       (("R", "i", "i"), "T"),
    "tensor_cat2":                      (("T", "T"), "T"),
    "tensor_narrow":                    (("T", "i", "i", "i"), "T"),
    "tensor_mm":                        (("T", "T"), "T"),
    "tensor_sdpa_2d":                   (("T", "T", "T", "i", "i", "i", "i"), "T"),
    "tensor_bmm":                       (("T", "T"), "T"),
    "tensor_bmm_3x3":                   (("T", "T"), "T"),
    "tensor_batch":                     (("R", "i"), "T"),
    "tensor_unbatch":                   (("T", "R"), "R"),
    "tensor_transpose_2d":              (("T",), "T"),
    "tensor_transpose_last2":           (("T",), "T"),
    "tensor_expand_mask":               (("T", "i"), "T"),
    "tensor_causal_mask":               (("i",), "T"),
    "tensor_tile_2d":                   (("T", "i", "i"), "T"),
    "tensor_masked_fill":               (("T", "T", "d"), "T"),
    "tensor_view_1d":                   (("T", "i"), "T"),
    "tensor_view_2d":                   (("T", "i", "i"), "T"),
    "tensor_item_1d":                   (("T", "i"), "d"),
    "tensor_item_2d":                   (("T", "i", "i"), "d"),

    # Autograd
    "tensor_backward":                  (("T",), "v"),
    "tensor_grad":                      (("T",), "T"),
    "tensor_zero_grad":                 (("T",), "v"),
    "tensor_requires_grad":             (("T",), "i"),
    "tensor_detach":                    (("T",), "T"),
    "tensor_with_grad":                 (("T",), "T"),
    "tensor_set_requires_grad":         (("T", "i"), "v"),
    "tensor_backward_return":           (("T",), "T"),
    "tensor_backward_conditional":      (("T",), "i"),
    "tensor_backward_return_loss":      (("T", "d"), "d"),

    # Device
    "tensor_to_device":                 (("T", "s"), "T"),
    "tensor_device":                    (("T",), "s"),

    # RNN
    "tensor_gru_cell":                  (("T", "T", "T", "i"), "T"),
    "tensor_lstm_cell":                 (("T", "T", "T", "T", "T", "T", "T", "R", "R"), "v"),
    "tensor_lstm_gates":                (("T", "T", "i", "R", "R"), "v"),
    "tensor_lstm_gates_pair":           (("T", "T", "i"), "R"),  # pair handle, not Tensor
    "tensor_pair_first":                (("R",), "T"),
    "tensor_pair_second":               (("R",), "T"),
    "tensor_pair_free":                 (("R",), "v"),

    # Param registry
    "param_register":                   (("s", "T"), "v"),
    "param_register_return":            (("s", "T"), "T"),  # returns same Tensor; re-wrap on return
    "param_tensor":                     (("i",), "T"),

    # Tensor constructors
    "tensor_one_hot":                   (("R", "i", "i", "i"), "T"),
    "tensor_create_1d":                 (("i", "R", "i"), "T"),
    "tensor_create_2d":                 (("i", "i", "R", "i"), "T"),
    "tensor_create_param_1d":           (("i", "R"), "T"),
    "tensor_create_param_2d":           (("i", "i", "R"), "T"),
    "tensor_create_param_3d":           (("i", "i", "i", "R"), "T"),
    "tensor_create_param_4d":           (("i", "i", "i", "i", "R"), "T"),
    "tensor_create_state_1d":           (("i", "R"), "T"),
    "tensor_create_state_2d":           (("i", "i", "R"), "T"),

    # Ptr array (raw array of Tensor handles)
    "tensor_ptr_array_set":             (("R", "i", "T"), "v"),
    "tensor_ptr_array_set_return":      (("R", "i", "T"), "R"),
    "tensor_stack_from_array":          (("R", "i", "i"), "T"),
    "tensor_cat_from_array":            (("R", "i", "i"), "T"),

    # Embedding / gather / scatter / sort / scan
    "tensor_embedding":                 (("T", "T", "i", "i"), "T"),
    "tensor_gather":                    (("T", "T", "i"), "T"),
    "tensor_scatter_add":               (("T", "T", "i"), "T"),
    "tensor_argsort":                   (("T", "i", "i"), "T"),
    "tensor_cumprod":                   (("T", "i"), "T"),

    # Cross-attention
    "tensor_cross_attention":           (("T", "T", "T", "T", "d"), "T"),

    # MNIST
    "mnist_get_image":                  (("R", "i", "i"), "T"),

    # Native train
    "native_train_step":                (("R", "i", "d", "T", "d"), "d"),
    "native_train_step_scaled":         (("R", "i", "d", "T", "d", "d"), "d"),

    # Print
    "tensor_print":                     (("T",), "v"),

    # idrisml_seq: void* → void*. Sequencing helper for ordering side-effecting
    # FFIs; both args are opaque, neither is a wrapped Tensor handle.
    "idrisml_seq":                      (("R", "R"), "R"),

    # Unified dtag create/cast (streamed). One symbol per shape; the trailing
    # `i` is the RuntimeDType tag (dtag), the one before it is the stream tag.
    # Supersede the per-dtype *_f32/_f64_streamed wrappers (which were never
    # in the manifest, hence lint-exempt). The Idris wrappers stay hand-written
    # (they thread dtag + lifecycle), but are now manifest-known so the
    # wrap-template lint validates their signature + wrap/retain invariants.
    "tensor_create_scalar_streamed":    (("d", "i", "i", "i"), "T"),
    "tensor_create_streamed":           (("R", "R", "i", "i", "i", "i"), "T"),
    "tensor_create_1d_streamed":        (("i", "R", "i", "i", "i"), "T"),
    "tensor_create_2d_streamed":        (("i", "i", "R", "i", "i", "i"), "T"),
    "tensor_create_param_1d_streamed":  (("i", "R", "i", "i"), "T"),
    "tensor_create_param_2d_streamed":  (("i", "i", "R", "i", "i"), "T"),
    "tensor_create_param_3d_streamed":  (("i", "i", "i", "R", "i", "i"), "T"),
    "tensor_create_param_4d_streamed":  (("i", "i", "i", "i", "R", "i", "i"), "T"),
    "tensor_create_state_1d_streamed":  (("i", "R", "i", "i"), "T"),
    "tensor_create_state_2d_streamed":  (("i", "i", "R", "i", "i"), "T"),
    "tensor_cast_dtype_streamed":       (("T", "i", "i"), "T"),

    # Fused param create + in-place init (added 2026-05-28, see commit
    # b38e71c). Allocates the param tensor in C and runs an in-place init
    # kernel (torch::nn::init::normal_ for normal, t.fill_ for const),
    # bypassing the per-element Idris-side sampler + per-element
    # prim__setDouble FFI that dominated HfLlama state construction.
    # Each takes (dims…, init-params, stream_tag, dtag).
    "tensor_create_param_1d_normal_streamed": (("i", "d", "d", "i", "i"), "T"),
    "tensor_create_param_2d_normal_streamed": (("i", "i", "d", "d", "i", "i"), "T"),
    "tensor_create_param_3d_normal_streamed": (("i", "i", "i", "d", "d", "i", "i"), "T"),
    "tensor_create_param_4d_normal_streamed": (("i", "i", "i", "i", "d", "d", "i", "i"), "T"),
    "tensor_create_param_1d_const_streamed":  (("i", "d", "i", "i"), "T"),
    "tensor_create_param_2d_const_streamed":  (("i", "i", "d", "i", "i"), "T"),
    "tensor_create_param_3d_const_streamed":  (("i", "i", "i", "d", "i", "i"), "T"),
    "tensor_create_param_4d_const_streamed":  (("i", "i", "i", "i", "d", "i", "i"), "T"),
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
    "mnist_get_image",
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
    "packages/idris-ml/src/Device.idr",
    "packages/idris-ml/src/Device/Mlx.idr",
    "packages/idris-ml/src/Device/Tape.idr",
    "packages/idris-ml/src/Device/Torch.idr",
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
