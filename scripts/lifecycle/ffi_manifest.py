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
    "mnist_get_image":                  (("R", "i"), "T"),

    # Native train
    "native_train_step":                (("R", "i", "d", "T", "d"), "d"),

    # Print
    "tensor_print":                     (("T",), "v"),

    # idrisml_seq: void* → void*. Sequencing helper for ordering side-effecting
    # FFIs; both args are opaque, neither is a wrapped Tensor handle.
    "idrisml_seq":                      (("R", "R"), "R"),
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
GUARDIAN_LAZY_INIT = (
    "(when (not (top-level-bound? 'idris-tensor-guardian))"
    " (set-top-level-value! 'idris-tensor-guardian (make-guardian)))"
)

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
    "tensor_causal_mask",
    "mnist_get_image",
}


def strip_suffix(cname):
    """Strip _mlx / _tape / _torch suffix from a C name to get the base."""
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
    fp = f'(foreign-procedure \\"{cname}\\" ({fp_arg_types}) {fp_ret_type})'

    if ret_class == "T":
        tag = backend_tag_of(cname)
        retain_sym = (
            "tensor_retain_handle"
            if tag == "primary"
            else f"tensor_retain_handle_{tag}"
        )
        body = (
            f" (let ((raw_r ({fp} {call_args_str})))"
            f" (let ((wr (vector 'tensor-handle-v2 \\\"{tag}\\\" raw_r)))"
            f" ((top-level-value 'idris-tensor-guardian) wr)"
            f" ((foreign-procedure \\\"{retain_sym}\\\" (void*) void) raw_r)"
            f" wr))"
        )
    else:
        body = f" ({fp} {call_args_str})"

    init = GUARDIAN_LAZY_INIT if strip_suffix(cname) in INIT_FFI else ""
    return f"(lambda ({' '.join(arg_names)}) {init}{body})"


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
