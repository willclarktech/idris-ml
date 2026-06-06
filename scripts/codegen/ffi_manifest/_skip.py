"""SKIP set + guardian/drain initialization tuples + INIT_FFI set.

These are family-orthogonal: SKIP names C functions excluded from
the wrap-template conversion entirely; the GUARDIAN/DRAIN constants
are inline Scheme literals shared by every wrapper; INIT_FFI is the
curated set of Tensor-creating FFIs whose wrapper carries the
guardian-lazy-init prelude.
"""

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
    " (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache)))"  # noqa: E501
    " (let ((rel (or (hashtable-ref cache tag #f)"
    ' (let ((sym (if (string=? tag \\"primary\\") \\"tensor_release_handle\\"'
    ' (string-append \\"tensor_release_handle_\\" tag))))'
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
    # Fused-init creators — same rule: each can be the first Tensor-
    # creating call in a program (HfBert's makeBertLinear is the first
    # FFI on the BERT path, etc.).
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
