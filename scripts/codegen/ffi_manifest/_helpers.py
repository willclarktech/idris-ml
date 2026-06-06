"""Stateless helpers — name munging, classifier mapping, Scheme wrapper
generation. Family-orthogonal: every consumer uses these regardless of
which typeclass family an FFI lives in.
"""

from ._skip import GUARDIAN_LAZY_INIT, INIT_FFI


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

    **FFI symbol caching:** each `foreign-procedure` is lazy-cached at
    first call via a per-FFI Chez top-level binding
    (`idris-ffi-<c-symbol-with-dashes>`). Without the cache every call
    re-evaluates `(foreign-procedure …)` → fresh dlsym → walks every
    loaded library's symbol table; on a Llama-3.2-1B forward pass that
    dominated 100% of CPU wall. The lazy-init
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
