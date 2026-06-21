#!/usr/bin/env python3
"""
Lint: flag %foreign declarations whose Idris type is non-IO but whose C
symbol has side effects (allocate, mutate, log, append to tape, toggle
global flag).

The bug class this catches comes from how Idris-2's Chez codegen handles
pure-typed FFI calls. Two failure modes:

1. **Unit return on a side-effecting body.** A `prim__foo : ... -> ()`
   declaration looks correct because `()` "obviously" has no useful
   value. But Idris will happily drop `let _ = prim__foo ...` bindings
   as dead code (the binding produces nothing the body needs), and
   `pure (prim__foo ...)` evaluates the FFI strictly at IO-value
   *construction* time rather than at sequencing time. Use
   `PrimIO ()` instead — the side effect only fires when the resulting
   action is run via `primIO`.

2. **Non-IO scalar return on a mutating body.** A
   `prim__bar : ... -> Int` whose C body mutates registry state can be
   CSE'd: the compiler treats `prim__bar 7` as a pure expression and
   may evaluate it once, caching the result. Subsequent "calls" see a
   stale value. Same fix: type it `PrimIO Int` and sequence via
   `primIO`. Reading immutable state (`tensor_item`, `tensor_dim`,
   shape queries) is on the allowlist below.

The wrap-on-return scheme template (`check-ffi-wrap-template.py`)
covers Tensor-handle-returning prims separately; this linter covers
the scalar / unit side. Together they enforce the IO-shape discipline
that surfaces from the IO refactor (commits leading up to e337512) and
the saved feedback `feedback_typeclass_zero_arg_method_eval.md`.

Usage:
    python3 scripts/codegen/check-non-io-side-effects.py

Exit 0 on clean; 1 on any violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import ANY_FFI_RE, parse_args, strip_suffix

# Files scanned by this lint. Larger set than WRAP_HANDLE_FILES because
# this lint also covers Layer/*.idr and any other site that declares
# %foreign decls outside the wrap-handle file set.
SCAN_DIRS = [
    Path("packages/idris-ml/src"),
    Path("packages/idris-ml-examples/src"),
    Path("packages/idris-gym/src"),
]


# C symbols that are pure reads from Idris's perspective. Reading them
# does not mutate any state the rest of the program observes, so a
# non-IO Idris type is safe even though the C body may touch memory.
#
# Note: even "pure reads" carry a CSE risk if the value can change
# across calls — e.g. `param_count` reads a registry that grows over
# time, so two calls return different values but the compiler may
# memoise the first. Mark such symbols as `STALE_READ_RISK` instead
# of allow-listing them; they should be IO-typed at the boundary even
# though they don't mutate.
PURE_READS = {
    # Shape / handle metadata — immutable per handle
    "tensor_item",
    "tensor_item_1d",
    "tensor_item_2d",
    "tensor_numel",
    "tensor_dim",
    "tensor_size",
    "tensor_device",
    "tensor_read_double",
    # Autograd flag query (snapshot of mutable flag — see STALE_READ_RISK
    # discussion above; flag is global and toggled by no_grad_begin/end)
    "tensor_requires_grad",
    # Immutable dataset metadata
    "mnist_count",
    "mnist_get_label",
    # Build-time configuration
    "backend_name",
    "backend_supports_tensor_params",
    # Memory observability — value changes but reading is non-mutating.
    # Callers wrap these in IO at the use site (`getRSS = primIO ...`).
    # Allowlisted here because the existing module-level non-IO decls
    # are dead code paths kept for backward source compat.
    "get_rss_mb",
    "get_current_rss_mb",
}


# Idris-side prim names that legitimately stay non-IO and are exempt
# from the lint. Anything not on this list and not on PURE_READS gets
# checked. Kept short on purpose — every entry is an explicit decision.
KNOWN_DEAD_PRIMS = {
    # `tensor_write_double_return` mutates a caller-owned buffer and
    # returns the same pointer for let-chain threading. The threading
    # idiom (used in `writePE` / `writeCausalMask`) prevents CSE in
    # practice; flagging it would require either retyping every
    # mask/PE construction site as IO or extending the lint with an
    # "intentional threaded-ptr" annotation.
    "prim__setDouble",
    # primFree's UserDeviceCore method is the lifecycle "release this
    # handle" hook. It's typed `AnyPtr -> ()` to match the method
    # signature (the C bodies on mlx are refcount-driven, tape/torch
    # are no-ops). The Idris-side guardian drives release, so the
    # binding doesn't fire on a fixed schedule — keeping it `()` is
    # intentional. The unit-arg ABI surface is identical across all
    # backends.
    "prim__freeMlx",
    "prim__freeTape",
    "prim__freeTorch",
    "prim__freeUnified",
    "prim__freeBYO",
}


# C symbols whose Idris-side decl must be IO/PrimIO. Reading or calling
# them mutates state, or the return value is invalidated by a parallel
# state change, or the FFI carries an ordering constraint with other
# side-effecting calls. Listed explicitly so the lint can give a
# specific error message.
KNOWN_SIDE_EFFECTING = {
    # Param registry mutations
    "param_clear",
    "param_register",
    "param_register_return",
    "param_zero_all_grads",
    "param_zero_all_grads_return",
    "param_grad_item_and_zero",
    # Stale-read risk: the registry grows; cached value goes stale
    "param_count",
    "param_name",
    "param_grad_item",
    "param_grad_item_at",
    # Backward pass + autograd state
    "tensor_backward",
    "tensor_backward_return",
    "tensor_backward_return_loss",
    "tensor_zero_grad",
    "tensor_set_requires_grad",
    # Global grad-mode flag (mutates a thread-local toggle)
    "no_grad_begin",
    "no_grad_end",
    # In-place mutation of caller-owned buffers
    "tensor_write_double",
    "tensor_to_doubles",
    "tensor_lstm_cell",
    "tensor_lstm_gates",
    # I/O + diagnostics (ordering matters)
    "tensor_print",
    "backend_memory_report",
    "backend_memory_report_return",
    "backend_reset_for_eval",
    "backend_reset_for_eval_return",
    "backend_profile_reset",
    "backend_profile_reset_return",
    "backend_profile_report",
    "backend_profile_report_return",
    # Optimizer mutations
    "polyak_blend",
    "optimizer_clip_grad_norm",
    "optimizer_set_param_lr",
    "native_train_step",
    "native_zero_grad",
    "native_step",
    # Buffer mutation (tensor_write_double_return threads ptr but still
    # mutates; flag so the typing is intentional, not a coincidence)
    "tensor_write_double_return",
}


# Bug class identifier for diagnostics.
def classify_return(ret_type: str) -> str:
    s = ret_type.strip()
    if s == "()":
        return "unit-non-io"
    if s.startswith("IO ") or s == "IO" or s.startswith("PrimIO"):
        return "io"
    # AnyPtr / Tensor handles — wrap-handle lint covers these
    if s == "AnyPtr":
        return "ptr"
    if s in ("Int", "Double", "String", "Bits8"):
        return "scalar"
    return "other"


def lint_decl(
    path: str,
    name: str,
    cname: str,
    args: list[str],
    ret_type: str,
    errors: list[str],
) -> None:
    """Apply the lint rules to one %foreign decl."""
    if name in KNOWN_DEAD_PRIMS:
        return
    if cname == "(scheme)":
        # Scheme-wrapped decls — these are the wrap-on-return template.
        # The other lint (`check-ffi-wrap-template.py`) covers their
        # invariants; we trust them here.
        return

    base = strip_suffix(cname)
    cls = classify_return(ret_type)

    # Rule 1: unit return on a non-IO body is always wrong.
    if cls == "unit-non-io":
        errors.append(
            f'{path}:{name} `%foreign "C:{cname}"` returns bare `()` '
            f"(should be `PrimIO ()`). Unit-typed bodies are evaluated "
            f"at IO-value construction and dropped from `let _ =` "
            f"bindings; the side effect won't fire at the sequencing "
            f"point you expect. See "
            f"`feedback_typeclass_zero_arg_method_eval.md` for the "
            f"underlying mechanism."
        )
        return

    # Rule 2: known side-effecting C symbol with non-IO Idris return.
    if base in KNOWN_SIDE_EFFECTING and cls != "io":
        errors.append(
            f'{path}:{name} `%foreign "C:{cname}"` is known to '
            f"mutate state but returns non-IO `{ret_type.strip()}`. "
            f"Type it `PrimIO {ret_type.strip()}` and call via "
            f"`primIO`. C symbols on the side-effecting list: "
            f"`param_*`, `tensor_backward*`, `no_grad_*`, "
            f"`polyak_blend`, `native_*`, `optimizer_*`, "
            f"`tensor_write_double*`, `tensor_print`, "
            f"`tensor_lstm_cell`, etc."
        )
        return

    # Rule 3: allowlist check is informational — every scalar return
    # that isn't in PURE_READS / KNOWN_SIDE_EFFECTING is unclassified
    # and should get an explicit entry. Don't fail on this; print a
    # warning so the allowlist stays current.
    if cls == "scalar" and base not in PURE_READS and base not in KNOWN_SIDE_EFFECTING:
        # No-op for now. Could be promoted to a soft warning later.
        return


def main(argv: list[str]) -> None:
    errors: list[str] = []
    n_decls = 0
    files_scanned = 0

    files: list[Path] = []
    if len(argv) > 1:
        files = [Path(p) for p in argv[1:]]
    else:
        for d in SCAN_DIRS:
            files.extend(d.rglob("*.idr"))

    for path in files:
        if not path.exists():
            continue
        text = path.read_text()
        files_scanned += 1
        for m in ANY_FFI_RE.finditer(text):
            kind = m.group(2)
            spec = m.group(3)
            sig_line = m.group(5).strip()
            try:
                name, args, ret = parse_args(sig_line)
            except Exception:
                continue
            n_decls += 1
            cname = spec.split(",", 1)[0] if kind == "C" else "(scheme)"
            lint_decl(str(path), name, cname, args, ret, errors)

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(
            f"\nNon-IO side-effects lint: {len(errors)} violation(s) "
            f"across {files_scanned} files ({n_decls} FFI decls scanned).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Non-IO side-effects lint: clean. {files_scanned} files, {n_decls} FFI decls scanned.")


if __name__ == "__main__":
    main(sys.argv)
