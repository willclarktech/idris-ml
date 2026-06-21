#!/usr/bin/env python3
"""Generate packages/backends/rename_<backend>.h files.

For each built-in backend (tape, torch, mlx), this script writes a header
of `#define <sym> <sym>_<backend>` lines, one per exported C function
declared in `packages/backends/backend.h`. Injected into each backend's
compile command via `-include rename_<backend>.h` so the resulting `.o`
exports backend-suffixed symbols. Enables multi-link (all three
backends co-existing in one `libidrisml.(so|dylib)`) per the pluggable-
Executor refactor's Phase 1.

Idempotent — re-running with no changes produces no diff. CI gates on
that property so the headers stay in sync as we add ops.

Usage:
    python3 scripts/codegen/gen-rename-headers.py [--check]

`--check` exits non-zero if the headers on disk differ from what we
would generate. Used by CI.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_H = REPO_ROOT / "packages" / "backends" / "backend.h"
OUT_DIR = REPO_ROOT / "packages" / "backends"

BACKENDS = ("tape", "torch", "mlx")

# Additional exported symbols not declared as functions in backend.h.
# These are extern variables / non-function exports that still need
# per-backend renaming to avoid multi-link collisions. The function-only
# DECL_RE above doesn't pick them up, so they're listed here.
EXTRA_EXPORTS = (
    # Shared training port struct — each backend's training adapter
    # defines its own `g_active_port`; the rename header maps the
    # unsuffixed name in shared/training/*.c TUs to the per-backend
    # symbol so multi-link gets distinct instances.
    "g_active_port",
)

# Match C function declarations. Permissive: any sequence of
# `(const)? <identifier>` words optionally followed by `*`s, plus
# trailing whitespace, then the function name + `(`. Skips macros /
# struct decls by anchoring at line start and requiring `(` after the
# name.
DECL_RE = re.compile(
    r"""
    ^\s*                                # line start (+ optional indent)
    (?:const\s+)?                       # optional const
    [A-Za-z_]\w*                        # return type identifier
    \s*\**\s*                           # optional * (and whitespace around it)
    (?P<name>[A-Za-z_]\w*)              # function name
    \s*\(                               # opening paren of param list
    """,
    re.MULTILINE | re.VERBOSE,
)


def extract_symbols(header_text: str) -> list[str]:
    """Return a sorted, de-duplicated list of function names declared in
    `backend.h`. Skips entries inside `#if 0` / commented-out blocks by
    stripping `//` and `/* ... */` comments first."""
    # Strip block comments.
    text = re.sub(r"/\*.*?\*/", "", header_text, flags=re.DOTALL)
    # Strip line comments.
    text = re.sub(r"//[^\n]*", "", text)
    names = sorted({m.group("name") for m in DECL_RE.finditer(text)})
    # Drop typedefs that the regex incidentally caught (none should
    # match the function-call pattern, but guard anyway) plus the
    # intentionally-unified shared utilities — these live in
    # `shared_utils.c`, are compiled without a rename header, and
    # emerge under their unified names directly.
    exclude = {
        "TensorHandle",
        "TensorPair",
        # Shared utilities (see packages/backends/shared_utils.c) +
        # the IDX-format dataset surface (packages/backends/idx.c).
        # IDX is intentionally NOT declared in backend.h so the
        # generator won't see it anyway; the entries below cover the
        # shared_utils.c surface that DOES still pass through the
        # backend.h scan.
        "create_index_array",
        "shuffle_index_array",
        "index_array_get",
        "create_seeded_index_array",
        "seeded_index_array_shuffle",
        "seeded_index_array_get",
        "get_rss_mb",
        "get_current_rss_mb",
        # C buffer helpers — byte-identical malloc wrappers across all
        # three backends; consolidated into shared_utils.c.
        "tensor_alloc_doubles",
        "tensor_free_doubles",
        "tensor_read_double",
        "tensor_write_double_return",
        "tensor_alloc_ints",
        "tensor_free_ints",
        "tensor_write_int_return",
        "tensor_ptr_array_alloc",
        "tensor_ptr_array_set_return",
        # Dropout RNG (drives process-global rand()).
        "dropout_random_seed",
        # Backend-agnostic raw-bytes safetensors reader. Unlike the others
        # above it lives in safetensors.c (a renamed shared TU), not
        # shared_utils.c — but it's pure file I/O with no tensor handles or
        # device side effects (see backend.h), so excluding it here keeps the
        # `#define` out of every rename header and the function emerges under
        # its unified name. The Idris FFI (Tensor/Handle.idr) binds the
        # unified `safetensors_read_raw_bytes` directly (no per-backend
        # dispatch); without this exclusion only `_<backend>` existed and the
        # HF BitNet ternary-weight load aborted with "no entry for
        # safetensors_read_raw_bytes".
        "safetensors_read_raw_bytes",
    }
    return [n for n in names if n not in exclude]


def render_header(backend: str, symbols: list[str]) -> str:
    """Render the `rename_<backend>.h` content for `backend`."""
    guard = f"IDRISML_RENAME_{backend.upper()}_H"
    lines = [
        "/* AUTO-GENERATED by scripts/codegen/gen-rename-headers.py — DO NOT EDIT. */",
        "/* Run `make rename-headers` to regenerate after editing backend.h. */",
        "/* */",
        f"/* Injected via `-include` on backend_{backend}.{'c' if backend == 'tape' else 'cpp'} compile.",  # noqa: E501
        f" * Renames each exported C symbol to `<sym>_{backend}` so all three",
        " * backends can co-exist linked into one libidrisml.(so|dylib).",
        " * Every Idris %foreign reaches these via the suffixed name from a",
        " * per-instance UserExecutor method; the former unified-name link-time",
        " * alias machinery has been removed. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
    ]
    for sym in symbols:
        lines.append(f"#define {sym} {sym}_{backend}")
    lines += [
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated content differs from disk.",
    )
    args = parser.parse_args()

    header_text = BACKEND_H.read_text()
    symbols = extract_symbols(header_text)
    # Merge in the manual EXTRA_EXPORTS (variables / non-function exports
    # that the function-only regex doesn't catch). Sort the union so the
    # output stays stable across regenerations.
    symbols = sorted(set(symbols) | set(EXTRA_EXPORTS))

    print(f"Extracted {len(symbols)} exported symbols ({len(EXTRA_EXPORTS)} extra)")

    drift = False
    for backend in BACKENDS:
        out_path = OUT_DIR / f"rename_{backend}.h"
        rendered = render_header(backend, symbols)
        existing = out_path.read_text() if out_path.exists() else ""
        if rendered == existing:
            print(f"  {out_path.name}: unchanged")
            continue
        if args.check:
            print(f"  {out_path.name}: DRIFT (regen would change content)", file=sys.stderr)
            drift = True
            continue
        out_path.write_text(rendered)
        print(f"  {out_path.name}: wrote {len(symbols)} renames")

    if args.check and drift:
        print(
            "\nRename headers are out of date. Run `make rename-headers` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
