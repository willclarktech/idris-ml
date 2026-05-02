#!/usr/bin/env python3
"""One-shot migrator: rewrite every wrap-handle Scheme body from the v1
3-slot-vector layout `(vector 'tensor-handle raw_r)` to the v2 4-slot
layout `(vector 'tensor-handle-v2 "TAG" raw_r)`, and shift every
`(vector-ref a<i> 1)` to `(vector-ref a<i> 2)`. The backend tag is
derived from the foreign-procedure name inside the same wrap body via
`ffi_manifest.backend_tag_of` — wraps whose inner `foreign-procedure`
calls a `_tape` / `_torch` / `_mlx` symbol get tagged accordingly;
unsuffixed names get tagged "primary" (the link-time alias to the
build's primary backend).

Retain calls are also rewritten: `tensor_retain_handle` → suffixed
`tensor_retain_handle_<tag>` (or kept unified when the tag is
"primary", since the unified symbol already aliases to primary's).

The script edits in place. After this runs once across the file set,
the codebase is on the v2 layout and `ffi-convert-to-scheme.py`
(which emits v2 directly via the updated `gen_scheme_wrapper`) keeps
new wraps consistent. The CI gate
(`check-ffi-wrap-template.py`) enforces v2 going forward.

Usage:
    python3 scripts/lifecycle/migrate-wrap-v2.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import (
    WRAP_HANDLE_FILES,
    backend_tag_of,
    ANY_FFI_RE,
)


# Inside a scheme body, foreign-procedure names appear as \"name\".
FP_NAME_RE = re.compile(r'\(foreign-procedure\s+\\"([a-zA-Z_0-9]+)\\"')


def find_tag_in_body(body):
    """Find the FIRST non-retain/release foreign-procedure call in the
    body and use its name to derive the backend tag. Skips retain /
    release calls themselves (which are about to be rewritten).
    """
    for m in FP_NAME_RE.finditer(body):
        cname = m.group(1)
        if cname in ("tensor_retain_handle", "tensor_release_handle"):
            continue
        if cname.startswith("tensor_retain_handle_") or cname.startswith(
            "tensor_release_handle_"
        ):
            continue
        return backend_tag_of(cname)
    return None


def migrate_body(body):
    """Rewrite one `%foreign "scheme:..."` body string in place.

    The body is the inner content (no surrounding `scheme:` prefix or
    quotes). All Idris `\\"` escapes are preserved.
    """
    # Decide the tag from the body's payload foreign-procedure call.
    # Bodies with no manifest call (bespoke helpers like the drain
    # function) are left untouched — those carry no wrap layout.
    tag = find_tag_in_body(body)
    if tag is None:
        return body, False

    new_body = body
    changed = False

    # Shift (vector-ref a<i> 1) → (vector-ref a<i> 2). This must run
    # before the wrap rewrite, since a single body can carry both an
    # arg unwrap (slot 1 → 2) and a return wrap (now 4-slot).
    def shift_vector_ref(m):
        return f"(vector-ref {m.group(1)} 2)"

    new_body, n_args = re.subn(
        r"\(vector-ref\s+(a\d+)\s+1\)",
        shift_vector_ref,
        new_body,
    )
    if n_args:
        changed = True

    # Rewrite the wrap-on-return: replace `(vector 'tensor-handle raw_r)`
    # with `(vector 'tensor-handle-v2 \"TAG\" raw_r)`.
    wrap_re = re.compile(
        r"\(vector\s+'tensor-handle\s+([a-zA-Z_][a-zA-Z_0-9]*)\)"
    )
    def rewrite_wrap(m):
        var = m.group(1)
        return f'(vector \'tensor-handle-v2 \\"{tag}\\" {var})'

    new_body, n_wraps = wrap_re.subn(rewrite_wrap, new_body)
    if n_wraps:
        changed = True

    # Rewrite retain: `tensor_retain_handle` → `tensor_retain_handle_<tag>`
    # when tag != "primary". The "primary" wrappers keep the unified
    # retain symbol (its link-time alias already routes to primary).
    if tag != "primary":
        suffixed = f"tensor_retain_handle_{tag}"
        # Only rewrite the BARE unified name; leave already-suffixed
        # names alone. Use a word boundary so we don't touch
        # tensor_retain_handle_mlx etc.
        retain_re = re.compile(
            r'(\\"tensor_retain_handle)(\\"\s*\(void\*\)\s*void\))'
        )
        def rewrite_retain(m):
            return f'\\"{suffixed}{m.group(2)}'
        new_body, n_retain = retain_re.subn(rewrite_retain, new_body)
        if n_retain:
            changed = True

    return new_body, changed


def migrate_file(path):
    text = Path(path).read_text()
    n_changed = 0

    def replace(m):
        nonlocal n_changed
        full_match = m.group(0)
        kind = m.group(2)
        spec = m.group(3)
        if kind != "scheme":
            return full_match
        new_spec, changed = migrate_body(spec)
        if not changed:
            return full_match
        n_changed += 1
        # Rebuild the original group(1) (the %foreign + "scheme:...\n" line)
        # with the new body.
        new_foreign = f'%foreign "scheme:{new_spec}"\n'
        return new_foreign + m.group(4) + m.group(5)

    new_text, _n_match = ANY_FFI_RE.subn(replace, text)
    if n_changed:
        Path(path).write_text(new_text)
    return n_changed


def main():
    total = 0
    for f in WRAP_HANDLE_FILES:
        n = migrate_file(f)
        total += n
        print(f"{f}: {n} wraps migrated")
    print(f"\nTotal: {total} wraps migrated to v2.")


if __name__ == "__main__":
    main()
