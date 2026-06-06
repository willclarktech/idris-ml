#!/usr/bin/env python3
"""
Convert/regenerate %foreign declarations in Tensor.idr + Device.idr +
Device/{Mlx,Tape,Torch}.idr to the canonical wrap-on-return Scheme template
emitted by `ffi_manifest.gen_scheme_wrapper`.

Handles two cases:

1. **First-time conversion** — `%foreign "C:cname,libidrisml"` → wrap-template
   `%foreign "scheme:..."`. Lookup C base name (suffix-stripped) in MANIFEST
   to learn which AnyPtr args are wrapped Tensor handles ('T') vs raw
   pointers ('R'). FFIs whose base is not in MANIFEST are left untouched.

2. **Regeneration** — `%foreign "scheme:..."` whose body already references a
   MANIFEST C symbol gets re-emitted from the current `gen_scheme_wrapper`
   template. Used when the template itself changes (e.g. adding the FFI
   symbol cache in 2026-05-27); preserves the lifecycle invariants the
   linter checks (wrap layout v2, guardian register, retain dispatch)
   while picking up any template improvements.

The manifest, helpers, and canonical wrapper generator all live in
`ffi_manifest.py` — that module is the single source of truth.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import (
    ANY_FFI_RE,
    MANIFEST,
    SKIP,
    gen_scheme_wrapper,
    idris_type_to_class,
    parse_args,
    strip_suffix,
)

# Locate the first foreign-procedure call inside a scheme: body. Inside the
# body the surrounding `%foreign "scheme:..."` literal escapes `"` as `\"`,
# so foreign-procedure names appear as `\"name\"`.
_FP_RE = re.compile(r'\(foreign-procedure\s+\\"([a-zA-Z_0-9]+)\\"\s*\([^)]*\)\s+[a-zA-Z*_]+\s*\)')


def _first_manifest_call(body):
    """Return the first foreign-procedure name in `body` whose base is in
    MANIFEST. Returns None if no such call exists (the scheme: body is a
    bespoke helper or only references SKIP-listed names).
    """
    for m in _FP_RE.finditer(body):
        cname = m.group(1)
        if cname in SKIP:
            continue
        base = strip_suffix(cname)
        if base in MANIFEST:
            return cname
    return None


def convert_file(path):
    text = Path(path).read_text()
    stats = {"converted": 0, "regenerated": 0, "skipped": 0}

    def replace(m):
        kind = m.group(2)  # "C" or "scheme"
        spec = m.group(3)  # body without "C:" / "scheme:" prefix
        export_line = m.group(4)
        sig_line = m.group(5)

        if kind == "C":
            # `spec` is e.g. `tensor_add_torch,libidrisml`.
            cname = spec.split(",", 1)[0]
        else:
            # `spec` is the full scheme body. Find the first manifest call
            # to determine the cname; if none, leave alone (bespoke helper).
            cname = _first_manifest_call(spec)
            if cname is None:
                stats["skipped"] += 1
                return m.group(0)

        base = strip_suffix(cname)
        if base in SKIP:
            stats["skipped"] += 1
            return m.group(0)
        if base not in MANIFEST:
            stats["skipped"] += 1
            return m.group(0)

        manifest_entry = MANIFEST[base]
        manifest_args, manifest_ret = manifest_entry.args, manifest_entry.ret
        name, idris_args, idris_ret = parse_args(sig_line.strip())

        # Cross-check arg count.
        if len(idris_args) != len(manifest_args):
            print(
                f"  WARN {cname}: arg count mismatch — Idris has {len(idris_args)}, "
                f"manifest has {len(manifest_args)}. Skipping.",
                file=sys.stderr,
            )
            stats["skipped"] += 1
            return m.group(0)

        # Each Idris arg's class is taken from the manifest (which knows
        # whether AnyPtr means T or R).
        arg_classes = list(manifest_args)
        ret_class = manifest_ret

        # Validate consistency with Idris signature.
        for i, (idris_t, cls) in enumerate(zip(idris_args, arg_classes, strict=True)):
            expected = idris_type_to_class(idris_t, cls)
            if expected != cls:
                print(
                    f"  WARN {cname} arg {i}: Idris type {idris_t!r} → {expected}, "
                    f"manifest says {cls}. Trusting manifest.",
                    file=sys.stderr,
                )

        lam = gen_scheme_wrapper(cname, arg_classes, ret_class)
        new_foreign_line = f'%foreign "scheme:{lam}"\n'
        if kind == "C":
            stats["converted"] += 1
        else:
            stats["regenerated"] += 1
        return new_foreign_line + export_line + sig_line

    new_text = ANY_FFI_RE.sub(replace, text)
    Path(path).write_text(new_text)
    return stats


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Default to the wrap-handle file set when no args given.
        from ffi_manifest import WRAP_HANDLE_FILES

        args = WRAP_HANDLE_FILES
    for f in args:
        stats = convert_file(f)
        print(
            f"{f}: converted={stats['converted']} "
            f"regenerated={stats['regenerated']} "
            f"skipped={stats['skipped']}"
        )
