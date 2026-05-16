#!/usr/bin/env python3
"""
Convert %foreign "C:..." declarations to %foreign "scheme:..." wrap-on-return
templates in Tensor.idr + Device.idr + Device/{Mlx,Tape,Torch}.idr.

For each FFI:
- Parse the immediately-following Idris signature.
- Look up the C base name (suffix-stripped) in MANIFEST to learn which
  AnyPtr args are wrapped Tensor handles ('T') vs raw pointers ('R').
- If no Tensor args nor Tensor return → leave as %foreign "C:..." unchanged.
- Otherwise generate the canonical Scheme wrapper via
  `gen_scheme_wrapper` from `ffi_manifest`.

The manifest, helpers, and canonical wrapper generator all live in
`ffi_manifest.py` — that module is the single source of truth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import (
    MANIFEST, SKIP,
    C_FFI_RE,
    gen_scheme_wrapper, parse_args, idris_type_to_class, strip_suffix,
)


def convert_file(path):
    text = Path(path).read_text()

    def replace(m):
        foreign_line, cname, export_line, sig_line = m.group(1), m.group(2), m.group(3), m.group(4)
        base = strip_suffix(cname)
        if base in SKIP:
            return m.group(0)
        if base not in MANIFEST:
            # Not in manifest — leave as-is.
            return m.group(0)

        manifest_args, manifest_ret = MANIFEST[base]
        name, idris_args, idris_ret = parse_args(sig_line.strip())

        # Cross-check arg count
        if len(idris_args) != len(manifest_args):
            print(
                f"  WARN {cname}: arg count mismatch — Idris has {len(idris_args)}, "
                f"manifest has {len(manifest_args)}. Skipping.",
                file=sys.stderr,
            )
            return m.group(0)

        # Each Idris arg's class is taken from the manifest (which knows
        # whether AnyPtr means T or R).
        arg_classes = list(manifest_args)
        ret_class = manifest_ret

        # Validate consistency with Idris signature
        for i, (idris_t, cls) in enumerate(zip(idris_args, arg_classes)):
            expected = idris_type_to_class(idris_t, cls)
            if expected != cls:
                print(
                    f"  WARN {cname} arg {i}: Idris type {idris_t!r} → {expected}, "
                    f"manifest says {cls}. Trusting manifest.",
                    file=sys.stderr,
                )

        lam = gen_scheme_wrapper(cname, arg_classes, ret_class)
        new_foreign_line = f'%foreign "scheme:{lam}"\n'
        return new_foreign_line + export_line + sig_line

    new_text, n = C_FFI_RE.subn(replace, text)
    Path(path).write_text(new_text)
    return n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        n = convert_file(f)
        print(f"{f}: processed (matches found in file: see warnings)")
