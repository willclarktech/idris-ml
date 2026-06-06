#!/usr/bin/env python3
"""
Lint check: every Tensor-touching FFI in the wrap-handle file set must
conform to the wrap-on-return Scheme template (see
docs/develop/tensor-lifecycle.md "FFI conventions").

For each `%foreign` declaration in the file set:

- `%foreign "C:cname,libidrisml"`: cname's base name must NOT be in
  MANIFEST. If it is, the decl is a missing conversion — wrap it via
  scripts/codegen/ffi-convert-to-scheme.py.

- `%foreign "scheme:..."`: locate the first foreign-procedure call whose
  name is in MANIFEST (a bespoke helper that only references things
  outside MANIFEST is allowed). Verify the body's structural invariants
  for the manifest's classification:
    * foreign-procedure arg/return type spec matches the classifier
      (T/R → void*, i → int, d → double, s → string, v → void).
    * Each T arg at position i appears as `(vector-ref a<i> 2)` in the
      body — wrap layout v2 keeps the raw pointer at slot 2 (slot 0
      is the sentinel symbol, slot 1 is the backend-tag string).
    * For T return: the body contains
        `(vector 'tensor-handle-v2 "TAG" …)`
      where TAG is the expected backend tag (derived from the C name's
      suffix: `_tape` → "tape", `_torch` → "torch", `_mlx` → "mlx",
      unsuffixed → "primary"). It also has a retain call to
      `tensor_retain_handle_<tag>` (or unified `tensor_retain_handle`
      when tag is "primary") and registers with idris-tensor-guardian.
    * No body may contain the legacy v1 sentinel `'tensor-handle ` —
      the migration to v2 is committed and stale consumers reading
      slot 1 for a raw pointer would see a tag string instead.
    * For non-T return: the body does NOT contain `(vector 'tensor-handle-v2 …)`
      (no spurious wrapping).

Cosmetic variations (extra init blocks, whitespace) are tolerated.
The linter enforces the lifecycle invariants, not the exact template.

Usage:
    python3 scripts/codegen/check-ffi-wrap-template.py

Exit code 0 on clean; 1 on any violation.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import (
    ANY_FFI_RE,
    MANIFEST,
    SKIP,
    WRAP_HANDLE_FILES,
    backend_tag_of,
    parse_args,
    scheme_type,
    strip_suffix,
)

# Inside a scheme body, foreign-procedure names appear as \"name\" — the
# enclosing %foreign "scheme:..." literal escapes the quotes.
FP_RE = re.compile(
    r'\(foreign-procedure\s+\\"([a-zA-Z_0-9]+)\\"\s*\(([^)]*)\)\s+([a-zA-Z*_]+)\s*\)'
)


def lint_scheme_body(body, cname, manifest_args, manifest_ret):
    """Verify body matches the wrap-on-return template for cname.

    Returns a list of issue strings (empty = clean).
    """
    issues = []

    # 1. Locate the foreign-procedure call for cname.
    fp_re = re.compile(
        r'\(foreign-procedure\s+\\"' + re.escape(cname) + r'\\"\s*\(([^)]*)\)\s+([a-zA-Z*_]+)\s*\)'
    )
    m = fp_re.search(body)
    if not m:
        return [f'missing (foreign-procedure "{cname}" ...) call in body']

    typespec = m.group(1).split()
    fp_ret = m.group(2)

    # 2. Arg type spec must match manifest.
    expected_arg_types = [scheme_type(c) for c in manifest_args]
    if typespec != expected_arg_types:
        issues.append(
            f"foreign-procedure arg types {typespec} != expected "
            f"{expected_arg_types} (from manifest)"
        )

    # 3. Return type must match manifest.
    expected_ret = scheme_type(manifest_ret)
    if fp_ret != expected_ret:
        issues.append(
            f"foreign-procedure return type {fp_ret!r} != expected {expected_ret!r} (from manifest)"
        )

    # 4. Every T arg at position i must be unwrapped via (vector-ref a<i> 2)
    # (v2 layout: slot 2 is the raw pointer; slots 0/1 are the sentinel
    # symbol and backend-tag string).
    for i, cls in enumerate(manifest_args):
        if cls == "T":
            ref_re = re.compile(rf"\(vector-ref\s+a{i}\s+2\)")
            if not ref_re.search(body):
                issues.append(
                    f"T arg at position {i}: missing (vector-ref a{i} 2) — "
                    f"wrapped Tensor handle must be unwrapped before "
                    f"foreign-procedure call (wrap-v2 stores raw at slot 2)"
                )

    # No body may carry the legacy v1 sentinel — any stale wrap would
    # silently corrupt a future consumer that reads slot 1 expecting a
    # raw pointer.
    if re.search(r"vector\s+'tensor-handle(?!-v2)\b", body):
        issues.append(
            "legacy v1 wrap sentinel `'tensor-handle` detected — "
            "every wrap must use the v2 layout "
            '(vector \'tensor-handle-v2 "TAG" raw)'
        )

    # 5. T return → wrap + retain + register; non-T return → no wrap.
    has_wrap = "vector 'tensor-handle-v2" in body
    expected_tag = backend_tag_of(cname)
    expected_retain = (
        "tensor_retain_handle"
        if expected_tag == "primary"
        else f"tensor_retain_handle_{expected_tag}"
    )
    has_retain = expected_retain in body
    has_guardian = "idris-tensor-guardian" in body
    if manifest_ret == "T":
        if not has_wrap:
            issues.append(
                "T return: missing (vector 'tensor-handle-v2 …) — the C "
                "result must be wrapped before returning to Idris"
            )
        # Check the wrap carries the right backend tag.
        tag_re = re.compile(r"vector\s+'tensor-handle-v2\s+\\\"([a-zA-Z_/]+)\\\"")
        m_tag = tag_re.search(body)
        if m_tag and m_tag.group(1) != expected_tag:
            issues.append(
                f"T return: wrap tag {m_tag.group(1)!r} != expected "
                f"{expected_tag!r} (derived from C name {cname!r})"
            )
        if not has_retain:
            issues.append(
                f"T return: missing {expected_retain} call — the wrap "
                f"must take the first refcount bump on the right backend"
            )
        if not has_guardian:
            issues.append(
                "T return: missing idris-tensor-guardian registration — "
                "the wrap must be tracked for GC-driven release"
            )
    else:
        if has_wrap:
            issues.append(
                f"non-T return ({manifest_ret!r}): unexpected (vector "
                f"'tensor-handle-v2 …) in body — only T returns are wrapped"
            )

    return issues


def first_manifest_call(body):
    """Return the first foreign-procedure name in body whose base is in
    MANIFEST. Returns None if no such call exists (bespoke helper).
    """
    for m in FP_RE.finditer(body):
        cname = m.group(1)
        if cname in SKIP:
            continue
        base = strip_suffix(cname)
        if base in MANIFEST:
            return cname
    return None


def check_file(path, errors):
    text = Path(path).read_text()
    for m in ANY_FFI_RE.finditer(text):
        kind = m.group(2)  # "C" or "scheme"
        spec = m.group(3)  # body without surrounding "C:" / "scheme:"
        sig_line = m.group(5).strip()
        try:
            name, idris_args, idris_ret = parse_args(sig_line)
        except Exception:
            continue  # bail; unrelated decl

        if kind == "C":
            # spec is e.g. `tensor_add,libidrisml` — pull the cname.
            cname = spec.split(",", 1)[0]
            base = strip_suffix(cname)
            if base in MANIFEST:
                # Only Tensor-touching FFIs (T arg or T return) need the
                # wrap-on-return scheme template. Pure primitive FFIs
                # (Int/Double/void in and out) can stay as raw %foreign "C:..."
                # — adding the scheme wrapper buys nothing and the manifest
                # carries them only for instance-method generation metadata.
                entry = MANIFEST[base]
                touches_tensor = entry.ret == "T" or "T" in entry.args
                if touches_tensor:
                    errors.append(
                        f'{path}: {name} uses %foreign "C:{cname}" but '
                        f"base {base!r} is in MANIFEST — should have been "
                        f"converted to wrap-on-return scheme template. "
                        f"Run scripts/codegen/ffi-convert-to-scheme.py."
                    )
            continue

        # scheme: body
        body = spec
        cname = first_manifest_call(body)
        if cname is None:
            # Bespoke scheme helper (e.g. drainManagedHandles, forceMajorGc,
            # initManagedHandles) — exempt from the wrap-on-return template.
            continue

        base = strip_suffix(cname)
        manifest_entry = MANIFEST[base]
        manifest_args, manifest_ret = manifest_entry.args, manifest_entry.ret

        # Cross-check Idris signature arg count.
        if len(idris_args) != len(manifest_args):
            errors.append(
                f"{path}: {name} (C:{cname}) Idris signature has "
                f"{len(idris_args)} args but manifest expects "
                f"{len(manifest_args)}"
            )
            continue

        issues = lint_scheme_body(body, cname, list(manifest_args), manifest_ret)
        for issue in issues:
            errors.append(f"{path}: {name} (C:{cname}): {issue}")


def main(argv):
    files = argv[1:] if len(argv) > 1 else WRAP_HANDLE_FILES
    errors = []
    n_decls = 0
    for f in files:
        text = Path(f).read_text()
        n_decls += len(list(ANY_FFI_RE.finditer(text)))
        check_file(f, errors)

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(
            f"\nFFI wrap-template lint: {len(errors)} violation(s) "
            f"across {len(files)} files "
            f"({n_decls} FFI decls scanned).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"FFI wrap-template lint: clean. {len(files)} files, {n_decls} FFI decls scanned.")


if __name__ == "__main__":
    main(sys.argv)
