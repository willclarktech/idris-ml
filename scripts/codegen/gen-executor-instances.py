#!/usr/bin/env python3
"""Generate the per-backend instance method body lines in
`Executor/{Tape,Torch,Mlx}.idr` from `ffi_manifest.py`.

For each `instance UserExecutorX where` block, locate the marker pair

    -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
    ...generated body lines...
    -- <<< END GENERATED <<<

and rewrite the content between the markers from the manifest's Entry
rows whose `slice` matches the block.

Per-backend body shape:
- `direct`   →  `<idris_method> = prim__<basecamel><Backend>`
- `streamed` (mlx only) →
                `<idris_method> a0 a1 … aN = prim__<basecamel>MlxStreamed a0 a1 … aN (streamTag s)`
- `bespoke`  → skipped; the body must live below `<<< END GENERATED <<<`.

Usage:
    python scripts/codegen/gen-executor-instances.py            # rewrite in place
    python scripts/codegen/gen-executor-instances.py --check    # exit 1 on diff
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "codegen"))
from ffi_manifest import MANIFEST, Entry  # noqa: E402

EXEC_DIR = REPO_ROOT / "packages" / "idris-ml" / "src" / "Executor"
FILES = {
    "tape": EXEC_DIR / "Tape.idr",
    "torch": EXEC_DIR / "Torch.idr",
    "mlx": EXEC_DIR / "Mlx.idr",
}

BACKEND_SUFFIX = {"tape": "Tape", "torch": "Torch", "mlx": "Mlx"}

BEGIN_MARKER = "  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>"
END_MARKER = "  -- <<< END GENERATED <<<"

# Match an instance block head, capturing the slice name.
#   `UserExecutorCore TapeExecutor where`
#   `{d : TorchHwDev} -> UserExecutorCore (TorchExecutor d) where`
#   `{s : MlxStream} -> UserExecutorCore (MlxExecutor s) where`
INSTANCE_RE = re.compile(
    r"^(?:\{[^}]+\}\s*->\s*)?"
    r"(UserExecutor[A-Z][A-Za-z]*)\s+"
    r"(?:\(?(?:TapeExecutor|TorchExecutor\s+d|MlxExecutor\s+s)\)?)"
    r"\s+where\s*$"
)


def camel_from_c(c_symbol: str) -> str:
    """tensor_add → tensorAdd; tensor_create_scalar → tensorCreateScalar."""
    parts = c_symbol.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def base_camel_for(entry: Entry, manifest_key: str) -> str:
    """The C-name-derived camelCase that backs `prim__<this>Backend`.

    The Idris `%foreign` decls in each backend file are named
    `prim__<base_camel><Backend>` where `<base_camel>` is the camelCase of
    the C function called (which is `entry.c_symbol` if set, otherwise
    the manifest key). For aliased entries (e.g. primCreateFromHost
    calling tensor_create), the prim__ name is camelCase of the manifest
    key (createFromHost), NOT of c_symbol (create) — because the Idris
    layer has its own named binding.
    """
    # The prim__ binding uses the manifest-key's camel, with `tensor_`
    # prefix stripped (so `tensor_add` → `add`, `tensor_create_scalar` → `createScalar`).
    base = manifest_key
    for prefix in ("tensor_",):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    parts = base.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def emit_method_line(entry: Entry, manifest_key: str, backend: str) -> str | None:
    """Return the generated body line for this (entry, backend), or None
    if the per-backend flavor is bespoke (caller hand-writes it).
    """
    flavor = getattr(entry, backend)
    if flavor == "bespoke":
        return None
    method = entry.idris_method
    base_camel = base_camel_for(entry, manifest_key)
    backend_suffix = BACKEND_SUFFIX[backend]
    if flavor == "direct":
        return f"  {method} = prim__{base_camel}{backend_suffix}"
    if flavor == "streamed":
        # Mlx streamed: bind args, pass through with (streamTag s).
        n_args = len(entry.args)
        arg_names = " ".join(f"a{i}" for i in range(n_args))
        arg_passthrough = " ".join(f"a{i}" for i in range(n_args))
        return (
            f"  {method} {arg_names} = "
            f"prim__{base_camel}MlxStreamed {arg_passthrough} (streamTag s)"
        )
    raise ValueError(f"Unknown flavor {flavor!r} for {manifest_key}/{backend}")


def entries_for_slice(slice_name: str) -> list[tuple[str, Entry]]:
    """Return all (key, Entry) pairs whose `slice == slice_name`,
    sorted by entry.idris_method for stable output."""
    out = [(k, e) for k, e in MANIFEST.items() if e.slice == slice_name]
    out.sort(key=lambda kv: kv[1].idris_method)
    return out


def generate_block(slice_name: str, backend: str) -> list[str]:
    """Return the list of generated body lines (with leading 2-space
    indent, no marker lines) for `slice_name` on `backend`."""
    lines = []
    for key, entry in entries_for_slice(slice_name):
        line = emit_method_line(entry, key, backend)
        if line is not None:
            lines.append(line)
    return lines


def rewrite_file(path: Path, backend: str, dry_run: bool = False) -> tuple[str, str]:
    """Rewrite all marker-bounded blocks in `path` from the manifest.

    Returns (original_text, new_text). When `dry_run`, the file is not
    written; the caller compares old vs new.
    """
    src = path.read_text()
    lines = src.split("\n")
    n = len(lines)
    out: list[str] = []
    i = 0
    current_slice: str | None = None
    while i < n:
        line = lines[i]
        m = INSTANCE_RE.match(line)
        if m:
            current_slice = m.group(1)
            out.append(line)
            i += 1
            continue
        # Detect BEGIN marker inside a slice block we know about.
        if current_slice and line.rstrip() == BEGIN_MARKER.rstrip():
            out.append(BEGIN_MARKER)
            # Skip everything up to the matching END marker.
            j = i + 1
            while j < n and lines[j].rstrip() != END_MARKER.rstrip():
                j += 1
            if j >= n:
                raise RuntimeError(f"{path.name}: BEGIN marker at line {i + 1} has no matching END")
            # Inject generated lines
            generated = generate_block(current_slice, backend)
            out.extend(generated)
            out.append(END_MARKER)
            i = j + 1
            continue
        # Detect end of instance block: a top-level line (no leading whitespace)
        # that follows blank/indented lines — resets current_slice.
        if line and not line.startswith(" ") and not line.startswith("\t"):
            current_slice = None
        out.append(line)
        i += 1
    new_src = "\n".join(out)
    if not dry_run and new_src != src:
        path.write_text(new_src)
    return src, new_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="Exit 1 if regeneration would change any file."
    )
    args = ap.parse_args()

    diffs = []
    for backend, path in FILES.items():
        src, new_src = rewrite_file(path, backend, dry_run=args.check)
        if src != new_src:
            diffs.append((backend, path))

    if args.check:
        if diffs:
            print(f"gen-executor-instances --check: {len(diffs)} file(s) would change:")
            for _backend, path in diffs:
                print(f"  - {path.relative_to(REPO_ROOT)}")
            print("Re-run scripts/codegen/gen-executor-instances.py to update.")
            sys.exit(1)
        print("gen-executor-instances --check: clean")
    else:
        if diffs:
            print(f"gen-executor-instances: regenerated {len(diffs)} file(s):")
            for _backend, path in diffs:
                print(f"  - {path.relative_to(REPO_ROOT)}")
        else:
            print("gen-executor-instances: no changes")


if __name__ == "__main__":
    main()
