#!/usr/bin/env python3
"""One-shot bootstrap extractor for CC2 — scans the three Executor
instance files (`Executor/{Tape,Torch,Mlx}.idr`), classifies each
`primX = ...` body by shape (direct / streamed / bespoke), maps it back
to a manifest base-name, and emits a `{base_name: {slice, idris_method,
tape, torch, mlx}}` Python dict that seeds the dataclass migration.

Run once, eyeball the output against the existing instance files, then
use it to populate the `Entry` literals in `ffi_manifest.py`. Delete (or
move to `_obsolete/`) after the seeding commit lands.

The classifier recognises:

  direct      : `primX = prim__xBackend`  (point-free)
  streamed    : `primX a0 a1 ... = prim__xMlxStreamed a0 a1 ... (streamTag s)`
  bespoke-ffi : anything else that still references a `prim__` symbol
                (multi-line bodies, device migration via `prim__toDevice*`)
  constant    : `methodName = "literal"` or `= 0` — non-FFI per-backend constant
                (deviceName, deviceStreamTag, hardwareClass)

Constants are excluded from the manifest (they stay hand-written above
the generator's marker pair in each instance block).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEC_DIR = REPO_ROOT / "packages" / "idris-ml" / "src" / "Executor"

FILES = {
    "tape":  EXEC_DIR / "Tape.idr",
    "torch": EXEC_DIR / "Torch.idr",
    "mlx":   EXEC_DIR / "Mlx.idr",
}

# Match an instance block head:
#   `UserExecutorCore TapeExecutor where`
#   `{d : TorchHwDev} -> UserExecutorCore (TorchExecutor d) where`
#   `{s : MlxStream} -> UserExecutorCore (MlxExecutor s) where`
INSTANCE_RE = re.compile(
    r"^(?:\{[^}]+\}\s*->\s*)?"
    r"(UserExecutor[A-Z][A-Za-z]*)\s+"
    r"(?:\(?(?:TapeExecutor|TorchExecutor\s+d|MlxExecutor\s+s)\)?)"
    r"\s+where\s*$"
)

# Match a method body line at 2-space indent.
#   point-free direct:   `  primAdd       = prim__addTape`
#   streamed:            `  primAdd a b   = prim__addMlxStreamed a b (streamTag s)`
#   constant:            `  deviceName    = "tape"` / `  deviceStreamTag = 0`
#   bespoke (catch-all): anything else still inside the block
METHOD_LINE_RE = re.compile(r"^  ([a-z][A-Za-z0-9']*)(.*)$")

# RHS patterns
DIRECT_RHS_RE = {
    "tape":  re.compile(r"^\s*=\s*prim__([A-Za-z0-9]+)Tape\s*$"),
    "torch": re.compile(r"^\s*=\s*prim__([A-Za-z0-9]+)Torch\s*$"),
    "mlx":   re.compile(r"^\s*=\s*prim__([A-Za-z0-9]+)Mlx\s*$"),
}
STREAMED_RHS_RE = re.compile(
    r"^\s+[a-z0-9_ ]+\s*=\s*prim__([A-Za-z0-9]+)MlxStreamed\s+[a-z0-9_ ]+\(streamTag\s+s\)\s*$"
)
CONSTANT_RHS_RE = re.compile(r"^\s*=\s*(\".*\"|\d+|True|False)\s*$")


def camel_to_snake(name: str) -> str:
    """primCreateScalar -> create_scalar (strip prim, snake-case).

    Handles the codebase's naming conventions:
    - Insert _ between lowercase→uppercase transitions: `CreateScalar` → `create_scalar`.
    - Insert _ before `<N>d` dimension descriptors: `Concat2dAxis1` → `concat_2d_axis1`,
      `Param1dNormalStreamed` → `param_1d_normal_streamed`.
    - Insert _ before `<N>x<N>` patterns: `Bmm3x3` → `bmm_3x3`.
    Trailing single-digit suffixes (`TransposeLast2`, `Axis1`) keep digits attached.
    """
    if name.startswith("prim__"):
        name = name[len("prim__"):]
    elif name.startswith("prim"):
        name = name[len("prim"):]
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and out and out[-1] != "_" and not out[-1].isupper():
            out.append("_")
        out.append(ch.lower())
    s = "".join(out)
    # Insert _ before `<N>d` (dimension descriptor)
    s = re.sub(r"([a-z])(\d+d)(?=[_$]|[A-Za-z]|$)", r"\1_\2", s)
    # Insert _ before `<N>x<N>` (e.g. bmm3x3 → bmm_3x3)
    s = re.sub(r"([a-z])(\d+x\d+)", r"\1_\2", s)
    return s


def manifest_key_candidates(base_camel: str):
    """Yield candidate manifest keys for a camelCase base, in priority order.

    MANIFEST's naming convention is inconsistent: math ops follow PyTorch
    style (`conv1d`, `avg_pool1d`, no underscore before the dimension
    suffix), but library utilities use explicit underscores (`create_1d`,
    `view_1d`, `reshape_2d`). Try both variants for any `<N>d` suffix.
    """
    snake = camel_to_snake(base_camel)
    # Special-case prefixes that don't get `tensor_` prepended
    has_special = False
    for prefix in ("mnist_", "native_", "idrisml_", "param_", "optimizer_",
                   "backend_", "polyak_"):
        if snake.startswith(prefix):
            has_special = True
            break
    base = snake if has_special else f"tensor_{snake}"
    yield base
    # Try collapsing `_<N>d` → `<N>d` for math-op style
    collapsed = re.sub(r"_(\d+d)", r"\1", base)
    if collapsed != base:
        yield collapsed
    # Try the inverse: insert `_` before any digit cluster glued to a letter
    expanded = re.sub(r"([a-z])(\d+)", r"\1_\2", base)
    if expanded != base:
        yield expanded


def manifest_key(base_camel: str, existing) -> tuple[str, bool]:
    """Look up the canonical manifest key for a base, falling back to the
    first candidate if none exists. Returns (key, in_manifest)."""
    for cand in manifest_key_candidates(base_camel):
        if cand in existing:
            return cand, True
    # Default to the first candidate (the convention-driven snake form)
    return next(manifest_key_candidates(base_camel)), False


def extract_blocks(file_path: Path, backend: str):
    """Yield (slice_name, block_body_lines, start_line)."""
    text = file_path.read_text()
    lines = text.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = INSTANCE_RE.match(line)
        if not m:
            i += 1
            continue
        slice_name = m.group(1)
        start = i + 1
        # Capture body: contiguous run of 2-space-indented lines + blank lines,
        # ending at next instance head, non-indented line, or EOF.
        body_lines = []
        j = i + 1
        while j < n:
            ln = lines[j]
            if ln.strip() == "":
                body_lines.append(ln)
                j += 1
                continue
            if ln.startswith("  ") or ln.startswith("\t"):
                body_lines.append(ln)
                j += 1
                continue
            # Non-indented line ends the block
            break
        yield slice_name, body_lines, start
        i = j


# Match `prim__XYZBackend : T1 -> T2 -> ... -> Tn` at file-toplevel.
PRIM_DECL_RE = re.compile(
    r"^prim__([A-Za-z0-9]+?)(Tape|Torch|MlxStreamed|Mlx)\s*:\s*(.+)$"
)


def idris_type_to_class(t: str) -> str:
    """Map a single Idris type token to a manifest class.

    Returns 'T' (Tensor handle / AnyPtr), 'i', 'd', 's', 'v'.
    Recognises `PrimIO X` and unwraps it.
    """
    t = t.strip()
    if t.startswith("PrimIO "):
        t = t[len("PrimIO "):].strip()
    if t == "AnyPtr":
        return "T"
    if t == "Int":
        return "i"
    if t in ("Bits64",):  # treat 64-bit unsigned as Int classifier
        return "i"
    if t == "Double":
        return "d"
    if t == "String":
        return "s"
    if t in ("()", "Unit"):
        return "v"
    # Default: treat as raw pointer / opaque tensor handle
    return "T"


def parse_prim_signature(sig: str):
    """Parse 'T1 -> T2 -> ... -> Tn' into (args_classes, ret_class).

    Handles parens via flat-arity assumption (no curried higher-order
    types in our %foreign declarations)."""
    parts = [p.strip() for p in sig.split("->")]
    args = [idris_type_to_class(p) for p in parts[:-1]]
    ret = idris_type_to_class(parts[-1])
    return tuple(args), ret


def extract_prim_decls(file_path: Path):
    """Scan file for all `prim__XYZBackend : sig` declarations.

    For each, also capture the canonical C symbol from the preceding
    `%foreign "C:cname,libidrisml"` or `%foreign "scheme:..."` line —
    Scheme bodies carry the C symbol in their `foreign-procedure "..."`
    call, which is more reliable than snake-casing the Idris prim name
    (which sometimes adds `At` / `Tensor` prefixes that the C symbol
    doesn't have).

    Returns dict[base_camel] = (suffix, (args, ret), c_symbol).
    """
    out = {}
    lines = file_path.read_text().split("\n")
    # Walk paired (foreign_decl, prim_decl) lines
    for i, ln in enumerate(lines):
        m = PRIM_DECL_RE.match(ln)
        if not m:
            continue
        base, suffix, sig = m.group(1), m.group(2), m.group(3)
        try:
            args, ret = parse_prim_signature(sig)
        except Exception:
            continue
        # Look at the preceding line for a %foreign declaration
        c_symbol = None
        if i > 0:
            prev = lines[i - 1]
            # %foreign "C:cname,libidrisml"
            mc = re.match(r'^%foreign\s+"C:([a-zA-Z_0-9]+),libidrisml"', prev)
            if mc:
                c_symbol = mc.group(1)
            else:
                # %foreign "scheme:..." — look for first foreign-procedure
                ms = re.search(r'foreign-procedure\s+\\"([a-zA-Z_0-9]+)\\"', prev)
                if ms:
                    c_symbol = ms.group(1)
        out[base] = (suffix, args, ret, c_symbol)
    return out


def classify(body_lines: list[str], backend: str):
    """Return list of (method_name, kind, base_name_or_None, raw_line).

    kind is one of: 'direct', 'streamed', 'constant', 'bespoke'
    """
    out = []
    i = 0
    n = len(body_lines)
    while i < n:
        line = body_lines[i]
        m = METHOD_LINE_RE.match(line)
        if not m or not line.startswith("  ") or line.startswith("   "):
            # Not a method definition line (or it's a continuation of a
            # multi-line body — count it as part of the previous bespoke)
            i += 1
            continue
        name = m.group(1)
        rhs = m.group(2)
        # Try direct match for this backend
        d_re = DIRECT_RHS_RE[backend]
        md = d_re.match(rhs)
        if md:
            out.append((name, "direct", md.group(1), line))
            i += 1
            continue
        # Try streamed (mlx only)
        if backend == "mlx":
            ms = STREAMED_RHS_RE.match(rhs)
            if ms:
                out.append((name, "streamed", ms.group(1), line))
                i += 1
                continue
        # Constant?
        mc = CONSTANT_RHS_RE.match(rhs)
        if mc:
            out.append((name, "constant", None, line))
            i += 1
            continue
        # Anything else: bespoke. Try to infer the base name from any
        # `prim__XXXBackend` reference in the body (could be a multi-line
        # body — scan forward through continuation lines).
        body_text = rhs
        j = i + 1
        while j < n and body_lines[j].startswith("    "):
            body_text += " " + body_lines[j].strip()
            j += 1
        # Look for any prim__XXX(Tape|Torch|Mlx|MlxStreamed) inside body_text
        suffix_re = re.compile(r"prim__([A-Za-z0-9]+?)(Tape|Torch|MlxStreamed|Mlx)\b")
        matches = suffix_re.findall(body_text)
        # Prefer one whose suffix matches our backend; otherwise first
        chosen = None
        for base, suf in matches:
            if (backend == "tape" and suf == "Tape") or \
               (backend == "torch" and suf == "Torch") or \
               (backend == "mlx" and suf in ("Mlx", "MlxStreamed")):
                chosen = base
                break
        if chosen is None and matches:
            chosen = matches[0][0]
        out.append((name, "bespoke", chosen, line))
        i = j if j > i + 1 else i + 1
    return out


def main():
    # base_name -> {slice, idris_method, tape, torch, mlx, lines: {b: line}}
    seed = {}
    # Also: for each backend, methods seen that don't appear in MANIFEST.
    unmatched = defaultdict(list)
    # Constants per backend per slice (informational)
    constants = defaultdict(list)

    # Load existing MANIFEST keys for cross-check
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "lifecycle"))
    from ffi_manifest import MANIFEST as EXISTING

    # Per-backend: base_camel -> (suffix, args, ret)
    decls_by_backend = {b: extract_prim_decls(p) for b, p in FILES.items()}

    for backend, path in FILES.items():
        for slice_name, body_lines, start in extract_blocks(path, backend):
            for method, kind, base_camel, raw in classify(body_lines, backend):
                if kind == "constant":
                    constants[(backend, slice_name)].append((method, raw.strip()))
                    continue
                # FFI method (direct/streamed/bespoke). Map to manifest key.
                if base_camel is None:
                    unmatched[backend].append((slice_name, method, kind, raw.strip()))
                    continue
                # For bespoke methods, the manifest key MUST come from the
                # Idris method LHS (e.g. `primCreateFromHost`), NOT from the
                # first prim__X reference in the body (which may point to an
                # aliased FFI like `tensor_to_device`). This avoids merging
                # bespoke entries into unrelated MANIFEST keys.
                key_camel = method[len("prim"):] if method.startswith("prim") else method
                key_camel = key_camel[0].lower() + key_camel[1:]
                lookup_camel = key_camel if kind == "bespoke" else base_camel
                key, in_manifest = manifest_key(lookup_camel, EXISTING)
                # Look up inferred (args, ret) from the FFI decl for this
                # backend. For Mlx-streamed, the args carry an extra stream
                # tag at the end vs the manifest signature — strip it.
                inferred_args = None
                inferred_ret = None
                c_symbol_base = None
                d = decls_by_backend.get(backend, {})
                if base_camel in d:
                    suffix, args, ret, c_symbol = d[base_camel]
                    if suffix == "MlxStreamed":
                        if args and args[-1] == "i":
                            args = args[:-1]
                    inferred_args = args
                    inferred_ret = ret
                    if c_symbol:
                        for suf in ("_mlx_streamed", "_tape", "_torch", "_mlx"):
                            if c_symbol.endswith(suf):
                                c_symbol_base = c_symbol[: -len(suf)]
                                break
                        if c_symbol_base is None:
                            c_symbol_base = c_symbol
                # The KEY is always derived from the Idris method name (snake
                # of base_camel). The c_symbol is recorded as a PROPERTY so
                # aliased FFIs (e.g. tensor_create + tensor_create_from_host
                # both calling tensor_create) get distinct Entry rows.
                entry = seed.setdefault(key, {
                    "slice": slice_name,
                    "idris_method": method,
                    "tape": None,
                    "torch": None,
                    "mlx": None,
                    "args": None,
                    "ret": None,
                    "_in_manifest": in_manifest,
                    "_seen_at": {},
                    "_sig_seen": {},
                })
                if inferred_args is not None:
                    if entry["args"] is None:
                        entry["args"] = inferred_args
                        entry["ret"] = inferred_ret
                    else:
                        if entry["args"] != inferred_args or entry["ret"] != inferred_ret:
                            entry.setdefault("_sig_conflicts", []).append(
                                (backend, inferred_args, inferred_ret)
                            )
                    entry["_sig_seen"][backend] = (inferred_args, inferred_ret)
                if c_symbol_base:
                    if entry.get("c_symbol") and entry["c_symbol"] != c_symbol_base:
                        entry.setdefault("_csym_conflicts", []).append(
                            (backend, c_symbol_base)
                        )
                    else:
                        entry["c_symbol"] = c_symbol_base
                # Sanity: slice should agree across backends
                if entry["slice"] != slice_name:
                    entry.setdefault("_slice_conflicts", []).append(
                        (backend, slice_name)
                    )
                if entry["idris_method"] != method:
                    entry.setdefault("_method_conflicts", []).append(
                        (backend, method)
                    )
                entry[backend] = kind
                entry["_seen_at"][backend] = raw.strip()

    # Render the seed dict in a stable order: manifest-known first, then unknown
    print("# --- SEED EXTRACTOR OUTPUT (seed-instance-meta.py) ---")
    print(f"# Files scanned: {', '.join(str(p) for p in FILES.values())}")
    print(f"# Entries found: {len(seed)}")
    print()
    in_manifest_keys = sorted(k for k, v in seed.items() if v["_in_manifest"])
    not_in_manifest_keys = sorted(k for k, v in seed.items() if not v["_in_manifest"])
    print(f"# In MANIFEST already: {len(in_manifest_keys)}")
    print(f"# Found in instances but NOT in MANIFEST: {len(not_in_manifest_keys)}")
    if not_in_manifest_keys:
        print("# (these are likely typeclass methods whose body is hand-coded")
        print("#  but doesn't go through the wrap-handle %foreign pipeline;")
        print("#  inspect each before deciding whether to add to MANIFEST.)")
        for k in not_in_manifest_keys:
            v = seed[k]
            print(f"#   {k}  slice={v['slice']}  method={v['idris_method']}")
            for b in ("tape", "torch", "mlx"):
                if v.get(b):
                    print(f"#     {b}: {v[b]} -- {v['_seen_at'].get(b, '?')[:80]}")
    print()

    print("# Per-slice constant lines (stay outside generator markers,")
    print("# hand-written above the BEGIN marker per backend):")
    for (b, sl), lines in sorted(constants.items()):
        print(f"#   {b}/{sl}: {[m for m, _ in lines]}")
    print()

    print("# Unmatched bodies (couldn't infer base name — investigate manually):")
    for b, items in unmatched.items():
        for sl, method, kind, raw in items:
            print(f"#   {b}/{sl}: {method} ({kind}) -- {raw[:100]}")
    print()

    # Also emit MANIFEST entries that don't appear in any instance block
    # (internal helpers used from Tensor.idr / lifecycle / etc.). These
    # keep `slice=None` in the migrated MANIFEST.
    manifest_only = sorted(k for k in EXISTING if k not in seed)
    print(f"# MANIFEST-only (no instance method backing): {len(manifest_only)}")
    print()

    # Emit the entries themselves
    print("# >>> SEED ENTRIES >>>")
    print("# Order: by manifest key (alphabetical), instance-backed first,")
    print("# then MANIFEST-only.")
    print()

    for k in manifest_only:
        args, ret = EXISTING[k]
        if len(args) == 1:
            args_lit = f'("{args[0]}",)'
        else:
            args_lit = "(" + ", ".join(f'"{a}"' for a in args) + ")"
        print(f'    "{k}": Entry(args={args_lit}, ret="{ret}"),')
    print()
    print("# --- instance-backed entries below ---")
    for k in in_manifest_keys + not_in_manifest_keys:
        v = seed[k]
        tape = v.get("tape") or "MISSING"
        torch = v.get("torch") or "MISSING"
        mlx = v.get("mlx") or "MISSING"
        # Default suppression: drop a backend kwarg if it matches the dataclass default
        kwargs = []
        # args/ret: emit positionally for the dataclass Entry
        args_t = v.get("args")
        ret_c = v.get("ret")
        if args_t is not None:
            args_literal = "(" + ", ".join(f'"{a}"' for a in args_t)
            if len(args_t) == 1:
                args_literal += ","
            args_literal += ")"
            kwargs.append(f'args={args_literal}')
            kwargs.append(f'ret="{ret_c}"')
        else:
            kwargs.append("args=??")
            kwargs.append("ret=??")
        kwargs.append(f'slice="{v["slice"]}"')
        kwargs.append(f'idris_method="{v["idris_method"]}"')
        if v.get("c_symbol") and v["c_symbol"] != k:
            # Emit c_symbol override only when it differs from the key.
            kwargs.append(f'c_symbol="{v["c_symbol"]}"')
        if tape != "direct":
            kwargs.append(f'tape="{tape}"')
        if torch != "direct":
            kwargs.append(f'torch="{torch}"')
        if mlx != "streamed":
            kwargs.append(f'mlx="{mlx}"')
        if v.get("_slice_conflicts"):
            kwargs.append(f'# SLICE CONFLICT: {v["_slice_conflicts"]}')
        if v.get("_method_conflicts"):
            kwargs.append(f'# METHOD CONFLICT: {v["_method_conflicts"]}')
        if v.get("_sig_conflicts"):
            kwargs.append(f'# SIG CONFLICT: seen={v["_sig_seen"]}')
        if v.get("_csym_conflicts"):
            kwargs.append(f'# C_SYM CONFLICT: {v["_csym_conflicts"]}')
        warn = "" if v["_in_manifest"] else "  # NEW (not in MANIFEST today)"
        print(f'    "{k}": Entry({", ".join(kwargs)}),{warn}')


if __name__ == "__main__":
    main()
