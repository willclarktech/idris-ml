"""Parsers for the backend C/C++ headers and source files.

The coverage-gap-probe + (eventually) the gen-rename-headers and
manifest-checking tools share the same parsing primitives:

  - `extract_ops(header_text)`
        Pull every `OP_<NAME>` token out of an `enum { ... }` block,
        excluding the OP_COUNT sentinel and the OP_CONST leaf marker
        (per the coverage-policy.md exclusion rules).

  - `extract_ffi_symbols_from_source(source_text)`
        Identify the non-`static` FFI entry points defined in a
        backend source file. These are the symbols the tests need
        to call into to exercise the backward path of an OP_*.

  - `extract_backend_h_symbols(header_text)`
        All declared FFI symbols in `packages/backends/backend.h`
        (the top-level public surface). Used by the coverage probe
        to enumerate everything we ship.

  - `find_op_source(op, backend_dir, anchor)`
        Locate the C/C++ source file holding the registration
        anchor for `op` (e.g. `TAPE_REGISTER_OP(OP_FOO, ...)`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

# Sentinels stripped from the OP_* enum scan.
OP_ENUM_EXCLUDE = frozenset({"OP_COUNT", "OP_CONST"})

# Allowed leading return types for an FFI entry point. Same set the bash
# version filtered on; conservative enough to skip helpers / static decls.
_FFI_RETURN_TYPES = "|".join(
    [
        r"TensorHandle\*?",
        r"void\*?",
        r"int",
        r"double",
        r"float",
        r"char\*?",
    ]
)

_FFI_SOURCE_LINE_RE = re.compile(
    rf'^(?:extern\s+"C"\s+)?(?:{_FFI_RETURN_TYPES})\s+([a-z_][a-z0-9_]*)\s*\(',
    re.MULTILINE,
)

# More permissive variant for backend.h decls: accept any leading return
# type built from identifiers / `*` / whitespace. Captures the function
# name as group 1.
_BACKEND_H_DECL_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_ *]*[ *]([a-z_][a-z0-9_]+)\s*\(",
    re.MULTILINE,
)

_ENUM_BLOCK_RE = re.compile(r"^enum\s*\{(.*?)^\};", re.MULTILINE | re.DOTALL)
_OP_TOKEN_RE = re.compile(r"\bOP_[A-Z][A-Z0-9_]*\b")


def extract_ops(header_text: str) -> set[str]:
    """Set of `OP_<NAME>` tokens declared in any `enum { ... }` block,
    minus `OP_COUNT` / `OP_CONST` sentinels."""
    ops: set[str] = set()
    for block in _ENUM_BLOCK_RE.findall(header_text):
        for token in _OP_TOKEN_RE.findall(block):
            if token in OP_ENUM_EXCLUDE:
                continue
            ops.add(token)
    return ops


def extract_ffi_symbols_from_source(source_text: str) -> set[str]:
    """Non-static FFI entry-point names defined in a backend C/C++ file.

    The bash version greps for `^(extern "C")?<rettype> name(`, then
    drops lines beginning with `static`. Mirror that behaviour here:
    we filter to lines that don't start with `static\b` and match the
    return-type allowlist.
    """
    out: set[str] = set()
    for line in source_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("static"):
            continue
        m = _FFI_SOURCE_LINE_RE.match(line)
        if m:
            out.add(m.group(1))
    return out


def extract_backend_h_symbols(header_text: str) -> set[str]:
    """All declared FFI symbols in backend.h's top-level decls."""
    return set(_BACKEND_H_DECL_RE.findall(header_text))


def find_op_source(
    op: str, backend_dir: Path, anchor: str
) -> Optional[Path]:
    """Return the first C/C++ source under `backend_dir` containing the
    registration anchor `<anchor>(<op>,`. Returns None on miss.
    """
    needle = f"{anchor}({op},"
    for ext in ("*.c", "*.cpp"):
        for path in backend_dir.rglob(ext):
            try:
                if needle in path.read_text():
                    return path
            except (OSError, UnicodeDecodeError):
                continue
    return None


def grep_word_in_dir(symbol: str, root: Path, suffixes: Iterable[str] = ("*.c",)) -> list[Path]:
    """All files under `root` (matching `suffixes`) that contain `symbol`
    as a whole word. Word-boundary semantics match `grep -w`.
    """
    rx = re.compile(rf"\b{re.escape(symbol)}\b")
    hits: list[Path] = []
    for ext in suffixes:
        for path in root.rglob(ext):
            try:
                if rx.search(path.read_text()):
                    hits.append(path)
            except (OSError, UnicodeDecodeError):
                continue
    return hits


def grep_word_in_dirs(
    symbol: str, roots: Iterable[Path], suffixes: Iterable[str] = ("*.c",)
) -> list[Path]:
    """Like `grep_word_in_dir` but across multiple roots; roots that
    don't exist are silently skipped. Lets the coverage probe declare a
    static list of test-tree roots without each caller having to guard
    on `path.exists()`.
    """
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        hits.extend(grep_word_in_dir(symbol, root, suffixes))
    return hits
