"""File paths and regex constants — where the manifest tools read Idris
sources from, and how they recognise `%foreign` declarations in them.
"""

from __future__ import annotations

import re
from pathlib import Path

# Files in the wrap-handle FFI set — the linter and converter both
# operate on these. Globbed (not hardcoded) so the per-module splits of
# Tensor.idr (Tensor/*.idr) and the per-slice Executor backend modules
# (Executor/<Backend>/*.idr) are picked up automatically, and future
# slices need no edit here. Paths are repo-root-relative POSIX strings.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "packages" / "idris-ml" / "src"


def _glob(*patterns: str) -> list[str]:
    out: set[str] = set()
    for pat in patterns:
        for p in _SRC.glob(pat):
            out.add(p.relative_to(_REPO_ROOT).as_posix())
    return sorted(out)


WRAP_HANDLE_FILES = _glob(
    "Tensor.idr",
    "Tensor/*.idr",
    "Executor/*.idr",
    "Executor/*/*.idr",
)


# Matches a `%foreign "C:cname,libidrisml"` declaration + its
# immediately-following Idris signature line.
C_FFI_RE = re.compile(
    r'(%foreign\s+"C:([a-zA-Z_0-9]+),libidrisml"\s*\n)'
    r"((?:[ \t]*export[ \t]*\n)?)"
    r"([ \t]*(?:export[ \t]+)?"
    r"[a-zA-Z_][a-zA-Z_0-9\']*"
    r"\s*:\s*[^\n]+\n)",
    re.MULTILINE,
)

# Matches any `%foreign "..."` declaration + its signature.
ANY_FFI_RE = re.compile(
    r'(%foreign\s+"(C|scheme):([^"]*(?:\\"[^"]*)*)"\s*\n)'
    r"((?:[ \t]*export[ \t]*\n)?)"
    r"([ \t]*(?:export[ \t]+)?"
    r"[a-zA-Z_][a-zA-Z_0-9\']*"
    r"\s*:\s*[^\n]+\n)",
    re.MULTILINE,
)
