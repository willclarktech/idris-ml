"""File paths and regex constants — where the manifest tools read Idris
sources from, and how they recognise `%foreign` declarations in them.
"""

import re


# Files in the wrap-handle FFI set — the linter and converter both
# operate on these.
WRAP_HANDLE_FILES = [
    "packages/idris-ml/src/Tensor.idr",
    "packages/idris-ml/src/Executor/Mlx.idr",
    "packages/idris-ml/src/Executor/Tape.idr",
    "packages/idris-ml/src/Executor/Torch.idr",
]


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
