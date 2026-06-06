#!/usr/bin/env python3
"""One-shot rename: Device kind → Executor kind across idris-ml.

Ordering matters: longest compound names first to avoid substring corruption
(`UserDeviceCore` must rename before bare `Device`, otherwise the latter
splits `User%sCore` and breaks). The `\\bDevice\\b` final pass mops up
remaining bare references in type signatures and module declarations.

Comments/docs containing the everyday word "device" (lowercase) are not
touched. Capitalised "Device" in docstrings/comments will be converted —
hand-review the diff before committing.

Run from repo root: `python3 scripts/rename_device_to_executor.py`.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Ordered substitution table. (pattern, replacement). Patterns are regexes;
# replacements are literals. Word boundaries on bare names prevent matches
# inside longer identifiers.
SUBS: list[tuple[str, str]] = [
    # ---- Compound names (longest first; word-boundary-safe). ----
    (r"\bMultiDeviceRegistry\b", "MultiExecutorRegistry"),
    (r"\bUserDeviceTraining\b", "UserExecutorTraining"),
    (r"\bUserDeviceTransfer\b", "UserExecutorTransfer"),
    (r"\bUserDeviceQuant\b",    "UserExecutorQuant"),
    (r"\bUserDeviceLinear\b",   "UserExecutorLinear"),
    (r"\bUserDeviceCore\b",     "UserExecutorCore"),
    (r"\bUserDeviceConv\b",     "UserExecutorConv"),
    (r"\bUserDeviceNN\b",       "UserExecutorNN"),
    (r"\bUserDevice\b",         "UserExecutor"),  # any remaining bare uses
    (r"\bavailableDevices\b",   "availableExecutors"),
    (r"\bbuiltinDevices\b",     "builtinExecutors"),
    (r"\bSomeDevice\b",         "SomeExecutor"),
    (r"\bsomeDevice\b",         "someExecutor"),
    (r"\btoDeviceChecked\b",    "toExecutorChecked"),
    (r"\btoDevice\b",           "toExecutor"),
    (r"\bDeviceError\b",        "ExecutorError"),
    (r"\bExampleDevice\b",      "ExampleExecutor"),
    (r"\bTestDevice\b",         "TestExecutor"),
    (r"\bHwDevices\b",          "HwExecutors"),
    (r"\bHWDEVICES\b",          "HWEXECUTORS"),  # uppercase Make var names
    (r"\bDeviceCore\b",         "ExecutorCore"),  # the example file name

    # ---- Backend tag types (TapeDev/TorchDev/MlxDev → *Executor). ----
    (r"\bTapeDev\b",  "TapeExecutor"),
    (r"\bTorchDev\b", "TorchExecutor"),
    (r"\bMlxDev\b",   "MlxExecutor"),

    # ---- Data constructors. ----
    (r"\bMkTapeDev\b",     "MkTapeExecutor"),
    (r"\bMkTorchDev\b",    "MkTorchExecutor"),
    (r"\bMkMlxDev\b",      "MkMlxExecutor"),
    (r"\bMkSomeDevice\b",  "MkSomeExecutor"),

    # ---- printf-string `\nbuiltinDevices` — the `\b` boundary fails
    # because `n` (in `\n`) is a word char. Catch the specific shape.
    (r"\\nbuiltinDevices\b", r"\\nbuiltinExecutors"),

    # ---- Module paths in import / module decls. ----
    (r"\bDevice\.Core\b",  "Executor.Core"),
    (r"\bDevice\.Tape\b",  "Executor.Tape"),
    (r"\bDevice\.Torch\b", "Executor.Torch"),
    (r"\bDevice\.Mlx\b",   "Executor.Mlx"),

    # ---- File-system paths (Makefile / scripts / comments). ----
    (r"\bDevice/Core\b",  "Executor/Core"),
    (r"\bDevice/Tape\b",  "Executor/Tape"),
    (r"\bDevice/Torch\b", "Executor/Torch"),
    (r"\bDevice/Mlx\b",   "Executor/Mlx"),
    (r"\bDevice\.idr\b",  "Executor.idr"),

    # ---- Bare kind name (must come last). ----
    (r"\bDevice\b", "Executor"),
]

# Files to rewrite. Discovered via shell glob; exclude generated artefacts.
TEXT_GLOBS = [
    "packages/**/*.idr",
    "packages/**/*.ipkg",
    "packages/**/*.in",
    "Makefile",
    "scripts/gen-rename-headers.py",
    # CI workflow comments referencing TestDevice / UserDeviceTransfer.
    ".github/workflows/*.yml",
]

# Excluded from C/C++ scope: the bare `\\bDevice\\b` substitution corrupts
# libtorch's `c10::Device` / `c10::DeviceType` C++ types. C++ doc comments
# that mention Idris-side `TapeDev`/`TorchDev`/`MlxDev` get stale — that's
# a small acceptable cost; the alternative (whitelisting safe contexts in
# C++ regex) is brittle. C++ comment hygiene lands as a follow-up.

# File renames (post-sed). Tuples of (old_path, new_path).
FILE_RENAMES: list[tuple[str, str]] = [
    ("packages/idris-ml/src/Device.idr",                   "packages/idris-ml/src/Executor.idr"),
    ("packages/idris-ml/src/Device/Core.idr",              "packages/idris-ml/src/Executor/Core.idr"),
    ("packages/idris-ml/src/Device/Tape.idr",              "packages/idris-ml/src/Executor/Tape.idr"),
    ("packages/idris-ml/src/Device/Torch.idr",             "packages/idris-ml/src/Executor/Torch.idr"),
    ("packages/idris-ml/src/Device/Mlx.idr",               "packages/idris-ml/src/Executor/Mlx.idr"),
    ("packages/idris-ml/src/HwDevices.idr.in",             "packages/idris-ml/src/HwExecutors.idr.in"),
    ("packages/idris-ml/src/Test/MultiDeviceRegistry.idr", "packages/idris-ml/src/Test/MultiExecutorRegistry.idr"),
    ("packages/idris-ml-examples/src/Example/DeviceCore.idr",
     "packages/idris-ml-examples/src/Example/ExecutorCore.idr"),
]


EXCLUDE_PATHS = {
    # This script's SUBS table contains the patterns as literal strings;
    # substituting them would corrupt the script. Skip self.
    "scripts/rename_device_to_executor.py",
}


def discover_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in TEXT_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and str(p.relative_to(ROOT)) not in EXCLUDE_PATHS:
                files.add(p)
    return sorted(files)


def rewrite_file(path: Path) -> tuple[int, bool]:
    """Apply all substitutions to a file. Returns (total_substitutions, touched)."""
    original = path.read_text()
    new = original
    total = 0
    for pat, repl in SUBS:
        new, n = re.subn(pat, repl, new)
        total += n
    if new != original:
        path.write_text(new)
        return total, True
    return 0, False


def main() -> int:
    files = discover_files()
    print(f"Scanning {len(files)} files…", file=sys.stderr)

    touched: list[Path] = []
    skipped: list[Path] = []
    total_subs = 0
    for f in files:
        n, did = rewrite_file(f)
        total_subs += n
        (touched if did else skipped).append(f)

    print(f"Substitutions: {total_subs}; files touched: {len(touched)}; files unchanged: {len(skipped)}",
          file=sys.stderr)

    # File renames via git mv (preserves history). Idempotent: if the
    # source is already gone (re-running after a partial state), skip
    # silently when the target exists.
    print("Renaming files via git mv…", file=sys.stderr)
    for old, new in FILE_RENAMES:
        old_p = ROOT / old
        new_p = ROOT / new
        if not old_p.exists():
            if new_p.exists():
                print(f"  already renamed: {old} → {new}", file=sys.stderr)
            else:
                print(f"  skip (missing): {old}", file=sys.stderr)
            continue
        new_p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], check=True)
        print(f"  {old} → {new}", file=sys.stderr)

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
