#!/usr/bin/env python3
"""Emit per-OP and per-FFI-symbol test coverage CSVs.

Usage:
    python3 scripts/coverage-gap-probe.py [OUTDIR]
    make coverage-gap-probe          # invokes the above

Algorithm (mirrors the predecessor `coverage-gap-probe.sh`; the
relational-join logic moved to Python so it's a sequence of set
operations rather than a 7-stage `sed | grep | awk` chain):

  1. Parse `OP_*` enum entries from each backend's `tape.h`
     (excluding `OP_COUNT` / `OP_CONST`).
  2. For each OP_FOO, find the source file holding the canonical
     registration anchor:
       - tape: `TAPE_REGISTER_OP(OP_FOO, ...)`
       - mlx:  `MLX_REGISTER_REPLAY(OP_FOO, ...)`
  3. Extract the non-static FFI symbols defined in that source file.
  4. If at least one of those symbols appears (word-boundary) in any
     test_*.c under TEST_DIR, the OP is covered; otherwise MISSING.
  5. Parse top-level decls from `backend.h`, apply the exclusion list
     (see `docs/develop/coverage-policy.md` "Exclusions"), and count
     test-file hits per remaining symbol.

CSV outputs land at `<OUTDIR>/coverage-gap-ops.csv` and
`<OUTDIR>/coverage-gap-symbols.csv` (default OUTDIR = `build/`).
Summary + sample MISSING rows print to stdout.

Exit code is advisory (always 0) until the OP_* gaps close per
coverage-policy.md; then flip to non-zero on any MISSING.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.header_parser import (  # noqa: E402
    extract_backend_h_symbols,
    extract_ffi_symbols_from_source,
    extract_ops,
    find_op_source,
    grep_word_in_dir,
)

BACKENDS = ROOT / "packages" / "backends"
TEST_DIR = BACKENDS / "test"

# Documented FFI exclusion list (lifted verbatim from
# coverage-gap-probe.sh; see docs/develop/coverage-policy.md
# "Exclusions" for the rationale per category).
FFI_EXCLUSIONS = frozenset(
    {
        "tensor_print",
        "tensor_live_count",
        "tensor_peak_live_count",
        "backend_profile_reset",
        "backend_profile_report",
        "backend_reset_for_eval",
        "backend_epoch_begin",
        "backend_name",
        "backend_profile_reset_return",
        "backend_profile_report_return",
        "backend_reset_for_eval_return",
        "get_rss_mb",
        "get_current_rss_mb",
        "mnist_load",
        "mnist_count",
        "mnist_get_image",
        "mnist_get_label",
        "mnist_free",
        "tensor_retain_handle",
        "tensor_release_handle",
        "idrisml_seq",
        "tensor_mlx_compile_enabled",
        "tensor_mlx_compile_invocations",
    }
)

BACKENDS_SPEC = [
    ("tape", "TAPE_REGISTER_OP"),
    ("mlx", "MLX_REGISTER_REPLAY"),
]


def _rel_to_backends(p: Path | None) -> str:
    if p is None:
        return "NO_REGISTRATION"
    try:
        return str(p.relative_to(BACKENDS))
    except ValueError:
        return str(p)


def build_ops_rows() -> list[dict]:
    rows: list[dict] = []
    for backend, anchor in BACKENDS_SPEC:
        header = BACKENDS / f"backend_{backend}" / "tape.h"
        backend_dir = BACKENDS / f"backend_{backend}"
        if not header.exists():
            continue
        ops = sorted(extract_ops(header.read_text()))
        for op in ops:
            source = find_op_source(op, backend_dir, anchor)
            symbols: list[str] = []
            if source is not None:
                symbols = sorted(extract_ffi_symbols_from_source(source.read_text()))
            test_path = None
            for sym in symbols:
                hits = grep_word_in_dir(sym, TEST_DIR)
                if hits:
                    test_path = hits[0]
                    break
            status = "present" if test_path is not None else "MISSING"
            rows.append(
                {
                    "backend": backend,
                    "op": op,
                    "source_file": _rel_to_backends(source),
                    # `|` separator matches the bash version's
                    # CSV-embed convention (commas would break columns).
                    "ffi_symbols": "|".join(symbols),
                    "test_file": _rel_to_backends(test_path) if test_path else "MISSING",
                    "status": status,
                }
            )
    return rows


def build_symbols_rows() -> list[dict]:
    backend_h = BACKENDS / "backend.h"
    if not backend_h.exists():
        return []
    syms = sorted(extract_backend_h_symbols(backend_h.read_text()))
    rows: list[dict] = []
    for sym in syms:
        if sym in FFI_EXCLUSIONS:
            continue
        hits = grep_word_in_dir(sym, TEST_DIR)
        rows.append({"symbol": sym, "test_hits": str(len(hits))})
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    # Use a StringIO buffer with explicit Unix newlines so the file
    # contents are byte-for-byte stable regardless of platform line-
    # ending defaults.
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    path.write_text(buf.getvalue())


def print_summary(ops_rows: list[dict], symbols_rows: list[dict],
                  ops_csv: Path, symbols_csv: Path) -> int:
    missing = [r for r in ops_rows if r["status"] == "MISSING"]
    tape_missing = sum(1 for r in missing if r["backend"] == "tape")
    mlx_missing = sum(1 for r in missing if r["backend"] == "mlx")
    symbols_zero = [r for r in symbols_rows if r["test_hits"] == "0"]

    print()
    print("=== Coverage gap probe ===")
    print(f"OP_* without any FFI test hit: {len(missing)}  (tape={tape_missing}, mlx={mlx_missing})")
    print(f"FFI symbols with 0 test hits:  {len(symbols_zero)}")
    print()
    print("Reports:")
    print(f"  {ops_csv}")
    print(f"  {symbols_csv}")

    if missing:
        print()
        print("Missing OP_* tests (sample):")
        for r in missing[:20]:
            print(f"  {r['backend']}  {r['op']}  (source: {r['source_file']}, symbols: {r['ffi_symbols']})")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more — see CSV")

    if symbols_zero:
        print()
        print("FFI symbols with 0 test hits (sample):")
        for r in symbols_zero[:15]:
            print(f"  {r['symbol']}")
        if len(symbols_zero) > 15:
            print(f"  ... and {len(symbols_zero) - 15} more — see CSV")

    # Advisory only; matches the predecessor.
    return 0


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else ROOT / "build"
    outdir.mkdir(parents=True, exist_ok=True)
    ops_csv = outdir / "coverage-gap-ops.csv"
    symbols_csv = outdir / "coverage-gap-symbols.csv"

    ops_rows = build_ops_rows()
    symbols_rows = build_symbols_rows()

    write_csv(ops_csv, ops_rows, ["backend", "op", "source_file", "ffi_symbols", "test_file", "status"])
    write_csv(symbols_csv, symbols_rows, ["symbol", "test_hits"])

    return print_summary(ops_rows, symbols_rows, ops_csv, symbols_csv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
