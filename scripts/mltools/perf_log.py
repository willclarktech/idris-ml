"""Perf-log entry construction, JSONL append, and parser helpers.

Single source of truth for the `docs/develop/perf-log.jsonl` schema
(documented in `docs/develop/perf-log.md`). Direct callers:

  Writers:  scripts/perf-{run,baseline,sweep,fast,nightly}.sh
            (via the `python3 -m mltools.perf_log <subcommand>` CLI)
  Readers:  scripts/render-benchmarks.py
            scripts/check-perf-regression.py
            (via `from mltools.perf_log import iter_entries, ...`)

The CLI subcommands map 1:1 onto the writer functions:

    append-run          → append_run(...)
    append-baseline     → append_baseline(...)
    parse-op-bench      → batch append_op_bench(...) over a bench-output file
    append-axis-row     → append_op_bench(...) (single row, for Axis C/D)

Bash callers are expected to set PYTHONPATH to include `scripts/`
(scripts/perf_lib.sh does this). Then:

    python3 -m mltools.perf_log append-run --example ntm-copy ...

Append is idempotent in the sense that each call writes one JSON object
followed by a newline; the log file is append-only by convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

LOG_PATH_REL = "docs/develop/perf-log.jsonl"

# ----------------------------------------------------------------------
# Line parsers — shared by writers and tests
# ----------------------------------------------------------------------

_EPOCH_RE = re.compile(r"epoch\s+(\d+)")
_COMPLETED_PARENS_RE = re.compile(r"\((\d+)\s+epochs?,\s+([\d.]+)\s*ms/epoch\)")
_COMPLETED_WALL_RE = re.compile(r"Completed in\s+([^()]+)\s+\(")
_STAGE_RE = re.compile(r"^\[stage\] \[(\d{2}):(\d{2}):(\d{2})\]\s+(.*)$")
_OP_LINE_RE = re.compile(r"^([A-Za-z][^\t:]*?):\s*([0-9.]+)\s*ms\s*\((\d+)\s*iters\)\s*$")
_OP_SECTION_RE = re.compile(r"^---\s*(.+?)\s*---\s*$")


def parse_perf_marker(text: str) -> Optional[float]:
    """Last `PERF_MS_PER_EP=<float>` value in text, or None."""
    last: Optional[float] = None
    for line in text.splitlines():
        if line.startswith("PERF_MS_PER_EP="):
            try:
                last = float(line[len("PERF_MS_PER_EP=") :])
            except ValueError:
                pass
    return last


def parse_axis_d_markers(text: str) -> tuple[Optional[int], Optional[float]]:
    """Last `PERF_GENERATE_TOKENS=<int>` and `PERF_GENERATE_WALL_MS=<float>`."""
    tokens: Optional[int] = None
    wall: Optional[float] = None
    for line in text.splitlines():
        if line.startswith("PERF_GENERATE_TOKENS="):
            try:
                tokens = int(line[len("PERF_GENERATE_TOKENS=") :])
            except ValueError:
                pass
        elif line.startswith("PERF_GENERATE_WALL_MS="):
            try:
                wall = float(line[len("PERF_GENERATE_WALL_MS=") :])
            except ValueError:
                pass
    return tokens, wall


def parse_epoch(line: str) -> Optional[int]:
    """Pull `epoch N` out of a Converged/Diverged log line."""
    if not line:
        return None
    m = _EPOCH_RE.search(line)
    return int(m.group(1)) if m else None


def parse_completed(line: str) -> dict:
    """Parse `Completed in 1m 7s (5500 epochs, 12ms/epoch)` → dict."""
    out: dict = {}
    if not line:
        return out
    m = _COMPLETED_PARENS_RE.search(line)
    if m:
        out["total_epochs"] = int(m.group(1))
        out["ms_per_epoch"] = float(m.group(2))
    m = _COMPLETED_WALL_RE.search(line)
    if m:
        out["wall"] = m.group(1).strip()
    return out


def parse_result(line: str) -> dict:
    """Parse `RESULT\\tk=v\\tk=v` → {k: v} with int / float / str coercion."""
    out: dict = {}
    if not line:
        return out
    for part in line.split("\t")[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def parse_stages(blob: str) -> list[dict]:
    """Parse multi-line `[stage] [hh:mm:ss] <label>` block."""
    out: list[dict] = []
    if not blob:
        return out
    for line in blob.splitlines():
        m = _STAGE_RE.match(line)
        if not m:
            continue
        h, mi, s, label = m.groups()
        elapsed = int(h) * 3600 + int(mi) * 60 + int(s)
        out.append({"label": label.strip(), "elapsed_s": elapsed})
    return out


def parse_op_bench_output(blob: str) -> list[dict]:
    """Parse `bench_ops` / `bench_layers` stdout.

    Returns one dict per measurement line:
        {"section": "...", "label": "...", "wall_ms": float, "iters": int}
    Section is the most recent `--- <name> ---` header (empty if none yet).
    """
    entries: list[dict] = []
    section = ""
    for line in blob.splitlines():
        line = line.rstrip()
        m = _OP_SECTION_RE.match(line)
        if m:
            section = m.group(1)
            continue
        m = _OP_LINE_RE.match(line)
        if not m:
            continue
        entries.append(
            {
                "section": section,
                "label": m.group(1).strip(),
                "wall_ms": float(m.group(2)),
                "iters": int(m.group(3)),
            }
        )
    return entries


# ----------------------------------------------------------------------
# Make-log extraction (perf-run.sh side)
# ----------------------------------------------------------------------

_CONVERGED_RE = re.compile(r"^\s*\[[^\]]+\]\s+Converged")
_DIVERGED_RE = re.compile(r"^\s*\[[^\]]+\]\s+Diverged")
_COMPLETED_LINE_RE = re.compile(r"^Completed")
_RESULT_LINE_RE = re.compile(r"^RESULT")
_STAGE_LINE_RE = re.compile(r"^\[stage\] \[\d{2}:\d{2}:\d{2}\]")


def extract_run_lines(blob: str) -> dict:
    """Pull the canonical lines out of a captured example stdout/stderr.

    Returns a dict with the *last* match for each (or empty string when
    absent) plus the full stage block joined by newline:

        {"converged": "...", "diverged": "...", "completed": "...",
         "result": "...", "stages": "[stage] ...\\n[stage] ..."}
    """
    converged = diverged = completed = result = ""
    stage_lines: list[str] = []
    for line in blob.splitlines():
        if _CONVERGED_RE.match(line):
            converged = line
        if _DIVERGED_RE.match(line):
            diverged = line
        if _COMPLETED_LINE_RE.match(line):
            completed = line
        if _RESULT_LINE_RE.match(line):
            result = line
        if _STAGE_LINE_RE.match(line):
            stage_lines.append(line)
    return {
        "converged": converged,
        "diverged": diverged,
        "completed": completed,
        "result": result,
        "stages": "\n".join(stage_lines),
    }


# ----------------------------------------------------------------------
# Timestamp helpers
# ----------------------------------------------------------------------

def now_ts(_clock=None) -> tuple[str, str]:
    """(ISO timestamp, ISO date) in UTC.

    The `_clock` hook is for tests; production callers leave it None.
    """
    n = _clock() if _clock else datetime.now(timezone.utc)
    return n.strftime("%Y-%m-%dT%H:%M:%SZ"), n.strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# Path resolution + append
# ----------------------------------------------------------------------

def resolve_log_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent  # mltools/ → scripts/ → repo
    return repo_root / LOG_PATH_REL


def _append(entry: dict, log_path: Optional[str | Path]) -> Path:
    path = resolve_log_path(log_path)
    if not path.exists():
        path.touch()
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return path


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def append_run(
    *,
    example: str,
    backend: str,
    device: str,
    commit: str,
    mlx_compile: str,
    args: str,
    exit_code: int,
    wall_ms: int,
    wall_human: str,
    parse_log: Optional[str | Path] = None,
    torch_dtype: Optional[str] = None,
    mlx_dtype: Optional[str] = None,
    tape_dtype: Optional[str] = None,
    log_path: Optional[str | Path] = None,
) -> dict:
    """Build a `kind="run"` entry and append it.

    When `parse_log` is given, the captured make-log file is scanned
    for Converged / Diverged / Completed / RESULT / [stage] lines and
    the structured fields are filled in.
    """
    ts, date = now_ts()
    entry: dict = {
        "ts": ts,
        "date": date,
        "kind": "run",
        "example": example,
        "backend": backend,
        "device": device,
        "mlx_compile": mlx_compile,
        "commit": commit,
        "args": args,
        "exit": exit_code,
        "wall_ms": wall_ms,
        "wall_human": wall_human,
    }
    if torch_dtype:
        entry["torch_dtype"] = torch_dtype
    if mlx_dtype:
        entry["mlx_dtype"] = mlx_dtype
    if tape_dtype:
        entry["tape_dtype"] = tape_dtype
    if parse_log:
        blob = Path(parse_log).read_text()
        lines = extract_run_lines(blob)
        conv = parse_epoch(lines["converged"])
        div = parse_epoch(lines["diverged"])
        if conv is not None:
            entry["converged_at_epoch"] = conv
        if div is not None:
            entry["diverged_at_epoch"] = div
        stats = parse_completed(lines["completed"])
        if stats:
            entry["stats"] = stats
        result = parse_result(lines["result"])
        if result:
            entry["result"] = result
        stages = parse_stages(lines["stages"])
        if stages:
            entry["stages"] = stages
    _append(entry, log_path)
    return entry


def append_baseline(
    *,
    example: str,
    backend: str,
    device: str,
    commit: str,
    idris_raw: str,
    pytorch_raw: str,
    ratio: str,
    n_long: int,
    seed: int,
    log_path: Optional[str | Path] = None,
) -> dict:
    """Build a `kind="baseline"` entry from raw markers.

    `idris_raw` / `pytorch_raw` are float strings, or the sentinel
    strings `"crashed"` / `"missing"` (preserved from the bash extractor).
    """
    ts, date = now_ts()

    def num(s: str) -> Optional[float]:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    idris_crashed = idris_raw in ("crashed", "missing")
    py_crashed = pytorch_raw in ("crashed", "missing")
    entry: dict = {
        "ts": ts,
        "date": date,
        "kind": "baseline",
        "methodology": "in_script_marker",
        "example": example,
        "backend": backend,
        "device": device,
        "commit": commit,
        "idris_ms_per_epoch": None if idris_crashed else num(idris_raw),
        "pytorch_ms_per_epoch": None if py_crashed else num(pytorch_raw),
        "ratio": None if (idris_crashed or py_crashed) else num(ratio),
        "n_long": n_long,
        "seed": seed,
    }
    notes: list[str] = []
    if idris_raw == "crashed":
        notes.append("idris binary aborted during timed run")
    elif idris_raw == "missing":
        notes.append("idris stdout had no PERF_MS_PER_EP marker")
    if pytorch_raw == "crashed":
        notes.append("pytorch ref aborted during timed run")
    elif pytorch_raw == "missing":
        notes.append("pytorch stdout had no PERF_MS_PER_EP marker")
    if notes:
        entry["notes"] = "; ".join(notes)
    _append(entry, log_path)
    return entry


def append_op_bench(
    *,
    axis: str,
    runtime: str,
    section: str,
    label: str,
    wall_ms: float,
    iters: int,
    commit: str,
    log_path: Optional[str | Path] = None,
) -> dict:
    ts, date = now_ts()
    entry = {
        "ts": ts,
        "date": date,
        "kind": "op_bench",
        "axis": axis,
        "section": section,
        "label": label,
        "runtime": runtime,
        "commit": commit,
        "wall_ms": wall_ms,
        "iters": iters,
        "ms_per_iter": wall_ms / iters if iters else None,
    }
    _append(entry, log_path)
    return entry


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------

def iter_entries(log_path: Optional[str | Path] = None) -> Iterator[dict]:
    """Yield each JSONL entry. Skips blank lines + JSON-decode errors."""
    path = resolve_log_path(log_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _cmd_append_run(args: argparse.Namespace) -> int:
    append_run(
        example=args.example,
        backend=args.backend,
        device=args.device,
        commit=args.commit,
        mlx_compile=args.mlx_compile,
        args=args.cli_args,
        exit_code=args.exit_code,
        wall_ms=args.wall_ms,
        wall_human=args.wall_human,
        torch_dtype=args.torch_dtype or None,
        mlx_dtype=args.mlx_dtype or None,
        tape_dtype=args.tape_dtype or None,
        parse_log=args.parse_log,
        log_path=args.log_path,
    )
    return 0


def _cmd_append_baseline(args: argparse.Namespace) -> int:
    append_baseline(
        example=args.example,
        backend=args.backend,
        device=args.device,
        commit=args.commit,
        idris_raw=args.idris_ms,
        pytorch_raw=args.pytorch_ms,
        ratio=args.ratio,
        n_long=args.n_long,
        seed=args.seed,
        log_path=args.log_path,
    )
    return 0


def _cmd_parse_op_bench(args: argparse.Namespace) -> int:
    blob = Path(args.input).read_text()
    rows = parse_op_bench_output(blob)
    for row in rows:
        append_op_bench(
            axis=args.axis,
            runtime=args.runtime,
            section=row["section"],
            label=row["label"],
            wall_ms=row["wall_ms"],
            iters=row["iters"],
            commit=args.commit,
            log_path=args.log_path,
        )
    print(f"appended {len(rows)} {args.runtime} axis-{args.axis} entries")
    return 0


def _cmd_append_axis_row(args: argparse.Namespace) -> int:
    append_op_bench(
        axis=args.axis,
        runtime=args.runtime,
        section=args.section or args.label,
        label=args.label,
        wall_ms=args.wall_ms,
        iters=args.iters,
        commit=args.commit,
        log_path=args.log_path,
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mltools.perf_log",
        description="Perf-log entry writer CLI (one subcommand per `kind`).",
    )
    p.add_argument(
        "--log-path",
        default=None,
        help=f"Override target JSONL path (default: {LOG_PATH_REL}).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("append-run", help="Append a kind=run entry.")
    pr.add_argument("--example", required=True)
    pr.add_argument("--backend", required=True)
    pr.add_argument("--device", required=True)
    pr.add_argument("--mlx-compile", required=True)
    pr.add_argument("--commit", required=True)
    pr.add_argument("--cli-args", default="", help="example CLI args (display only).")
    pr.add_argument("--exit-code", type=int, required=True)
    pr.add_argument("--wall-ms", type=int, required=True)
    pr.add_argument("--wall-human", required=True)
    pr.add_argument("--torch-dtype", default="")
    pr.add_argument("--mlx-dtype", default="")
    pr.add_argument("--tape-dtype", default="")
    pr.add_argument(
        "--parse-log",
        default=None,
        help="Path to make stdout/stderr log to grep for "
        "RESULT / Completed / Converged / Diverged / [stage] lines.",
    )
    pr.set_defaults(func=_cmd_append_run)

    pb = sub.add_parser("append-baseline", help="Append a kind=baseline entry.")
    pb.add_argument("--example", required=True)
    pb.add_argument("--backend", required=True)
    pb.add_argument("--device", required=True)
    pb.add_argument("--commit", required=True)
    pb.add_argument(
        "--idris-ms",
        required=True,
        help="Float string, or sentinel 'crashed' / 'missing'.",
    )
    pb.add_argument("--pytorch-ms", required=True)
    pb.add_argument("--ratio", required=True)
    pb.add_argument("--n-long", type=int, required=True)
    pb.add_argument("--seed", type=int, required=True)
    pb.set_defaults(func=_cmd_append_baseline)

    po = sub.add_parser(
        "parse-op-bench",
        help="Parse a bench_ops/bench_layers stdout file; append all op_bench rows.",
    )
    po.add_argument("--axis", required=True, choices=["A", "B", "C", "D"])
    po.add_argument("--runtime", required=True, choices=["tape", "pytorch"])
    po.add_argument("--commit", required=True)
    po.add_argument("--input", required=True)
    po.set_defaults(func=_cmd_parse_op_bench)

    par = sub.add_parser(
        "append-axis-row", help="Append one op_bench row (Axis C/D pre-computed metric)."
    )
    par.add_argument("--axis", required=True, choices=["A", "B", "C", "D"])
    par.add_argument("--runtime", required=True, choices=["tape", "pytorch"])
    par.add_argument("--label", required=True)
    par.add_argument("--section", default="")
    par.add_argument("--wall-ms", type=float, required=True)
    par.add_argument("--iters", type=int, required=True)
    par.add_argument("--commit", required=True)
    par.set_defaults(func=_cmd_append_axis_row)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
