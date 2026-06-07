#!/usr/bin/env python3
"""Render BENCHMARKS.md from docs/develop/perf-log.jsonl.

Reads the JSONL log, picks the latest entry per (axis, label, runtime)
tuple, and emits a Markdown file with one table per axis showing the
idris vs PyTorch ratio for each workload. The output is repo-front-
page-visible (BENCHMARKS.md at root) so external readers can compare
idris-ml's performance to PyTorch at a glance.

This is part of the testing-taxonomy "Axis A" framework — see
docs/develop/testing-taxonomy.md and CLAUDE.md "Performance docs"
for the broader context.

Usage:
  python3 scripts/render-benchmarks.py            # rewrites BENCHMARKS.md
  python3 scripts/render-benchmarks.py --check    # exit 1 if would change

Today only Axis A (op kernels) is populated. Axes B (single-layer
fwd+bwd), C (e2e training), D (HF inference) will be added as their
benches land — the renderer already keys on `axis` so it picks up
new axes automatically.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from mltools.perf_log import resolve_log_path  # noqa: E402

# One JSONL log entry (json.loads boundary — values stay Any here and are
# narrowed at the consuming sites below).
Entry = dict[str, Any]
# (axis, label, runtime) selection key — fields come from `.get`, so each
# may be absent (None) in a malformed log line.
Key = tuple[str | None, str | None, str | None]

if TYPE_CHECKING:
    # Typed overlay: mltools.perf_log annotates iter_entries as a bare
    # `Iterator[dict]`, which strict mode treats as partially unknown.
    from collections.abc import Iterator

    def iter_entries(log_path: str | Path | None = None) -> Iterator[Entry]: ...
else:
    from mltools.perf_log import iter_entries

LOG_PATH = resolve_log_path()
OUT_PATH = ROOT / "BENCHMARKS.md"

AXIS_TITLES = {
    "A": "Axis A — Op kernels (vs PyTorch)",
    "B": "Axis B — Single-layer forward + backward (vs PyTorch)",
    "C": "Axis C — End-to-end training (vs PyTorch)",
    "D": "Axis D — End-to-end HF inference (vs HF transformers)",
}

AXIS_ORDER = ["A", "B", "C", "D"]


def select_op_bench_latest(entries: list[Entry]) -> dict[Key, Entry]:
    """Pick the latest entry per (axis, label, runtime). 'Latest' = last
    occurrence in the log (the log is append-only and naturally ordered
    by time)."""
    latest: dict[Key, Entry] = {}
    for e in entries:
        if e.get("kind") != "op_bench":
            continue
        key = (e.get("axis"), e.get("label"), e.get("runtime"))
        latest[key] = e
    return latest


AXIS_BLURBS = {
    "A": (
        "Wall-clock per iteration on the C backend (tape) vs the same\n"
        "kernel in PyTorch on the same hardware. Both measured in-process\n"
        "after a warmup. Lower ratios are better; ≈1.0 means parity."
    ),
    "B": (
        "Wall-clock per layer-scope fwd+bwd+step on idris-ml's typed-layer\n"
        "API (tape backend, F64) vs an equivalent PyTorch reference at the\n"
        "same shape. Captures FFI + tape wrap + autograd graph overhead\n"
        "that Axis A's pure C-kernel timings don't see."
    ),
    "C": (
        "Wall-clock per epoch on a representative end-to-end training\n"
        "workload per training mode (supervised / RNN / transformer /\n"
        "NTM-class / RL). One entry per distinct compute pattern."
    ),
    "D": (
        "Per-token wall-clock on HuggingFace inference workloads (encoder\n"
        "fwd / decoder fwd / cached-decode generation) vs HF transformers\n"
        "Python on the same hardware."
    ),
}


def render_axis_table(latest: dict[Key, Entry], axis: str) -> str:
    """One row per (label) — pairs the tape entry with the pytorch entry.

    Works for any axis; entries are filtered by `e.get("axis") == axis`.
    Grouped by `section` field (header line in bench output)."""
    pairs: defaultdict[str | None, dict[str | None, Entry]] = defaultdict(dict)
    sections_for_label: dict[str | None, str] = {}
    for (eaxis, label, runtime), entry in latest.items():
        if eaxis != axis:
            continue
        pairs[label][runtime] = entry
        sections_for_label[label] = entry.get("section", "")
    sections: defaultdict[str, list[tuple[str | None, dict[str | None, Entry]]]] = defaultdict(list)
    for label, runtimes in pairs.items():
        sections[sections_for_label[label]].append((label, runtimes))

    out: list[str] = []
    out.append(f"## {AXIS_TITLES[axis]}")
    out.append("")
    out.append(AXIS_BLURBS[axis])
    out.append("")
    for section in sorted(sections.keys()):
        rows = sections[section]
        if not rows:
            continue
        if section:
            out.append(f"### {section}")
            out.append("")
        out.append(
            "| Workload | tape (ms/iter) | pytorch (ms/iter) | ratio (tape / pytorch) | iters | commit |"  # noqa: E501
        )
        out.append("|---|---:|---:|---:|---:|---|")
        for label, runtimes in sorted(rows):
            tape = runtimes.get("tape")
            pyt = runtimes.get("pytorch")
            tape_msi: float | None = tape["ms_per_iter"] if tape else None
            pyt_msi: float | None = pyt["ms_per_iter"] if pyt else None
            ratio = (tape_msi / pyt_msi) if (tape_msi and pyt_msi) else None
            meta: Entry = tape or pyt or {}
            commit = meta.get("commit", "?")
            iters = meta.get("iters", "?")
            tape_str = f"{tape_msi:.4f}" if tape_msi is not None else "—"
            pyt_str = f"{pyt_msi:.4f}" if pyt_msi is not None else "—"
            ratio_str = f"{ratio:.2f}×" if ratio is not None else "—"
            out.append(f"| {label} | {tape_str} | {pyt_str} | {ratio_str} | {iters} | `{commit}` |")
        out.append("")
    return "\n".join(out)


def render_placeholder_axis(axis: str) -> str:
    title = AXIS_TITLES.get(axis, f"Axis {axis}")
    return "\n".join(
        [
            f"## {title}",
            "",
            f"_No entries yet — Axis {axis} benches not yet wired up._",
            "",
        ]
    )


def render() -> str:
    entries = list(iter_entries())
    latest = select_op_bench_latest(entries)
    axes_present = {axis for (axis, _, _) in latest}

    parts: list[str] = []
    parts.append("# idris-ml benchmarks")
    parts.append("")
    parts.append("Auto-generated by `scripts/render-benchmarks.py` from")
    parts.append("`docs/develop/perf-log.jsonl`. Do not hand-edit — re-run via")
    parts.append("`make bench-fast` (Tier 1) or `make bench-deep`")
    parts.append("(Tier 2). The framework + selection rule are documented in")
    parts.append("`docs/develop/testing-taxonomy.md` (testing taxonomy → Axis")
    parts.append("A/B/C/D coverage).")
    parts.append("")
    parts.append("Ratios show idris-ml wall-clock divided by the PyTorch")
    parts.append("reference on the same hardware. **1.0× is parity**; lower")
    parts.append("is faster. VM noise floor is ±15–20% (per `CLAUDE.md`).")
    parts.append("")

    for axis in AXIS_ORDER:
        if axis in axes_present:
            parts.append(render_axis_table(latest, axis))
        else:
            parts.append(render_placeholder_axis(axis))

    parts.append("---")
    parts.append("")
    parts.append("_See `docs/develop/perf-log.md` for the JSONL schema, and")
    parts.append("`docs/develop/perf-baseline.md` for the broader example-")
    parts.append("level baseline this complements._")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    check = "--check" in sys.argv[1:]
    new_text = render()
    old_text = OUT_PATH.read_text() if OUT_PATH.exists() else ""
    if check:
        if new_text != old_text:
            print(
                f"FAIL: {OUT_PATH.relative_to(ROOT)} disagrees with log "
                f"({LOG_PATH.relative_to(ROOT)}). Run scripts/render-benchmarks.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK: {OUT_PATH.relative_to(ROOT)} is in sync with log.")
        return
    if new_text != old_text:
        OUT_PATH.write_text(new_text)
        print(
            f"Wrote {OUT_PATH.relative_to(ROOT)} ({sum(1 for _ in new_text.splitlines())} lines)."
        )
    else:
        print(f"No changes; {OUT_PATH.relative_to(ROOT)} already up to date.")


if __name__ == "__main__":
    main()
