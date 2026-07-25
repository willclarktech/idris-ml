#!/usr/bin/env python3
"""Gate: paired examples train on the same data distribution.

Why this exists: `Example.SeqClassify` generated noise-free waveforms while its
reference added N(0, 0.1) to every timestep. The Idris side was solving a
strictly easier problem, and nothing could see it — the models matched, the
hyperparameters matched, the metrics matched, and each side passed its own
convergence bar. The data generators are the one part of a paired example no
other gate looks at.

How: run both sides with `IDRISML_DUMP_DATA` set, which makes each print one
batch's shape and moments and exit before training. Values cannot be compared
(the RNGs differ), so this compares element count and moments.

Element count, not shape: the Idris `Seq` surface takes flattened inputs, so a
`[32, 1, 32]` reference batch is `[32, 32]` on the Idris side. That is a
representational difference, unlike a parameter shape, where rank IS the
architecture.

Usage: scripts/check-data-manifest.py [--only <name>] [--tolerance 0.15]
Exit 0 = aligned, 1 = divergence, 2 = a side failed to dump.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402

# One batch is a small sample, so the band is wider than the init gate's and
# scaled by the batch's own size.
SAMPLING_TERM = 4.0


def parse_manifest(text: str) -> dict[str, tuple[int, float, float, float, float, float]]:
    """{tag: (numel, mean, std, d1_std, min, max)} from DATA_MANIFEST lines."""
    out: dict[str, tuple[int, float, float, float, float, float]] = {}
    for line in text.splitlines():
        if not line.startswith("DATA_MANIFEST\t"):
            continue
        _, tag, shape, mean, sd, d1, lo, hi = line.split("\t")
        dims = [int(d) for d in shape.strip("[]").split(",") if d.strip()]
        numel = 1
        for d in dims:
            numel *= d
        out[tag] = (numel, float(mean), float(sd), float(d1), float(lo), float(hi))
    return out


def run(cmd: list[str], cwd: Path) -> str:
    env = dict(os.environ, IDRISML_DUMP_DATA="1")
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def compare(
    idris: dict[str, tuple[int, float, float, float, float, float]],
    python: dict[str, tuple[int, float, float, float, float, float]],
    tolerance: float,
) -> list[str]:
    problems: list[str] = []
    for tag in sorted(set(idris) | set(python)):
        if tag not in idris or tag not in python:
            problems.append(f"{tag}: dumped by only one side")
            continue
        (inum, imean, isd, id1, ilo, ihi) = idris[tag]
        (pnum, pmean, psd, pd1, plo, phi) = python[tag]
        if inum != pnum:
            problems.append(f"{tag}: batch holds {inum} values on idris, {pnum} on python")
            continue

        band = tolerance + SAMPLING_TERM / math.sqrt(inum)
        if isd < 1e-12 and psd < 1e-12:
            continue
        if isd < 1e-12 or psd < 1e-12:
            problems.append(
                f"{tag}: one side is constant and the other is not "
                f"(idris std {isd:.6g}, python std {psd:.6g})"
            )
            continue
        if abs(math.log(isd / psd)) > math.log1p(band):
            problems.append(
                f"{tag}: std ratio {isd / psd:.3f} (idris {isd:.6g}, python {psd:.6g}); "
                f"range idris [{ilo:.4g}, {ihi:.4g}] python [{plo:.4g}, {phi:.4g}]"
            )
        scale = max(abs(isd), abs(psd))
        if abs(imean - pmean) > band * scale:
            problems.append(
                f"{tag}: mean differs by {abs(imean - pmean):.6g} "
                f"(idris {imean:.6g}, python {pmean:.6g}; std ~{scale:.4g})"
            )
        if pd1 > 1e-12 and abs(math.log(max(id1, 1e-12) / pd1)) > math.log1p(band):
            problems.append(
                f"{tag}: lag-1 difference std ratio {id1 / pd1:.3f} "
                f"(idris {id1:.6g}, python {pd1:.6g})"
            )
        # Range: the statistic that separates a bounded signal from the same
        # signal plus additive noise. Adding N(0, 0.1) to a waveform of std
        # 0.82 shifts the batch std by 0.7% and the lag-1 std by 1%, but takes
        # the range from exactly [-1, 1] to [-1.35, 1.25].
        for label, iv, pv in (("min", ilo, plo), ("max", ihi, phi)):
            if abs(iv - pv) > band * scale:
                problems.append(
                    f"{tag}: {label} differs by {abs(iv - pv):.6g} "
                    f"(idris {iv:.6g}, python {pv:.6g}; std ~{scale:.4g})"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="check a single example by table name")
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args()

    specs = [
        s for s in EXAMPLES if args.only in (None, s["name"]) and s.get("data_manifest", False)
    ]
    if not specs:
        print(f"no data-bearing example matched {args.only!r}", file=sys.stderr)
        return 2

    failed = False
    for spec in specs:
        name = spec["name"]
        target = spec.get("target", f"example-{name}")
        module = Path(spec["python"]).stem
        idris = parse_manifest(run(["make", target], REPO_ROOT))
        python = parse_manifest(
            run(
                ["uv", "run", "python", "-u", "-m", f"torch_ref.scripts.{module}"],
                REPO_ROOT / "packages" / "pytorch",
            )
        )
        if not idris or not python:
            side = "idris" if not idris else "python"
            print(f"{name:<20} [DUMP FAILED] no DATA_MANIFEST from the {side} side")
            failed = True
            continue

        problems = compare(idris, python, args.tolerance)
        if problems:
            failed = True
            print(f"{name:<20} [{len(problems)} divergence(s)]")
            for problem in problems:
                print(f"    {problem}")
        else:
            print(f"{name:<20} [OK]")

    if failed:
        print("", file=sys.stderr)
        print(
            "Data generators diverge. The two sides are training on different",
            file=sys.stderr,
        )
        print(
            "distributions, so their convergence numbers are not comparable.",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(specs)} data-bearing examples generate matching batches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
