#!/usr/bin/env python3
"""Gate: paired examples build the same parameter shapes with the same init.

Why this exists: four separate init divergences reached main before anyone
noticed — dense (2026-07-29), then conv, recurrent and attention (2026-07-31).
Each was caught by a human reading two files side by side, and each had been
sitting there for months. Every existing gate compares one side against a
written contract, so a contract that quietly stops being true is invisible;
this compares the two sides against each other.

How: run both sides with `IDRISML_DUMP_INIT` set, which makes each dump its
freshly-constructed parameters to safetensors and exit before training. Then
diff the ordered sequence of (shape, mean, std, min, max).

Shapes are compared exactly and in order — that catches an architecture change
(a hidden size, a layer count) as well as a missing or extra parameter.
Moments are compared within a band, because the two sides draw from the same
*distribution*, never the same numbers: the RNGs differ. `std` is the
discriminating one — the conv divergence was a 2.45x ratio, dense was 1.73x,
and both sit far outside any sampling noise at these sizes.

Names are reported but NOT compared: the Idris registry derives them from the
init scope (`conv2d_0.weight`) while the reference uses attribute names
(`conv1.weight`). Making those agree is a bigger change than this gate needs.

Usage: scripts/check-init-manifest.py [--only <name>] [--tolerance 0.25]
Exit 0 = aligned, 1 = divergence, 2 = a side failed to dump.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402

# Below this, a sample std says essentially nothing. Shapes are still compared;
# only the moment check is skipped. Kept low because the tolerance widens with
# sample size (see `std_band`) rather than the cutoff doing that job — the
# seq-classify conv1 kernel is 12 elements and still carries a real check.
MIN_ELEMS_FOR_MOMENTS = 8

# Constant-init parameters (zero bias, unit LayerNorm gain) must match exactly;
# there is no sampling involved.
EXACT_TOLERANCE = 1e-12


def std_band(elems: int, tolerance: float) -> float:
    """Allowed |ln(idris_std / python_std)| for a parameter of `elems` values.

    Log space so the test is symmetric: a 2.45x divergence has to fail whichever
    side is the wider one, and a raw ratio check is not symmetric (0.41 and 2.45
    are the same divergence). The sampling term is 3 standard errors of a sample
    std, ~1/sqrt(2n) — that is what lets a 12-element conv kernel still carry a
    meaningful check.
    """
    return math.log1p(tolerance) + 3.0 / math.sqrt(2.0 * elems)


def load_manifest(path: Path) -> list[tuple[str, tuple[int, ...], float, float, float, float]]:
    """(name, shape, mean, std, min, max) per tensor, in file order."""
    from safetensors import safe_open

    out = []
    with safe_open(str(path), framework="pt") as f:  # pyright: ignore[reportUnknownMemberType]
        for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__
            t = f.get_tensor(key).double()
            out.append(
                (
                    key,
                    tuple(t.shape),
                    float(t.mean()),
                    float(t.std()) if t.numel() > 1 else 0.0,
                    float(t.min()),
                    float(t.max()),
                )
            )
    return out


def dump_idris(target: str, out_path: Path) -> str | None:
    """Run the Idris example under IDRISML_DUMP_INIT. Returns an error or None."""
    env = dict(os.environ, IDRISML_DUMP_INIT=str(out_path))
    proc = subprocess.run(
        ["make", target],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if not out_path.exists():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        return f"no dump written by `make {target}`:\n{tail}"
    return None


def dump_python(module: str, out_path: Path) -> str | None:
    env = dict(os.environ, IDRISML_DUMP_INIT=str(out_path))
    proc = subprocess.run(
        ["uv", "run", "python", "-u", "-m", f"torch_ref.scripts.{module}"],
        cwd=REPO_ROOT / "packages" / "pytorch",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if not out_path.exists():
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        return f"no dump written by torch_ref.scripts.{module}:\n{tail}"
    return None


def compare(
    idris: list[tuple[str, tuple[int, ...], float, float, float, float]],
    python: list[tuple[str, tuple[int, ...], float, float, float, float]],
    tolerance: float,
) -> list[str]:
    problems: list[str] = []

    ishapes = [row[1] for row in idris]
    pshapes = [row[1] for row in python]
    if ishapes != pshapes:
        problems.append(f"shape sequence differs\n    idris:  {ishapes}\n    python: {pshapes}")
        return problems  # moment comparison is meaningless once shapes disagree

    # no strict=: lengths are equal by the shape check above, and strict= is
    # unavailable on the macOS system python 3.9 the other gates run under.
    pairs = zip(idris, python)  # noqa: B905
    for (iname, shape, imean, istd, imin, imax), (pname, _, pmean, pstd, pmin, pmax) in pairs:
        elems = 1
        for d in shape:
            elems *= d

        # A constant init (zero bias, unit gain) has zero spread on both sides
        # and must match exactly — no sampling to excuse a difference.
        if istd < EXACT_TOLERANCE and pstd < EXACT_TOLERANCE:
            if abs(imean - pmean) > EXACT_TOLERANCE:
                problems.append(
                    f"{iname} / {pname} {shape}: constant init differs "
                    f"(idris {imean:.6g}, python {pmean:.6g})"
                )
            continue

        if elems < MIN_ELEMS_FOR_MOMENTS:
            continue

        if istd < EXACT_TOLERANCE or pstd < EXACT_TOLERANCE:
            problems.append(
                f"{iname} / {pname} {shape}: one side is constant and the other is not "
                f"(idris std {istd:.6g}, python std {pstd:.6g})"
            )
            continue

        ratio = istd / pstd
        if abs(math.log(ratio)) > std_band(elems, tolerance):
            problems.append(
                f"{iname} / {pname} {shape}: init std ratio {ratio:.3f} "
                f"(idris {istd:.6g}, python {pstd:.6g}); "
                f"range idris [{imin:.4g}, {imax:.4g}] python [{pmin:.4g}, {pmax:.4g}]"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="check a single example by table name")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="allowed |1 - idris_std/python_std| (default 0.25)",
    )
    args = parser.parse_args()

    specs = [s for s in EXAMPLES if args.only in (None, s["name"])]
    if not specs:
        print(f"no example named {args.only!r} in the paired table", file=sys.stderr)
        return 2

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            name = spec["name"]
            module = Path(spec["python"]).stem
            ipath = Path(tmp) / f"{name}-idris.safetensors"
            ppath = Path(tmp) / f"{name}-python.safetensors"

            err = dump_idris(f"example-{name}", ipath) or dump_python(module, ppath)
            if err:
                print(f"{name:<20} [DUMP FAILED] {err}", file=sys.stderr)
                failed = True
                continue

            problems = compare(load_manifest(ipath), load_manifest(ppath), args.tolerance)
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
            "Init diverges. The two sides draw from different distributions, so any",
            file=sys.stderr,
        )
        print(
            "comparison between them measures init noise rather than the",
            file=sys.stderr,
        )
        print(
            "implementation. See reference-alignment.md for the agreed contracts.",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(specs)} paired examples build matching shapes and init.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
