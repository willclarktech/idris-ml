#!/usr/bin/env python3
# The safetensors/torch handles below are untyped third-party objects, and
# scripts/pyrightconfig.json covers an otherwise stdlib-only tree with no stubs
# for them. Silence only the unknown-type family here rather than drop the file
# out of strict mode — every other rule still applies.
# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
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

Parameters pair by NAME, through the `params` map in `paired_examples.py`. The
two sides name things differently and neither is wrong — Idris derives names
from the init scope (`conv1d_0.weight`), PyTorch from the attribute you
assigned (`conv1.weight`) — so the correspondence is a property of the pairing,
not of either side, and it lives with the other pairing facts. The map is
verified, not trusted: every Idris name a key exactly once, every reference
name a value exactly once, shapes equal per pair. That makes it strictly
stronger than matching sorted shapes, which cannot see two same-shaped layers
swapped.

`--propose` prints a candidate map for an example that has none yet, pairing
unique shapes directly and same-shaped parameters in registration order. Review
it before pasting: that ordering is a guess, and it is exactly the guess this
map exists to replace.

Usage: scripts/check-init-manifest.py [--only <name>] [--propose] [--tolerance 0.25]
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
from typing import Any, cast

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

    out: list[tuple[str, tuple[int, ...], float, float, float, float]] = []
    # safetensors ships no py.typed marker, so everything reached through the
    # handle is Unknown under pyright strict. Cast once at the boundary rather
    # than annotate every access.
    with safe_open(str(path), framework="pt") as raw:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        handle: Any = raw
        for key in cast("list[str]", handle.keys()):  # noqa: SIM118 — no __iter__
            t = cast("Any", handle.get_tensor(key)).double()
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


def propose_map(
    idris: list[tuple[str, tuple[int, ...], float, float, float, float]],
    python: list[tuple[str, tuple[int, ...], float, float, float, float]],
) -> tuple[dict[str, str], list[str]]:
    """Candidate {idris_name: reference_name}, plus warnings about guesses.

    Pairs on (leaf name, shape) first: both sides follow the PyTorch
    `state_dict` leaf convention, so `lstm_0.bias_ih` and `0.lstm.bias_ih`
    agree on `bias_ih` even though the prefixes never will. Only where that
    leaves several candidates does it fall back to registration order, and
    every such pair is flagged — that ordering is exactly the guess this map
    exists to replace.
    """

    def leaf(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    def skeleton(name: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """(kind tokens, index sequence), with the dump's model-index prefix dropped.

        `block_0.attn_0.key_2.weight` and `0.blocks.0.key_ws.2.weight` both
        reduce to kinds ('block','attn','key','weight') / ('blocks','key_ws',
        'weight') with indices (0, 0, 2). Indices are what actually
        disambiguate the per-head projections; kinds are compared by prefix
        because the two sides pluralise differently.
        """
        parts = name.split(".")
        if parts and parts[0].isdigit():
            parts = parts[1:]
        kinds: list[str] = []
        idxs: list[int] = []
        for part in parts:
            if part.isdigit():
                idxs.append(int(part))
                continue
            head, sep, tail = part.rpartition("_")
            if sep and tail.isdigit():
                kinds.append(head)
                idxs.append(int(tail))
            else:
                kinds.append(part)
        return tuple(kinds), tuple(idxs)

    def kinds_compatible(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        """Same number of kind tokens, each a prefix of its counterpart."""
        if len(a) != len(b):
            return False
        return all(x.startswith(y) or y.startswith(x) for x, y in zip(a, b))  # noqa: B905

    buckets: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    by_shape: dict[tuple[int, ...], list[str]] = {}
    for name, shape, *_ in python:
        buckets.setdefault((leaf(name), shape), []).append(name)
        by_shape.setdefault(shape, []).append(name)

    mapping: dict[str, str] = {}
    warnings: list[str] = []
    taken: set[str] = set()
    for name, shape, *_ in idris:
        key = (leaf(name), shape)
        pool = [c for c in buckets.get(key, []) if c not in taken]
        ikinds, iidx = skeleton(name)
        exact = [
            c for c in pool if skeleton(c)[1] == iidx and kinds_compatible(skeleton(c)[0], ikinds)
        ]
        if exact:
            chosen = exact[0]
            if len(exact) > 1:
                warnings.append(
                    f"{name} -> {chosen} {shape}: {len(exact)} share leaf+shape+indices"
                )
        elif pool:
            chosen = pool[0]
            if len(pool) > 1:
                warnings.append(f"{name} -> {chosen} {shape}: {len(pool)} share this leaf+shape")
        else:
            fallback = [c for c in by_shape.get(shape, []) if c not in taken]
            if not fallback:
                warnings.append(f"{name} {shape}: no reference parameter left of this shape")
                continue
            chosen = fallback[0]
            warnings.append(f"{name} -> {chosen} {shape}: shape-only guess, leaf names differ")
        taken.add(chosen)
        mapping[name] = chosen

    for name, shape, *_ in python:
        if name not in taken:
            warnings.append(f"reference {name} {shape}: unmatched")
    return mapping, warnings


def check_map(
    mapping: dict[str, str],
    idris: list[tuple[str, tuple[int, ...], float, float, float, float]],
    python: list[tuple[str, tuple[int, ...], float, float, float, float]],
) -> list[str]:
    """Verify the params map is a shape-consistent bijection. Never trust it.

    A map that silently rots is the failure mode this whole gate exists to
    kill, so every way it can be wrong is an error: a parameter it forgets, one
    it invents, two Idris names claiming the same reference parameter, or a
    pair whose shapes disagree.
    """
    problems: list[str] = []
    ishapes = {name: shape for name, shape, *_ in idris}
    pshapes = {name: shape for name, shape, *_ in python}

    for missing in sorted(set(ishapes) - set(mapping)):
        problems.append(f"params map has no entry for idris {missing} {ishapes[missing]}")
    for extra in sorted(set(mapping) - set(ishapes)):
        problems.append(f"params map names idris {extra}, which the model does not register")

    seen: dict[str, str] = {}
    for iname, pname in sorted(mapping.items()):
        if pname in seen:
            problems.append(f"params map points {seen[pname]} and {iname} at the same {pname}")
        seen[pname] = iname
        if pname not in pshapes:
            problems.append(f"params map names reference {pname}, which the model does not have")
            continue
        if iname in ishapes and ishapes[iname] != pshapes[pname]:
            problems.append(
                f"{iname} {ishapes[iname]} maps to {pname} {pshapes[pname]}: shapes differ"
            )
    for unmapped in sorted(set(pshapes) - set(mapping.values())):
        problems.append(f"reference {unmapped} {pshapes[unmapped]} is not in the params map")
    return problems


def compare(
    mapping: dict[str, str],
    idris: list[tuple[str, tuple[int, ...], float, float, float, float]],
    python: list[tuple[str, tuple[int, ...], float, float, float, float]],
    tolerance: float,
) -> list[str]:
    problems = check_map(mapping, idris, python)
    if problems:
        return problems  # comparing moments through a broken map says nothing

    irows = {row[0]: row for row in idris}
    prows = {row[0]: row for row in python}
    idris = [irows[i] for i in mapping]
    python = [prows[mapping[i]] for i in mapping]

    # no strict=: lengths are equal by the shape check above, and strict= is
    # unavailable on the macOS system python 3.9 the other gates run under.
    pairs = zip(idris, python)  # noqa: B905
    for (iname, shape, imean, istd, imin, imax), (pname, _, pmean, pstd, pmin, pmax) in pairs:
        elems = 1
        for d in shape:
            elems *= d

        # A single value has no spread to measure, so `std` is 0 by definition
        # and says nothing about the distribution it came from. All that can be
        # checked is zero-vs-nonzero: a bias the reference draws at random
        # against one Idris pins to zero IS a divergence, two different random
        # draws are not.
        if elems == 1:
            if (imean == 0.0) != (pmean == 0.0):
                problems.append(
                    f"{iname} / {pname} {shape}: one side inits this to zero and the "
                    f"other draws it (idris {imean:.6g}, python {pmean:.6g})"
                )
            continue

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
        "--propose",
        action="store_true",
        help="print a candidate params map instead of checking (review before pasting)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="allowed |1 - idris_std/python_std| (default 0.25)",
    )
    args = parser.parse_args()

    specs = [s for s in EXAMPLES if args.only in (None, s["name"]) and s.get("init_manifest", True)]
    skipped = [s["name"] for s in EXAMPLES if not s.get("init_manifest", True)]
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

            target = spec.get("target", f"example-{name}")
            err = dump_idris(target, ipath) or dump_python(module, ppath)
            if err:
                print(f"{name:<20} [DUMP FAILED] {err}", file=sys.stderr)
                failed = True
                continue

            imanifest, pmanifest = load_manifest(ipath), load_manifest(ppath)

            if args.propose:
                mapping, warnings = propose_map(imanifest, pmanifest)
                print(f"        # {name}")
                print('        "params": {')
                for k, v in mapping.items():
                    print(f'            "{k}": "{v}",')
                print("        },")
                for warning in warnings:
                    print(f"        # REVIEW: {warning}")
                continue

            problems = compare(spec.get("params", {}), imanifest, pmanifest, args.tolerance)
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

    if skipped and not args.only:
        print(f"\nskipped (no parameters to compare): {', '.join(skipped)}")
    print(f"All {len(specs)} paired examples build matching shapes and init.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
