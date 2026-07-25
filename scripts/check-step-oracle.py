#!/usr/bin/env python3
"""Gate: paired examples compute the same step from the same starting point.

The other four gates compare *descriptions* — shapes, moments, flag defaults,
metric keys. All of them pass while the two sides compute different things, and
that has happened repeatedly here: dropout left on during inference, a
`log_softmax` applied to an already-softmaxed output, an optimizer that never
updated one of two networks. Each was found by a human, late.

This one compares arithmetic. Three runs:

  1. Idris dumps its init weights and its first batch to one file.
  2. The reference loads both, takes exactly one optimizer step, and dumps.
  3. Idris takes exactly one step on the same batch and dumps.

Both sides started from identical numbers on identical inputs, so the
post-step parameters must agree to floating-point round-off. Any difference is
forward, backward or optimizer semantics.

Post-step weights rather than gradients: it needs no hook between `backward`
and the step, and it covers the optimizer too. Weights flow Idris -> reference
for the transfer, which is the direction that needs no rename machinery on the
Idris side; once both sides hold the same numbers it does not matter who
authored them.

Usage: scripts/check-step-oracle.py [--only <name>] [--tolerance 1e-9]
Exit 0 = the step agrees, 1 = divergence, 2 = a run failed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402


def run(cmd: list[str], cwd: Path, env_extra: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=dict(os.environ, **env_extra),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def tail(text: str, n: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def compare_step(
    idris_path: Path, ref_path: Path, params: dict[str, str], tolerance: float
) -> list[str]:
    from safetensors import safe_open

    problems: list[str] = []
    with (
        safe_open(str(idris_path), framework="pt") as i_raw,  # pyright: ignore[reportUnknownMemberType]
        safe_open(str(ref_path), framework="pt") as r_raw,  # pyright: ignore[reportUnknownMemberType]
    ):
        i_handle: Any = i_raw
        r_handle: Any = r_raw
        for idris_name, ref_name in params.items():
            a = i_handle.get_tensor(idris_name).double()
            # The oracle dump prefixes by model index; strip it.
            b = r_handle.get_tensor(ref_name.split(".", 1)[1]).double()
            if tuple(a.shape) != tuple(b.shape):
                problems.append(f"{idris_name}: {tuple(a.shape)} vs {ref_name} {tuple(b.shape)}")
                continue
            diff = float((a - b).abs().max())
            scale = max(1.0, float(a.abs().max()))
            if diff / scale > tolerance:
                problems.append(
                    f"{idris_name} vs {ref_name}: max|diff| = {diff:.3e} "
                    f"after one step (relative {diff / scale:.3e})"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="check a single example by table name")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    specs = [s for s in EXAMPLES if args.only in (None, s["name"]) and s.get("step_oracle", False)]
    if not specs:
        print(f"no step-oracle example matched {args.only!r}", file=sys.stderr)
        return 2

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            name = spec["name"]
            target = spec.get("target", f"example-{name}")
            module = Path(spec["python"]).stem
            oracle = Path(tmp) / f"{name}-oracle.safetensors"
            ref_after = Path(tmp) / f"{name}-ref-after.safetensors"
            idris_after = Path(tmp) / f"{name}-idris-after.safetensors"

            rc, out = run(["make", target], REPO_ROOT, {"IDRISML_ORACLE_DUMP": str(oracle)})
            if not oracle.exists():
                print(f"{name:<20} [FAILED] idris did not dump the oracle\n{tail(out)}")
                failed = True
                continue

            rc, out = run(
                ["uv", "run", "python", "-u", "-m", f"torch_ref.scripts.{module}"],
                REPO_ROOT / "packages" / "pytorch",
                {"IDRISML_ORACLE_LOAD": str(oracle), "IDRISML_ORACLE_STEP": str(ref_after)},
            )
            if not ref_after.exists():
                print(f"{name:<20} [FAILED] reference did not take the oracle step\n{tail(out)}")
                failed = True
                continue

            rc, out = run(["make", target], REPO_ROOT, {"IDRISML_ONE_STEP": str(idris_after)})
            if not idris_after.exists():
                print(f"{name:<20} [FAILED] idris did not dump after one step\n{tail(out)}")
                failed = True
                continue

            problems = compare_step(
                idris_after,
                ref_after,
                cast("dict[str, str]", spec.get("params", {})),
                args.tolerance,
            )
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
            "One step diverges. Both sides started from the same weights on the",
            file=sys.stderr,
        )
        print(
            "same batch, so the difference is in the forward, the backward, or",
            file=sys.stderr,
        )
        print("the optimizer.", file=sys.stderr)
        return 1

    print(f"\nAll {len(specs)} step-oracle examples compute the same step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
