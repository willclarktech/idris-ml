#!/usr/bin/env python3
# The safetensors/torch handles below are untyped third-party objects, and
# scripts/pyrightconfig.json covers an otherwise stdlib-only tree with no stubs
# for them. Silence only the unknown-type family here rather than drop the file
# out of strict mode — every other rule still applies.
# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Gate: paired examples compute the same step from the same starting point.

The other four gates compare *descriptions* — shapes, moments, flag defaults,
metric keys. All of them pass while the two sides compute different things, and
that has happened repeatedly here: dropout left on during inference, a
`log_softmax` applied to an already-softmaxed output, an optimizer that never
updated one of two networks. Each was found by a human, late.

This one compares arithmetic. Two runs:

  1. The reference runs its own code path unmodified, writing the fixture it
     started from — its parameters, plus (for RNG-driven examples) every draw
     it made, recorded to a `.replay` sidecar — and then its own post-step
     parameters.
  2. Idris loads that fixture, replays the recorded draws through the
     example's own --replay flag instead of sampling, takes one step and
     dumps.

Both sides started from identical numbers on identical inputs, so the
post-step parameters must agree to floating-point round-off. Any difference is
forward, backward or optimizer semantics — and for the RL examples the
regenerated rollout additionally covers the env dynamics and the value
channel, since Idris recomputes both from the replayed draws.

The fixture travels reference -> Idris, and is keyed by Idris registry names
so `Ml.Checkpoint.load` needs no remap. That direction matters: the reference
is the ground truth and the thing written first when a new architecture lands,
so it should never be bent to consume the implementation's output. It also
removes the need for the two sides' RNGs to agree — replaying the reference's
draws gets identical inputs without requiring the generators, the consumption
order and the sampling algorithms to match (recorded *decisions* for the
categorical channel; this library's inverse CDF and torch.multinomial would
never map the same uniform to the same index).

Post-step weights rather than gradients: it needs no hook between `backward`
and the step, and it covers the optimizer too.

Bit-identity is not the bar and could not be: the two sides run different
sequences of floating-point operations for the same expression (FMA
contraction, reduction order, libm against torch's vectorised transcendentals),
so agreement is to a few ULP rather than exactly. Most parameters do come out
bit-identical; the ones that do not sit at 1e-19 to 1e-18 on the SGD models.
a2c is eight orders looser because global-norm clipping turns one scalar
derived from every gradient into a uniform relative error, and Adam's first
step divides by `sqrt(v) + 1e-8`, which is ill-conditioned where a gradient is
comparable to that epsilon.

Each example therefore carries its own `tolerance` in `paired_examples.py`,
measured with `--tolerance 0` (which prints the exact difference on every
parameter) and left with ~1000x headroom for platform variation. A single
loose bound would waste most of the sensitivity the clean models have.

Usage: scripts/check-step-oracle.py [--only <name>] [--tolerance <float>]
Exit 0 = the step agrees, 1 = divergence, 2 = a run failed.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_examples import EXAMPLES, REPO_ROOT  # noqa: E402

if TYPE_CHECKING:
    from paired_examples import ExampleSpec  # noqa: F401


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
        for idris_name in params:
            # Both post-step dumps are keyed by Idris registry name — the
            # reference translates when it writes, so nothing maps here.
            a = i_handle.get_tensor(idris_name).double()
            b = r_handle.get_tensor(idris_name).double()
            ref_name = params[idris_name]
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


def check_params_mirror(spec: ExampleSpec) -> list[str]:
    """The reference script keeps its own PAIRED_PARAMS copy (it runs from the
    pytorch package, which cannot import this directory). Read it back and
    compare, so a rename in the table cannot silently desync the oracle into
    comparing nothing."""
    module_path = REPO_ROOT / spec["python"]
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "PAIRED_PARAMS" for t in node.targets):
            continue
        script_params = cast("dict[str, str]", ast.literal_eval(node.value))
        table_params = spec.get("params", {})
        if script_params != table_params:
            missing = sorted(set(table_params) - set(script_params))
            extra = sorted(set(script_params) - set(table_params))
            changed = sorted(
                k
                for k in set(table_params) & set(script_params)
                if table_params[k] != script_params[k]
            )
            return [
                f"{module_path.name} PAIRED_PARAMS differs from paired_examples.py: "
                f"missing={missing} extra={extra} changed={changed}"
            ]
        return []
    return [f"{module_path.name} defines no PAIRED_PARAMS, so the oracle cannot load weights"]


DEFAULT_TOLERANCE = 1e-9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="check a single example by table name")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=f"override every example's tolerance (default: per-example, else "
        f"{DEFAULT_TOLERANCE:g}). Pass 0 to print the exact difference on every "
        f"parameter, which is how the per-example values are measured.",
    )
    args = parser.parse_args()

    specs = [s for s in EXAMPLES if args.only in (None, s["name"]) and s.get("step_oracle", False)]
    if not specs:
        print(f"no step-oracle example matched {args.only!r}", file=sys.stderr)
        return 2

    mirror_problems = [p for s in specs for p in check_params_mirror(s)]
    if mirror_problems:
        for problem in mirror_problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            name = spec["name"]
            target = spec.get("target", f"example-{name}")
            module = Path(spec["python"]).stem
            oracle = Path(tmp) / f"{name}-oracle.safetensors"
            ref_after = Path(tmp) / f"{name}-ref-after.safetensors"
            idris_after = Path(tmp) / f"{name}-idris-after.safetensors"

            # Some examples need a non-default config for their first epoch to
            # exercise the update at all (sac: no warmup, batch small enough
            # to fill from one lockstep step). Applied to BOTH sides, so the
            # compared run is still one experiment.
            oracle_args = [str(a) for a in spec.get("oracle_args", [])]

            # One reference run writes both halves: the fixture it started
            # from (parameters plus every random input it drew, keyed for the
            # Idris registry) and its own post-step parameters.
            _rc, out = run(
                ["uv", "run", "python", "-u", "-m", f"torch_ref.scripts.{module}", *oracle_args],
                REPO_ROOT / "packages" / "pytorch",
                {"IDRISML_ORACLE_DUMP": str(oracle), "IDRISML_ORACLE_STEP": str(ref_after)},
            )
            if not oracle.exists():
                print(f"{name:<20} [FAILED] reference did not write the fixture\n{tail(out)}")
                failed = True
                continue
            if not ref_after.exists():
                print(f"{name:<20} [FAILED] reference did not take the oracle step\n{tail(out)}")
                failed = True
                continue

            # Idris replays that fixture rather than generating its own.
            make_cmd = ["make", target]
            if spec.get("replay", False):
                # The reference wrote its recorded draws beside the fixture;
                # hand them to the example through its own --replay flag
                # (the per-example make args var, e.g. A2C_ARGS).
                replay_path = Path(f"{oracle}.replay")
                if not replay_path.exists():
                    print(f"{name:<20} [FAILED] reference did not write {replay_path.name}")
                    failed = True
                    continue
                # The recipe's args variable is named after the make target
                # (which can differ from the table name — ntm-recall builds
                # example-ntm-associative-recall).
                args_var = target.removeprefix("example-").upper().replace("-", "_") + "_ARGS"
                extra = (" " + " ".join(oracle_args)) if oracle_args else ""
                make_cmd.append(f"{args_var}=--replay {replay_path}{extra}")
            _rc, out = run(
                make_cmd,
                REPO_ROOT,
                {"IDRISML_ORACLE_LOAD": str(oracle), "IDRISML_ONE_STEP": str(idris_after)},
            )
            if not idris_after.exists():
                print(f"{name:<20} [FAILED] idris did not dump after one step\n{tail(out)}")
                failed = True
                continue

            tolerance = (
                args.tolerance
                if args.tolerance is not None
                else spec.get("tolerance", DEFAULT_TOLERANCE)
            )
            problems = compare_step(
                idris_after,
                ref_after,
                spec.get("params", {}),
                tolerance,
            )
            if problems:
                failed = True
                print(f"{name:<20} [{len(problems)} divergence(s) at tol {tolerance:g}]")
                for problem in problems:
                    print(f"    {problem}")
            else:
                print(f"{name:<20} [OK at tol {tolerance:g}]")

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
