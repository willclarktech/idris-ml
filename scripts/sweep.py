#!/usr/bin/env python3
"""Generic hyperparameter grid / random search.

Reads a sweep spec from a JSON file and runs Cartesian-product (or
randomly-sampled) configs against the named example, capturing each
run's RESULT line into a CSV.

Usage:
    python3 scripts/sweep.py --grid <spec.json> [--parallel N]
                              [--epochs N] [--patience N]
                              [--random N] [--skip-build]

Convenience entry points (load scripts/sweeps/<task>.json):
    python3 scripts/sweep.py --task copy [...]
    python3 scripts/sweep.py --task recall [...]
    python3 scripts/sweep.py --task lstm [...]

Spec format (JSON):
    {
      "name":        "ntm-copy",
      "src":         "packages/idris-ml-examples/src/Example/NtmCopy.idr",
      "exec":        "ntm-copy",
      "fixed_flags": ["--alpha", "0.95"],
      "grid": {
        "--lr":    [0.0001, 0.0003, 0.001],
        "--batch": [4, 16],
        "--seed":  [1, 2, 42]
      }
    }

Behavior:
  * Cartesian product of grid values, optionally subsampled with
    `--random N` (uniform without replacement, deterministic if seeded
    externally via PYTHONHASHSEED is unrelated; pass `--seed` to control
    sampler).
  * `--epochs` and `--patience` are global flags (passed to every run);
    leave the spec's grid free for the actual hyperparameters.
  * RESULT lines are converted CSV-style: header = grid keys (without
    `--`) + first non-empty RESULT's keys. Configs that crash / time
    out emit a row with the grid values + empty result cells, so
    failures don't drop.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.sweep_grid import (  # noqa: E402
    config_to_cli_args,
    expand_grid,
    first_nonempty_result_keys,
    parse_result_line,
    random_sample,
    to_csv,
)


def _setup_idris_paths() -> None:
    """Mirror Makefile's idris2 package-path setup so this script works
    without `make` being involved."""
    idris2_local = os.environ.get("IDRIS2_LOCAL")
    if not idris2_local:
        idris2_local = str(ROOT / ".idris2")
    try:
        out = subprocess.run(
            ["idris2", "--paths"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except FileNotFoundError:
        out = ""
    m = re.search(r'Installation Prefix\s*=\s*"([^"]+)"', out)
    sys_prefix = m.group(1) if m else ""
    if sys_prefix:
        os.environ["IDRIS2_PACKAGE_PATH"] = f"{idris2_local}/idris2-0.8.0:{sys_prefix}/idris2-0.8.0"
    else:
        os.environ["IDRIS2_PACKAGE_PATH"] = f"{idris2_local}/idris2-0.8.0"


def _resolve_grid_path(grid: str | None, task: str | None) -> Path:
    if grid:
        return Path(grid)
    if task:
        return ROOT / "scripts" / "sweeps" / f"{task}.json"
    raise SystemExit("Error: must pass --grid <spec.json> or --task <name>")


def _build_idris(exec_name: str, src: str) -> None:
    print(f"Building {exec_name} from {src}...")
    subprocess.run(
        [
            "idris2",
            "--source-dir",
            "packages/idris-ml-examples/src",
            "-p",
            "contrib",
            "-p",
            "idris-ml",
            "-p",
            "idris-gym",
            "-p",
            "idris-ml-examples",
            "-o",
            exec_name,
            src,
        ],
        check=True,
    )


def _stage_dylib(exec_name: str) -> None:
    candidates = sorted(Path("build").glob("libidrisml.dylib")) + sorted(
        Path("build").glob("libidrisml*.dylib")
    )
    dylib = candidates[0] if candidates else None
    app_dir = Path(f"build/exec/{exec_name}_app")
    if dylib and app_dir.is_dir():
        shutil.copy(dylib, app_dir / dylib.name)


def _run_one(
    args: tuple[str, dict[str, str], int, int, list[str], str],
) -> tuple[dict[str, str], dict[str, object]]:
    """Worker: run the binary for one config; return (cfg, parsed RESULT).

    Args packed as a tuple to be `ProcessPoolExecutor`-friendly.
    """
    exe, cfg, epochs, patience, fixed_flags, tmpdir = args
    flags = config_to_cli_args(cfg)
    cmd = [exe, *flags, "--epochs", str(epochs), "--patience", str(patience), *fixed_flags]
    tag = "_".join(f"{k.lstrip('-')}={v}" for k, v in cfg.items())
    out_path = Path(tmpdir) / f"{tag}.out"
    with out_path.open("w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    result_line = ""
    for line in out_path.read_text().splitlines():
        if line.startswith("RESULT"):
            result_line = line
            break
    return cfg, parse_result_line(result_line) if result_line else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--quick", action="store_true", help="Shortcut for --epochs 2000.")
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--random", type=int, default=None, dest="random_n")
    p.add_argument("--grid", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--seed", type=int, default=None, help="Sampler seed (only used with --random).")
    args = p.parse_args(argv)
    if args.quick:
        args.epochs = 2000

    os.chdir(ROOT)
    _setup_idris_paths()

    grid_path = _resolve_grid_path(args.grid, args.task)
    if not grid_path.is_file():
        raise SystemExit(f"Error: grid spec not found at {grid_path}")
    spec = json.loads(grid_path.read_text())

    name = spec["name"]
    src = spec["src"]
    exec_name = spec["exec"]
    fixed_flags: list[str] = list(spec.get("fixed_flags", []))
    grid: dict[str, list] = spec["grid"]

    exe = f"./build/exec/{exec_name}"
    if not args.skip_build:
        _build_idris(exec_name, src)
    if not os.access(exe, os.X_OK):
        raise SystemExit(f"Error: {exe} not found. Run without --skip-build.")
    _stage_dylib(exec_name)

    configs = expand_grid(grid)
    total = len(configs)
    if args.random_n is not None and args.random_n < total:
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        configs = random_sample(configs, args.random_n, rng=rng)
        total = args.random_n

    Path("results").mkdir(exist_ok=True)
    results_file = Path("results") / f"sweep-{name}.csv"

    print(
        f"Running {total} configs with {args.parallel} parallel jobs "
        f"(epochs={args.epochs}, patience={args.patience})..."
    )
    print()

    with tempfile.TemporaryDirectory(prefix="sweep-") as tmpdir:
        work = [(exe, cfg, args.epochs, args.patience, fixed_flags, tmpdir) for cfg in configs]
        configs_with_results: list[tuple[dict[str, str], dict[str, object]]] = []
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(_run_one, w) for w in work]
            for fut in as_completed(futures):
                configs_with_results.append(fut.result())

    # Preserve grid order in the output (re-sort by enumerating the
    # original cartesian-product order).
    cfg_to_idx = {tuple(sorted(c.items())): i for i, c in enumerate(configs)}
    configs_with_results.sort(key=lambda pair: cfg_to_idx.get(tuple(sorted(pair[0].items())), 0))

    grid_keys = list(grid.keys())
    result_keys = first_nonempty_result_keys(configs_with_results)
    body = to_csv(configs_with_results, grid_keys, result_keys)
    results_file.write_text(body)

    print()
    print(f"Wrote {total} configs to {results_file}")
    print()

    # Pretty-print with `column` (if available) — purely cosmetic.
    column = shutil.which("column")
    if column:
        subprocess.run([column, "-t", "-s,", str(results_file)], check=False)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
