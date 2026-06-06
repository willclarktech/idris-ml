"""Hyperparameter-sweep grid expansion and result-table assembly.

Used by `scripts/sweep.py`; designed to be reusable for any future
grid-style harness:

  - `expand_grid(spec)`     Cartesian product over a `{flag: [values]}`
                            map → list of OrderedDict configs.
  - `random_sample(configs, n, rng=...)`  Uniform-without-replacement
                                          subsample.
  - `parse_result_line(line)`  Parse `RESULT\\tk=v\\tk=v` (with int /
                               float / str coercion). Re-exported from
                               `mltools.perf_log.parse_result` so we
                               keep a single source of truth.
  - `to_csv(configs_with_results, grid_keys, result_keys)`
                            Emit a header + one row per config; missing
                            result cells are empty strings.

The grid spec format is the one consumed by `scripts/sweep.py`:

    {
      "name": "ntm-copy",
      "exec": "ntm-copy",
      "src":  "packages/idris-ml-examples/src/Example/NtmCopy.idr",
      "fixed_flags": ["--alpha", "0.95"],
      "grid": {"--lr": [0.0001, 0.001], "--seed": [1, 42]}
    }
"""

from __future__ import annotations

import csv
import io
import random
from typing import Iterable, Optional

# Re-export parse_result from perf_log so callers see one canonical
# function. Both perf-log entries and sweep RESULT lines use the same
# format.
from mltools.perf_log import parse_result as parse_result_line  # noqa: F401


def expand_grid(grid: dict[str, list]) -> list[dict[str, str]]:
    """Cartesian product over the values of each grid key.

    Preserves key order (Python 3.7+ dict ordering). Values are
    stringified so they can be passed straight on the CLI.

    Returns a list of dicts, one per config: `{flag: stringified value}`.
    """
    keys = list(grid.keys())
    if not keys:
        return [{}]
    # Stringify values up-front; the runner only cares about CLI form.
    value_lists = [[str(v) for v in grid[k]] for k in keys]
    out: list[dict[str, str]] = [{}]
    for k, vs in zip(keys, value_lists):
        out = [{**cfg, k: v} for cfg in out for v in vs]
    return out


def random_sample(
    configs: list[dict[str, str]],
    n: int,
    rng: Optional[random.Random] = None,
) -> list[dict[str, str]]:
    """Uniform without-replacement subsample, deterministic if `rng`
    is supplied."""
    if n >= len(configs):
        return configs
    r = rng if rng is not None else random.Random()
    return r.sample(configs, n)


def config_to_cli_args(cfg: dict[str, str]) -> list[str]:
    """Flatten {flag: value} → ["flag", "value", ...] for subprocess."""
    out: list[str] = []
    for k, v in cfg.items():
        out.append(k)
        out.append(v)
    return out


def to_csv(
    configs_with_results: Iterable[tuple[dict[str, str], dict[str, object]]],
    grid_keys: list[str],
    result_keys: list[str],
) -> str:
    """Build a CSV body: header = grid keys (without `--`) + result keys.

    Each row's grid cells use the config's value for that key; result
    cells use the RESULT dict's value or empty string when missing.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    header = [k.lstrip("-") for k in grid_keys] + result_keys
    w.writerow(header)
    for cfg, result in configs_with_results:
        row = [cfg.get(k, "") for k in grid_keys]
        for rk in result_keys:
            v = result.get(rk, "")
            row.append("" if v == "" else str(v))
        w.writerow(row)
    return buf.getvalue()


def first_nonempty_result_keys(
    configs_with_results: Iterable[tuple[dict[str, str], dict[str, object]]]
) -> list[str]:
    """Mimics the bash version's "sample the first RESULT line's keys".

    The bash sweep took the first non-empty RESULT line and used its
    key set as the CSV header for the result columns. Replicated here
    so the CSV column set is stable across re-runs that hit the same
    schema.
    """
    for _, result in configs_with_results:
        if result:
            return list(result.keys())
    return []
