#!/usr/bin/env python3
"""Gate: examples must not construct environment states by hand.

An RL example resets its environment in two places — the rollout's auto-reset
when an episode ends, and the start of each evaluation episode. Both must go
through the `Env` interface (`reset`, `Gym.Vector.resetAll`,
`Gym.Vector.stepAutoReset`), which draws from the env's own start
distribution.

Writing the state literal instead pins every episode to one point. That is
invisible in the metrics: a greedy policy evaluated from a fixed start replays
one trajectory N times and reports it as an N-episode mean. Every Idris
deep-RL example did this (2026-08-01) while every PyTorch reference called
`env.reset()`, so the two sides had been measuring different tasks — Pendulum
from the downward equilibrium against Pendulum from a uniformly random angle,
CartPole from dead centre against CartPole from U(-0.05, 0.05)^4.

The forbidden constructor list is read out of idris-gym rather than written
here, so a new environment is covered the day it lands.

Run: python3 scripts/check-env-reset-literals.py
Exit 0 = clean, 1 = a literal state, 2 = no constructors found (bad glob).
"""

from __future__ import annotations

import glob
import os
import re
import sys

ENV_SOURCE_GLOB = "packages/idris-gym/src/Gym/**/*.idr"
EXAMPLES_GLOB = "packages/idris-ml-examples/src/**/*.idr"
EXAMPLES_ROOT = "packages/idris-ml-examples/src"

# Files allowed to name a state constructor directly. Add an entry only with a
# reason, and prefer a helper in idris-gym instead.
EXEMPT: set[str] = {
    # Sequential-vs-batched rollout parity: the two paths must start from the
    # same state for the comparison to mean anything, so a literal is what the
    # test is for. It is not an episode reset.
    "Test/Reinforce.idr",
}

CONSTRUCTOR_RE = re.compile(r"^\s*constructor\s+(Mk\w+)\s*$")
COMMENT_RE = re.compile(r"^\s*(--|\|\|\|)")


def env_state_constructors() -> set[str]:
    """Every `constructor MkX` declared under Gym/ClassicControl.

    Those records are the env states: CartPole's `MkCP`, Pendulum's `MkP`, and
    so on. `Gym.Vector`'s `MkVecEnv` is a container, not a state, and lives
    outside that directory, so it is not swept up.
    """
    found: set[str] = set()
    for path in glob.glob(ENV_SOURCE_GLOB, recursive=True):
        if "ClassicControl" not in path:
            continue
        with open(path) as handle:
            for line in handle:
                match = CONSTRUCTOR_RE.match(line)
                if match:
                    found.add(match.group(1))
    return found


def main() -> int:
    constructors = env_state_constructors()
    if not constructors:
        print("no env state constructors found — check ENV_SOURCE_GLOB", file=sys.stderr)
        return 2

    # \b on both sides so MkA does not match MkA2C, and MkMC does not match MkMCC.
    pattern = re.compile(r"\b(" + "|".join(sorted(constructors)) + r")\b")

    failures: list[str] = []
    for path in sorted(glob.glob(EXAMPLES_GLOB, recursive=True)):
        rel = os.path.relpath(path, EXAMPLES_ROOT)
        if rel in EXEMPT:
            continue
        with open(path) as handle:
            for lineno, line in enumerate(handle, start=1):
                if COMMENT_RE.match(line):
                    continue
                match = pattern.search(line)
                if match:
                    failures.append(f"{rel}:{lineno}: {match.group(1)} — {line.strip()}")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Reset through the Env interface instead: `reset` for a single env,",
            file=sys.stderr,
        )
        print(
            "`Gym.Vector.resetAll` for a VecEnv, `Gym.Vector.stepAutoReset` to step",
            file=sys.stderr,
        )
        print(
            "and reset the terminated envs in one pass. Each threads a Seed and draws",
            file=sys.stderr,
        )
        print("from the environment's own start distribution.", file=sys.stderr)
        return 1

    names = ", ".join(sorted(constructors))
    print(f"env-reset-literals OK (no example constructs {names})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
