#!/usr/bin/env python3
"""Ratchet gate: prim__ usage in examples must not grow.

Examples should construct tensors through the typed facade
(tensor/param x InitSpec, bulkToTensor*, tparam*) rather than raw
prim__ FFI calls. The per-file baseline below was captured at the
gate's introduction (2026-06-12, 53 occurrences); the example-migration sweep drives
it to zero, at which point the baseline collapses to {}.

Rules:
  - a file may not exceed its baseline count
  - a file absent from the baseline may not use prim__ at all
  - counts below baseline are reported as ratchet opportunities
    (update the baseline downward in the same change)
  - Example/BringYourOwn.idr is fully exempt: it IS the
    bring-your-own-backend recipe, and raw prims are its subject

Run: python3 scripts/check-prim-in-examples.py
"""

import glob
import os
import sys

EXAMPLES_GLOB = "packages/idris-ml-examples/src/**/*.idr"
EXEMPT = {"Example/BringYourOwn.idr"}

# file (relative to packages/idris-ml-examples/src) -> max allowed prim__ count
BASELINE = {
    "Example/BertClassifySst2Finetune.idr": 2,
    "Example/BertClassifySst2Lora.idr": 2,
    "Example/BertMlmFinetune.idr": 2,
    "Example/Gpt.idr": 7,
    "Example/Gpt2LmFinetune.idr": 2,
    "Example/HfBitNetInference.idr": 1,
    "Example/MatmulBench.idr": 3,
    "Example/PrecisionDemo.idr": 3,
    "Example/RankBroadcastBench.idr": 7,
    "Example/Transfer.idr": 4,
    "Example/Transformer.idr": 4,
}


def main() -> int:
    root = "packages/idris-ml-examples/src"
    failures: list[str] = []
    opportunities: list[str] = []
    seen: set[str] = set()

    for path in sorted(glob.glob(EXAMPLES_GLOB, recursive=True)):
        rel = os.path.relpath(path, root)
        if rel in EXEMPT:
            continue
        with open(path) as f:
            count = sum(line.count("prim__") for line in f)
        seen.add(rel)
        allowed = BASELINE.get(rel)
        if allowed is None:
            if count > 0:
                failures.append(
                    f"{rel}: {count} prim__ occurrence(s) in a file outside the baseline "
                    "(new examples must use the typed construction facade)"
                )
        elif count > allowed:
            failures.append(f"{rel}: {count} prim__ occurrence(s), baseline allows {allowed}")
        elif count < allowed:
            opportunities.append(f"{rel}: {count} < baseline {allowed} — ratchet the baseline down")

    stale = [rel for rel in BASELINE if rel not in seen]
    for rel in sorted(stale):
        opportunities.append(f"{rel}: in baseline but no longer present — drop the entry")

    for msg in opportunities:
        print(f"note: {msg}")
    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    exempt = ", ".join(sorted(EXEMPT))
    print(f"prim-ratchet OK ({len(BASELINE)} baselined files, exempt: {exempt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
