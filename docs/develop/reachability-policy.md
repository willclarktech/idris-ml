# Idris reachability policy

This doc defines what the **Idris reachability gap-finder** measures, what
it deliberately does not, and how to act on its output. It is the Idris
counterpart to [`coverage-policy.md`](coverage-policy.md) (which governs the
C/C++ backends), but it answers a narrower question and must not be mistaken
for line/branch coverage.

## What it is

Idris 2 has no gcov-equivalent execution-coverage instrumentation. What it
*does* expose is `idris2 --dumpcases`, which emits a **tree-shaken** dump of
a compiled program — one `Fully.Qualified.Name = …` line per definition
**reachable from `main`**. Unreachable definitions are dropped by the
compiler's own dead-code elimination, so they never appear.

The gap-finder unions the dumps of all our entry points and subtracts them
from the source universe:

- **REACHABLE** — every FQN appearing in a dump of an entry point. Entry
  points (v1): the `idris-ml` unit-test main (`Test.Main`) + every example
  main under `packages/idris-ml-examples/src/Example/*.idr`. The union is
  what stops example-only code being flagged as untested.
- **UNIVERSE** — every top-level definition declared in
  `packages/idris-ml/src/**/*.idr` (excluding `Test/`), as `Module.name`.
- **GAP = UNIVERSE − REACHABLE − EXCL** — definitions no test or example
  exercises.

Implementation: `scripts/reach-gap-probe.py` + `scripts/mltools/idris_parser.py`;
dumps produced by the `reach-dump` Make target; run via
`make test-coverage-reach-gap`.

## What it is NOT

**This is a binary "exercised at all" signal, not a coverage percentage.**
"Reachable" means a static call path from a `main` exists — not that a test
ran the code, hit a particular branch, or asserted on the result. A function
called in a dead `if` arm, or whose output is discarded, counts as reachable.
So:

- A reachable def can still be **under-tested**; this tool will not catch it.
- The valuable artifact is the **unreachable LIST** (actionable: "wire this
  to a test or example"), never the percentage. Most definitions are
  trivially reachable, so chasing the % to 100 is near-meaningless and is
  **not a goal**. Do not treat it like the C gcov stack.

Granularity is **per-definition** (whole function), never per-line/branch.

## Exclusions (`scripts/reach-exclusions.txt`)

Some definitions never get a `Name = ` dump line even when used, so they are
false gaps. List them in `scripts/reach-exclusions.txt` (one normalized FQN
per line, `#` comment, rationale required). Categories:

- **`%inline` / heavily auto-inlined defs** — body spliced into callers; no
  standalone line emitted.
- **Erased / type-level-only defs** (0-quantity, used only in types) — no
  runtime code generated.
- **`%foreign` / `prim__` shims** that inline to their raw FFI call.

An exclusion is a claim that a def is *exercised but invisible to
`--dumpcases`* — **not** a way to silence a genuinely-untested def. For the
latter, write a test.

## Known accuracy limits

- **Inlining/erasure** → false gaps (the main ongoing EXCL maintenance).
- **Single-`BACKEND` dump** omits defs reachable only through another
  backend's interface dispatch (tape/torch/mlx) → false gaps for
  backend-specific code. Mitigation (deferred): union dumps from multiple
  `BACKEND=` builds.
- **Defs without a type signature** (`Foo = …` with no `Foo : …`) are not in
  the universe — under-reports, never over-reports.
- **Nested-namespace and interface-method defs** are indented, so the
  column-0 universe scan skips them; they are not audited in v1.
- **Operator/qualified-name normalization** (`Array.(++)` ↔ `Array.+`) is the
  highest-risk parse step — guarded by `scripts/tests/test_reach_gap_probe.py`.

## Gate status & the ratchet follow-up

**v1 is advisory**: `make test-coverage-reach-gap` always exits 0 and is
**not** in the `test-coverage` aggregator. It produces
`build/<BUILD_KEY>/reach-gap.csv` + a stdout summary for triage.

The intended follow-up — once the gap list is pruned (false positives moved
to EXCL) and trusted — is a **ratchet gate**, matching the C policy's
"fails if a new hole opens, and fails if a baselined hole is closed without
being removed":

- commit a baseline (`scripts/reach-baseline.txt`) of grandfathered
  unreachable FQNs;
- fail on `GAP − BASELINE` (a newly-unreachable def: wire it to a
  test/example, EXCL it with rationale, or add to baseline);
- fail on `BASELINE − GAP` (a baselined def became reachable → drop it, so
  the list only tightens);
- add `test-coverage-reach-gap` to the `test-coverage` aggregator + the CI
  coverage job.

The percentage is never the gate; the list-not-growing is.

## Scope (v1)

`packages/idris-ml/src` only. The other Idris packages (`idris-gym`,
`idris-transformers`, `idris-args`) are separate universes with their own
test mains and are deferred to follow-up probe configs.
