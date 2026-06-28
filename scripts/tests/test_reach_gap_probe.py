#!/usr/bin/env python3
"""Tests for the Idris reachability gap-finder parsers.

Fixtures are real lines captured from `idris2 --dumpcases` and minimal
`.idr` snippets exercising the universe scanner's edge cases (operators,
modifier placement, nested-scope skipping). These guard the highest-risk
part of the probe: making the dump's FQN spelling agree with the source
scan's (see `normalize_fqn`).

Run via:
    python3 scripts/tests/test_reach_gap_probe.py
    make test-integration-py-scripts   # pytest collects this
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mltools.idris_parser import (  # noqa: E402
    is_excluded,
    normalize_fqn,
    parse_exclusions,
    parse_reachable,
    scan_universe,
    scan_universe_text,
)

# --- real dump lines (from a tape build of the idris-ml test main) -------
DUMP_SAMPLE = """\
prim__sub_Integer = [{arg:0}, {arg:1}]: (-Integer [!{arg:0}, !{arg:1}])
Tensor.Core.weakenGrad = [{arg:0}]: (some body here)
Tensor.Internal.dtCreateScalar = [{arg:0}]: (body)
Array.+ = [{arg:0}, {arg:1}]: (body)
Array.(++) = [{arg:0}, {arg:1}]: (body)
Floating.(^) = [{arg:0}]: (body)
Train.Engine.6321:1000:padZeros = [{arg:0}]: (lifted local)
"""


def test_parse_reachable_captures_plain_fqn() -> None:
    r = parse_reachable(DUMP_SAMPLE)
    assert "Tensor.Core.weakenGrad" in r
    assert "Tensor.Internal.dtCreateScalar" in r


def test_parse_reachable_normalizes_operators() -> None:
    """Operators render inconsistently in the dump — `Array.+` bare but
    `Array.(++)` parenthesized. Both must normalize to a paren-free FQN."""
    r = parse_reachable(DUMP_SAMPLE)
    assert "Array.+" in r
    assert "Array.++" in r  # from `Array.(++)`
    assert "Floating.^" in r  # from `Floating.(^)`
    assert "Array.(++)" not in r  # the un-normalized form must NOT survive


def test_parse_reachable_keeps_lifted_locals_harmless() -> None:
    """where/let-lifted locals appear as `Mod.<num>:<num>:name`; they go
    into the reachable set verbatim (they never match a universe entry, so
    they're harmless noise) — just confirm the ` = ` split didn't choke."""
    r = parse_reachable(DUMP_SAMPLE)
    assert "Train.Engine.6321:1000:padZeros" in r
    assert "prim__sub_Integer" in r


def test_normalize_fqn() -> None:
    assert normalize_fqn("Array.(++)") == "Array.++"
    assert normalize_fqn("(>>=)") == ">>="
    assert normalize_fqn("Array.+") == "Array.+"
    assert normalize_fqn("Tensor.Core.weakenGrad") == "Tensor.Core.weakenGrad"


# --- exclusions: exact + prefix rules -----------------------------------
EXCL_SAMPLE = """\
# a comment
Tensor.Core.weakenGrad     # exact, inline comment

Executor.Mlx.*
Executor.Torch.*
"""


def test_parse_exclusions_splits_exact_and_prefix() -> None:
    exact, prefixes = parse_exclusions(EXCL_SAMPLE)
    assert exact == frozenset({"Tensor.Core.weakenGrad"})
    assert prefixes == ("Executor.Mlx.", "Executor.Torch.")


def test_is_excluded_exact_and_prefix() -> None:
    exact, prefixes = parse_exclusions(EXCL_SAMPLE)
    assert is_excluded("Tensor.Core.weakenGrad", exact, prefixes)  # exact
    assert is_excluded("Executor.Mlx.Nn.linearForward", exact, prefixes)  # prefix
    assert is_excluded("Executor.Torch.Linear.matmul", exact, prefixes)  # prefix
    # Tape executor must NOT be caught by the Mlx/Torch prefixes.
    assert not is_excluded("Executor.Tape.Transfer.toHost", exact, prefixes)
    assert not is_excluded("Tensor.Core.retypeGrad", exact, prefixes)


# --- universe scan: a synthetic module covering the edge cases -----------
MODULE_SAMPLE = """\
module Demo.Mod

import Data.Vect

public export
data Colour : Type where
  Red : Colour
  Green : Colour

public export
record Point where
  constructor MkPoint
  xCoord : Double
  yCoord : Double

interface Greet a where
  greet : a -> String

export
shapeOf : {dims : Vect rank Nat} -> Array dims ty -> Vect rank Nat
shapeOf {dims = ds} _ = ds

length : Nat -> Nat
length n = n

export bar : Int -> Int
bar x = x

export %inline
fastPath : Int -> Int
fastPath x = x

%inline
quux : Int -> Int
quux x = x

public export
(++) : List a -> List a -> List a
(++) xs ys = xs

namespace Inner
  hidden : Int
  hidden = 0
"""


def test_scan_universe_text_captures_top_level_sigs() -> None:
    names = scan_universe_text(MODULE_SAMPLE)
    assert "shapeOf" in names
    assert "length" in names


def test_scan_universe_text_handles_inline_modifier() -> None:
    """`export bar : Int -> Int` — modifier on the same line as the sig."""
    names = scan_universe_text(MODULE_SAMPLE)
    assert "bar" in names


def test_scan_universe_text_drops_pct_inline_defs() -> None:
    """`%inline` defs are spliced into callers and never get a --dumpcases
    line, so they're unmeasurable and must be dropped from the universe
    (else they're permanent false gaps — the `tadd` case)."""
    names = scan_universe_text(MODULE_SAMPLE)
    assert "fastPath" not in names  # `export %inline` on the line above
    assert "quux" not in names  # bare `%inline` on the line above


def test_scan_universe_text_normalizes_operator_def() -> None:
    names = scan_universe_text(MODULE_SAMPLE)
    assert "++" in names  # from `(++) : ...`
    assert "(++)" not in names


def test_scan_universe_text_skips_constructors_fields_methods() -> None:
    """Constructors, record fields, the record ctor, and interface methods
    are all indented (nested scope) — none get their own dump line, so they
    must be absent from the universe to avoid false gaps."""
    names = scan_universe_text(MODULE_SAMPLE)
    for nested in ("Red", "Green", "MkPoint", "xCoord", "yCoord", "greet"):
        assert nested not in names, f"{nested} (nested scope) leaked into universe"


def test_scan_universe_text_skips_namespaced_and_headers() -> None:
    names = scan_universe_text(MODULE_SAMPLE)
    assert "hidden" not in names  # indented under `namespace Inner`
    for kw in ("Colour", "Point", "Greet", "data", "record", "interface", "module"):
        assert kw not in names


def test_scan_universe_prefixes_module_and_excludes_test() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "Foo").mkdir()
        (root / "Foo" / "Bar.idr").write_text("module Foo.Bar\n\nexport\nbaz : Int\nbaz = 1\n")
        (root / "Test").mkdir()
        (root / "Test" / "Thing.idr").write_text("module Test.Thing\n\nhelper : Int\nhelper = 1\n")
        # .idr.in templates must be skipped
        (root / "Tmpl.idr.in").write_text("module Tmpl\n\ntmpl : Int\ntmpl = 1\n")
        uni = scan_universe(root)
    assert "Foo.Bar.baz" in uni
    assert uni["Foo.Bar.baz"].endswith("Bar.idr")
    assert "Test.Thing.helper" not in uni  # Test/ excluded by default
    assert "Tmpl.tmpl" not in uni  # .idr.in skipped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK ({len(fns)} tests)")
