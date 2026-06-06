#!/usr/bin/env python3
"""Regression test for the ffi_manifest package's per-family split.

Asserts the merge invariant — every family file contributes its
entries and no two families silently collide on a C symbol key. The
package's `__init__.py` also asserts this at import time; this test
gives it an explicit failure surface.

Run via:
    python3 scripts/tests/test_ffi_manifest_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "codegen"))

from ffi_manifest import MANIFEST, Entry  # noqa: E402
from ffi_manifest.families import (  # noqa: E402
    core, linear, nn, conv, tensor_create, transfer, autograd, optimizer,
    optimizations, serialize, quant, param_registry, memory_hygiene,
    profiling, diagnostics, internal,
)


_FAMILIES = {
    "core": core,
    "linear": linear,
    "nn": nn,
    "conv": conv,
    "tensor_create": tensor_create,
    "transfer": transfer,
    "autograd": autograd,
    "optimizer": optimizer,
    "optimizations": optimizations,
    "serialize": serialize,
    "quant": quant,
    "param_registry": param_registry,
    "memory_hygiene": memory_hygiene,
    "profiling": profiling,
    "diagnostics": diagnostics,
    "internal": internal,
}


def test_no_duplicate_keys_across_families() -> None:
    """A C symbol appearing in two family files would let the later
    module's Entry silently win. Sum-of-counts must equal MANIFEST.
    """
    total = sum(len(m.ENTRIES) for m in _FAMILIES.values())
    assert total == len(MANIFEST), (
        f"family sum {total} != MANIFEST {len(MANIFEST)} — duplicate keys?"
    )


def test_every_entry_is_an_Entry() -> None:
    """Cheap sanity: a typo in a family file (e.g. accidentally writing
    `"foo": (args, ret)` instead of `Entry(...)`) would land a tuple
    here.
    """
    for k, v in MANIFEST.items():
        assert isinstance(v, Entry), f"{k!r}: expected Entry, got {type(v).__name__}"


def test_slice_consistency_per_family() -> None:
    """Every entry in `families/<name>.py` must either:
      - have `slice == 'UserExecutor<CamelName>'` matching the file, OR
      - live in `internal.py` (and have `slice is None`).
    Catches a paste-error where an Optimizer entry lands in the Linear
    file with its slice still set to UserExecutorOptimizer.
    """
    expected_slices = {
        "core": "UserExecutorCore",
        "linear": "UserExecutorLinear",
        "nn": "UserExecutorNN",
        "conv": "UserExecutorConv",
        "tensor_create": "UserExecutorTensorCreate",
        "transfer": "UserExecutorTransfer",
        "autograd": "UserExecutorAutograd",
        "optimizer": "UserExecutorOptimizer",
        "optimizations": "UserExecutorOptimizations",
        "serialize": "UserExecutorSerialize",
        "quant": "UserExecutorQuant",
        "param_registry": "UserExecutorParamRegistry",
        "memory_hygiene": "UserExecutorMemoryHygiene",
        "profiling": "UserExecutorProfiling",
        "diagnostics": "UserExecutorDiagnostics",
        "internal": None,
    }
    for fam, mod in _FAMILIES.items():
        want = expected_slices[fam]
        for key, entry in mod.ENTRIES.items():
            assert entry.slice == want, (
                f"families/{fam}.py: {key!r} has slice={entry.slice!r}, "
                f"expected {want!r}"
            )


if __name__ == "__main__":
    test_no_duplicate_keys_across_families()
    test_every_entry_is_an_Entry()
    test_slice_consistency_per_family()
    print("OK")
