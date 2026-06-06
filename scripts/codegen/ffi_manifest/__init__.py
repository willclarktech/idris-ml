"""Shared manifest + helpers for FFI wrap-template tooling and instance
generation.

This package is the single source of truth for two related concerns:
1. Which Idris-side `%foreign` declarations are Tensor-touching (and
   therefore must use the wrap-on-return Scheme template) — read by
   `ffi-convert-to-scheme.py` and `check-ffi-wrap-template.py`.
2. Which typeclass methods on `Executor/{Tape,Torch,Mlx}.idr` are
   generated from which FFI — read by `gen-executor-instances.py` and
   `check-executor-method-drift.py`.

Each entry is an `Entry` dataclass:
- `args` / `ret`: classifier tuple for the FFI's C signature.
- `slice`: typeclass sub-interface the method lives in (None if the FFI
  is internal-only and not bound to any instance method).
- `idris_method`: name of the typeclass method that calls this FFI.
- `c_symbol`: canonical C function name (defaults to the manifest key;
  override for aliased FFIs where one C symbol backs multiple methods).
- `tape` / `torch` / `mlx`: per-backend generation flavor.

Internal layout (underscore-prefixed modules are implementation detail;
consumers import via this package's top-level surface):

  _entry.py     — `Entry` dataclass + classifier alphabet
  _skip.py      — `SKIP`, `INIT_FFI`, guardian/drain initialization tuples
  _helpers.py   — stateless utilities (`gen_scheme_wrapper` and friends)
  _paths.py     — `WRAP_HANDLE_FILES`, `C_FFI_RE`, `ANY_FFI_RE`
  families/     — per-typeclass-family ENTRIES sub-dicts merged into MANIFEST
"""

from ._entry import Entry
from ._skip import (
    SKIP,
    INIT_FFI,
    GUARDIAN_ONLY_INIT,
    DRAIN_ONCE_INSTALL,
    GUARDIAN_LAZY_INIT,
)
from ._helpers import (
    strip_suffix,
    parse_args,
    idris_type_to_class,
    scheme_type,
    cache_var,
    backend_tag_of,
    gen_scheme_wrapper,
)
from ._paths import WRAP_HANDLE_FILES, C_FFI_RE, ANY_FFI_RE
from .families import (
    core,
    linear,
    nn,
    conv,
    tensor_create,
    transfer,
    autograd,
    optimizer,
    optimizations,
    serialize,
    quant,
    param_registry,
    memory_hygiene,
    profiling,
    diagnostics,
    internal,
)


_FAMILY_MODULES = (
    core, linear, nn, conv, tensor_create, transfer, autograd, optimizer,
    optimizations, serialize, quant, param_registry, memory_hygiene,
    profiling, diagnostics, internal,
)

MANIFEST: dict[str, Entry] = {}
for _mod in _FAMILY_MODULES:
    MANIFEST.update(_mod.ENTRIES)
del _mod

# Integrity guard: two families silently colliding on a key would let
# the later module's Entry win and the earlier one disappear. Catch at
# import time rather than waiting for a codegen diff.
assert sum(len(m.ENTRIES) for m in _FAMILY_MODULES) == len(MANIFEST), (
    "duplicate key across ffi_manifest families — see "
    "scripts/codegen/ffi_manifest/families/<name>.py"
)
