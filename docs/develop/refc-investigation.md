# RefC Backend Investigation

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

## Goal

Determine if Idris 2's RefC backend (`--cg refc`) can replace Chez Scheme for idris-ml, potentially eliminating the 18-40% Chez runtime overhead.

## Compatibility Work Done

### Scheme FFI Elimination (completed)

All 23 `%foreign "scheme:..."` bindings in the codebase were replaced with portable `%foreign "C:...,libidrisml"` calls. New C helper functions added to all three backends (tape, MLX, torch):

- `tensor_backward_return`, `param_register_return`, `native_train_step`, `optimizer_step_with_clip`, `idrisml_seq`, `dropout_random_seed`, and 10 more `*_return` wrappers
- Zero Scheme-specific FFI remaining in idris-ml source code
- All examples produce identical results on Chez after the change

### Basic RefC Compilation (verified)

Simple "hello world" compiles and runs with RefC:
```bash
IDRIS2_CFLAGS="-I$GMP_INC" IDRIS2_LDFLAGS="-L$GMP_LIB" IDRIS2_LDLIBS="-lgmp" \
  idris2 --cg refc -o test src/Example/Test.idr
./build/exec/test  # "hello RefC"
```

RefC requires GMP (GNU Multiple Precision) for big integer support. On nix: `nix build nixpkgs#gmp.dev`.

## Blocker 1 (resolved): contrib's System.Random

**The idris-ml codebase cannot currently compile with RefC** due to:

```
Error: INTERNAL ERROR: [refc] FFI not found for System_Random_prim__srand
```

`System.Random` from the `contrib` package uses Scheme-specific FFI for `srand` and `randomIO`. This affects 24 files across the project (every example, most core modules).

### Options to resolve

1. **Provide our own System.Random replacement** — write a `Sampler.Random` module with C FFI bindings for `srand` and `rand`. Replace all `import System.Random` with `import Sampler.Random`. Medium effort (~2 hours), but diverges from upstream.

2. **Patch contrib upstream** — submit a PR to idris-lang/Idris2 adding RefC-compatible FFI to contrib's System.Random. Best long-term solution but depends on upstream acceptance timeline.

3. **Use `--directive refc:<name>=<c_function>`** — RefC supports custom FFI directives that map Idris FFI names to C functions. Could potentially map `System_Random_prim__srand` to a C `srand` wrapper without changing any Idris code. Needs investigation.

**Resolution**: Created `Compat.Random` module with C FFI (`srand`/`rand` from libc). Replaced all 23 `import System.Random` with `import Compat.Random`. Library type-checks and examples produce correct results on Chez.

## Blocker 2 (open): RefC runtime library incomplete

After resolving the Random blocker, RefC generates C code successfully but compilation fails:

1. **Missing `idris2_negate_Double`** — The RefC runtime (`libidris2_refc.a`) on Idris 0.8.0 only has `idris2_negate_Integer`, not the Double variant. Our code uses `negate` on Doubles (e.g., in Sampler's uniform distribution). This is a gap in the RefC runtime library.

2. **Missing `idris2_cast_string_to_Integer`** — Similar runtime function gap.

3. **`-include csrc/backend.h`** needed — RefC's generated C file doesn't include our FFI headers. Workaround: pass `-include csrc/backend.h` via `IDRIS2_CFLAGS`.

4. **const char* vs char*** — RefC generates `char*` for String returns but our `backend_name()` returns `const char*`. Minor, suppressed with `-Wno-incompatible-pointer-types-discards-qualifiers`.

### Resolution

Shims provided in `csrc/refc_shims.c` for all three missing functions. Supervised example now compiles and links successfully.

## Blocker 3 (open): RefC trampoline crash

The compiled Supervised binary segfaults immediately in `idris2_trampoline` — the RefC runtime's closure dispatch function. Investigation:

- `__mainExpression_0()` creates a closure wrapping `PrimIO_unsafePerformIO` → `Example_Supervised_main`
- `idris2_trampoline` tries to read the tag byte from the closure's Value_header
- Register `x19` contains a garbage pointer (different each run: `0xffffffff8d020002`, `0x5b020002`, etc.)
- All functions resolve at link time — no missing symbols
- No implicit function declaration warnings — the generated C is clean
- Simple RefC programs (hello world, tensor creation + read) work fine
- The crash is BEFORE any idris-ml code runs — it's in the RefC runtime dispatching the initial closure

**Diagnosis** (via ASAN): The crash happens in `Tensor_map` (Functor instance for Tensor) called from `Example_Supervised_dataPoints` when constructing `VTensor [1.5, -2.7]`. The Functor map returns a pointer into the zero page (`x19 = 0x1490`). This is a RefC runtime bug with nested algebraic data types (Vect of STensor constructors), not an FFI issue. Simple programs (scalar tensor creation, reads) work; constructing nested Idris data structures crashes.

Stack trace:
```
#0 idris2_trampoline
#1 Tensor_map_Functor (supervised-refc.c:12149)
#3 Example_Supervised_dataPoints (supervised-refc.c:5756)
#6 Example_Supervised_main
```

### Assessment

This is an upstream RefC runtime bug, not fixable on our side. Options:
- Wait for a newer Idris 2 release with RefC fixes
- File upstream issue with reproduction case
- Build Idris 2 from `main` branch (which has updated RefC runtime)

## Decision

**Status: Blocked on RefC runtime library gaps (Idris 0.8.0).**

The FFI portability work and Compat.Random are complete and independently valuable. RefC is close to working — the Idris→C code generation succeeds, and all our FFI calls resolve correctly. The remaining issues are in Idris 2's own RefC runtime, not in our code.

**Recommendation**: File an upstream issue, provide shim implementations for `idris2_negate_Double` etc. to unblock, or revisit when Idris 2 updates its RefC runtime. The investigation shows RefC is architecturally viable for idris-ml but needs runtime library fixes.

## What was independently valuable

Even without RefC benchmarks, this investigation delivered:
1. **Zero Scheme FFI remaining** — all 23 bindings replaced with portable C calls
2. **Compat.Random module** — C-based random generation, no Chez dependency
3. **Portable C helpers** on all 3 backends (tape, MLX, torch)
4. **Clear understanding** of RefC's limitations and what's needed to unblock

## Addendum (2026-07-27): Blocker 3 refuted as an upstream bug

An evidence pass re-tested the trampoline-crash hypothesis on the current pack
toolchain (`0.8.0-b2d2cf40d`, collection nightly-260604 — a different runtime
from the nix 0.8.0 release the crash was observed on):

1. The minimal repro from [refc-upstream-bug.md](refc-upstream-bug.md)
   (`Vect 3 (Wrapper Double)` + Functor map + traverse) builds with
   `idris2 --cg refc` and runs correctly.
2. A faithful `Ml.Array`-shaped repro — the rank-indexed GADT with the recursive
   `Functor` instance, the exact `VArray [SArray 1.5, SArray (-2.7)]`
   construction Blocker 3 blamed, a rank-2 nested map, and a map-to-String
   closure through the trampoline — also runs correctly. Neither build needed
   `refc_shims.c`.
3. Upstream `support/refc` history between the 0.8.0 release and the pin has no
   fix that would explain a vanished nested-ADT crash (formatting/renames, the
   negation-header fix [#3751](https://github.com/idris-lang/Idris2/pull/3751),
   a WASM32 integer-comparison fix, aligned_alloc portability), and no matching
   issue was ever reported by anyone else.

Conclusion: the SEGV was almost certainly environmental, with the prime suspect
being `refc_shims.c`'s hand-copied `_datatypes.h` struct layouts linked into the
crash binary (a layout mismatch producing a corrupt `Value` matches the
zero-page pointer in the ASan trace) on the old nix runtime. No upstream issue
will be filed. Runtime-gap status on the pinned nightly:
`idris2_cast_String_to_{Double,Integer}` now exist (capital-S spellings);
`idris2_negate_Double` is still absent from `libidris2_refc.a`, but the test
programs (which negate and multiply Doubles) no longer reference it — whether a
full idris-ml build still does is part of the re-validation below.

**Path forward for RefC adoption** (replaces the "file an upstream issue"
recommendation): retry the full Supervised build on the pack nightly — first
WITHOUT `refc_shims.c`; if link errors name missing runtime symbols, re-derive
the shims against the CURRENT `_datatypes.h` (the old hand-copied layouts must
not be reused — note upstream renamed internals with an `Idris2_` prefix in
2026-04, commit `d11e2af`).
