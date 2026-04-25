# RefC Backend Investigation

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

### Assessment

The missing `idris2_negate_Double` is a RefC runtime bug/gap in Idris 0.8.0. Options:
- Wait for a newer Idris 2 release with a more complete RefC runtime
- Provide our own shim implementations of the missing functions
- File an upstream issue

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
