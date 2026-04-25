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

## Blocker: contrib's System.Random

**The idris-ml codebase cannot currently compile with RefC** due to:

```
Error: INTERNAL ERROR: [refc] FFI not found for System_Random_prim__srand
```

`System.Random` from the `contrib` package uses Scheme-specific FFI for `srand` and `randomIO`. This affects 24 files across the project (every example, most core modules).

### Options to resolve

1. **Provide our own System.Random replacement** — write a `Sampler.Random` module with C FFI bindings for `srand` and `rand`. Replace all `import System.Random` with `import Sampler.Random`. Medium effort (~2 hours), but diverges from upstream.

2. **Patch contrib upstream** — submit a PR to idris-lang/Idris2 adding RefC-compatible FFI to contrib's System.Random. Best long-term solution but depends on upstream acceptance timeline.

3. **Use `--directive refc:<name>=<c_function>`** — RefC supports custom FFI directives that map Idris FFI names to C functions. Could potentially map `System_Random_prim__srand` to a C `srand` wrapper without changing any Idris code. Needs investigation.

## Decision

**Status: Blocked on contrib System.Random.**

The FFI portability work (eliminating all 23 Scheme bindings) is complete and was worth doing regardless — it removes fragile inline Scheme code from the codebase. When the System.Random blocker is resolved (via option 1, 2, or 3 above), RefC compilation should work.

**No performance data collected** — couldn't reach the benchmarking step.

## Recommendation

**Option 1 (our own Random module) is the fastest path** to unblocking. If RefC benchmarks show a meaningful improvement, we can upstream the fix to contrib afterwards. If benchmarks show no improvement, we save the effort.

Next steps:
1. Implement option 1 (own Random module) OR investigate option 3 (RefC directives)
2. Get Supervised + Transformer compiling under RefC
3. Run 5x benchmarks on each, compare wall time and C time
4. Document results and make adopt/reject decision
