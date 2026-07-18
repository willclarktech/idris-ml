# Upstream Bug Report: RefC Trampoline Crash on Nested ADTs

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

> **DO NOT FILE — refuted 2026-07-27.** The evidence pass re-ran both the minimal
> repro below and a faithful `Ml.Array`-shaped repro (rank-indexed GADT, recursive
> Functor, the exact `VArray [SArray 1.5, SArray (-2.7)]` construction, nested map +
> map-to-String closure dispatch) on the pack toolchain (`0.8.0-b2d2cf40d`,
> collection nightly-260604): both build and run correctly, no crash. No matching
> upstream issue exists, and the `support/refc` history between the 0.8.0 release
> and the pin contains nothing that would have fixed a nested-ADT trampoline crash
> (only formatting/renames, the negation-header fix
> [#3751](https://github.com/idris-lang/Idris2/pull/3751), a WASM32 comparison fix,
> and aligned_alloc portability) — so "was a real bug, silently fixed" is unlikely.
> The remaining suspect for the original SEGV is environmental: the crash binary
> linked `refc_shims.c` (hand-copied `_datatypes.h` struct layouts) on the old nix
> 0.8.0-release runtime; a layout mismatch there producing a corrupt `Value` is
> exactly consistent with the zero-page pointer in the ASan trace. Two of the three
> shimmed symbols now exist upstream under capital-S spellings
> (`idris2_cast_String_to_{Double,Integer}`); only `idris2_negate_Double` remains
> absent from the pinned runtime. See the addendum in
> [refc-investigation.md](refc-investigation.md).

**For filing on:** https://github.com/idris-lang/Idris2/issues (superseded — see above)

## Title

RefC backend: SEGV in idris2_trampoline when mapping Functor over nested constructors

## Description

The RefC backend (`--cg refc`) on Idris 2 version 0.8.0 crashes with a segmentation fault when executing programs that use the Functor `map` over nested algebraic data types (specifically, `Vect` of single-field constructors).

Simple programs (IO, FFI calls to C libraries, basic data types) work correctly. The crash occurs specifically when dispatching closures returned by Functor map over nested constructors.

## Minimal reproduction

```idris
module Main

import Data.Vect

data Wrapper : Type -> Type where
  MkW : ty -> Wrapper ty

-- A nested structure: Vect of Wrappers
testData : Vect 3 (Wrapper Double)
testData = [MkW 1.5, MkW 2.7, MkW 3.9]

-- Mapping over nested constructors triggers the crash
showW : Wrapper Double -> String
showW (MkW x) = show x

main : IO ()
main = do
  let results = map showW testData
  traverse_ putStrLn results
```

Build with:
```bash
idris2 --cg refc -o test-refc Main.idr
./build/exec/test-refc
```

**Expected:** Prints `1.5`, `2.7`, `3.9`
**Actual:** Segmentation fault

Note: The above is a minimal reproduction hypothesis — the actual crash was observed in a larger project (idris-ml) where `Tensor` is defined as nested `Vect`/constructor types and the Functor instance's `map` triggers the crash. If the minimal repro above doesn't crash, a full reproduction is available at the project repository.

## Actual crash context

In idris-ml, the crash occurs in:
```
Example_Supervised_dataPoints → Tensor_map_Functor → idris2_trampoline → SEGV
```

The `dataPoints` function constructs values like:
```idris
VTensor [STensor 1.5, STensor (-2.7)]
```

which involves `Vect` construction with nested `STensor` constructors, triggering Functor map internally.

## ASAN stack trace

```
==96281==ERROR: AddressSanitizer: SEGV on unknown address 0x000000001492
    #0 idris2_trampoline+0x44
    #1 Tensor_map_Functor__parenOpenTensor__dollardims_parenClose (supervised-refc.c:12149)
    #2 idris2_trampoline+0x140
    #3 Example_Supervised_dataPoints (supervised-refc.c:5756)
    #6 Example_Supervised_main (supervised-refc.c:5371)
    #8 main (supervised-refc.c:20465)
```

Register `x19 = 0x1490` — a pointer into the zero page, indicating a corrupted or uninitialized Value pointer returned from the Functor map closure.

## Environment

- Idris 2 version: 0.8.0
- Platform: macOS 26.2 (arm64, Apple M4 Pro)
- RefC runtime: libidris2_refc.a from nix package idris2-0.8.0
- GMP: 6.3.0

## Notes

- The Chez Scheme backend handles this code correctly
- The generated C code compiles and links without warnings (all symbols resolve)
- Three runtime functions missing from 0.8.0 but present in `main` branch were shimmed: `idris2_negate_Double`, `idris2_cast_string_to_Double`, `idris2_cast_string_to_Integer`
- contrib's `System.Random` uses Scheme-only FFI (`"scheme:blodwen-random"`) with no `"C:..."` fallback, blocking RefC compilation of any program using random numbers — required a workaround (custom Random module with C FFI)

## Additional context

Our project (idris-ml, a deep learning library) has zero Scheme-specific FFI — all 147 C FFI bindings and 23 former Scheme bindings are now portable C calls. We have shims for the missing runtime functions. The only remaining blocker is this trampoline crash on nested ADTs.

Full investigation: https://github.com/<user>/idris-ml/blob/main/docs/develop/refc-investigation.md
