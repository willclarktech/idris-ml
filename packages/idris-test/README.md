# idris-test

The shared Idris test harness for the monorepo — one source of truth for assertions, suite
runners, and property testing. Every package's `*-tests.ipkg` (idris-ml, idris-transformers,
idris-gym, idris-ml-examples) depends on it. Depends on `base` + `hedgehog`.

The C-side counterpart is [idris-test-c](../idris-test-c/); the test-layer taxonomy lives in
[docs/develop/testing.md](../../docs/develop/testing.md).

## Modules

| Module | Provides |
| --- | --- |
| `Test.Harness` | `check`, `checkClose` (tolerance), `runSuite`, `runAll` (exits non-zero on failure) |
| `Test.Property` | Hedgehog adapter: `checkProperty`, `checkPropertyN`, and `checkPropertyIO` (FFI-safe variant for tests that build real tensors via IO) |
| `Test.Property.Golden` | comparison against stored golden values |

## Usage

```idris
import Test.Harness

main : IO ()
main = runAll
  [ ("arithmetic", [ check "1+1=2" (1 + 1 == 2)
                   , checkClose "pi" 3.14159 (myPi) 1.0e-5 ])
  ]
```
