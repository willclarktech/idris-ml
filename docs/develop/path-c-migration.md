# Path C migration

The Path C migration moved the autograd value's tensor shape onto the value
itself, deleted the V1 surface, and renamed the canonical types. Bit-identical
numerics, ~5,000 fewer lines, single API.

## Why

V1 had two related problems:

1. **The autograd value was shape-erased.** `record Variable (0 d : Device)` carried
   a `tensorPtr : AnyPtr` and a cached `value : Double` but no shape. Shape lived
   on the outer `Tensor dims (Variable d)` (Vect-of-Vect of scalar Variables).
   Every op packed/unpacked that Vect-of-Vect at the autograd boundary. This was
   slow (~60× slower at hidden=256 on the boundary path) and footgun-prone
   (`forwardVar` was the slow path; `forwardVarTensor` was the fast path; same
   type signatures, different speeds).
2. **Two parallel paths.** The codebase grew "fast paths" (`applyVarTensor`,
   `forwardVarTensor`, `epochNativeTensorPre`, `epochRecurrentNativeTensor`, …)
   that operated on raw `AnyPtr` to dodge the Vect-of-Vect cost, but still
   carrying the V1 `Variable d` interface for the user-facing `LayerLike`. Two
   surfaces, one of them unsafe.

## What changed

| V1 | V2 |
|---|---|
| `record Variable (0 d : Device)` (shape-erased) | `record Tensor (dims : Vect rank Nat) (0 d : Device)` |
| `Vector n (Variable d)`, `Matrix m n (Variable d)` (Vect of Variable) | `Tensor [n] d`, `Tensor [m, n] d` |
| `data Tensor : Vect rank Nat -> Type -> Type` (structural Vect-of-Vect with `Functor`/`Num` instances) | renamed to `Array` (same shape, freed the `Tensor` namespace) |
| `STensor x` / `VTensor xs` (constructors of structural type) | `SArray x` / `VArray xs` |
| `Var ptr (Just pid) value` (autograd constructor with cached value) | `MkTensor ptr (Just pid)` (no cached value — read via `tensorItem`) |
| `forwardVar` (Vect-of-Vect, slow) and `forwardVarTensor` (fast) | `forwardVar` (single fast path) |
| `LayerLike` with 13 methods | `LayerLike` with 4 methods (`applyVar`, `applyVarBatch`, `layerPrefix`, `resetState`) |
| `autoName $ ll ~> OutputLayer ...` (name after construction) | `linearLayerAny "ll0"` (name at construction) |
| `epochNative` / `epochNativeTensorPre` / `epochNativeTensorBatch` / `epochRecurrentNative` / `epochTwoPhaseBceNative` | `epochVar` / `epochVarTensor` / `epochVarTensorBatch` / `epochRecurrentVar` / `epochTwoPhaseVar` |
| `nameLayer`, `applyDeltas`, `setParamId`, `prefixParamId` | gone (V2 names at construction) |
| `toDoubleNetwork`, pure-Double `forward` | gone (V2 has no element-type polymorphism on Network) |
| `Endofunctor.emap` | gone (no V2 use case) |
| Pure-Idris `Optimizer` (sgd / adam / rmsprop / `applyDeltas`) | gone — V2 uses C-side `nativeSgd` / `nativeAdamGlobalClip` / etc. exclusively |
| V1 `Debug` (`debugForward`, `toDoubleNetwork`-based) | replaced with `forwardVarTraced` (lightweight stderr min/max/mean tracer) |
| `clampMinTensor` (Math.idr, on `Tensor dims ty`) | renamed `clampMinArray` (on `Array dims ty`) |

## Branch commits

The migration shipped over five logical commits:

1. **Path C steps 1–4** (commits `fe16ce2` and ancestors): every example migrated
   to V2 (Variable→shape-indexed surface), V1 deleted, V2→no-suffix rename done.
2. **Phase 1** (`cf57b90`): structural `Tensor` → `Array`. Pure identifier swap.
3. **Phase 2** (`3d734d6`): `Variable` → `Tensor`. Pure identifier swap.
4. **Phase 3** (`cd3aba5`): re-add `Layer.idr` re-export hub. `import Layer` works again.
5. **Phase 4** (`7854244`): port `Curriculum.idr` to V2. Multi-stage trainer back.
6. **Phase 5** (`4519909`): re-add `toDevice` for explicit CPU↔CUDA↔MPS bridging.
7. **Tracer** (`01a4150`): `forwardVarTraced` debug walker.

## What was preserved

- **Bit-identical numerics**. `make test-examples` is 76/76 OK on tape + mlx +
  torch with seed=42 losses matching the pre-migration baseline exactly.
- **Multi-seed crash-free**. Reinforce / DQN / A2C / PPO / SAC / LSTM tested at
  5 seeds each; all examples crash-free, all pass smoke thresholds.
- **`paramId` registry semantics**. Same C-side optimizer registry; same
  per-parameter LR override (`setParamLR`); same multi-network paramId scoping
  (`nativeAdamGroup "actor_" ...`).
- **`Curriculum` multi-stage trainer**. Re-ported on top of `epochRecurrentVar`.
- **`toDevice`** CPU↔CUDA↔MPS bridge. Re-added at Tensor level.
- **`TVec` / `TMat` aliases**. Type-checker hang workaround on multiplicative
  Nat shape arithmetic still real; aliases preserved.

## What's gone

| Capability | Why dropped |
|---|---|
| `forward` (pure-Double Network evaluation) | V2 `Network i hs o d` is parameterised by Device only, not by element type |
| `toDoubleNetwork` | same as above — no `Variable d` → `Double` conversion path |
| `Endofunctor.emap` over Network | V2 names at construction, so `emap (prefixParamId "actor_") net` is unnecessary |
| Pure-Idris `Optimizer` | V1's per-paramId-Double interface couldn't express per-element updates on consolidated tensor params; was vestigial |
| `Debug` module (`debugForward`, NTM diagnostics) | V1's `debugForward` required `toDoubleNetwork`. NTM-specific diagnostics (`addrEntropy`, `peakMass`, sequential-walker detection) — see TODO.md follow-up |
| Network-level `toDevice` | Per-layer mapping over typed state records is non-trivial; tracked as follow-up. Single-tensor `toDevice` covers manual transfer |

## Identifier search-and-replace cheat sheet

For grep-replacing notebook code or external docs:

```
Variable d                       → Tensor [...] d   (and pick the right shape)
Vector n (Variable d)            → Tensor [n] d
Matrix m n (Variable d)          → Tensor [m, n] d
the (Vector n (Variable CPU)) (VTensor [...])
                                 → MkTensor (bulkToTensor (VArray [...])) Nothing
linearLayer {ty = Variable CPU} {i=I, o=O}
                                 → linearLayerAny {i=I} {o=O} "<name>"
autoName (ll ~> OutputLayer x)   → ll ~~> OutputLayer x  (name `ll` at construction)
forwardVar / forwardVarTensor    → forwardVar
epochNative                      → epochVar
epochRecurrentNative             → epochRecurrentVar
epochTwoPhaseBceNative           → epochTwoPhaseVar
epochNativeTensorPre             → epochVarTensor
epochNativeTensorBatch           → epochVarTensorBatch
emap refreshValue trained        → trained  (no refresh needed)
toDoubleNetwork model            → (gone — V2 has only the autograd path)
applyDeltas opt                  → nativeTrainStep opt loss  (single fused call)
VTensor [...]                    → VArray [...]
STensor x                        → SArray x
clampMinTensor                   → clampMinArray
Tensor (dims : Vect rank Nat) -> Type -> Type   (structural)
                                 → Array (dims : Vect rank Nat) -> Type -> Type
```

## Pareto status vs V1

The branch is a Pareto improvement over `main`:

**Strictly better**
- Type-safe tensor shapes on the autograd value
- No silent slow path (V1's `forwardVar` was ~60× slower than `forwardVarTensor`
  with same type signatures; V2 has one fast path)
- No `refreshValue` cache-staleness, no `autoName` double-init footgun
- ~5,000 fewer lines
- Modern PyTorch-aligned naming (`Tensor [dims] d` ≅ `torch.Tensor`)

**Same**
- All examples bit-identical at seed=42
- Multi-seed pass rates match `reference-alignment.md`
- All unit tests + smoke gate green on tape + mlx + torch
- Curriculum, toDevice surface preserved

**Not blocking but tracked in `TODO.md`**
- TensorBoard-style file sink for debugging (Medium priority)
- Precision type parameter for F32/BF16/FP16 mixed-precision (Medium priority,
  out of scope until CUDA support lands)
- NTM/DNC addressing-pattern diagnostics (port from V1's `Debug.idr` summarizer)
