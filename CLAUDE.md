# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## References

- [Neural Turing Machines (Graves, Wayne, Danihelka 2014)](https://arxiv.org/abs/1410.5401) — original NTM paper
- [Implementing Neural Turing Machines (Collier & Beel 2018)](https://isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/) — stability findings: constant memory init (1e-6) converges 3.5x faster, tanh memory bounding, grad clip norm 50

## Build Commands

```bash
# Build C tape backend (default, no external dependencies)
make backend

# Build MLX backend (Apple Metal GPU, requires python3Packages.mlx from nix)
make BACKEND=mlx MLX_SITE=/path/to/mlx backend

# Build libtorch backend (optional, requires libtorch)
make BACKEND=torch backend

# Type-check all library modules
idris2 --build idris-ml.ipkg

# Type-check a single module
idris2 --source-dir src -p contrib --check src/<File>.idr

# Build and run an example (all examples accept --epochs, --lr, --seed)
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr && ./build/exec/<name>

# Tests
make test                # Idris unit tests
make test-backend-tape   # C tape backend tests
make test-backend-torch  # libtorch backend tests (requires BACKEND=torch)

# Benchmarks
make bench           # Idris benchmark (Supervised + RNN + NTM)
make bench-compare   # Side-by-side Idris vs PyTorch

# PyTorch reference (requires uv)
make ref-setup       # One-time: install Python deps
make ref-test        # Correctness tests
make ref-lint        # Lint (ruff)
make ref-typecheck   # Type-check (pyright)
make ref-convergence # NTM convergence verification

# Hyperparameter sweep
bash scripts/sweep.sh --task copy --parallel 4         # full
bash scripts/sweep.sh --task copy --parallel 4 --quick  # 2000 epochs for screening
```

## Architecture

### Module dependency order (leaves first)

1. **Floating** - Extended `Floating` interface adding `sqrt`
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`, `formatElapsed`, `formatDuration`, `sigD`
3. **Sampler** - Distribution samplers: `uniform`, `normal` (Box-Muller), `normalSample`
3b. **Init** - Weight initialization strategies composable with samplers: `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`
4. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
5. **Math** - Loss functions, activations, linear algebra
6. **Memory** - NTM read/write head operations
7. **Variable** - Autograd variables wrapping libtorch tensors. `NativeOptimizer` for training
8. **DataPoint** - `DataPoint`, `RecurrentDataPoint`, and `TwoPhaseDataPoint` records
8b. **Generate** - Random data generation: `copyTaskBinary`/`recallTaskBinary`, `randomBatchVect`, `patternData`
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Re-export hub for the interface-based layer system:
    - **Layer.Core** - `LayerLike` interface, `AnyLayer` existential, `Network` type, network-level ops
    - **Layer.Linear**, **Layer.Rnn**, **Layer.Lstm**, **Layer.Activation**, **Layer.Normalization** - per-layer `LayerLike` instances
    - **Layer.LayerNorm** - `LayerNormState` with learnable gamma/beta (used as sub-component)
    - **Layer.Ntm** - `NtmState` + NTM head ops (imports Lstm and Linear for sub-layers)
    - **Layer.Transformer** - `TransformerState` with causal self-attention (single-head)
    - **Layer.Transformer** - `TransformerState` with multi-head attention, layer norm, learned embeddings, sinusoidal PE
11. **Optimizer** - SGD, Adam, RMSprop (Idris-side), plus `NativeOptimizer` (libtorch torch::optim)
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`
13. **Backprop** - Epoch functions: `epochNative`, `epochRecurrentNative`, `epochTwoPhaseBceNative`
14. **Train** - Unified training runner: `runTraining`, `TrainConfig`, `EarlyStopConfig`, `ArgSpec`/`parseArgs`, `formatResult`
15. **Curriculum** - Multi-stage curriculum training: `Stage` record, `runCurriculum`
16. **Debug** - Forward-pass diagnostics: `debugForward`, `debugForwardRecurrent`, `toDoubleNetwork`

### Core type signatures

```idris
-- Tensor.idr
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []
Vector elems = Tensor [elems]
Matrix rows columns = Tensor [rows, columns]

-- Variable.idr (libtorch autograd)
record Variable where
  constructor Var
  tensorPtr : AnyPtr      -- libtorch tensor (carries autograd graph)
  paramId : Maybe String  -- parameter name (Nothing = intermediate)
  value : Double          -- cached forward result
```

The `LayerLike` interface + `AnyLayer` existential wrapper provides dynamic dispatch over layer types. `Network` chains `AnyLayer`s via `(~>)`. `Endofunctor`'s `emap` applies type-preserving transforms (e.g., `applyDeltas`). Adding a new layer type = one file implementing `LayerLike`, zero edits elsewhere.

## Key Patterns

### Network composition

```idris
ll <- linearLayer
let model = autoName $ ll ~> OutputLayer softmaxLayer

ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
let model = autoName $ OutputLayer ntm
```

### Forward pass (state threading)

```idris
forward : Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)
let (updatedModel, output) = forward model input
```

### Training (Train.idr)

All examples use `runTraining` from the `Train` module:

```idris
-- Simple: run N epochs, no early stopping
(trained, epochs, loss) <- runTraining
  (\m, d => epochNative opt d lossFn m) (pure data) (simpleConfig 1000) model

-- Patience-based early stopping (LSTM)
(trained, epochs, loss) <- runTraining
  (\m, d => epochRecurrentNative opt d lossFn m) (pure data) (patienceConfig 2000 500) model

-- Windowed convergence + per-epoch data gen + metrics (NTM)
let cfg = MkTrainConfig epochs 100 (WindowedAvg threshold window patience) evalMetrics
(trained, epochs, loss) <- runTraining
  (\m, d => epochTwoPhaseBceNative opt d m) genBatch cfg model
```

`runTraining` handles: epoch loop, NaN detection, progress logging, early stopping, timing summary.

### CLI arg parsing

Declarative via `ArgSpec` + `parseArgs`:
```idris
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c) ]
cfg = parseArgs defaultConfig specs (drop 1 args)
```

### Training modes

| Mode | Epoch function | Data type | Use case |
|------|---------------|-----------|----------|
| Supervised | `epochNative` | `DataPoint i o ty` | Feedforward nets |
| Recurrent | `epochRecurrentNative` | `RecurrentDataPoint i o ty` | RNN/LSTM sequences |
| TwoPhase | `epochTwoPhaseBceNative` | `TwoPhaseDataPoint i o ty` | NTM copy/recall |

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training. Use `autoName` (preferred):

```idris
let model = autoName $ ll ~> OutputLayer softmaxLayer  -- ll0_weight0, ll0_bias0, ...
```

### Curriculum training

Multi-stage training via the `Curriculum` module. Each `Stage` has a label, advancement threshold, and `IO` data generator. `runCurriculum` handles stage progression and two-level early stopping. Not required for LSTM-controller NTMs.

### Type safety conventions

The codebase has **zero `believe_me`** and **zero `unsafePerformIO`**. Keep it that way.

- **Nat arithmetic**: `(S k) * n` reduces to `n + (k * n)` as `Refl`. Use `Tensor.splitAt` for reshape/flatten.
- **Erased proofs in records**: Transformer carries `0 inputPrf : i = seqLen * dModel` for type-safe reshape at layer boundaries. Zero runtime cost.
- **`decEq` + `Refl`**: When a generic `{n : Nat}` must equal a specific value, use `case decEq n expected of Yes Refl => ...` to unify types in the branch.
- **`rewrite`**: Convert between provably-equal types: `rewrite sym prf in expr`.
- **`coerceLastGate`**: For the `o + 0 = o` case after repeated `splitAt`, use `rewrite plusZeroRightNeutral`.
- **Never add `believe_me`**: If a type won't unify, prove it. If you can't prove it, the types might actually be wrong.

### Debug / diagnostics

```idris
let dblModel = toDoubleNetwork trained
let (_, _, snapshots) = debugForwardRecurrent dblModel inputs
printDiagnostics "label" snapshots
```

## Workflows

### Adding new examples

1. **Source reference** — find paper/implementation for ground truth. Add to References
2. **PyTorch implementation** — port to `pytorch/torch_ref/models/`, add tests + benchmark. Verify: `make ref-test && make ref-lint && make ref-typecheck`
3. **Idris implementation** — implement in `src/Example/`, add to `Bench.idr` + Makefile. Verify: `make test && make bench-compare`

Commit at each step. PyTorch is the correctness oracle.

### Performance optimization

- **Profile first**: `make profile` — per-epoch timing
- **Benchmark**: `make bench-compare` — always compare at same batch size (current: 16)
- **Sweep**: `bash scripts/sweep.sh` — systematic grid search, never manually loop
- **Convergence**: `make ref-convergence-copy` vs `./build/exec/ntm-copy` at matched settings
- **Document**: update `docs/performance-analysis.md` with fresh profile data + results

See `docs/performance-analysis.md` for current baseline and optimization history.

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc. Keep subject concise (~50 chars), imperative present tense. Commit work regularly in meaningful chunks — one logical change per commit. Never include ads, branding, or promotional text in commit messages or PR descriptions
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style
- **Documentation**: Always update CLAUDE.md, docs/design-decisions.md, and TODO.md when adding features, changing architecture, or making design decisions

## Gotchas

See [`docs/gotchas.md`](docs/gotchas.md) for detailed explanations of each entry.

### Idris 2 / Chez Scheme traps

- **`total` is a keyword**: never use as a variable name — cryptic parse error. Use `numEpochs`, `totalEpochs`
- **Build flags**: forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Temporary test files**: Idris2 requires source files in `--source-dir`. Put temp files in `src/Example/`, not `/tmp`
- **Elementwise `(*)`**: `Tensor`'s `Num` uses elementwise multiply. Use `matrixVectorMultiply` for matvec
- **Tensor Foldable reversal**: `foldr`/`toList` produce reversed order. Use direct `Vect` traversal for ordered packing
- **Zero-arg FFI CSE trap**: zero-arg `%noinline` defs are constants (evaluated once). Pass a dummy arg through to the FFI call
- **FFI side-effect threading**: `let _ = ffiCall` is dropped. FFI must return a value consumed by later computation
- **`fst`/`snd` re-evaluation**: separate projections may re-evaluate FFI calls. Use `case ... of (a, b) =>` destructuring
- **`prim__seq` ordering**: use `prim__seq a b` to force evaluation order when no data dependency exists
- **`foreign-set! 'void*` corruption**: do NOT store C pointers via `foreign-set! 'void*` — corrupts memory. Use C helpers
- **Chez output buffering**: stdout fully buffered when piped. Use `stdbuf -oL ./build/exec/<name>`
- **Backend library required**: `make backend` builds `libidrisml.dylib` (C tape backend by default). Manual builds need `cp build/libidrisml.dylib build/exec/<name>_app/`
- **Scheme-side allocation reordering**: `foreign-alloc`/`foreign-set!` can be reordered by Chez — use C-side allocation (`tensor_alloc_doubles`/`tensor_write_double`) instead
- **`prim__seq` must use concrete types**: polymorphic `a -> b -> b` causes Chez arg count mismatch. Use `AnyPtr -> AnyPtr -> AnyPtr`

### Training & numerics

- **`paramId` / autoName**: variables without paramId are invisible to gradients. Always `autoName` before training
- **Double `nameLayer`**: calling `nameLayer` then `autoName` creates TWO sets of parameter tensors. The first set becomes stale (optimizer only updates the second). If holding a direct state reference, name once and skip `autoName`
- **`setParamId` enables requires_grad**: Variables from `fromDouble` have `requires_grad=false`. `nameParam`/`setParamId` must upgrade them
- **`logSoftmax` + `nllLoss`**: separate softmax+CE creates 1/pp intermediates (up to 1e6). Use `logSoftmaxLayer` + `nllLoss`
- **Gradient clipping**: use `NormClip` for recurrent models (preserves direction). `ValueClip` per-param
- **Native optimizer**: preferred for training — `nativeRmsprop`/`nativeSgd`/`nativeAdamGlobalClip`. Single `optimizer.step()` updates all params
- **Stale Variable.value**: after native optimizer step, cached `value` fields are stale. Use `emap refreshValue` before `toDoubleNetwork`

### NTM-specific

- **Dimension calculations**: `ReadParamWidth m = m + ShiftKernelSize + 3`, `WriteParamWidth m = ReadParamWidth m + m`. LSTM input: `m + inputSize`, output FC input: `h + m`
- **Head parameters**: β=softplus, g=sigmoid, γ=1+softplus (unbounded). Add vectors are raw linear. See Memory.idr
- **State flow**: previous read output + current input → LSTM. Cell state → head FCs. Hidden + read output → output FC
- **Two-phase training**: `epochTwoPhaseBceNative` — encode with outputs discarded, decode with loss on targets. No output activation layer (fused sigmoid+BCE via libtorch)
- **Batch size**: copy and recall use batch=1 (seed-sensitive). Larger batches dilute per-sequence addressing signal
- **No tanh memory bounding**: raw interpolation write matches PyTorch reference. Tanh was for erase+add, causes cumulative degradation with interpolation
- **Initial addressing**: weights initialized to zeros (projected to simplex), read output to Kaiming uniform. Non-learnable, reset per sequence
- **Early stopping**: windowed-average convergence (`--es-threshold`, `--es-window`, `--es-patience`). LSTM example uses old best-loss patience

### Architecture & infrastructure

- **Interface-based layer system**: `LayerLike` + `AnyLayer` existential. Explicit `{i, o : Nat}` needed on all methods (QTT erases Nat params). Adding a layer = one file, zero edits elsewhere
- **libtorch backend**: `csrc/backend.h` (abstract C API) + `csrc/backend_torch.cpp` (libtorch implementation). ~50 tensor ops, parameter registry, native optimizers. Autograd delegated entirely to libtorch
- **Test suite**: `make test` (Idris), `make test-backend` (C backend). Tests in `test/src/Test/*.idr`, `Harness.idr` for assertions
- **Curriculum learning**: available via `Curriculum` module. Not needed for LSTM-controller NTMs — converges directly with two-phase training
