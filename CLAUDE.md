# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## References

- [Neural Turing Machines (Graves, Wayne, Danihelka 2014)](https://arxiv.org/abs/1410.5401) — original NTM paper
- [Implementing Neural Turing Machines (Collier & Beel 2018)](https://isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/) — stability findings: constant memory init (1e-6) converges 3.5x faster, tanh memory bounding, grad clip norm 50

## Build Commands

```bash
# Build C shared library (required before running examples)
make build/libidrisml.dylib

# Type-check all library modules
idris2 --build idris-ml.ipkg

# Type-check a single module
idris2 --source-dir src -p contrib --check src/<File>.idr

# Build and run an example (all examples accept --help for CLI flags)
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr && ./build/exec/<name>

# Tests
make test            # Idris unit tests
make test-c          # C library tests

# Benchmarks
make bench           # Idris benchmark (Supervised + RNN + NTM)
make bench-compare   # Side-by-side Idris vs PyTorch
make profile         # Per-sub-phase timing + tape histogram

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
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`
3. **Sampler** - Distribution samplers: `uniform`, `normal` (Box-Muller), `normalSample`
3b. **Init** - Weight initialization strategies composable with samplers: `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`
4. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
5. **Math** - Loss functions, activations, linear algebra
6. **Memory** - NTM read/write head operations
7. **Variable** - Tape-based autograd (Wengert list) with hybrid Scheme/C storage and C backward pass
8. **DataPoint** - `DataPoint`, `RecurrentDataPoint`, and `TwoPhaseDataPoint` records
8b. **Generate** - Random data generation: `copyTaskBinary`/`recallTaskBinary`, `randomBatchVect`
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Re-export hub for the interface-based layer system:
    - **Layer.Core** - `LayerLike` interface, `AnyLayer` existential, `Network` type, network-level ops
    - **Layer.Linear**, **Layer.Rnn**, **Layer.Lstm**, **Layer.Activation**, **Layer.Normalization** - per-layer `LayerLike` instances
    - **Layer.Ntm** - `NtmState` + NTM head ops (imports Lstm and Linear for sub-layers)
11. **Optimizer** - SGD, Adam, and RMSprop with per-parameter, global norm, or value gradient clipping
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`
13. **Backprop** - Training loop: `epoch`, `epochRecurrent`, `epochTwoPhaseDenseBce`, and `train*` variants
14. **Curriculum** - Multi-stage curriculum training: `Stage` record, `runCurriculum`
15. **Debug** - Forward-pass diagnostics: `debugForward`, `debugForwardRecurrent`, `toDoubleNetwork`

### Core type signatures

```idris
-- Tensor.idr
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []
Vector elems = Tensor [elems]
Matrix rows columns = Tensor [rows, columns]

-- Variable.idr (tape-based autograd)
record Variable where
  constructor Var
  tapeIdx : Nat           -- index into global tape (Wengert list)
  tapeGen : Nat           -- tape generation (staleness detection)
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

### Training cycle

```idris
-- Basic: forward -> backward (tape reset) -> optimizer -> apply deltas
epoch opt dataPoints lossFn model st =
  let loss = calculateLoss lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
  in (emap (applyDeltas deltas) model, st', loss)

-- NTM production: dense optimizer with C-backed BCE
let opt = rmspropValueClipMomentumDense 0.0001 0.95 1.0e-8 10.0 0.9
let (m', s', loss) = epochTwoPhaseDenseBce opt dataPoints model st0
```

### Training modes

| Mode | Data type | Use case |
|------|-----------|----------|
| Supervised | `DataPoint i o ty` | Feedforward nets |
| Recurrent | `RecurrentDataPoint i o ty` | RNN/LSTM sequences |
| TwoPhase | `TwoPhaseDataPoint i o ty` | NTM copy/recall (output-phase loss only) |

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training. Use `autoName` (preferred):

```idris
let model = autoName $ ll ~> OutputLayer softmaxLayer  -- ll0_weight0, ll0_bias0, ...
```

### Curriculum training

Multi-stage training via the `Curriculum` module. Each `Stage` has a label, advancement threshold, and `IO` data generator. `runCurriculum` handles stage progression and two-level early stopping. Not required for LSTM-controller NTMs.

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

- **Profile first**: `make profile` — sub-phase timing + tape histogram. Keep `Profile.idr` in sync with production config
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
- **C shared library required**: build `libidrisml.dylib` first. Manual builds need `cp build/libidrisml.dylib build/exec/<name>_app/`

### Training & numerics

- **`paramId` / autoName**: variables without paramId are invisible to gradients. Always `autoName` before training
- **Tape generation staleness**: after `collectGrads` (tape reset, gen++), old Variables are stale. `ensureOnTape` handles this transparently
- **`logSoftmax` + `nllLoss`**: separate softmax+CE creates 1/pp intermediates (up to 1e6). Use `logSoftmaxLayer` + `nllLoss`
- **`pow` zero-base NaN**: `pow(0, k)` backward = `0^k * log(0) = NaN`. Fixed: return 0 when base is 0
- **Detached max in `logSoftmax`**: max subtraction uses detached constant to avoid corrupting max element's gradient
- **Gradient clipping**: `adamGlobalClip` for recurrent models (preserves direction). Per-param clipping distorts direction
- **Periodic GC**: NTM 50K+ epochs OOMs without `forceGC` every 10 epochs. Uses `heap-reserve-ratio 1.0`
- **RSS tracking**: `getRssMB` (peak, monotonic) and `getCurrentRssMB` (current, macOS only). Both take dummy args to prevent CSE

### NTM-specific

- **Dimension calculations**: `ReadParamWidth m = m + ShiftKernelSize + 3`, `WriteParamWidth m = ReadParamWidth m + m`. LSTM input: `m + inputSize`, output FC input: `h + m`
- **Head parameters**: β=softplus, g=sigmoid, γ=1+softplus (unbounded). Add vectors are raw linear. See Memory.idr
- **State flow**: previous read output + current input → LSTM. Cell state → head FCs. Hidden + read output → output FC
- **Two-phase training**: `epochTwoPhaseDenseBce` — encode with outputs discarded, decode with loss on targets. No output activation layer (C-backed fused sigmoid+BCE)
- **Batch size**: copy and recall use batch=1 (seed-sensitive). Larger batches dilute per-sequence addressing signal
- **No tanh memory bounding**: raw interpolation write matches PyTorch reference. Tanh was for erase+add, causes cumulative degradation with interpolation
- **Initial addressing**: weights initialized to zeros (projected to simplex), read output to Kaiming uniform. Non-learnable, reset per sequence
- **Early stopping**: windowed-average convergence (`--es-threshold`, `--es-window`, `--es-patience`). LSTM example uses old best-loss patience

### Architecture & infrastructure

- **Interface-based layer system**: `LayerLike` + `AnyLayer` existential. Explicit `{i, o : Nat}` needed on all methods (QTT erases Nat params). Adding a layer = one file, zero edits elsewhere
- **Dense optimizer**: NTM uses C-array optimizer (`epochTwoPhaseDenseBce`). Call `getNumPids 0` after `autoName`. Call `readFromBuffersNetwork` before `toDoubleNetwork` (C buffers don't update Variable.value)
- **Persistent NtmMemBuf**: memory matrix in C struct across timesteps. Reset per sequence via `resetNtmMemBufs`. `initial_vals` snapshotted after optimizer deltas
- **Test suite**: `make test` (Idris), `make test-c` (C). Tests in `test/src/Test/*.idr`, `Harness.idr` for assertions
- **Curriculum learning**: available via `Curriculum` module. Not needed for LSTM-controller NTMs — converges directly with two-phase training
