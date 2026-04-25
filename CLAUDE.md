# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## References

- [Neural Turing Machines (Graves, Wayne, Danihelka 2014)](https://arxiv.org/abs/1410.5401) — original NTM paper
- [Implementing Neural Turing Machines (Collier & Beel 2018)](https://isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/) — stability findings: constant memory init (1e-6) converges 3.5x faster, tanh memory bounding, grad clip norm 50
- [Hybrid computing using a neural network with dynamic external memory (Graves et al. 2016)](https://www.nature.com/articles/nature20101) — DNC paper: temporal link matrix, usage-based allocation, multiple read heads with mode mixture

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
make test-all            # Everything: Idris + C backends + integration + PyTorch ref
make test                # Idris unit tests (pure logic, tape backend)
make test-backend-tape   # C backend API tests on tape
make test-backend-mlx    # C backend API tests on MLX
make test-backend-torch  # C backend API tests on torch
make test-safetensors    # SafeTensors serialization round-trip
make test-ntm-grad       # NTM gradient NaN detection
make test-ntm-timestep   # NTM full timestep integration
make test-examples       # All examples on all backends, validates RESULT lines

# Benchmarks
make example-bench           # Idris benchmark (Supervised + RNN + NTM)
make bench-compare           # Side-by-side Idris vs PyTorch (end-to-end training)
make bench-ops-compare       # Operator-level C backend vs PyTorch (raw speed)
make bench-ops               # Operator-level C backend only
make bench-ops-py            # Operator-level PyTorch only

# PyTorch reference (requires uv)
make ref-setup       # One-time: install Python deps
make ref-test        # Correctness tests
make ref-lint        # Lint (ruff)
make ref-typecheck   # Type-check (pyright)
make ref-convergence # NTM convergence verification

# Jupyter kernel (interactive REPL notebooks)
make jupyter-install      # Install kernel + deps
make jupyter-lab          # Launch JupyterLab with notebooks
make test-jupyter         # Full kernel tests (REPL + FFI integration)
make test-jupyter-unit    # Cell parser unit tests only (no backend needed)
make test-notebooks       # Run all notebooks headless (catches API breakage)

# Hyperparameter sweep
bash scripts/sweep.sh --task copy --parallel 4         # full
bash scripts/sweep.sh --task copy --parallel 4 --quick  # 2000 epochs for screening
```

## Architecture

### Module dependency order (leaves first)

1. **Device** - `data Device = CPU | CUDA Nat | MPS` — phantom type for compile-time device safety
1b. **Floating** - Extended `Floating` interface adding `sqrt`
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`, `formatElapsed`, `formatDuration`, `sigD`
3. **Sampler** - Distribution samplers: `uniform`, `normal` (Box-Muller), `normalSample`, `categoricalSample` (cumulative sum)
3b. **Init** - Weight initialization strategies composable with samplers: `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`
4. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
5. **Math** - Loss functions (MSE, BCE, cross-entropy, NLL, L1, Huber, KL divergence), activations, linear algebra
6. **Memory** - NTM read/write head operations
7. **Variable** - Autograd variables wrapping backend tensor pointers (tape/MLX/torch). `NativeOptimizer` for training
8. **DataPoint** - `DataPoint`, `RecurrentDataPoint`, and `TwoPhaseDataPoint` records
8b. **DataLoader** - Reusable batched data pipeline: `mkGeneratorLoader` (synthetic), `mkIndexedLoader` (file-backed with shuffle/repeat via IORef + C Fisher-Yates)
8c. **Generate** - Random data generation: `copyTaskBinary`/`recallTaskBinary`, `randomBatchVect`, `patternData`
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Re-export hub for the interface-based layer system:
    - **Layer.Core** - `LayerLike` interface, `AnyLayer` existential, `Network` type, network-level ops
    - **Layer.Linear**, **Layer.Rnn**, **Layer.Lstm**, **Layer.Gru**, **Layer.Activation**, **Layer.Normalization** - per-layer `LayerLike` instances
    - **Layer.LayerNorm** - `LayerNormState` with learnable gamma/beta (used as sub-component)
    - **Layer.BatchNorm** - `BatchNormState` with running mean/var, training/eval mode. Per-channel normalization (instance norm when batch=1)
    - **Layer.Conv** - `Conv1DState`, `Conv2DState`, `MaxPool1DState`, `MaxPool2DState`, `AvgPool1DState`, `AvgPool2DState` with type-level `ConvOutDim`/`PoolOutDim`
    - **Layer.Dropout** - `DropoutState` with inverted dropout, training/eval toggle via `setTraining`
    - **Layer.Embedding** - `EmbeddingState` for token index lookup (gather forward, scatter_add backward)
    - **Layer.Residual** - `ResidualState` wrapping `AnyLayer n n ty`. Forward: `add(input, inner(input))`
    - **Layer.Ntm** - `NtmState` + NTM head ops (imports Lstm and Linear for sub-layers)
    - **Layer.Dnc** - `DncState` + DNC head ops (Graves et al. 2016: temporal link matrix, usage-based allocation, multi-head read with mode mixture). R read heads parameterized at type level
    - **Layer.Transformer** - `TransformerState` with multi-head attention, layer norm, learned embeddings, sinusoidal PE
11. **Optimizer** - SGD, Adam, AdamW, RMSprop (Idris-side), plus `NativeOptimizer` (C-side, all backends). AdamW has decoupled weight decay. Per-parameter LR overrides via `setParamLR`
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`, `withWarmup`, `cosineWithWarmup`, `stepLR`, `exponentialLR`
13. **Backprop** - Epoch functions: `epochNative`, `epochRecurrentNative`, `epochTwoPhaseBceNative`, `epochNativeTensorPreAccum` (gradient accumulation)
14. **Train** - Unified training runner: `runTraining`, `TrainConfig`, `EarlyStopConfig`, `ArgSpec`/`parseArgs`, `formatResult`
15. **Curriculum** - Multi-stage curriculum training: `Stage` record, `runCurriculum`
16. **Debug** - Forward-pass diagnostics: `debugForward`, `debugForwardRecurrent`, `toDoubleNetwork`
17. **Checkpoint** - SafeTensors serialization: `saveModel`/`loadModel`, `saveOptimizer`/`loadOptimizer`
18. **Notebook.Prelude** - Re-exports all library modules via `import public` for Jupyter kernel interactive use

### Jupyter notebooks

Two categories in `jupyter/notebooks/`:
- **`tutorials/`** (01-06): Library concepts — tensors, types, layers, loss, training, devices
- **`models/`** (9 notebooks): Per-architecture walkthroughs — supervised, rnn_lstm, transformer, gpt, ntm, dnc, cnn, reinforce, seq_classify. Each covers architecture, types, and training (interactive where feasible, CLI instructions for heavy models)

`make test-notebooks` runs all notebooks headless to catch API breakage. CLI examples (`src/Example/`) remain the authoritative validation/benchmark targets via `make test-examples`.

### Core type signatures

```idris
-- Tensor.idr
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []
Vector elems = Tensor [elems]
Matrix rows columns = Tensor [rows, columns]

-- Device.idr (type-safe device placement)
data Device = CPU | CUDA Nat | MPS

-- Variable.idr (backend-agnostic autograd)
record Variable (0 d : Device) where  -- phantom Device, erased at runtime
  constructor Var
  tensorPtr : AnyPtr      -- libtorch tensor (carries autograd graph)
  paramId : Maybe String  -- parameter name (Nothing = intermediate)
  value : Double          -- cached forward result
-- Network i hs o (Variable CPU) can't accept Vector i (Variable (CUDA 0))
-- toDevice : (d2 : Device) -> Variable d1 -> IO (Variable d2)
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
| Supervised (accum) | `epochNativeTensorPreAccum` | `Vect n (TensorDataPoint i o)` | Gradient accumulation (larger effective batch) |
| Recurrent | `epochRecurrentNative` | `RecurrentDataPoint i o ty` | RNN/LSTM/GRU sequences |
| TwoPhase | `epochTwoPhaseBceNative` | `TwoPhaseDataPoint i o ty` | NTM copy/recall |
| RL (REINFORCE) | `epochRL` (custom) | `List (List Double)` (random pool) | Policy gradient |

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training. Use `autoName` (preferred):

```idris
let model = autoName $ ll ~> OutputLayer softmaxLayer  -- ll0_weight0, ll0_bias0, ...
```

### Model serialization (SafeTensors)

Save/load weights and optimizer state using the SafeTensors format (`.safetensors`). Backend-agnostic — works on tape, MLX, and torch. Python interop: PyTorch loads via `safetensors.torch.load_file()`, MLX via `mx.load()`.

```idris
import Checkpoint

-- Save trained model
ok <- saveModel "model.safetensors"

-- Load into existing model (after autoName, before or after training)
ok <- loadModel "model.safetensors"
let model' = emap refreshValue model  -- update cached Variable.value fields

-- Save/load optimizer state for training resumption
ok <- saveOptimizer "model.optimizer.safetensors" opt
ok <- loadOptimizer "model.optimizer.safetensors" opt
```

C-level API: `param_save`/`param_load` for weights, `optimizer_save`/`optimizer_load` for optimizer buffers. Shared implementation in `csrc/safetensors.c` using `csrc/cJSON.{c,h}` (vendored, MIT).

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
- **Device phantom type**: `Variable (0 d : Device)` — erased at runtime, prevents mixing CPU/CUDA/MPS tensors at compile time. Use `Variable CPU` in type annotations. `toDevice` is the only intentional device bridge. `LossFnTensor d` is parameterized by device.

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

- **Profile first**: `make example-profile` — per-epoch timing
- **Benchmark**: `make bench-compare` — always compare at same batch size (current: 16)
- **Sweep**: `bash scripts/sweep.sh` — systematic grid search, never manually loop
- **Convergence**: `make ref-convergence-copy` vs `./build/exec/ntm-copy` at matched settings
- **Document**: update `docs/develop/performance-analysis.md` with fresh profile data + results

See `docs/develop/performance-analysis.md` for current baseline and optimization history.

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc. Keep subject concise (~50 chars), imperative present tense. Commit work regularly in meaningful chunks — one logical change per commit. Never include ads, branding, or promotional text in commit messages or PR descriptions
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style
- **Documentation**: Always update CLAUDE.md, docs/develop/design-decisions.md, and TODO.md when adding features, changing architecture, or making design decisions

## Gotchas

See [`docs/develop/gotchas.md`](docs/develop/gotchas.md) for detailed explanations of each entry.

### Idris 2 / Chez Scheme traps

- **`total` is a keyword**: never use as a variable name — cryptic parse error. Use `numEpochs`, `totalEpochs`
- **Build flags**: forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Temporary test files**: Idris2 requires source files in `--source-dir`. Put temp files in `src/Example/`, not `/tmp`
- **Elementwise `(*)`**: `Tensor`'s `Num` uses elementwise multiply. Use `(<>)` for matmul: `w <> x` for mat-vec, `a <> b` for mat-mat. Equivalent to PyTorch's `@` operator
- **Tensor Foldable reversal**: `foldr`/`toList` produce reversed order. Use direct `Vect` traversal for ordered packing
- **Zero-arg FFI CSE trap**: zero-arg `%noinline` defs are constants (evaluated once). Pass a dummy arg through to the FFI call
- **FFI side-effect threading**: `let _ = ffiCall` is dropped. FFI must return a value consumed by later computation
- **`fst`/`snd` re-evaluation**: separate projections may re-evaluate FFI calls. Use `case ... of (a, b) =>` destructuring
- **`prim__seq` ordering**: use `prim__seq a b` to force evaluation order when no data dependency exists
- **`foreign-set! 'void*` corruption**: do NOT store C pointers via `foreign-set! 'void*` — corrupts memory. Use C helpers
- **Chez output buffering**: stdout fully buffered when piped. Use `stdbuf -oL ./build/exec/<name>`
- **Backend library required**: `make backend` builds `libidrisml.dylib` (tape by default, `BACKEND=mlx|torch` for others). Per-backend caching: `libidrisml_tape.dylib`, `libidrisml_mlx.dylib`, `libidrisml_torch.dylib` with symlink switching. Manual builds need `cp build/libidrisml.dylib build/exec/<name>_app/`
- **Scheme-side allocation reordering**: `foreign-alloc`/`foreign-set!` can be reordered by Chez — use C-side allocation (`tensor_alloc_doubles`/`tensor_write_double`) instead
- **`prim__seq` must use concrete types**: polymorphic `a -> b -> b` causes Chez arg count mismatch. Use `AnyPtr -> AnyPtr -> AnyPtr`
- **Large Nat type-level reduction**: Idris 2 Peano Nats hang the type-checker for dims > ~1000. Identity layers (dropout, batch norm) at conv output dims (e.g., 16*576=9216) never compile. Place them only at smaller dims (≤512). See `docs/develop/gotchas.md` for thresholds

### Training & numerics

- **`paramId` / autoName**: variables without paramId are invisible to gradients. Always `autoName` before training
- **Double `nameLayer`**: calling `nameLayer` then `autoName` creates TWO sets of parameter tensors. The first set becomes stale (optimizer only updates the second). If holding a direct state reference, name once and skip `autoName`
- **`setParamId` enables requires_grad**: Variables from `fromDouble` have `requires_grad=false`. `nameParam`/`setParamId` must upgrade them
- **`logSoftmax` + `nllLoss`**: separate softmax+CE creates 1/pp intermediates (up to 1e6). Use `logSoftmaxLayer` + `nllLoss`
- **Gradient clipping**: use `NormClip` for recurrent models (preserves direction). `ValueClip` per-param
- **Native optimizer**: preferred for training — `nativeRmsprop`/`nativeSgd`/`nativeAdamGlobalClip`. Single `optimizer.step()` updates all params
- **Stale Variable.value**: after native optimizer step, cached `value` fields are stale. Use `emap refreshValue` before `toDoubleNetwork`

### MLX backend

- **`mx::transpose` requires explicit axes**: `mx::transpose(x)` reverses ALL axes (not just last two). Use `mx::transpose(x, {1, 0})` for 2D transpose, `mx::transpose(x, {0, 2, 1})` for batched 3D
- **`mx::array(double)` defaults to float32**: Use `mx::array(value, mx::float64)` for double-precision scalars. Without the explicit dtype, `item<double>()` returns 0.0
- **Metal float32 transcendentals**: `mx::exp`, `mx::sigmoid`, etc. compute on GPU in float32 even with float64 inputs. Expect ~1e-6 precision, not 1e-10
- **Lazy eval use-after-free**: `tape_reset` must `mx::eval` ALL tensors before deleting intermediates — surviving tensors' lazy graphs may reference deleted arrays
- **Non-contiguous views**: `mx::transpose` returns a view with swapped strides. `data<double>()` pointer arithmetic assumes contiguous layout — use `mx::flatten` first or MLX indexing
- **`tensor_free` must check `all_tensors` membership**: after `tape_reset` deletes a tensor, a subsequent `tensor_free` on the same pointer is a double-free. Skip if not in `all_tensors`

### MLX replay autograd

The MLX backend uses **replay-based native autograd** via `mlx::vjp`. Forward ops record to a tape. `tensor_backward` replays the tape inside a closure and passes it to `mlx::vjp` — zero hand-written backward rules. Key performance constraints:

- **Constant pool from tape, not all_tensors**: backward must build the replay pool by scanning tape entries, not iterating `all_tensors`. Scanning `all_tensors` causes O(N²) degradation as persistent tensors accumulate
- **Non-grad tensors must be non-persistent**: `tensor_create_scalar`/`tensor_create` with `requires_grad=0` must NOT set `persistent=1` — they'd accumulate in `all_tensors` forever. Exception: `tensor_create_state_*` is explicitly persistent (NTM memory, LSTM state)
- **Pool indices must be compact**: `tape_reset` reassigns surviving tensors' `pool_idx` to contiguous 0..N. Without this, pool vectors grow unboundedly across epochs

### NTM-specific

- **Dimension calculations**: `ReadParamWidth m = m + ShiftKernelSize + 3`, `WriteParamWidth m = ReadParamWidth m + m`. LSTM input: `m + inputSize`, output FC input: `h + m`
- **Head parameters**: β=softplus, g=sigmoid, γ=1+softplus (unbounded). Add vectors are raw linear. See Memory.idr
- **State flow**: previous read output + current input → LSTM. Cell state → head FCs. Hidden + read output → output FC
- **Two-phase training**: `epochTwoPhaseBceNative` — encode with outputs discarded, decode with loss on targets. No output activation layer (fused sigmoid+BCE via libtorch)
- **Batch size**: copy and recall use batch=1 (seed-sensitive). Larger batches dilute per-sequence addressing signal
- **No tanh memory bounding**: raw interpolation write matches PyTorch reference. Tanh was for erase+add, causes cumulative degradation with interpolation
- **Initial addressing**: weights initialized to zeros (projected to simplex), read output to Kaiming uniform. Non-learnable, reset per sequence
- **Early stopping**: windowed-average convergence (`--es-threshold`, `--es-window`, `--es-patience`). LSTM example uses old best-loss patience

### DNC-specific

- **Extends NTM**: Same LSTM controller + two-phase training, but replaces shift-based addressing with usage allocation + temporal links + multi-mode reads
- **Dimension calculations**: Controller input: `r * m + inputSize`. Output FC input: `h + r * m`. Separate FC layers for each parameter group (write key, beta, erase, add, free gates, alloc gate, write gate, read keys, read betas, read modes)
- **Link matrix is O(n²)**: `Matrix n n ty`. Default N=32 (link matrix = 1024 elements). Larger N increases capacity but slows training. DNC is significantly slower than NTM per epoch on tape backend
- **Allocation uses argsort + cumprod**: `tensor_argsort` (non-differentiable integer indices) + `tensor_cumprod` (differentiable with backward rule). Allocation weights sum to ≤1. Sorted usage clamped to [1e-6, inf) to prevent cumprod underflow
- **R read heads**: Type-level `r : Nat`. R=1 exercises all DNC mechanisms. R=4 matches the paper but needs more epochs
- **applyGeneric (eval path)**: Simplified — uses content-based addressing only (no temporal links or allocation). Sufficient for eval accuracy measurement since the model learns content addressing as primary mechanism
- **Numerical stability clamping**: Six clamping points prevent forward-pass explosion: link matrix decay clamped to [0, inf), link entries clamped non-negative, allocation usage clamped to [1e-6, inf), retention clamped to [1e-10, inf), read weights clamped and normalized. Without these, multi-timestep state accumulation causes NaN at seqLen >= 4
- **Weight projection**: `syncBuffers` and `applyDeltasAndSync` project write weights and read weights onto the probability simplex (clamp to [1e-8, inf) + renormalize), matching the NTM pattern
- **Output FC uses current reads**: The output is computed from the CURRENT timestep's read outputs (after memory access), not the previous timestep's. This matches the paper and PyTorch reference

### Architecture & infrastructure

- **Interface-based layer system**: `LayerLike` + `AnyLayer` existential. Explicit `{i, o : Nat}` needed on all methods (QTT erases Nat params). Adding a layer = one file, zero edits elsewhere. `ActivationState` has tensor-level `applyVarTensor` overrides for tanh/sigmoid (1 tape entry vs ~7n for scalar path)
- **libtorch backend**: `csrc/backend.h` (abstract C API) + `csrc/backend_torch.cpp` (libtorch implementation). ~50 tensor ops, parameter registry, native optimizers. Autograd delegated entirely to libtorch
- **Autograd strategy per backend**: tape = manual Wengert tape + hand-written backward rules (reference, fastest for small tensors). torch = native `tensor.backward()` (2-line backward, zero rules). MLX = replay-based native autograd via `mlx::vjp` (forward ops recorded to tape, replayed inside closure for `mlx::vjp`, zero backward rules). Adding a new op: tape needs forward + backward rule, MLX needs only forward replay case (~2 lines), torch needs nothing (native autograd)
- **Test suite**: `make test-all` runs everything. `make test` (Idris unit tests), `make test-backend-{tape,mlx,torch}` (C API tests per backend), `make test-safetensors` / `test-ntm-grad` / `test-ntm-timestep` (specialized C tests), `make test-examples` (integration: all examples on all backends with RESULT line validation). Tests in `test/src/Test/*.idr`, `Harness.idr` for assertions
- **Curriculum learning**: available via `Curriculum` module. Not needed for LSTM-controller NTMs — converges directly with two-phase training
