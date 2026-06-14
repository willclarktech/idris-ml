# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## References

- [Neural Turing Machines (Graves, Wayne, Danihelka 2014)](https://arxiv.org/abs/1410.5401) — original NTM paper
- [Implementing Neural Turing Machines (Collier & Beel 2018)](https://isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/) — stability findings: constant memory init (1e-6) converges 3.5x faster, tanh memory bounding, grad clip norm 50
- [Hybrid computing using a neural network with dynamic external memory (Graves et al. 2016)](https://www.nature.com/articles/nature20101) — DNC paper: temporal link matrix, usage-based allocation, multi-head reads

## Monorepo Structure

```
packages/
  idris-ml/           # Core ML library (Idris ipkg)
  idris-ml-notebook/  # Notebook Prelude (re-exports all idris-ml modules for Jupyter)
  idris-gym/          # Pure Idris RL environments (Gymnasium-parity API)
  idris-transformers/ # HF-aligned model library on top of idris-ml (HfBert, HfGpt2, HfLlama)
  idris-ml-examples/  # Example programs (depends on idris-ml + idris-gym + idris-transformers), plus Generate.idr
  idris-args/         # Typed CLI flag parsing (general-purpose, zero deps beyond base)
  backends/           # C/C++ backends (tape, MLX, torch)
  jupyter/            # Jupyter kernel (Python)
  pytorch/            # PyTorch reference implementations (Python)
```

## Build Commands

`BACKEND` is a comma-separated list of backends linked into one `libidrisml.{so,dylib}`. First item is the **primary** — its symbols are exported under both unified (`tensor_add`) and suffixed (`tensor_add_<primary>`) names, so existing Idris `%foreign "C:tensor_add,libidrisml"` calls resolve to it. Other backends contribute only suffixed symbols (reachable by future `UserDevice` instance methods). See `docs/develop/design-decisions.md` "Pluggable Device".

```bash
make backend                              # Default: BACKEND=tape (lean, no C++ deps)
make BACKEND=tape,torch backend           # Multi-link: both built into one dylib, tape primary
make BACKEND=tape,torch,mlx backend       # macOS full build (all three)
make BACKEND=torch backend                # Torch-only build (CI lane)
make BACKEND=mlx MLX_SITE=... backend     # MLX-only (Apple Metal)
make BACKEND=mlx MLX_DEVICE=gpu install   # F32 mode: examples target Tensor [..] (MlxExecutor MGpu) F32
make BACKEND=torch TORCH_DEVICE=mps install # F32 on Metal via libtorch: Tensor [..] (TorchExecutor TMps) F32
make BACKEND=torch TORCH_DEVICE=cuda install # CUDA (when on a CUDA box): Tensor [..] (TorchExecutor (TCuda 0)) F64
make rename-headers                       # Regen packages/backends/rename_<b>.h from backend.h
make test-integration-lint-rename-headers # CI gate: errors if regen would change anything
make install                              # Install core lib + gym (required for examples/tests)
make example-<name>                       # Build and run an example (all accept --epochs, --lr, --seed)

# Tests — see docs/develop/testing.md for the full layer breakdown
make test-examples              # Smoke gate: every example × 4 lanes (tape, mlx, mlx-gpu, torch), ~30-60 min
make test-examples-convergence  # Every example to convergence (hours, tape only)
make test                       # Idris unit tests
make test-unit-c-{tape,mlx,torch}   # C backend unit tests per backend (criterion)

make bench-compare              # Side-by-side Idris vs PyTorch (end-to-end training)
make bench-ops-compare          # Operator-level C backend vs PyTorch (raw speed)

# PyTorch reference
make ref-setup / test-e2e-pytorch-ref / ref-lint / ref-typecheck / ref-convergence

# Python typecheck gates (pyright strict, per package — mirrors lint-py-<pkg>)
make typecheck-py               # umbrella: pytorch + scripts + transformers + examples + jupyter
make typecheck-py-<pkg>         # one surface (typecheck-py-pytorch ≡ ref-typecheck)

python3 scripts/sweep.py --task copy --parallel 4 [--quick]  # hyperparameter sweep
```

The root `Makefile` is wiring only; the build logic lives in `mk/*.mk` fragments
included in dependency order (config → backends → genconfig → lint → tests →
install → examples → bench → ref → jupyter → e2e). Find a target by its domain
fragment (e.g. `example-*` in `mk/examples.mk`, lint gates in `mk/lint.mk`);
target names are a public API (CI spec, perf scripts, docs) — keep them stable.
The two big e2e recipes live in `scripts/test-e2e-examples.sh` /
`scripts/test-convergence.sh`, shelled out from `mk/e2e.mk`.

Build artifacts live under `build/<BUILD_KEY>/` where
`BUILD_KEY := <backend-list>-mlx<MLX_DEVICE>-torch<TORCH_DEVICE>` (e.g.
`tape-mlxcpu-torchcpu`, `torch-mlxcpu-torchmps`, `tape-torch-mlxcpu-torchmps`).
Each distinct `(BACKEND, MLX_DEVICE, TORCH_DEVICE)` tuple keeps its own warm
ttc/install/dylib/exec tree, so switching between sets (e.g. `make test` on
tape ↔ `BACKEND=torch TORCH_DEVICE=mps make example-hf-llama-inference`) is
near-free instead of triggering 60-min cascading re-elaboration. `clean`
removes every backend set's tree (plus `build-cov/`, the per-package
`build/` dirs from pack-driven test builds, and legacy `.idris2/`);
`clean-set` removes just the active set; `clean-all` cascades to
`clean-models` + `clean-datasets` (root `data/` + `packages/pytorch/data/`)
+ `clean-venvs` (pytorch/jupyter `.venv`s) + removes `vendored/` and the
run-output dirs (`logs/`, `results/`, `.tmp/`).

## Architecture

Module dependency order (leaves first): **Device → Floating → Util → Sampler → Init → Array → Math → Schedule → Tensor → Optimizer → DataPoint → DataLoader → Dataset → DataStream → Layer.\* → Hpo → Backprop → Train.Engine → Train → Fit → Curriculum → Checkpoint → Notebook.Prelude**. Single `import Layer` brings in all layer modules (Linear, Activation, LayerNorm, BatchNorm, Conv, Dropout, Embedding, Residual, Rnn/Lstm/Gru, Ntm, Dnc, Transformer).

### Core type signatures

```idris
-- Array.idr (structural data — Vect-of-Vect, rank-N instances)
data Array : Vect rank Nat -> Type -> Type
Scalar       = Array []
Vector elems = Array [elems]
Matrix r c   = Array [r, c]

-- Device.idr (open device kind — pick a Type with a UserDeviceCore instance)
0 Device : Type
Device = Type
-- Built-in backend tags:
--   TapeExecutor               — tape backend (CPU only, no hardware variants)
--   TorchExecutor d            — libtorch; d : TorchHwDev = TCpu | TMps | TCuda Nat
--   MlxExecutor s              — mlx; s : MlxStream = MCpu | MGpu
-- `UserDeviceTransfer` makes the generic `toDevice` work between any
-- pair: matching backendTag → fast intra-backend HW migration; differing
-- → host buffer round-trip. Declare your own backend by adding a tag type
-- with these instances + a unique `backendTag`. See design-decisions.md
-- "Open `d` parameter".
--
-- Availability gating (design-decisions.md "Device-availability gating";
-- full doc device-availability-gating.md). Two gates, each where the fact
-- lives:
--   • Linkage (compile-time): empty `Linked ex` marker gates construction;
--     instances emitted per build by the generated `HwConfig`, so a
--     tape-only build can't even spell `MlxExecutor _`.
--   • Hardware presence (runtime, EAFP): construction shims catch the
--     backend's exception → NULL handle; `toDeviceChecked` / `attemptOn`
--     lift NULL → `Left DeviceError`; `availableDevices builtinDevices`
--     probes the build's candidates (`builtinDevices` is generated per
--     build into `HwDevices.idr`, the value-level mirror of `Linked`).
--     Degrades to "always Right" on tape/mlx (their construction can't fail).

-- DType.Core (open dtype kind — pick a Type with an IsDType / Compatible instance)
0 DType : Type
DType = Type
-- Float n / BFloat n / IntN n / UInt n / Bool are types with built-in
-- IsDType instances. Aliases F32 = Float 32, F64 = Float 64, etc.
-- `Compatible ex t` gates admissible (device, dtype) pairs at construction.
-- `Compatible (MlxExecutor MGpu) F64` and `Compatible (TorchExecutor TMps) F64`
-- deliberately don't exist — Metal GPU is F32-only (mlx 0.31; libtorch
-- rejects F64 at MPS *construction*). See design-decisions.md "Open `dt`".

-- Tensor.idr (autograd handle — backend-agnostic)
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr      -- wrapped handle: Chez vector #(tensor-handle-v2 tag raw)
  paramId   : Maybe String  -- parameter name (Nothing = intermediate)
-- Aliases TVec n ex dt g / TMat m n ex dt g dodge the Idris-2 type-checker hang on multiplicative-Nat shape literals
```

`Array` is the structural type used for input-data marshalling and Math.idr's pure-Idris ops; it is NOT the autograd type. `Tensor` is the daily user-facing autograd handle.

**Canonical constraint**: exported single-dtype signatures use the `Backend ex dt` bundle (`Backend.idr`: `(UserExecutorTraining ex, RuntimeDType dt, Linked ex, Compatible ex dt) => Backend ex dt` + blanket impl) — write `Backend ex dt => ...`, not the four-leaf stack. Leaf constraints survive only where deliberate: dt-less signatures (`withNoGrad` — its `{ex=...}` pin is unsolvable by design, the bracket result must hold no live tensors), mixed dual-dtype signatures (two `Backend ex _` dicts make leaf searches ambiguous — see design-decisions.md), `targsort`-style weaker tiers, and the cross-executor transfer path. Bridge code holding two dtype dict pairs assembles explicitly via `backendFrom`.

The library is **fully polymorphic in dt** — every interface method, smart constructor, and layer state record binds `dt` as an implicit and uses it. Callers pin the concrete dtype at the leaf use site. Hardcoding F64 in method bodies while leaving the record's slot polymorphic caused a 30+ GB elaborator memory blowup; see `docs/develop/gotchas.md` "Polymorphic type-parameter slot vs concrete value in method body."

Examples don't hardcode device or dtype. They reference `ExampleDevice` / `ExampleDType` from `packages/idris-ml-examples/src/BuildConfig.idr` — a Makefile-generated source file (template at `BuildConfig.idr.in`, version-controlled). The generator reads `BACKEND` + `MLX_DEVICE` + `TORCH_DEVICE` at build time and picks the right `(ExampleDevice, ExampleDType)` cell:

  - `BACKEND=tape`                       → `TapeExecutor`, `F64`
  - `BACKEND=torch TORCH_DEVICE=cpu`      → `TorchExecutor TCpu`, `F64`
  - `BACKEND=torch TORCH_DEVICE=mps`      → `TorchExecutor TMps`, `F32`
  - `BACKEND=torch TORCH_DEVICE=cuda`     → `TorchExecutor (TCuda 0)`, `F64`
  - `BACKEND=mlx MLX_DEVICE=cpu`          → `MlxExecutor MCpu`, `F64`
  - `BACKEND=mlx MLX_DEVICE=gpu`          → `MlxExecutor MGpu`, `F32`

Idris-2 can't drive type-level selection from a runtime env var (types fix at elaboration), so the env is observed at build time and baked into `BuildConfig.idr`. Switching modes is just a different `make install` — no source edits. (Same trick generates the per-build `Linked` instances in `HwConfig.idr`.)

The `LayerLike` interface (4 methods: `applyVar`, `applyVarBatch`, `layerPrefix`, `resetState`) + `AnyLayer` existential provides dynamic dispatch over layer types. `Network` chains `AnyLayer`s via `(~~>)`. Adding a new layer = one file implementing `LayerLike`, zero edits elsewhere.

#### `Nn/` — the v1 models-as-records surface

The successor to `Layer/` (coexists; `Layer/` dies at the example sweep). Models are plain records of layers; `Nn.Module` is a batched-first `forward : l i o ex dt -> Tensor [b,i] -> IO (Tensor [b,o])` (no `applyVarBatch`, no `idris_crash` default — a layer that can't batch simply isn't a `Module`, killing the crash hole structurally), `Nn.Params` is the param traversal, `Nn.Seq` (endpoints-only index, `~~>`/list-literal) chains `Module`s. Both `Module`/`Params` are **higher-kinded** over the `Nat -> Nat -> Executor -> DType -> Type` constructor (instances written unapplied, like `LayerLike`); layers with extra config Nats lead them and trail the `(i,o)` pin (Conv2D, TransformerBlock). `GradMode` is off model types (params `WithGrad` by construction; `g` on activations only); freeze/unfreeze flip C `requires_grad` via `Frozen m`. Recurrent/memory layers (RNN/LSTM/GRU/NTM/DNC) implement `Nn.Recurrent` (`recurStep`/`recurReset`, state in the record) instead of `Module`. Param names derive over the **unchanged** C registry via `Nn.Init` (`scoped`/`scopedChild`/`freshChild`/`named`/`runInit`); `Nn.Group.groupOf` returns a submodel's exact registry names for optimizer scoping (replaces substring-prefix matching). 19 layers ported; the Transformer is decomposed (`Nn.Attention` + a `TransformerBlock` that stacks via `Seq`), not the legacy monolith. Hand-written 3-line `Params` instances (the deriveParams spike chose this); see design-decisions.md "models-as-records: the `Nn` surface".

### Tensor lifecycle (wrapped-handle ABI)

`tensorPtr` is a Chez vector `#(tensor-handle-v2 tag raw)` (slot 1 = backend tag, slot 2 = raw pointer), not a raw pointer. Every Tensor-touching `%foreign` binds to a Scheme wrapper that unwraps via `(vector-ref a<i> 2)` on Tensor args and wraps + retains + registers with `idris-tensor-guardian` on Tensor returns. The wrap IS the value — Idris-Chez codegen can't elide it without eliding the Tensor. C-side refcount drives freeing on mlx; tape and torch carry no-op retain/release stubs. New FFIs go through `scripts/codegen/ffi_manifest.py` + `ffi-convert-to-scheme.py`; `make check-ffi-wrap-template` (CI preflight) enforces the template. Full model in `docs/develop/tensor-lifecycle.md`.

## Key Patterns

### Network composition

```idris
ll <- linearLayerAny {i=2} {o=3} "ll0"      -- naming happens at construction
let model = ll ~~> OutputLayer reluLayerAny -- registers "ll0_weights" + "ll0_bias"
```

Each `*LayerAny` constructor takes a paramPrefix and registers parameters in the C-side optimizer registry. **Parameters without a paramId are invisible to the optimizer** — always pass a prefix. For multi-network examples, scope each network's prefix distinctly (`actor_ll0`, `critic_ll0`).

### Construction

`tensor {dims=[2,3]} (Const 0.5)` / `param "w" (Normal 0.0 0.02)` — one construction surface over `InitSpec` (`Zeros | Const x | Normal mu sd | Uniform lo hi | FromVect xs`; `fromRows` stacks a `Vect b (Vect i Double)` for the batch case). `FromVect`'s length is tied to `Numel dims` at compile time; `param` requires rank <= 4 (compile error past the C surface's ceiling) and always registers. Raw `prim__*` + `dtCreate*` construction lives in `Tensor.Internal` (backend authors only); the prim ratchet gate keeps examples from growing new raw-prim call sites.

### Forward pass

```idris
forwardVar : Network i hs o d g -> Tensor [i] d g -> IO (Network i hs o d g, Tensor [o] d g)
```

`forwardVar` (and every Tensor-handle-touching smart constructor: `tadd`, `tmul`, `ttanh`, etc.) is `IO`-typed. This is load-bearing: `withNoGrad (pure (forwardVar …))` would have fired the FFI *before* `noGradBegin` since `pure`'s argument is evaluated strictly. With IO typing the FFI body fires only on `<-` sequencing — inside the bracket. The helper `ioRerun : (() -> a) -> IO a` defers a pure body to IO without using the prelude's private `MkIO`; `Lazy a` was rejected because it memoizes.

Swap `forwardVar` for `forwardVarTraced "label"` to dump per-layer min/max/mean/NaN to stderr without affecting numerics.

**Expression ops**: row-select-by-index and elementwise arithmetic compose without hand recursions — `tgatherRows` ([b,n] × [b] double-valued-int indices → [b]; PyTorch `gather(1, ·)`), `tmaxRows` ([b,n] → [b]; `max(1).values`), and the infix aliases `(+.)` `(-.)` `(*.)` (elementwise) / `(*:)` (scalar-left) on plain evaluated tensors with bang notation: `tgt <- r +. !(gamma *: !(tmaxRows qNext))`. No `Num` instance, no IO-carrier operators (roadmap.md decision 5). Note `tmseLoss` is a *sum* reduction — scale by `1/n` for PyTorch's mean default. (`tgather` is the separate torch-only integer-dtyped 1-D surface.)

**Long eval loops on mlx need per-sequence `withNoGrad`**: a single outer bracket around `traverse evalOne batch` lets mlx Metal MTLBuffer count blow past the Tart/GHA VM ceiling before exit-drain fires. Push the bracket inside: `evalOne dp = withNoGrad $ do { ... }` (NTM-style) or `withNoGrad (evalEp …)` inside `evalN`'s recursion (RL-style). Tape/torch don't need this; the per-sequence pattern is cheap on both.

### Training — `fit` driver (v1, Fit.idr)

```idris
(trained, epochs, loss) <- fitSupervised opt lossFn (batched dataStream) (simpleConfig 1000) model
```

One driver for everything. `EpochStep m batch = m -> batch -> IO (m, Double)`: the step owns
control-flow + the optimizer step + (optional) model-state threading; `fit` owns the epoch loop,
schedule `tick`, early stop, checkpointing, NaN handling, and mlx generation hygiene (all via
`Train.Engine.runEpochLoop`, the shared engine `runTrainingIO` also uses). `fit` reuses `TrainConfig`.

- **Supervised (90%)**: `fitSupervised opt lossFn stream cfg model` — pass a loss fn, never call
  `nativeTrainStep`. `fitSupervisedMixed opt gradScaler lossFn …` for mixed precision.
- **Recurrent / two-phase**: a `Step` that folds over timesteps into one loss — no driver variant.
- **RL / custom**: pass your own `EpochStep` to `fit` (rollout + your own `nativeTrainStep`s + state
  threading), or compose the exported engine pieces (`runEpochLoop`, `withEpoch`, `postEpoch`,
  `earlyStopMachine`) directly for multi-step loops fit can't express (DQN replay, PPO K-epoch).

This refines api-critique §N6 (which had `fit` own the step) — see design-decisions.md
"`fit` driver". Data: `Dataset { size : Nat; item : Fin size -> IO sample }` (`fromVect`/
`fromIndexed`/`idxDataset`) + `DataStream` (`stream shuffleSpec ds` / `generate ioAction` /
`batched` collating `(Tensor [i], Tensor [o])` pairs into `([b,i],[b,o])` C-side). See **Data
redesign** below.

### Training modes — legacy (pre-migration; deleted at the example sweep)

`runTraining`/`runTrainingIO` + the `epoch*` family still exist (examples use them until the sweep);
`runTrainingIO`'s internals now route through `Train.Engine.runEpochLoop` (behaviour-identical, see
the equivalence oracle). New library code uses `fit`.

| Mode | Epoch function | Data type | Use case |
|------|---------------|-----------|----------|
| Supervised | `epochVar` | `DataPoint i o ty` | Feedforward nets |
| Supervised (batched) | `epochVarTensorBatch` | `Vect n (TensorDataPoint i o)` | MNIST/Transformer/GPT |
| Recurrent | `epochRecurrentVar` | `RecurrentDataPoint i o ty` | RNN/LSTM/GRU |
| TwoPhase | `epochTwoPhaseVar` | `TwoPhaseDataPoint i o ty` | NTM/DNC copy/recall |
| RL | custom (uses `runTrainingIO`) | varies | REINFORCE / DQN / A2C / PPO / SAC / tabular |

### Data redesign (v1: Dataset.idr / DataStream.idr)

PyTorch's three orthogonal joints: `Dataset` (indexed access) / `ShuffleSpec` (order) / `DataStream`
(batching+collation). `Dataset { size : Nat; item : Fin size -> IO sample }` — `Fin` makes
out-of-bounds unrepresentable; `fromVect` (in-memory), `fromIndexed size cb` (file/IO), `idxDataset`
(MNIST-family, lifts the idx C reader). `DataStream { next : IO a; epochLen : Maybe Nat }` —
`stream spec ds` iterates a dataset in (shuffled) index order via the Fisher-Yates C engine
(reshuffle on epoch wrap), `generate ioAction` wraps a raw feed (synthetic/RL), `batched` collates
`(Tensor [i], Tensor [o])` pairs into `([b,i],[b,o])` C-side (catAllTensors + reshape, no readback;
`batched1` for the single-tensor shape). **Named `DataStream` not `Data.Stream`** — the `Data.*`
namespace collides with `data/` (gitignore × case-insensitive APFS), base `Data.Stream`, and
`Prelude.Stream.Stream`. Legacy `DataPoint`/`TensorDataPoint`/`RecurrentDataPoint`/`TwoPhaseDataPoint`
+ `DataLoader` coexist until the example sweep.

### Optimizer

`Optimizer.idr`: four IO constructors `sgd` / `rmsprop` / `adam` / `adamW` × `OptimOpts` (beta1/beta2/eps/clip/groups, `defaultOpts` = PyTorch defaults, record-update to override). Algorithm-specific knobs sit on the constructor that owns them — `rmsprop {alpha} {momentum}`, `adamW lr weightDecay opts`, `adam {scope="actor_"} lr opts` for per-network optimizers (scope routes to the AdamGroup prim). `groups := [("bert.", 0.0)]` sets per-prefix LR overrides at construction (0 freezes; params registered after construction miss the walk — construct optimizers after the networks). Schedules: `withSchedule sched opt` + `tick opt epoch` (interim driver spelling `{ beforeEpoch := tick opt }`). Single `nativeTrainStep opt loss` runs zero_grad → backward → clip → step; use `NormClip` for recurrent models. The `native*` constructors and `applySchedule` still exist for current examples and die at the migration sweep.

### Model serialization

Backend-agnostic SafeTensors (`.safetensors`) via `Checkpoint` module: `saveAll` + `load path opts : IO (Either LoadError ())` with `LoadOpts {allowCast = False, only : Maybe String}` (`only` = prefix-filtered warm-start; registry-miss is a skip, not an error); `saveOptimizer` / `loadOptimizer` for optimizer state. The Bool-returning `loadModel*` wrappers persist for current examples until the migration sweep. Python interop: PyTorch loads via `safetensors.torch.load_file()`, MLX via `mx.load()`.

Training-loop integration: attach a `CheckpointPolicy` (built by `fileCheckpoint dir everyN keepBest opt`) to a `TrainConfig` via `withCheckpoint`. `runTrainingIO` then auto-saves every N epochs to `<dir>/last`, keeps the best to `<dir>/best`, resumes from `<dir>/last` if present, and reloads best at the end (return-best). Resume scalars (epoch, best metric) live in a `trainer_state.json` sidecar; safetensors stays the only on-disk format. Examples expose `--checkpoint-dir` / `--resume` / `--checkpoint-every` (gpt, transformer, ntm-copy, dnc-copy). See design-decisions.md "Training-loop checkpointing".

Foreign HuggingFace checkpoints — where param names + storage shapes diverge from idris-ml's conventions — are handled by `packages/idris-transformers/`. That package contains one Idris module per HF architecture (`HfBert.idr`, etc.) whose params and shapes match HF on-disk, so loading is plain `loadModel "model.safetensors"` with no remap or shape-split machinery in core. The module IS the adapter, expressed as type-checked code. User guide at `docs/users/idris-transformers.md`; design rules at `packages/idris-transformers/CONVENTIONS.md`. The worked example `Example/HfBertInference.idr` loads `google/bert_uncased_L-2_H-128_A-2` and matches HF transformers' Python forward output to within 4e-4 (gated by `make test-hf-bert-roundtrip`).

Fine-tuning HF-loaded models is supported as of the 2026-06-07 closure. Three primitives: (a) `loadModelPrefix path pfx` in `Checkpoint.idr` — load only safetensors keys whose name starts with `pfx` (warm-start a backbone while keeping a fresh head at its init). (b) `freezeByPrefix opt pfx` in `Train/Freeze.idr` — bulk-freeze every registered param whose name starts with `pfx` by zeroing its per-param LR override on the optimizer (composes with a single optimizer; no two-optimizer plumbing). (c) `BertForSequenceClassification` head module in `HfBertForClassification.idr` — registers `classifier.weight` / `classifier.bias` alongside the backbone, forward composes `hfBertForward`'s pooled `[CLS]` with a 1-D `tlinear`. The worked example `Example/BertClassifyFinetune.idr` trains a tiny BERT from scratch on a synthetic 3-class task to 100% in seconds across tape / torch / mlx-cpu; the real-text path (tokenizer + attention mask) and LoRA / PEFT are parked as TODO follow-ups. See `docs/users/idris-transformers.md` "Fine-tuning HF-loaded models".

### Type-safety conventions

The codebase has **zero `believe_me`** and **zero `unsafePerformIO`**. Keep it that way.
- Nat arithmetic: prefer `Tensor.splitAt` for reshape/flatten; route multiplicative shape arithmetic through `TVec`/`TMat` aliases (raw `Tensor [4 * o] d` hangs the type-checker).
- `decEq`+`Refl` to unify a generic `{n : Nat}` with a specific value in a branch.
- `rewrite sym prf in expr` to convert between provably-equal types.
- Device phantom: `Tensor dims (0 ex : Executor)` is erased at runtime; `toExecutor` (or `toExecutorChecked` for the EAFP-gated variant) is the only intentional device bridge.

## Workflows

### Adding new examples

1. Find paper/implementation for ground truth, add to References.
2. Port to `packages/pytorch/torch_ref/models/`, add tests + benchmark. Verify `make test-e2e-pytorch-ref && make ref-lint && make ref-typecheck`.
3. Implement in `packages/idris-ml-examples/src/Example/`, add to `Bench.idr` + Makefile. Verify `make test && make bench-compare`.

Commit at each step. PyTorch is the correctness oracle.

### Test-driven development (cross-cutting — governs all new capability work)

**Default to TDD** for any new behaviour-bearing change: a new kernel, a new dtype rung, a new layer instance, a new example, a new bug fix. Write the test first; run it; **observe it failing for the right reason** (a value mismatch, a wrong `tensor_dtype_name`, an `abort`, a NaN, a wrong gradient). A compile error or missing-symbol/link error does **not** count as red — "it links" is not "it works". If the unit is too coarse to fail-for-the-right-reason because the symbol doesn't exist yet, shrink to the smallest behavioural probe that can fail, or use the skip-flag pattern below.

**Record the red** in the conversation and in the implementing commit's body (`RED before this commit: <assertion>`). This is the evidence the cycle happened; without it the step was skipped.

**Reconciling TDD with build-green-per-commit + "test gates must run in CI"** (`feedback_test_gates_must_run_in_ci`): a test must never be *pushed* red. Exactly two allowed commit shapes:

- **(i) skip-flag** — commit the test present-but-skipped (CI green), then the implementing commit *removes the skip* and turns it green. The red is observed locally before the skip is added. Used when the implementation lands across multiple commits (Phase 3's per-rung gradcheck ladder, Phase 5's `PRECISION_DEMO_READY=0/1` gate).
- **(ii) paired commit** — observe red locally, then commit test + implementation together in one commit whose body records the red. Used when a skip flag would be more ceremony than the change warrants.

**Test layer to use** (pick the one the change actually drives):
- **C unit tests** (criterion `test_*.c` files next to the backend sources, `make test-unit-c-{tape,torch,mlx}`) — backend-side dtype/kernel/lifetime work. Add assertions under the relevant `#ifdef`; verify on **all three** backends, not just the primary (regression on the non-primary backend is the bug class this catches).
- **F32 gradcheck oracle** (tape T29 block) — when extending F32 routing to a new kernel: paired F32-vs-F64 contract with tag-propagation + forward-tol + grad-tol asserts.
- **Idris unit tests** (`packages/idris-ml/test`, `make test`) — typed-surface, smart-constructor, training-loop work.
- **`.expect` example outputs** (`make test-examples`) — user-visible example behaviour. Author the fixed expected stdout first; the example is RED until step 2 writes it (gated by a `<EXAMPLE>_READY` Makefile var for the skip-flag shape).

**No "linked = green"**. The Phase 1 (unified FFI dispatch) slip — entry-point commits 1.1–1.4 shipped with only compile/link coverage; the behavioural test (`db1e4fb`) followed days later — is the failure mode this section exists to prevent.

**Coverage policy** — what "covered" means for the three backends, the principled-exclusion list, and the contributor checklist for adding new ops live in [`docs/develop/coverage-policy.md`](docs/develop/coverage-policy.md). Run `make coverage-gap-probe` to see current OP_* and FFI-symbol gaps. The three-axis target (symbol coverage + OP_* backward coverage + F32 paired oracle) is the yardstick; C-line % is advisory only.

### Verification procedure on completion (cross-cutting — governs every landed change)

**Every completed piece of work ends with an explicit, runnable verification procedure handed to the user.** Not "it should work" / "the tests passed" — the actual commands the user types to confirm the change does what was claimed, ordered cheapest-first, with the expected output called out for each layer.

The procedure has three layers when applicable; pick the ones that match the change:

- **Cheapest — automated gate already in CI**: name the `make test-*` / `scripts/check-*` target that exercises the change, and the expected pass line. If no existing target covers it, that's a hole — wire one in this commit (or file the gap as a follow-up row, not "skip verification"). Per `feedback_test_gates_must_run_in_ci`, test gates must run in CI; an unverified gate is not a gate.
- **End-to-end — observable behavior**: the actual user-facing invocation (`make example-X` / a perf-run / a roundtrip test) with the env vars + flags that exercise the change, plus what to look for in stdout / on disk / in the perf log. Wrap heavy commands with `caffeinate -i nice -n 19 env MAKEFLAGS=-j2` per the heavy-command convention. If the run is more than a few seconds, include the paired `scripts/perf-run.sh` call per `feedback_no_expensive_run_without_log`.
- **External-tool inspection** (when the change produces artifacts users will interrogate): the Python / shell snippet that loads the artifact and shows the expected shape / keys / values. Worked example: the activation-dump landing (2026-06-09, see CHANGELOG entry) — `python3 -c "from safetensors.numpy import load_file; ..."` showing the `__act/<label>/<i>` keys and per-layer shapes.

**Don't conflate verification with TDD.** TDD's red-then-green is *during* implementation; verification is what the user runs *after* the landing. A passing test in CI is necessary, not sufficient — the verification block tells the user how to convince themselves the feature does what the commit message claims, not just that the tests didn't regress.

**Don't outsource it.** "Run `make test`" is not a verification procedure for a new behavior; it's a regression check. Name the specific assertion / line / artifact that proves *this* change works, not a generic gate that would pass whether or not the change landed.

**Tone**: terse, copy-paste ready, the user shouldn't have to assemble it. Three log levels / build flags / env vars get a table, not prose. Match the response shape the activation-dump "how do I test this?" answer used.

### Alignment policy (cross-cutting — governs all example work)

**Identical defaults**: Idris examples and PyTorch references MUST use identical defaults for all hyperparameters (lr, batch size, epochs, seed, architecture, init). When a discrepancy is found, adopt the better practice in BOTH. When changing an example, always update both sides. See `docs/develop/reference-alignment.md`.

**Multi-seed convergence is required**: a single-seed pass is not a convergence claim. Validate on ≥ 5 seeds and report the pass rate (e.g. "converges 5/5 for REINFORCE on CartPole", "converges 4/7 for A2C on CartPole"). Single-seed convergence at seed=42 has hidden real bugs here — A2C shipped as "converges to 200" based on one seed; multi-seed testing later exposed that the Idris-side optimizer wasn't updating the actor's weights at all. RL is noisy; use PyTorch's pass rate as the target, flag when Idris differs.

**Architectural alignment — DO NOT pivot silently**: divergent architectures destroy the implementation-vs-configuration signal. If Idris' Network chain can't express the PyTorch architecture, update PyTorch to match Idris, not the other way around. Hyperparameter changes must land on both sides in the same commit, and PyTorch must still converge after the change. Divergences are a **refactor**, not a fix: commit explicitly, name the divergence, record it in `docs/develop/reference-alignment.md`. **Process**: when a convergence issue appears on one side, the first action is to align configs so both sides run the same experiment — only then can the remaining gap be attributed to implementation.

### Performance documentation regime

Four files, each with a distinct role. Don't conflate them; updating the wrong one loses information.

- **`docs/develop/perf-log.jsonl`** — *append-only* raw measurement log. One JSON object per line. `scripts/perf-run.sh` and `scripts/perf-baseline.sh` auto-append. Tagged with short commit hash (`+dirty` if uncommitted). **Never edit/delete prior entries** — historical numbers are regression evidence. If a measurement is invalid, append a follow-up entry saying so. Query via `jq` (cookbook in `perf-log.md`).
- **`docs/develop/perf-log.md`** — schema documentation for the JSONL. Don't add new entries here.
- **`docs/develop/perf-baseline.md`** — *current-state* table. Re-written, not appended. Snapshot of every example × backend ratio at the latest commit.
- **`docs/develop/perf-changes.md`** — *append-only* log of every perf *change*. One entry per change with motivation / change / impact / commit / outcome (landed / reverted / partial). Reverted attempts stay — negative results save future redoing.

### Performance optimization workflow

**Post-change measurement (required after every landable commit)**: use the auto-logging scripts so results land in `perf-log.jsonl`. **Never** hand-write JSONL entries; **never** use `make bench-compare` for post-change gating (it doesn't log).
- `scripts/perf-run.sh <example-key> <backend>` — single (example, backend) measurement.
- `scripts/perf-baseline.sh <example-key> <backend>` — Idris-vs-PyTorch ratio with two-point timing.
- `scripts/perf-sweep.sh [--examples …] [--cells tape,torch,mlx-cpu,mlx-gpu]` — **canonical for cross-backend cascade changes** (typeclass cascade, C ABI, lifecycle work). One PyTorch ref per example, cached across cells. A single-backend `bench-compare` on cross-backend work hides per-backend regressions and leaves no log entry.

**No expensive run without a perf log.** Any inference or training run that takes more than a few seconds — whether you're verifying correctness, eyeballing output, comparing backends, debugging, or doing ad-hoc exploration — MUST land in `perf-log.jsonl`. Run via `scripts/perf-run.sh` (or the equivalent script) instead of `make example-*` directly; if the example isn't yet wired into `perf-run.sh`, wire it in first (case arms are one line each). Build-only invocations (`make install`, `make backend`) are exempt — only the *run* must be logged. The rule applies to correctness gates too: `test-hf-bert-roundtrip` etc. are correctness scaffolding, but the underlying inference run is expensive and a paired `perf-run.sh` entry is the cheapest way to keep "yesterday's wall" comparable. Re-running a measurement that's already in the log is fine; missing measurements are not.

**Ad-hoc local exploration**: `make example-profile` → `make bench-compare` (same batch, currently 16) → `python3 scripts/sweep.py` for systematic grids → update `docs/develop/performance-analysis.md` with fresh data. `bench-compare` is convenient for eyeballing but does *not* log to `perf-log.jsonl` — pair it with a `perf-run.sh` measurement if you're keeping the result.

**Heavy-command convention — always wrap with caffeinate + nice + capped parallelism.** Any command that takes more than ~1 minute on this codebase (Idris elaboration of a real example, full backend rebuild, perf sweeps, multi-example test runs) is "heavy" and competes for CPU + holds the laptop awake. Wrap *every* such invocation:

```bash
caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make …
```

- **`caffeinate -i`** (macOS) prevents idle sleep — without it, a closed lid or display sleep suspends/kills the build mid-run. Observed 2026-06-03/04 multiple times: harness-spawned builds stranded with no exit code after the laptop slept.
- **`nice -n 19`** is the lowest CPU priority. Foreground apps preempt the build; the build still progresses, just slower. Without it, Chez elaboration peaks at 17-23 GB and saturates cores, making the host VM unresponsive (the user can't browse / edit / run tests in parallel).
- **`MAKEFLAGS="-j2"`** caps the parallel C++ compile to 2 cores. The Idris elaboration phase is single-threaded so `-j` doesn't help it; capping the C++ side stops the parallel-link phase from pegging all cores.

The wrapper `scripts/perf-run-quiet.sh` bakes all three in for perf measurements (`scripts/perf-run-quiet.sh hf-llama-generate torch` etc.); use it as the default for any `perf-run`-style invocation. For gates / Make targets that aren't wrapped via perf-run, inline the trio:

```bash
caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make BACKEND=torch test-hf-llama-roundtrip
caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make BACKEND=mlx test-transformers
caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make test-examples
```

Exceptions (~rare): `make install` / `make backend` of a hot tree (already-cached), short `make probe-foo` debug targets, anything that finishes in seconds. When in doubt, wrap — the overhead of `nice` + `caffeinate` is negligible on a fast command, and the cost of forgetting on a slow one is a slow host + a stranded build.

## Conventions

- **Indentation**: governed by `.editorconfig` per-extension. Honour it when writing new files — no editor enforces it in your environment. Quick read: `.idr` 2 spaces, `.py` 4 spaces (ruff format), `.c`/`.h`/`.cpp`/`.hpp` tabs (clang-format `ForIndentation`), `.{yml,yaml}` 2 spaces (spec), everything else (`.sh`, `.json`, `.md`, …) tabs (repo-wide default).
- **Formatters**: `make fmt` rewrites every source file in place (all languages); `make check-fmt` fails if anything is unformatted — run it before considering a change done. The `fmt` family is cross-language and lives in `mk/fmt.mk`; format is kept distinct from lint (per testing-taxonomy.md "format != lint"). Subtargets: `fmt-idris`/`check-fmt`'s `test-integration-lint-fmt` — **idris-fmt**, the repo's own compiler-native formatter (`packages/idris-fmt/`; parses with the compiler's own parser, gates every reformat behind a round-trip oracle, so it can never change a file's meaning; scope is whitespace hygiene + import-sort + `:`/`=`/`=>` alignment + FC-driven reindentation, each pass independently oracle-gated with identity fallback — reindent uses the deep `deepSig` oracle since whitespace-only edits leave `codeSig` unchanged). `fmt-py`/`check-fmt-py` — `ruff format`. `fmt-c`/`check-fmt-c` — `clang-format -i` over all backends. The **linters** are separate: `make lint-py` (ruff check + vulture) **and `make typecheck-py`** (pyright strict on every Python-bearing package; config roots are `packages/pytorch/pyproject.toml` for the pytorch tree and per-surface `pyrightconfig.json` / `[tool.pyright]` selected via `-p` for the rest — pyright discovers config from the project root, not per-file like ruff); `make lint-c` (cppcheck + clang-tidy with deny-list config in `.clang-tidy`). No formatter for shell / markdown / YAML — the `.editorconfig` indent rule is the only contract.
- **clang-tidy include hygiene**: `misc-include-cleaner` is disabled in `.clang-tidy` (the `-include rename_tape.h` flag hides every renamed `backend.h` symbol from include-cleaner, producing ~250 architectural FPs) but is exercised as a separate gate `make lint-c-include-cleaner` that runs WITHOUT the rename. Suppression conventions when adding a new BLAS/macro-provider include: `// IWYU pragma: keep` as a **trailing** comment on the include line (the preceding-line form doesn't suppress reliably in clang-tidy 21); `// NOLINTNEXTLINE(misc-include-cleaner)` per call site for macOS-Apple-SDK FPs that don't fire on Linux (`abort`, `cblas_*`, `vDSP_*`). Don't NOLINTBEGIN/END regions unless the call spans multiple lines past NEXTLINE's scope.
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first, then internal modules alphabetically
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`. ~50 char subject, imperative present tense. One logical change per commit. No ads/branding in messages or PRs.
- **Section dividers**: `----------------------------------------------------------------------` with titles, Layer.idr style
- **Documentation**: Update CLAUDE.md, `docs/develop/design-decisions.md`, and `TODO.md` when adding features, changing architecture, or making design decisions. `TODO.md` holds the open backlog only; when a row is finished, move its closure entry to `CHANGELOG.md` (most-recent-first) rather than into a Done section in `TODO.md`.
- **No ephemeral plan labels in committed docs**: don't reference "Phase 2.1b" / "Job 4 Phase B" / "Step 3" etc. in committed prose (design-decisions, gotchas, CLAUDE.md, in-code comments). Those labels live in the working plan file and are meaningless six months later. Anchor to the *commit* hash (`e67fe15`), a *date* (`2026-05-13`), or the *feature name* ("the multi-link refactor", "the rename + alias machinery") instead. Plan labels are fine in commit messages and conversation — that's their natural lifetime.

## Gotchas

See [`docs/develop/gotchas.md`](docs/develop/gotchas.md) for the comprehensive list (Idris 2 / Chez Scheme traps, training & numerics, MLX backend, NTM/DNC-specific, Gym, RL algorithms, architecture & infrastructure). High-leverage entries to remember:

- **`total` is a keyword in Idris 2** — use `numEpochs` / `totalEpochs` etc.
- **`paramId` is required for gradient flow** — tensors without paramId are invisible to the optimizer.
- **ParamId scoping for multi-network examples** (A2C / PPO / SAC) — scope each network's prefix distinctly (`actor_`, `critic_`, `q1_`, `q2_`, `q1tgt_`, `q2tgt_`). The bug class is silent gradient leakage between networks.
- **`logSoftmax` + `nllLoss`** — apply `tlogSoftmax1d` to raw logits and feed into `tnllLoss`; do NOT put a softmax layer in the network chain (creates 1/pp intermediates up to 1e6).
- **Elementwise `(*)`** — `Tensor`'s `Num` uses elementwise multiply; use `(<>)` for matmul (PyTorch's `@`).
- **Chez output buffering** — `stdout` is fully buffered when piped. Prefix long-running background commands with `stdbuf -oL`.
- **Large Nat type-level reduction** — Idris-2 Peano Nats hang the type-checker for dims > ~1000 and choke on multiplicative shape literals. Route through `TVec`/`TMat` aliases; place identity layers (dropout, batch norm) only at smaller dims.
- **`Data.Nat` stdlib functions are recursive at runtime too** — `Data.Nat.lte`/`divNat`/`modNatNZ` compile to recursive Peano walks even though `Nat` is `Integer` underneath (`div 256 2` = 128 `cond`/`sub1` cycles). Cast to `Int` and use `Int div`/`Int mod` in hot paths; the `Ord_Nat` comparators (`<`, `<=`) are fine. Details + the `posEncVal` incident in `docs/develop/gotchas.md`.
- **Gaussian policy entropy must be Tensor-typed** — building entropy from `prim__item1d` scalars silently zeroes the gradient (this was the V1 A2C bug).
