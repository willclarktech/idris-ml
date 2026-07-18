# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## Writing style — user-facing prose (IMPORTANT: read before touching any docs/*.md)

**LLM output drifts into a recognizable house style that this repo's docs must not have.** These rules were extracted from repeated review feedback on `docs/users/why-idris-ml.md` (2026-07); every one of them was violated more than once before being named. Re-read this section before writing or editing user-facing prose, and sweep your own draft against it before presenting it.

**Voice — the banned patterns (each observed in real drafts):**
- **No reveal/announcer framing.** Never introduce a point by announcing its significance — "the real cost", "the deeper point", "that's the telling part", "This is worth pausing on", "here's the part hasktorch can't reach", "the deciding fact:". State the significant thing directly; if the sentence's only job is to promise the next sentence matters, delete it.
- **No epigrams or punchlines.** "The extension exists; the ecosystem was never built on it", "dodges one instance without touching the class", "doesn't survive a type system that counts", "sits exactly where the footgun lives". If a sentence reads like a mic-drop, rewrite it to carry information instead.
- **No abstract metaphors.** "Floats free of reality", "no rung to stand on", "you pay full price", "the language's day job". One exception: an *established, cited* metaphor may be reused as a callback (the ["keyhole"](https://www.georgeho.org/tensor-computation-libraries/) in why-idris-ml.md).
- **No anthropomorphized machinery.** "Nothing looked", "nothing objects", "the graph builder never looks", "the failure waiting for execution". Say what is or isn't checked, and when.
- **No intensifiers.** "Truly", "genuinely", "really", filler-"actually". They add doubt, not force.
- **No narrated make-believe; present tense throughout.** Don't describe a hypothetical as a concrete past event ("the shapes were fixed the moment you wrote the layers" → "both shapes are known the moment you write the layers"), and don't let scenario setup slip into past tense mid-sentence ("your model got big, so you drop it" → "your model is too big for memory, so you drop it"). State conditions as properties ("hardware that is linked into the build", not "was linked"). Verbatim captured error text is exempt.
- **No imputed reader opinions.** Not "we want the mismatch caught early" — say "ideally a program with this bug is unrepresentable", anchored to the project mantra (*make illegal states unrepresentable*).
- **Plain words.** No cute variation ("flavour" → "kind"); en-US spelling; simple and familiar beats clever. No tool-grabbing idioms ("you reach for the experimental linear-base library", "its QualifiedDo sugar" → "it requires the linear-base library and the QualifiedDo extension").
- **Ration em-dashes.** Dash-heavy prose is an LLM signature, and the dash is the joint most of the banned patterns hinge on ("— linear types do" is an epigram pivot). Prefer parentheses for asides, a semicolon or colon for a pivot, or more, simpler sentences. Rough budget: one dash construction per paragraph, and only where it does narrative work.

**Devices that ARE house style** (use these, not the ones above): direct questions as section pivots ("Does the static graph help this time?"); "To be fair to X" concessions before criticism; numbered enumerations "(1) … (2) …"; `> [!TIP]` blockquotes reserved for idris-ml thesis statements; second-person bug scenarios ("you move the model to the GPU and forget one tensor").

**Structure (framework-comparison sections):** five beats per feature — (1) the bug class, shown concretely; (2) how the alternative framework approaches it; (3) how far that gets, conceded plainly; (4) the remaining limitation; (5) how idris-ml addresses it. Framework order: PyTorch → TF 1.x → (Pyright where relevant) → hasktorch → idris-ml. The SAME example threads through every framework in a section (the 784/256/128/10 layers, batch 64). Be honest in both directions: competitors' wins are stated ("To be fair to Pyright, this actually catches our planted bug"), and so are idris-ml's limits (Idris can't guess `k + n = n + k` either — the difference is the fix is ordinary library code). Distinguish plain language vs extensions vs plugins vs library encodings when crediting a capability.

**Evidence:** every error message and every "compiles clean" claim is captured from a real toolchain run — never composed from memory or predicted. Code and error live in separate blocks (` ```text ` for errors); captured text stays verbatim, ugly internals included; the provenance footnote lists every toolchain used and gets updated when a new one enters.

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
  idris-transformers/ # HF-aligned model library on top of idris-ml (Transformers.Bert, Transformers.Gpt2, Transformers.Llama)
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
`tape-mlxcpu-torchcpu`). Each `(BACKEND, MLX_DEVICE, TORCH_DEVICE)` tuple keeps
its own warm ttc/install/dylib/exec tree, so switching sets is near-free instead
of triggering 60-min cascading re-elaboration. `clean` removes every set's tree;
`clean-set` just the active set; `clean-all` cascades to models + datasets +
venvs + `vendored/` + run-output dirs.

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
-- `UserDeviceTransfer` makes the generic `toDevice` work between any pair
-- (matching backendTag → fast intra-backend HW migration; differing → host
-- round-trip). Declare your own backend via a tag type with these instances
-- + a unique `backendTag`. See design-decisions.md "Open `d` parameter".
--
-- Availability gating (two gates; full doc device-availability-gating.md):
-- compile-time linkage via the empty `Linked ex` marker (emitted per build
-- into `HwConfig`, so a tape-only build can't even spell `MlxExecutor _`),
-- and runtime EAFP hardware-presence (construction shims → NULL handle;
-- `toDeviceChecked`/`attemptOn` lift NULL → `Left DeviceError`;
-- `availableDevices builtinDevices` probes the build's candidates).

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

Examples don't hardcode device/dtype — they reference `ExampleDevice` / `ExampleDType` from the Makefile-generated `BuildConfig.idr` (template `BuildConfig.idr.in`), whose `(BACKEND, MLX_DEVICE, TORCH_DEVICE)` → cell is:

  - `BACKEND=tape`                       → `TapeExecutor`, `F64`
  - `BACKEND=torch TORCH_DEVICE=cpu`      → `TorchExecutor TCpu`, `F64`
  - `BACKEND=torch TORCH_DEVICE=mps`      → `TorchExecutor TMps`, `F32`
  - `BACKEND=torch TORCH_DEVICE=cuda`     → `TorchExecutor (TCuda 0)`, `F64`
  - `BACKEND=mlx MLX_DEVICE=cpu`          → `MlxExecutor MCpu`, `F64`
  - `BACKEND=mlx MLX_DEVICE=gpu`          → `MlxExecutor MGpu`, `F32`

Types fix at elaboration, so the env is observed at build time and baked into `BuildConfig.idr` — switching modes is just a different `make install`, no source edits. (Same trick generates the per-build `Linked` instances in `HwConfig.idr`.)

#### `Nn/` — the models-as-records surface

The sole layer surface (`Layer/` is gone). Models are plain records of layers, and **a model is a single-owner LINEAR resource** threaded through `Control.Linear.LIO.L IO` — every `forward`/`recurStep`/`eval`/`freeze`/`fit` *consumes* the handle `(1 _ : l …)` and threads it back, so reusing a stale handle after freeze/eval (a silent no-op against the shared C params) is a **compile-time linearity error**. Tensors stay **unrestricted** (reverse-mode AD shares them); the linear discipline is only at model granularity. Gate: `make test-integration-typegate-linear-model`. `Nn.Module` is batched-first (`forward` sig under **Forward pass** below; a layer that can't batch isn't a `Module`). `Nn.Params` (base of `Module`/`Recurrent`) carries a flat read-only `params : l … -> List SomeParam` (ω, the `.parameters()` analogue) plus the linear `reflect`/`castGrad`/`discard`; `Nn.Seq` chains `Module`s (`~~>`/list-literal, endpoints-only index). `Module`/`Params` are **higher-kinded** over the `Nat -> Nat -> Executor -> DType -> GradMode -> Type` constructor; layers with extra config Nats lead them and trail the `(i,o)` pin (Conv2D, TransformerBlock). `GradMode` is on the model type; `eval`/`freeze`/`unfreeze`/`trainable` are linear. Recurrent/memory layers (RNN/LSTM/GRU/NTM/DNC) implement `Nn.Recurrent` (linear `recurStep`/`recurReset`, state in the record); mixed precision (`Nn.LinearMixed`'s `ModuleMixed`/`ParamsMixed`) lives on the same linear surface. Param names derive over the C registry via `Nn.Init` (`scoped`/`runInit`, the PyTorch `state_dict` convention); `Nn.Group.groupOf` returns a submodel's exact registry names for optimizer scoping (replaces substring-prefix matching). 19 layers ported; the Transformer is decomposed (`Nn.Attention` + a `TransformerBlock` stacked via `Seq`), not the legacy monolith. `Params` instances are hand-written (~3 lines each). Full design: `docs/develop/linear-types-and-effects.md`.

### Tensor lifecycle (wrapped-handle ABI)

`tensorPtr` is a Chez vector `#(tensor-handle-v2 tag raw)` (slot 1 = backend tag, slot 2 = raw pointer), not a raw pointer. Every Tensor-touching `%foreign` binds to a Scheme wrapper that unwraps via `(vector-ref a<i> 2)` on Tensor args and wraps + retains + registers with `idris-tensor-guardian` on Tensor returns. The wrap IS the value — Idris-Chez codegen can't elide it without eliding the Tensor. C-side refcount drives freeing on mlx; tape and torch carry no-op retain/release stubs. New FFIs go through `scripts/codegen/ffi_manifest.py` + `ffi-convert-to-scheme.py`; `make check-ffi-wrap-template` (CI preflight) enforces the template. Full model in `docs/develop/tensor-lifecycle.md`.

## Key Patterns

### Model composition

A model is a record of `Nn` layers or a `Nn.Seq` chain (`~~>` / list-literal, endpoints-only index, hidden dims existential). Layers are built in the `Init` monad; `runInit` / `runInitL` populates the C param registry, deriving names from the scope path — no hand-typed prefixes:

```idris
model <- runInitL (linear {i=2} {o=3})          -- a one-layer model

Model = Seq InputDim NumClasses Ex F WithGrad    -- a chain (hidden dims existential)
mkModel : Init Model
mkModel = pure (c1 ~~> reluA ~~> l ~~> Nil)
trained <- runInitL mkModel
```

`Nn.Init`'s `scoped` / `scopedChild` / `runInit` own naming (the PyTorch `state_dict` convention). **Parameters reach the optimizer only through the registry `runInit` populates** — there is no separate paramId to pass. `Nn.Group.groupOf submodel` returns a submodel's exact registry names for per-network optimizer scoping (`actor` / `critic`), replacing substring-prefix matching.

### Construction

`tensor {dims=[2,3]} (Const 0.5)` / `param "w" (Normal 0.0 0.02)` — one construction surface over `InitSpec` (`Zeros | Const x | Normal mu sd | Uniform lo hi | FromVect xs`; `fromRows` stacks a `Vect b (Vect i Double)` for the batch case). `FromVect`'s length is tied to `Numel dims` at compile time; `param` requires rank <= 4 (compile error past the C surface's ceiling) and always registers. Raw `prim__*` + `dtCreate*` construction lives in `Tensor.Internal` (backend authors only); the prim ratchet gate keeps examples from growing new raw-prim call sites.

### Forward pass

```idris
forward    : (1 _ : l i o ex dt g) -> Tensor [b,i] ex dt g -> L IO (LPair (!* (Tensor [b,o] ex dt g)) (l i o ex dt g))
forwardSeq : (1 _ : Seq i o ex dt g) -> Tensor [b,i] ex dt g -> L IO (LPair (!* (Tensor [b,o] ex dt g)) (Seq i o ex dt g))
```

`forward` / `forwardSeq` consume the linear model handle and thread it back (the output rides the `(!*)` bang). Every Tensor-handle-touching smart constructor (`tadd`, `tmul`, `ttanh`, etc.) is `IO`-typed. This is load-bearing: `withNoGrad (pure (tadd …))` would have fired the FFI *before* `noGradBegin` since `pure`'s argument is evaluated strictly. With IO typing the FFI body fires only on `<-` sequencing — inside the bracket. The helper `ioRerun : (() -> a) -> IO a` defers a pure body to IO without using the prelude's private `MkIO`; `Lazy a` was rejected because it memoizes.

**Expression ops**: row-select-by-index and elementwise arithmetic compose without hand recursions — `tgatherRows` ([b,n] × [b] double-valued-int indices → [b]; PyTorch `gather(1, ·)`), `tmaxRows` ([b,n] → [b]; `max(1).values`), and the infix aliases `(+.)` `(-.)` `(*.)` (elementwise) / `(*:)` (scalar-left) on plain evaluated tensors with bang notation: `tgt <- r +. !(gamma *: !(tmaxRows qNext))`. No `Num` instance, no IO-carrier operators (roadmap.md decision 5). Note `tmseLoss` is a *sum* reduction — scale by `1/n` for PyTorch's mean default. (`tgather` is the separate torch-only integer-dtyped 1-D surface.)

**Long eval loops on mlx need per-sequence `withNoGrad`**: a single outer bracket around `traverse evalOne batch` lets mlx Metal MTLBuffer count blow past the Tart/GHA VM ceiling before exit-drain fires. Push the bracket inside: `evalOne dp = withNoGrad $ do { ... }` (NTM-style) or `withNoGrad (evalEp …)` inside `evalN`'s recursion (RL-style). Tape/torch don't need this; the per-sequence pattern is cheap on both.

### Training — `fit` driver (Fit.idr)

```idris
(trained, epochs, loss) <- fitSupervised opt lossFn (batched dataStream) (simpleConfig 1000) model
```

One driver for everything. `EpochStep m batch = m -> batch -> IO (m, Double)`: the step owns
control-flow + the optimizer step + (optional) model-state threading; `fit` owns the epoch loop,
schedule `tick`, early stop, checkpointing, NaN handling, and mlx generation hygiene (all via
`Train.Engine.runEpochLoop`). `fit` reuses `TrainConfig`.

- **Supervised (90%)**: `fitSupervised opt lossFn stream cfg model` — pass a loss fn, never call
  `trainStep` yourself. `fitSupervisedMixed opt gradScaler lossFn …` for mixed precision.
- **Recurrent / two-phase**: a `Step` that folds over timesteps into one loss — no driver variant.
- **RL / custom**: pass your own `EpochStep` to `fit` (rollout + your own `trainStep`s + state
  threading), or compose the exported engine pieces (`runEpochLoop`, `withEpoch`, `postEpoch`,
  `earlyStopMachine`) directly for multi-step loops fit can't express (DQN replay, PPO K-epoch).

See design-decisions.md "`fit` driver". Data: `Dataset { size : Nat; item : Fin size -> IO sample }`
(`fromVect`/`fromIndexed`/`idxDataset`) + `DataStream` (`stream shuffleSpec ds` / `generate ioAction`
/ `batched` collating `(Tensor [i], Tensor [o])` pairs into `([b,i],[b,o])` C-side). See **Data**
below.

### Data (Dataset.idr / DataStream.idr)

PyTorch's three orthogonal joints: `Dataset` (indexed access) / `ShuffleSpec` (order) / `DataStream`
(batching+collation). `Dataset { size : Nat; item : Fin size -> IO sample }` — `Fin` makes
out-of-bounds unrepresentable; `fromVect` (in-memory), `fromIndexed size cb` (file/IO), `idxDataset`
(MNIST-family, lifts the idx C reader). `DataStream { next : IO a; epochLen : Maybe Nat }` —
`stream spec ds` iterates a dataset in (shuffled) index order via the Fisher-Yates C engine
(reshuffle on epoch wrap), `generate ioAction` wraps a raw feed (synthetic/RL), `batched` collates
`(Tensor [i], Tensor [o])` pairs into `([b,i],[b,o])` C-side (catAllTensors + reshape, no readback;
`batched1` for the single-tensor shape). **Named `DataStream` not `Data.Stream`** — the `Data.*`
namespace collides with `data/` (gitignore × case-insensitive APFS), base `Data.Stream`, and
`Prelude.Stream.Stream`.

### Optimizer

`Optimizer.idr`: four IO constructors `sgd` / `rmsprop` / `adam` / `adamW` × `OptimOpts` (beta1/beta2/eps/clip/groups, `defaultOpts` = PyTorch defaults, record-update to override). Algorithm-specific knobs sit on the constructor that owns them — `rmsprop {alpha} {momentum}`, `adamW lr weightDecay opts`. Per-network optimizers are scoped after construction via `Train.Freeze`, not a constructor field. `groups := [("bert.", 0.0)]` sets per-prefix LR overrides at construction (0 freezes; params registered after construction miss the walk — construct optimizers after the networks). Schedules: `withSchedule sched opt` + `tick opt epoch`. Single `trainStep opt loss` runs zero_grad → backward → clip → step; use `NormClip` for recurrent models.

### Model serialization

Backend-agnostic SafeTensors (`.safetensors`) via `Checkpoint` module: `saveAll` + `load path opts : IO (Either LoadError ())` with `LoadOpts {allowCast = False, only : Maybe String}` (`only` = prefix-filtered warm-start; registry-miss is a skip, not an error); `saveOptimizer` / `loadOptimizer` for optimizer state. Python interop: PyTorch loads via `safetensors.torch.load_file()`, MLX via `mx.load()`.

Training-loop integration: attach a `CheckpointPolicy` (built by `fileCheckpoint dir everyN keepBest opt`) to a `TrainConfig` via `withCheckpoint`. `fit` then auto-saves every N epochs to `<dir>/last`, keeps the best to `<dir>/best`, resumes from `<dir>/last` if present, and reloads best at the end (return-best). Resume scalars (epoch, best metric) live in a `trainer_state.json` sidecar; safetensors stays the only on-disk format. Examples expose `--checkpoint-dir` / `--resume` / `--checkpoint-every` (gpt, transformer, ntm-copy, dnc-copy). See design-decisions.md "Training-loop checkpointing".

Foreign HuggingFace checkpoints (param names/shapes diverge from idris-ml's) are handled by `packages/idris-transformers/` — one Idris module per HF architecture (`Transformers.Bert`, …) whose params/shapes match HF on-disk, so loading is plain `fromPretrained "<dir>"` (reads `config.json` + `model.safetensors`), no remap machinery in core. The module IS the adapter. Guide: `docs/users/idris-transformers.md`; rules: `packages/idris-transformers/CONVENTIONS.md`. `Example/BertInference.idr` matches HF's Python forward to 4e-4 (`make test-e2e-bert-roundtrip`).

Fine-tuning HF-loaded models has three primitives — prefix-filtered subset-load (`load … {only := Just pfx}` in `Checkpoint.idr`), `freezeByPrefix opt pfx` (`Train/Freeze.idr`, zeroes per-param LR on a single optimizer), and a `BertForSequenceClassification` head — with the worked example `Example/BertClassifyFinetune.idr`. Full detail: `docs/users/idris-transformers.md` "Fine-tuning HF-loaded models".

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

**Default to TDD** for any behaviour-bearing change (kernel, dtype rung, layer, example, bug fix). Write the test first; **observe it failing for the right reason** (value mismatch, wrong `tensor_dtype_name`, `abort`, NaN, wrong gradient). A compile/link error does **not** count as red — "it links" is not "it works". If the unit is too coarse to fail because the symbol doesn't exist yet, shrink to the smallest behavioural probe, or use the skip-flag shape below.

**Record the red** in the conversation and the implementing commit body (`RED before this commit: <assertion>`) — without it the step was skipped.

**Never push a test red** (`feedback_test_gates_must_run_in_ci`). Two allowed commit shapes: **(i) skip-flag** — commit the test present-but-skipped (CI green), then the implementing commit removes the skip; used when the impl lands across multiple commits. **(ii) paired commit** — observe red locally, commit test + impl together with the red recorded in the body.

**Test layer** (pick what the change drives):
- **C unit tests** (criterion `test_*.c`, `make test-unit-c-{tape,torch,mlx}`) — backend dtype/kernel/lifetime work; verify on **all three** backends, not just the primary.
- **F32 gradcheck oracle** (tape T29 block) — extending F32 routing: paired F32-vs-F64 contract (tag-propagation + forward-tol + grad-tol).
- **Idris unit tests** (`packages/idris-ml/test`, `make test`) — typed-surface / smart-constructor / training-loop work.
- **`.expect` outputs** (`make test-examples`) — user-visible example behaviour; author the expected stdout first (`<EXAMPLE>_READY` gates the skip-flag shape).

**No "linked = green"** — compile/link coverage alone has shipped broken behaviour here; the behavioural test is the gate.

**Coverage policy** — the "covered" definition, principled-exclusion list, and new-op checklist live in [`docs/develop/coverage-policy.md`](docs/develop/coverage-policy.md). `make test-coverage-gap-probe` shows OP_*/FFI-symbol gaps; the three-axis target (symbol + OP_* backward + F32 oracle) is the yardstick, C-line % advisory.

### Verification procedure on completion (cross-cutting — governs every landed change)

**Every completed change ends with an explicit, runnable verification procedure handed to the user** — the actual commands to confirm the change does what was claimed, cheapest-first, with expected output per layer. Not "it should work" / "the tests passed". Three layers (pick what matches):

- **Cheapest — CI gate**: name the `make test-*` / `scripts/check-*` target + expected pass line. No covering target = a hole; wire one in this commit (or file a follow-up row, not "skip verification") — an unverified gate is not a gate.
- **End-to-end — observable behavior**: the user-facing invocation (`make example-X` / perf-run / roundtrip) with the env+flags that exercise the change and what to look for (stdout / disk / perf log). Wrap heavy commands per the heavy-command convention; >few-seconds runs need a paired `perf-run.sh` entry.
- **External-tool inspection** (for artifacts users interrogate): the Python/shell snippet that loads the artifact and shows the expected shape/keys/values.

**Verification ≠ TDD**: red-then-green is *during* implementation; verification is what the user runs *after*. **Don't outsource it** — "run `make test`" is a regression check, not verification; name the specific assertion/line/artifact that proves *this* change works. Tone: terse, copy-paste ready; tables for >2 flags/levels.

### Alignment policy (cross-cutting — governs all example work)

**Identical defaults**: Idris examples and PyTorch references MUST share all hyperparameter defaults (lr, batch, epochs, seed, architecture, init). On a discrepancy, adopt the better practice in BOTH, same commit. See `docs/develop/reference-alignment.md`.

**Multi-seed convergence required**: a single-seed pass is not a convergence claim — validate on ≥ 5 seeds and report the pass rate (e.g. "5/5 REINFORCE on CartPole"). Single-seed at seed=42 has hidden real bugs here (A2C "converged" on one seed while the Idris optimizer wasn't updating the actor at all). RL is noisy; use PyTorch's pass rate as the target.

**Don't pivot architecture silently**: divergent architectures destroy the implementation-vs-config signal. If Idris' chain can't express the PyTorch architecture, change PyTorch to match Idris (not vice-versa), same commit, PyTorch must still converge. Divergences are a **refactor** — commit explicitly, name it, record in `reference-alignment.md`. First action on any one-sided convergence issue: align configs so both sides run the same experiment.

### Performance documentation regime

Four files, distinct roles (don't conflate; schema + jq cookbook in `perf-log.md`):
`perf-log.jsonl` (**append-only** raw measurements, auto-appended by the perf
scripts — **never edit prior entries**, they're regression evidence),
`perf-log.md` (JSONL schema), `perf-baseline.md` (current-state ratio table,
re-written not appended), `perf-changes.md` (**append-only** log of each perf
change, incl. reverted attempts — negative results save future redoing).

### Performance optimization workflow

**Post-change measurement (required after every landable commit)**: use the
auto-logging scripts so results land in `perf-log.jsonl`. **Never** hand-write
JSONL; **never** gate on `make bench-compare` (it doesn't log).
- `scripts/perf-run.sh <example-key> <backend>` — single measurement.
- `scripts/perf-baseline.sh <example-key> <backend>` — Idris-vs-PyTorch ratio.
- `scripts/perf-sweep.sh [--examples …] [--cells …]` — **canonical for
  cross-backend cascade changes**; a single-backend `bench-compare` hides
  per-backend regressions and leaves no log entry.

Ad-hoc grids: `python3 scripts/sweep.py`, then update `docs/develop/performance-analysis.md`.

**No expensive run without a perf log.** Any inference/training run over a few
seconds — correctness, eyeballing, debugging, ad-hoc — MUST land in
`perf-log.jsonl` via `perf-run.sh` (wire the example into the script first if
missing; case arms are one line). Build-only (`make install`/`backend`) is
exempt; correctness gates (`test-e2e-bert-roundtrip` …) need a paired
`perf-run.sh` entry. Re-running a logged measurement is fine; missing ones aren't.

**Heavy-command convention** — wrap any >~1-min command (real-example
elaboration, full rebuild, perf sweeps, multi-example test runs):

```bash
caffeinate -i nice -n 19 env MAKEFLAGS=-j2 make …
```

`caffeinate -i` stops idle-sleep killing the build; `nice -n 19` keeps the host
responsive (Chez elaboration peaks 17-23 GB); `MAKEFLAGS=-j2` caps the parallel
C++ compile (the elaboration phase is single-threaded). `scripts/perf-run-quiet.sh`
bakes all three in — default for any perf-style run; inline the trio for
Make-target gates. Exempt: hot-tree `make install`/`backend`, short debug
targets, anything finishing in seconds.

## Conventions

- **Indentation**: governed by `.editorconfig` per-extension. Honour it when writing new files — no editor enforces it in your environment. Quick read: `.idr` 2 spaces, `.py` 4 spaces (ruff format), `.c`/`.h`/`.cpp`/`.hpp` tabs (clang-format `ForIndentation`), `.{yml,yaml}` 2 spaces (spec), everything else (`.sh`, `.json`, `.md`, …) tabs (repo-wide default).
- **Formatters**: `make fmt` rewrites every source file in place; `make check-fmt` fails if anything is unformatted — run it before considering a change done (`mk/fmt.mk`; format ≠ lint, per testing-taxonomy.md). Per-language: `fmt-idris` (**idris-fmt**, the repo's compiler-native formatter in `packages/idris-fmt/` — round-trip-oracle-gated so it can't change meaning; whitespace + import-sort + `:`/`=`/`=>` alignment + reindentation), `fmt-py` (ruff format), `fmt-c` (clang-format). Linters are separate: `make lint-py` (ruff check + vulture), `make typecheck-py` (pyright strict per package), `make lint-c` (cppcheck + clang-tidy, deny-list in `.clang-tidy`). No formatter for shell/markdown/YAML — `.editorconfig` indent is the only contract.
- **clang-tidy include hygiene**: `misc-include-cleaner` is disabled in `.clang-tidy` (the `-include rename_tape.h` flag causes ~250 architectural FPs) and run rename-free as a separate gate `make lint-c-include-cleaner`. Suppression conventions (IWYU-pragma placement, per-call-site `NOLINTNEXTLINE` for Apple-SDK FPs) are documented in `.clang-tidy`.
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first, then internal modules alphabetically
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`. ~50 char subject, imperative present tense. One logical change per commit. No ads/branding in messages or PRs.
- **Section dividers**: `----------------------------------------------------------------------` with titles, Layer.idr style
- **Documentation**: Update CLAUDE.md, `docs/develop/design-decisions.md`, and `TODO.md` when adding features, changing architecture, or making design decisions. `TODO.md` holds the open backlog only; when a row is finished, move its closure entry to `CHANGELOG.md` (most-recent-first) rather than into a Done section in `TODO.md`.
- **Prose style**: user-facing docs are governed by the **"Writing style — user-facing prose"** section at the top of this file. Re-read it before writing any docs/*.md prose — its banned patterns are the ones LLM drafts reliably drift into.
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
