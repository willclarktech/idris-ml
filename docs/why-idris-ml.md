# Why idris-ml

idris-ml gives you **the ergonomics of a dynamic graph (PyTorch) with safety guarantees
stronger than any static graph (TensorFlow 1.x) ever offered** — and a few guarantees no
mainstream framework offers at all. The computation graph is dynamic (define-by-run
autograd, ordinary `if`/`for`/`while`, normal debugging), but the *constraints* —
shapes, devices, grad-mode, dtype — live in the type system, checked at compile time and
erased at runtime.

The thesis of this page: **one mechanism — dependently-typed indices plus linear
resource types — covers five separate guarantees uniformly.** Each section shows the same
program in PyTorch (dynamic), TensorFlow 1.x / JAX (static), Haskell (Grenade /
hasktorch `Torch.Typed`, which get further than people expect), and idris-ml.

And it isn't a toy: `idris-transformers` loads real HuggingFace **BERT / GPT-2 /
Llama-3.2-1B / BitNet** checkpoints by name and matches PyTorch's forward pass to within
**4e-4** — see [Not a toy](#not-a-toy--real-huggingface-models) at the bottom.

Every "the compiler rejects this" snippet below is lifted from a **CI-enforced fixture**
(`packages/idris-ml/src/Test/neg/`, gated by `make test-integration-typegate-*`). The
docs can't drift from compiling truth: if the type system stopped rejecting these, CI
goes red.

## The five guarantees at a glance

| Guarantee | PyTorch (dynamic) | TF 1.x / JAX (static) | Haskell (Grenade / hasktorch) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| **Shape** mismatch | runtime error | TF1 graph-build / JAX trace-time | **compile** (plugins + singletons) | **compile** (native) |
| **Device** mismatch | runtime error | runtime | **compile** (phantom) | **compile** |
| **Multi-backend in one program** | n/a (one runtime) | n/a | no precedent | **compile-tracked + explicit bridge** |
| **Grad-mode / model ownership** | runtime / silent no-op | n/a | partial (`LinearTypes` incomplete) | **compile** (linear types) |
| **Lossy dtype cast** | silent | silent | manual instances | **compile** (must opt in) |

The honest reading: Haskell reaches three of these (with friction); the genuinely
out-of-reach-today ones are **multi-backend in one program** and **model ownership**.
What makes idris-ml different even where Haskell *can* compete is that it needs **no
typechecker plugins, no `singletons` boilerplate, no `unsafeCoerce`** — proofs are
ordinary terms, runtime values flow into types directly, and the *same* index machinery
does all five jobs.

---

## 1. Shape mismatches

### The bug (PyTorch — runtime)

```python
class MyModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)   # bug: should be 256

    def forward(self, x):
        return self.fc2(self.fc1(x))
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

The error fires when that code path executes — potentially deep into training, on a rare
branch, or (worse) never, if broadcasting makes a wrong shape "compatible" and you get
plausible garbage instead of a crash.

### Static graphs (TF 1.x / JAX)

TensorFlow 1.x catches *some* of this at graph-construction time — `tf.matmul` on
statically-known dims raises before the session runs. But the batch dim is usually
`None` (`tf.placeholder(tf.float32, [None, 784])`), so anything that depends on it slips
to session-run. JAX's `jit` raises shape errors at **trace** time, which is earlier than
PyTorch eager — but still when you run the traced function, not when you compile the
program, and a shape bug on an untaken branch goes untraced.

### Haskell (genuinely possible — Grenade / hasktorch)

This is the surprising column: Haskell *does* type shapes, via GHC's type-level `Nat`
literals (`GHC.TypeLits`, since GHC 7.8). Grenade puts the whole network shape in the
type:

```haskell
type MNIST =
  Network '[ FullyConnected 784 256, Relu, FullyConnected 128 10 ]
          '[ 'D1 784, 'D1 256, 'D1 256, 'D1 10 ]
-- The 128 ≠ 256 typo is a compile error. Shapes really are static here.
```

hasktorch's `Torch.Typed` carries shape, dtype, and device as type parameters:

```haskell
linearForward
  :: Linear inF outF dtype dev
  -> Tensor dev dtype '[n, inF]
  -> Tensor dev dtype '[n, outF]
```

So why isn't this the end of the story? The friction:
- **Type-level arithmetic isn't a solver.** `n + m ~ m + n` doesn't hold definitionally;
  you add `-fplugin GHC.TypeLits.Normalise` + `ghc-typelits-knownnat`, or write
  `Data.Type.Equality` proofs by hand. idris-ml's `Nat` arithmetic *is* the language's.
- **Runtime shapes need `singletons` / existential reflection.** A dimension read from a
  file becomes `SomeNat`, unpacked with `withSomeSing` ceremony at every site.
- **No first-class `Fin`.** Bounds-safe indexing leans on `finite-typelits` and is
  clunky.
- **Error messages** ("Could not deduce `KnownNat (n + 1)`") are notoriously opaque.

### idris-ml (compile, native)

The shape is part of the tensor type; the operation's signature is the check:

```idris
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode)

-- a Linear layer maps [b, i] to [b, o]; the chain operator (~~>) unifies
-- one layer's output dim with the next's input dim. 128 ≠ 256 won't unify.
```

A mismatched chain fails to elaborate with `Mismatch between: 256 and 128` — no plugin,
no proof obligation you write by hand.

**Where idris-ml goes past Haskell-today** — runtime values flowing into types with *no*
singleton ceremony. `Gpt2.fromPretrained` reads `n_embd` and `n_head` from `config.json`
at runtime, then uses `decEq` to obtain a *proof* that `hidden = numHeads * headDim` and
threads it straight into the model builder:

```idris
fromPretrained dir = do
  Right cfg <- readGpt2Config (dir ++ "/config.json")
    | Left e => pure (Left e)
  case decEq (hidden cfg) (numHeads cfg * headDim cfg) of
    No _      => pure (Left (ConfigError "n_embd not divisible by n_head"))
    Yes prfH  => do
      model <- hfGpt2Model {vocab = vocabSize cfg} {hidden = hidden cfg}
                 {numHeads = numHeads cfg} {headDim = headDim cfg} {prfH} ""
      ...
```

A runtime value (`hidden cfg`) refines the model's *type*, with the divisibility witness
passed as an ordinary argument. In Haskell that's an existential-unpacking dance every
time. Two more everyday examples: the dataset index is a `Fin`, so out-of-bounds is
unrepresentable —

```idris
record Dataset (sample : Type) where
  constructor MkDataset
  size : Nat
  item : Fin size -> IO sample   -- no runtime bounds check, no partiality
```

— and `reshape` carries a `product dims2 = product dims1` proof you `rewrite` through,
rather than a `coerce`/`unsafeCoerce`.

---

## 2. Device mismatches

### The bug (PyTorch — runtime)

```python
a = torch.randn(4, device="cuda:0")
b = torch.randn(4)                       # defaults to cpu
a + b
# RuntimeError: Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu!
```

### idris-ml: three flavours of device safety

**(a) Cross-executor ops don't typecheck.** Every binary op requires both operands to
share the `ex` parameter — `tadd : Tensor dims ex dt g -> Tensor dims ex dt g -> ...`.
Mixing a `TapeExecutor` tensor with a `TorchExecutor TCpu` one is a unification failure,
not a runtime crash. (Device-as-phantom is the one box Haskell ticks trivially with
DataKinds — so the *phantom* isn't the differentiator. The next two are.)

**(b) "I targeted CUDA on an M-series Mac."** PyTorch answers at runtime
(`AssertionError: Torch not compiled with CUDA enabled`). idris-ml has a **two-gate**
answer:

- *Compile-time linkage.* `Linked ex` is an empty marker whose instances are generated
  per build from the `BACKEND` list (into `HwConfig.idr`). Every tensor constructor
  carries `Linked ex =>`. On a build without CUDA, `TorchExecutor (TCuda 0)` is literally
  unspellable — naming it fails with `Can't find an implementation for Linked …`. The
  fixture `Test/neg/BackendRequiresLinked.idr` locks this: an executor missing only its
  `Linked` instance can't resolve the `Backend` bundle.
- *Runtime hardware presence (EAFP).* A device that's *linked but absent* (cuda:1 on a
  one-GPU box, MPS on a non-Apple host) is handled by attempting the construction; the
  backend's C shim catches its own exception and returns a NULL handle, which lifts to a
  typed `Left`:

```idris
data ExecutorError = DeviceUnavailable String

toExecutorChecked : ... -> Tensor dims d1 dt WithGrad
                 -> IO (Either ExecutorError (Tensor dims d2 dt WithGrad))
-- absent device → Left (DeviceUnavailable "cuda:1"), never an abort deep in the backend
```

`availableExecutors` probes a build's candidates the same way. One source of truth (the
real allocation), no TOCTOU, no separate `is_available` to drift.

**(c) Metal is F32-only.** mlx 0.31 dropped float64 on the GPU; libtorch rejects F64 at
MPS construction. PyTorch tells you at runtime (`TypeError: Cannot convert a MPS Tensor
to float64 dtype …`). idris-ml encodes it as a *missing* `Compatible` instance:

```idris
-- from Example/DTypePitch.idr — uncommenting either line is a compile error:
-- badMlxGpuF64   = compatOK {ex=MlxExecutor MGpu}    {dt = F64}
-- badTorchMpsF64 = compatOK {ex=TorchExecutor TMps}  {dt = F64}
-- Can't find an implementation for Compatible (MlxExecutor MGpu) F64
```

---

## 3. Multi-backend in a single program

This is the guarantee with **no precedent in any mainstream framework**. PyTorch is one
runtime; you can't hold a type-tracked libtorch tensor and an MLX array in the same
program and have the compiler police transfers between them.

`Executor` is an *open* type parameter, and `BACKEND=tape,torch,mlx` links all three
backends into **one** `libidrisml.dylib`. A single program can then hold tensors on
different executors at once, each tracked in its type, with `toExecutor` as the *only*
(explicit, checked) bridge. From the CI fixture `Test/Transfer.idr`:

```idris
-- one program, three backends, in flight together:
roundtripF64Smoke : IO Bool
roundtripF64Smoke = do
  v0 <- makeVec4 {ex=TapeExecutor}        expected     -- pure-C tape
  v1 <- toExecutor (TorchExecutor TCpu) v0             -- → libtorch
  v2 <- toExecutor (MlxExecutor MCpu)   v1             -- → MLX
  v3 <- toExecutor TapeExecutor         v2             -- → back to tape
  check "F64 roundtrip Tape→Torch→Mlx→Tape preserves value"
        (matchesExpected (read4 v3))
```

`toExecutor` dispatches on backend tag: same backend → fast in-place hardware migration;
different backend → host-buffer round-trip, with the type-level dtype threaded so an F32
tensor stays F32 storage across the hop. You *cannot* accidentally feed a tape tensor to
an mlx op — the executors don't unify — and you cannot silently cross backends; the only
crossing is the `toExecutor` you wrote. Haskell has no equivalent today: it would need an
open, runtime-extensible device kind with per-backend FFI dictionaries and checked
transfer, which the ecosystem hasn't built.

---

## 4. Grad-mode and single-owner model ownership

This is where idris-ml uses **linear types**, and where typed-FP languages other than
Idris fall short today.

### The grad-mode bug (PyTorch — runtime)

```python
with torch.no_grad():
    loss = compute_loss(model, batch)
loss.backward()
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

### The ownership bug (PyTorch — *silent*)

```python
for p in model.parameters():
    p.requires_grad = False          # "freeze"
optimizer.step()                     # still references model's params — silently no-ops
```

Freezing flips C-side flags; the stale Python handle still "works" and quietly trains
nothing. No error, ever.

### idris-ml: grad-mode in the type, model as a linear resource

A model is a **single-owner linear resource** threaded through `Control.Linear.LIO.L IO`.
`forward`, `eval`, `freeze` all *consume* the handle `(1 _ : …)` and hand back a fresh
one:

```idris
forward : (1 _ : l i o ex dt g) -> Tensor [b, i] ex dt g
       -> L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (l i o ex dt g))

eval    : (1 _ : l i o ex dt WithGrad) -> L IO {use=1} (l i o ex dt NoGrad)
```

So the PyTorch "freeze, then train via the stale handle" footgun becomes a **compile-time
linearity error**. From the CI fixture `Test/neg/ReuseAfterFreeze.idr`:

```idris
badReuse : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad)
        -> L IO {use=1} (Linear 2 3 TapeExecutor F64 NoGrad)
badReuse m = do
  m1 <- eval m
  m2 <- eval m        -- ^ EXPECTED ERROR: m already consumed above
  discard m1
  pure1 m2
-- There are 2 uses of linear name m
```

And the grad-mode mistake — feeding a `NoGrad` loss to the optimizer — is also a compile
error, because `trainStep` demands a `WithGrad` loss (`Test/neg/GateRejectsNoGrad.idr`):

```idris
brokenStep opt = trainStep opt fakeNoGradLoss   -- fakeNoGradLoss : Tensor [] _ _ NoGrad
-- Mismatch between: WithGrad and NoGrad
```

### Haskell (partial)

GHC has had `LinearTypes` since 9.0 (2021), so `%1 ->` exists. But linear `do`-notation
is awkward (you reach for the experimental `Control.Functor.Linear` in `linear-base`),
multiplicity polymorphism is incomplete, and essentially no ML library threads models
linearly. The grad-mode *phantom* is easy in Haskell; the *ownership* discipline that
makes "reuse after consume" a type error is not idiomatic today. (Tensors stay
unrestricted in idris-ml — reverse-mode AD shares them freely — so the linear discipline
applies only at model granularity, exactly where the footgun lives.)

---

## 5. Lossy dtype conversions must be explicit

### The bug (everyone — silent)

PyTorch autocast and `.half()` narrow precision silently; JAX's weak typing will happily
downcast. There is no error — that's the whole problem. A stray F64→F32 (or F32→BF16)
mid-graph just quietly loses bits.

### idris-ml: a derived partial order, upcast-yes / downcast-no

This is the cleanest illustration of "first-class type-level arithmetic buys you
something even where Haskell could compete." A **single** `LosslessTo` instance, with
`LTE` premises over the dtype families' bit-widths, defines the whole lossless lattice —
and Idris's proof search either finds the `LTE` witness or refuses:

```idris
public export
{from, to : Type} ->
FloatPrecision from => FloatPrecision to =>
LTE (mantissaBits {t=from}) (mantissaBits {t=to}) =>
LTE (exponentBits {t=from}) (exponentBits {t=to}) =>
LosslessTo from to where
```

So `F32 → F64` resolves (widening); `F64 → F32` does not. The lossy directions are CI
fixtures that **must not compile**:

```idris
-- Test/neg/LossyDirectionRejected.idr — F32 → BF16 shrinks mantissa 23 → 7:
proofF32ToBF16Lossy : LosslessTo (Float 32) (BFloat 16)
proofF32ToBF16Lossy = %search        -- requires LTE 23 7 — no inhabitant

-- Test/neg/IntOverflowToFloatRejected.idr — I64 → F32 overflows exact-int range:
proofI64ToF32Lossy : LosslessTo (IntN 64) (Float 32)
proofI64ToF32Lossy = %search         -- requires LTE 64 25 — no inhabitant
```

A narrowing cast isn't forbidden — it just has to be *code-visible* (an explicit
`tcastUnsafe`), so the lossy edge shows up in review instead of hiding in an autocast
context manager. In Haskell you'd hand-enumerate a per-pair instance matrix or lean on
the natnormalise plugin; here it's one instance with an arithmetic premise.

---

## The ergonomics you keep

None of the above costs you the dynamic-graph experience that made PyTorch win:

- **Standard control flow** — `if`, `for`, `while` are ordinary Idris; variable-length
  sequences, data-dependent architectures (RNN/LSTM/NTM/DNC) are natural.
- **Define-by-run autograd** — each forward builds a fresh tape; no `tf.cond`,
  no `tf.while_loop`, no session/placeholder boilerplate.
- **Normal debugging** — errors point at your code, not graph nodes.

The static-graph era conflated *shape safety* with *graph structure*. You don't need a
static graph to get static shape checking — you need a type system that can express
dimensional (and device, and grad-mode, and dtype) constraints. That's what dependent +
linear types provide.

## The uniformity argument

The deeper point is that **one mechanism does all five jobs**. Shape needs genuinely
dependent types (type-level `Nat` arithmetic). Once the compiler is already computing on
type indices, device, grad-mode, and dtype come along for free — they reuse the exact
same parameter machinery, no new language features, no `|G| × |G|` overload tables, no
plugins. Add linear resources for model ownership and you've covered the last guarantee
with the same type system. In PyTorch, four of these are runtime-only (and one is
unattainable); in idris-ml they're a single, uniform compile-time discipline.

---

## Not a toy — real HuggingFace models

The guarantees above would be academic if the library only ran toy tasks. It doesn't.
`packages/idris-transformers/` loads real HuggingFace checkpoints **by name** — each HF
architecture is one Idris module whose params and shapes match HF on disk, so loading is
plain `fromPretrained "<dir>"` (parse `config.json`, fill from `model.safetensors`) with
no remap or shape-split machinery:

| Model | Checkpoint loaded | Correctness gate |
|---|---|---|
| **BERT** | `google/bert_uncased_L-2_H-128_A-2` | matches HF Python forward to **4e-4** (`make test-e2e-bert-roundtrip`) |
| **GPT-2** | `distilgpt2` | matches to 1e-3 (`make test-e2e-gpt2-roundtrip`) |
| **Llama** | `unsloth/Llama-3.2-1B` | macro forward/RoPE/param-load gate |
| **BitNet** | `microsoft/bitnet-b1.58-2B-4T` | argmax-match + macro tolerance |

```idris
fromPretrained : Backend ex dt => KnownGrad g
              => (modelDir : String)
              -> IO (Either LoadError (cfg : Gpt2Config ** Gpt2Model cfg ex dt g))
```

These gates regenerate the Python oracle and compare per-element in CI, so the "matches
PyTorch" claim is verified on every publication push, not asserted. Fine-tuning is
supported too: `BertForSequenceClassification` heads, prefix-freeze (`freezeGroup`),
subset warm-start (`load {only := Just pfx}`), and LoRA / PEFT adapters
(peft-compatible on disk). See the full guide:
[**docs/users/idris-transformers.md**](users/idris-transformers.md).

---

## Go deeper

- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) — the dependent-types-for-shapes
  argument in full, with the NTM dimension-threading worked example.
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md) — phantom enums vs
  dependent types vs linear types: what each guarantee actually requires, and what it
  looks like in Python.
- [PyTorch Mapping](pytorch-mapping.md) — concept-by-concept translation for PyTorch users.
- [Jupyter notebooks](../packages/jupyter/README.md) — run the compile-error demos live.
