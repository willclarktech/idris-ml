# Why idris-ml

idris-ml gives you **the ergonomics of a dynamic graph (PyTorch) with safety guarantees
stronger than any static graph (TensorFlow 1.x) ever offered** — and a few guarantees no
mainstream framework offers at all. The computation graph is dynamic (define-by-run autograd,
ordinary `if`/`for`/`while`, normal debugging), but the *constraints* — shapes, devices,
grad-mode, dtype — live in the type system, checked at compile time and erased at runtime.

The thesis: **one mechanism — dependently-typed indices plus linear resource types — covers
five separate guarantees uniformly.** And it isn't a toy: `idris-transformers` loads real
HuggingFace **BERT / GPT-2 / Llama-3.2-1B / BitNet** checkpoints by name and matches PyTorch's
forward to within **4e-4** — see [Not a toy](#not-a-toy--real-huggingface-models).

To keep it concrete, one model runs through the whole article: a **Neural Turing Machine**. An
NTM is a small recurrent controller wired to an external memory, and its layer dimensions all
derive from one number — the memory width `m`. That single coupling is enough to hit every bug
class below in turn: change `m` and the shapes must agree; move it to the Mac's GPU and the
device must agree; freeze the controller and the grad-mode must agree; drop to fp16 and the
dtype must agree; and finally prototype it on one backend while training on another. Each
section is the same project, one step later — with the **literal error each tool produces** at
that step, so you can read the difference off the output, not take it on faith.[^prov]

Every idris-ml rejection below is lifted from a **CI-enforced fixture**
(`packages/idris-ml/src/Test/neg/`, gated by `make test-integration-typegate-*`): if the type
system stopped rejecting these, CI goes red.

## The five guarantees at a glance

| Guarantee | PyTorch (dynamic) | TF 1.x (static) | Haskell (Grenade / hasktorch) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| **Shape** mismatch | runtime error | graph-build | **compile** (plugins + singletons) | **compile** (native) |
| **Device** mismatch | runtime error | runtime | **compile** (phantom, hasktorch) | **compile** |
| **Grad-mode / model ownership** | runtime / silent no-op | n/a (static graph) | partial (`LinearTypes` incomplete) | **compile** (linear types) |
| **Lossy dtype cast** | silent | silent | silent (no lossless order) | **compile** (must opt in) |
| **Multi-backend in one program** | none (one runtime) | none | none (libtorch-only) | **compile-tracked + explicit bridge** |

Where each lands: Haskell reaches shape, device, and (with effort) grad-mode; the two genuinely
out-of-reach-today are **multi-backend in one program** and **first-class model ownership**.
What sets idris-ml apart even where Haskell competes is that it needs **no typechecker plugins,
no `singletons` boilerplate, no `unsafeCoerce`** — proofs are ordinary terms, runtime values
flow into types directly, and the *same* index machinery does all five jobs.

---

## 1. Shape mismatches

You wire up the NTM. Its read head emits `m + 6` numbers, the controller takes `memory_width +
9`, the output projection takes `hidden + m` — five layer dimensions, all functions of `m`. You
bump `m` and miss one.

**PyTorch — runtime error.** The mismatch fires only when that path executes; if broadcasting
makes a wrong shape "compatible" you get plausible garbage instead of a crash. Minimal repro of
the bug hiding in those NTM dimensions:

```python
fc1 = nn.Linear(784, 256); fc2 = nn.Linear(128, 10)   # bug: should be 256
fc2(fc1(torch.randn(64, 784)))
```
```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

**TensorFlow 1.x — graph-build (a point in its favour).** TF1 catches *statically-known* dim
mismatches at graph construction; but the batch dim is usually `None`, so anything depending on it
slips to session-run, and untaken branches are never built.

```python
# TF 1.x
h = tf.matmul(x, tf.Variable(tf.zeros([784, 256])))
y = tf.matmul(h, tf.Variable(tf.zeros([128, 10])))    # bug: 128 ≠ 256
```
```text
ValueError: Dimensions must be equal, but are 256 and 128 for 'MatMul_1'
  (op: 'MatMul') with input shapes: [?,256], [128,10].          †
```

**Haskell — compile error, with plugins + singletons (Grenade / hasktorch).** GHC *does* type
shapes via type-level `Nat` literals:

```haskell
-- Grenade: the 128 ≠ 256 typo won't compile
type MNIST = Network '[ FullyConnected 784 256, Relu, FullyConnected 128 10 ]
                     '[ 'D1 784, 'D1 256, 'D1 256, 'D1 10 ]
-- hasktorch Torch.Typed: shape carried as a type-level [Nat]
linearForward :: Linear inF outF dtype dev
              -> Tensor dev dtype '[n, inF] -> Tensor dev dtype '[n, outF]
```
```text
• Couldn't match type ‘256’ with ‘128’                          †
    arising from the second argument of ‘Network’
```

The friction: GHC's type-level `Nat` arithmetic isn't a solver (`n + m ~ m + n` needs `-fplugin
GHC.TypeLits.Normalise` or hand `:~:` proofs); runtime-derived shapes need `singletons`/`SomeNat`
reflection; there's no first-class `Fin`; and errors read like `Could not deduce KnownNat (n +
1)`.

**idris-ml — compile error, native.** The shape is part of the type; the operation's signature
is the check. A `tadd` of mismatched shapes:

```idris
v4 <- tensor {dims=[4]} (Const 1.0)
v5 <- tensor {dims=[5]} (Const 1.0)
s  <- tadd v4 v5
```
```text
Error: While processing right hand side of bad. When unifying:
    Tensor [5] TapeExecutor F64 NoGrad
and:
    Tensor [4] TapeExecutor F64 NoGrad
Mismatch between: 1 and 0.
```

The same check threads through a whole `Seq` chain (`l1 ~~> reluA ~~> l2 ~~> Nil`): swap a layer
so the dims don't line up — exactly the NTM-`m` mistake — and elaboration fails with `Mismatch
between: 10 and 5` before anything runs.

And the part Haskell can't reach ergonomically today — **runtime values flowing into types with
no singleton ceremony**. `Gpt2.fromPretrained` reads `n_embd`/`n_head` from `config.json` at
runtime and `decEq`'s a proof straight into the model builder:

```idris
case decEq (hidden cfg) (numHeads cfg * headDim cfg) of
  No _     => pure (Left (ConfigError "n_embd not divisible by n_head"))
  Yes prfH => hfGpt2Model {hidden = hidden cfg} {numHeads = numHeads cfg} {prfH} ...
```

Plus `Fin`-indexed datasets (`item : Fin size -> IO sample`) and `rewrite`-based reshape proofs
you pass as ordinary terms. A runtime value off disk refines the model's *type* — no
`singletons`, no `SomeNat`, no `unsafeCoerce`.

---

## 2. Device mismatches

The NTM runs. You move it to your Mac's GPU to train faster — and leave one tensor (the memory
init, say) on the CPU.

**PyTorch — runtime error** (captured here with MPS, since this is an Apple-silicon box):

```python
a = torch.randn(4, device="mps"); b = torch.randn(4)   # b on cpu
a + b
```
```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

**TensorFlow 1.x — runtime.** Device placement resolves at session-run, not graph construction:

```text
# TF 1.x:  InvalidArgumentError: Cannot assign a device for operation … (at session.run)   †
```

**Haskell — compile error (hasktorch Torch.Typed).** Device is a type parameter `'(DeviceType,
Nat)`; a CPU tensor won't unify with a CUDA one. This is a plain phantom — no dependent types
needed (and Grenade is CPU-only):

```haskell
addCuda :: Tensor '( 'CUDA, 0) dt sh -> Tensor '( 'CUDA, 0) dt sh -> Tensor '( 'CUDA, 0) dt sh
```
```text
• Couldn't match type ‘'( 'CPU, 0)’ with ‘'( 'CUDA, 0)’          †
```

**idris-ml — compile error.** Device-as-phantom is the easy part; idris-ml gets it the same way
(both operands of `tadd` share the executor `ex`, so the CPU-memory-init bug doesn't typecheck),
and then goes further in two directions a phantom alone can't.

```idris
-- both operands share `ex`, so a cross-executor add doesn't unify:
tadd : Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
```

First, **a device you didn't build is unspellable.** Suppose you reach for CUDA on that
M-series Mac. PyTorch tells you at runtime:

```python
torch.randn(4, device="cuda")
```
```text
AssertionError: Torch not compiled with CUDA enabled
```

idris-ml has a two-gate answer. The `Linked` constraint makes an un-built backend a *type error*
— `TorchExecutor (TCuda 0)` can't even be named in a non-CUDA build (`Can't find an
implementation for Linked (TorchExecutor (TCuda 0))`). And a backend that *is* linked but absent
at runtime degrades to a typed `Left (DeviceUnavailable …)` via `toExecutorChecked`/`attemptOn`,
rather than aborting deep in the C layer.

Second, **illegal (device, dtype) pairs are unspellable.** Metal is F32-only; in PyTorch that's a
runtime `TypeError` (`Cannot convert a MPS Tensor to float64 dtype …`). idris-ml encodes it as a
*missing* `Compatible` instance, so the bad pair never constructs:

```idris
bad = compatOK {ex=MlxExecutor MGpu} {dt = F64}
```
```text
Error: While processing right hand side of bad.
Can't find an implementation for Compatible (MlxExecutor MGpu) (Float 64).
```

---

## 3. Grad-mode and single-owner model ownership

Training works. Now you fine-tune: freeze the NTM's controller and train only the memory
read/write heads. In PyTorch this is where the *silent* bugs live — and it's where idris-ml uses
**linear types**, the guarantee every other setting falls short of.

**PyTorch — runtime error, or a silent no-op.** A loss built under `no_grad` then sent to
`backward`:

```python
with torch.no_grad():
    loss = (w * 2).sum()       # w requires_grad=True
loss.backward()
```
```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

And the *silent* one — freeze flips C-side flags, the stale Python handle still "works" and
quietly trains nothing, with **no error at all**:

```python
for p in model.parameters(): p.requires_grad = False
optimizer.step()               # references the same params — silent no-op, no traceback
```

**TensorFlow 1.x — n/a, but no ownership types.** TF1's static graph has no in-place-mutation
footgun (params are graph variables updated by ops, not Python handles you can stale out), but no
notion of a spent handle either — nothing tracks at compile time that an optimizer step references
the *current* parameters rather than a discarded copy.

**Haskell — partial.** GHC 9 has the linear arrow `%1 ->`, but linear `do`-notation isn't in
`base` (you reach for experimental `linear-base`), multiplicity polymorphism is incomplete, and
no ML library threads models linearly. The capability exists; the idiom doesn't.

```haskell
{-# LANGUAGE LinearTypes #-}
eval :: Model %1 -> (Model, Output)   -- the arrow is real; the monadic linear plumbing isn't
```

**idris-ml — compile error.** A model is a **single-owner linear resource** threaded through `L
IO`; `forward`/`eval`/`freeze` consume the handle and return a fresh one. Freeze the controller,
then reach for the stale handle to keep training — exactly the PyTorch silent no-op — and it's a
compile error (fixture `Test/neg/ReuseAfterFreeze.idr`):

```idris
badReuse m = do
  m1 <- eval m
  m2 <- eval m        -- reuse of a consumed model
  ...
```
```text
Error: While processing right hand side of badReuse.
There are 2 uses of linear name m.
```

And an `eval`'d (`NoGrad`) model's output can't be fed to `trainStep` — the loss/optimizer
surface demands `WithGrad`, so the grad-mode mismatch (the `no_grad`-then-`backward` bug above) is
a compile error too (`Test/neg/GateRejectsNoGrad.idr`). Tensors stay *unrestricted* — the linear
discipline is only at model granularity, exactly where the aliasing footgun lives.

---

## 4. Lossy dtype conversions must be explicit

Your NTM's memory got big, so you switch it to fp16 to fit. Half the dtype edges you cross are
lossless (widening); the dangerous ones narrow precision — and nothing warns you which is which.

**PyTorch — silent.** `.half()` / autocast narrow precision with no error — that's the whole
problem:

```python
x = torch.randn(4, dtype=torch.float64)
y = x.half()
print(y.dtype)
```
```text
torch.float16          # no error; ~13 bits of mantissa silently gone
```

**TensorFlow 1.x — silent.** `tf.cast` narrows in either direction with no error or warning:

```python
tf.cast(tf.constant([1.0], dtype=tf.float64), tf.float16)   # → float16, ~13 bits gone, silently  †
```

**Haskell — silent (no lossless order).** hasktorch carries dtype as a type parameter, but
`toDType` changes it in either direction with no partial-order gate:

```haskell
x :: Tensor dev 'Float sh
y = toDType @'Half x      -- compiles; silently narrows F32 → F16     †
```

**idris-ml — compile error unless you opt in.** A *single* `LosslessTo` instance defines the
whole lossless lattice for the float families — widen iff mantissa **and** exponent bits don't
shrink — and proof search finds the witness or refuses:

```idris
FloatPrecision from => FloatPrecision to =>
LTE (mantissaBits {t=from}) (mantissaBits {t=to}) =>
LTE (exponentBits {t=from}) (exponentBits {t=to}) =>
LosslessTo from to where
```

That one instance covers all four float cross-products (F→F, BF→BF, F→BF, BF→F). `F32 → F64`
resolves; the lossy directions are CI fixtures that must *not* compile:

```idris
proofF32ToBF16Lossy : LosslessTo (Float 32) (BFloat 16)   -- mantissa 23 → 7
proofF32ToBF16Lossy = %search
```
```text
Error: While processing right hand side of proofF32ToBF16Lossy.
Can't find an implementation for LosslessTo (Float 32) (BFloat 16).
```
```idris
proofI64ToF32Lossy : LosslessTo (IntN 64) (Float 32)      -- 2^63 ≫ F32 exact-int range
proofI64ToF32Lossy = %search
```
```text
Error: While processing right hand side of proofI64ToF32Lossy.
Can't find an implementation for LosslessTo (IntN 64) (Float 32).
```

A narrowing cast isn't forbidden — it just has to be *code-visible* (an explicit `tcastUnsafe`),
so the lossy edge in your fp16 switch shows up in review instead of hiding in an autocast context.

---

## 5. Multi-backend in a single program

The finale — the guarantee with **no precedent in any mainstream framework**. You prototyped the
NTM on the pure-C tape backend (no deps, easy to debug), you want to train it on libtorch, and
deploy inference on Apple MLX. Normally that's three programs. Here it's one.

`Executor` is an *open* kind, and a single build links every backend you ask for into **one**
dylib:

```bash
make BACKEND=tape,torch,mlx backend     # tape, libtorch, and MLX in one libidrisml.{so,dylib}
```

One type-checked program can then hold tensors from different backends at once — they simply have
different types, and the only way across is an explicit, checked `toExecutor`:

```idris
tapeVec  : Tensor [4] TapeExecutor        F64 g   -- pure-C tape
torchVec : Tensor [4] (TorchExecutor TMps) F32 g  -- libtorch on Metal
-- tapeVec and torchVec cannot be added: the executors don't unify.
```

**PyTorch / TF1 / Haskell — none.** Each is a single runtime. The closest in PyTorch is a manual,
untyped host copy to another array library — nothing tracks which backend a value lives on, and
there's no error to show because there's no check:

```python
import torch, mlx.core as mx
t = torch.randn(4)              # libtorch
a = mx.array(t.numpy())         # manual host round-trip; no type says "this is MLX now"
# t + a goes through numpy or errors at runtime — the type system is blind either way
```

hasktorch is libtorch-only, with no open device kind to add a second backend.

**idris-ml — compile-tracked, with an explicit checked bridge.** From the CI fixture
`Test/Transfer.idr`, the NTM's data moving tape → torch → mlx → tape, value preserved at every
hop:

```idris
roundtripF64Smoke = do
  v0 <- makeVec4 {ex=TapeExecutor}        expected     -- pure-C tape
  v1 <- toExecutor (TorchExecutor TCpu) v0             -- → libtorch
  v2 <- toExecutor (MlxExecutor MCpu)   v1             -- → MLX
  v3 <- toExecutor TapeExecutor         v2             -- → back to tape
  check "F64 roundtrip Tape→Torch→Mlx→Tape preserves value" (matchesExpected (read4 v3))
```

You can't feed a tape tensor to an mlx op (the executors don't unify), the only backend crossing
is the `toExecutor` you wrote, and `Linked` keeps un-built backends unspellable. Prototype,
train, and deploy across three runtimes — in one program the compiler keeps honest.

---

## The ergonomics you keep

None of this costs you the dynamic-graph experience that made PyTorch win:

- **Standard control flow** — `if`/`for`/`while` are ordinary Idris; variable-length sequences
  and data-dependent architectures (RNN/LSTM/NTM/DNC) are natural.
- **Define-by-run autograd** — each forward builds a fresh tape; no `tf.cond`, `tf.while_loop`,
  sessions, or placeholders.
- **Normal debugging** — errors point at your code, not graph nodes.

The static-graph era conflated *shape safety* with *graph structure*. You don't need a static
graph to get static shape checking — you need a type system that can express dimensional (and
device, grad-mode, dtype) constraints.

## The uniformity argument

The deeper point: **one mechanism does all five jobs.** Shape needs genuinely dependent types
(type-level `Nat` arithmetic). Once the compiler is already computing on type indices, device,
grad-mode, and dtype reuse the *same* parameter machinery — no new language features, no `|G| ×
|G|` overload tables, no plugins. Add linear resources for model ownership and the last guarantee
falls out of the same type system. In PyTorch four of these are runtime-only (and one is
unattainable); in idris-ml they're a single, uniform compile-time discipline — the same NTM,
checked at every step.

---

## Not a toy — real HuggingFace models

The guarantees above would be academic if the library only ran toy tasks. It doesn't.
`packages/idris-transformers/` loads real HuggingFace checkpoints **by name** — each HF
architecture is one Idris module whose params and shapes match HF on disk, so loading is plain
`fromPretrained "<dir>"` (parse `config.json`, fill from `model.safetensors`) with no remap or
shape-split machinery:

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

These gates regenerate the Python oracle and compare per-element in CI, so "matches PyTorch" is
verified on every publication push, not asserted. Fine-tuning is supported too:
`BertForSequenceClassification` heads, prefix-freeze (`freezeByPrefix`), subset warm-start (`load
{only := Just pfx}`), and LoRA / PEFT adapters (peft-compatible on disk). Full guide:
[**docs/users/idris-transformers.md**](users/idris-transformers.md).

---

## Go deeper

- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) — the dependent-types-for-shapes
  argument in full, with the NTM dimension-threading worked example.
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md) — phantom enums vs dependent
  types vs linear types: what each guarantee requires, and what it looks like in Python.
- [PyTorch Mapping](pytorch-mapping.md) — concept-by-concept translation for PyTorch users.
- [Getting Started](getting-started.md) / [Jupyter notebooks](../packages/jupyter/README.md) —
  run the compile-error demos live.

---

[^prov]: PyTorch and idris-ml errors are captured verbatim in this repo's environment —
    torch 2.11, Idris 2 0.8. Lines marked **†** are the frameworks' documented output
    (TF 1.x and Haskell toolchains aren't reproducible here), shown to illustrate the shape of the
    error rather than as captured evidence.
