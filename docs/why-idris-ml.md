# Why idris-ml

idris-ml gives you **the ergonomics of a dynamic graph (PyTorch) with safety guarantees
stronger than any static graph (TensorFlow 1.x) ever offered** — and a few guarantees no
mainstream framework offers at all. The computation graph is dynamic (define-by-run
autograd, ordinary `if`/`for`/`while`, normal debugging), but the *constraints* —
shapes, devices, grad-mode, dtype — live in the type system, checked at compile time and
erased at runtime.

The thesis: **one mechanism — dependently-typed indices plus linear resource types —
covers five separate guarantees uniformly.** Each section below shows the *same program*
in four settings — PyTorch (dynamic), TensorFlow 1.x / JAX (static), Haskell (Grenade /
hasktorch `Torch.Typed`), and idris-ml — with the **literal error each one produces**, so
you can read the difference off the tool output, not take it on faith. (Provenance:
PyTorch, JAX, and idris-ml errors are captured verbatim in this repo's environment — torch
2.11, jax 0.10, Idris 2 0.8; TF 1.x and Haskell error text is the frameworks' documented
output, marked *representative*, since those toolchains aren't reproducible here.)

It isn't a toy, either: `idris-transformers` loads real HuggingFace **BERT / GPT-2 /
Llama-3.2-1B / BitNet** checkpoints by name and matches PyTorch's forward to within
**4e-4** — see [Not a toy](#not-a-toy--real-huggingface-models).

Every idris-ml rejection below is lifted from a **CI-enforced fixture**
(`packages/idris-ml/src/Test/neg/`, gated by `make test-integration-typegate-*`): if the
type system stopped rejecting these, CI goes red.

## The five guarantees at a glance

| Guarantee | PyTorch (dynamic) | TF 1.x / JAX (static) | Haskell (Grenade / hasktorch) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| **Shape** mismatch | runtime error | TF1 graph-build / JAX trace-time | **compile** (plugins + singletons) | **compile** (native) |
| **Device** mismatch | runtime error | runtime | **compile** (phantom, hasktorch) | **compile** |
| **Multi-backend in one program** | none (one runtime) | none | none (libtorch-only) | **compile-tracked + explicit bridge** |
| **Grad-mode / model ownership** | runtime / silent no-op | n/a (TF1) / functional (JAX) | partial (`LinearTypes` incomplete) | **compile** (linear types) |
| **Lossy dtype cast** | silent | silent | silent (no lossless order) | **compile** (must opt in) |

The honest reading: Haskell reaches shape, device, and (with effort) grad-mode; the
genuinely out-of-reach-today ones are **multi-backend in one program** and **first-class
model ownership**. What sets idris-ml apart even where Haskell competes is that it needs
**no typechecker plugins, no `singletons` boilerplate, no `unsafeCoerce`** — proofs are
ordinary terms, runtime values flow into types directly, and the *same* index machinery
does all five jobs.

---

## 1. Shape mismatches

**PyTorch — runtime error.** Fires only when that path executes; if broadcasting makes a
wrong shape "compatible" you get plausible garbage instead of a crash.

```python
fc1 = nn.Linear(784, 256); fc2 = nn.Linear(128, 10)   # bug: should be 256
fc2(fc1(torch.randn(64, 784)))
```
```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

**TensorFlow 1.x — graph-build (a genuine point in its favour); JAX — trace-time.** TF1
catches *statically-known* dim mismatches at graph construction; but the batch dim is
usually `None`, so anything depending on it slips to session-run, and untaken branches are
never built. JAX raises when the jitted function is traced (first call), not at definition.

```python
# TF 1.x
h = tf.matmul(x, tf.Variable(tf.zeros([784, 256])))
y = tf.matmul(h, tf.Variable(tf.zeros([128, 10])))    # bug: 128 ≠ 256
```
```text
ValueError: Dimensions must be equal, but are 256 and 128 for 'MatMul_1'
  (op: 'MatMul') with input shapes: [?,256], [128,10].          # representative (TF 1.15)
```
```python
# JAX
@jax.jit
def f(x): return jnp.dot(jnp.dot(x, W1), W2)            # W2:[128,10], h:[...,256]
f(x)
```
```text
TypeError: dot_general requires contracting dimensions to have the same shape, got (256,) and (128,).
```

**Haskell — compile error, with plugins + singletons (Grenade / hasktorch).** GHC *does*
type shapes via type-level `Nat` literals:

```haskell
-- Grenade: the 128 ≠ 256 typo won't compile
type MNIST = Network '[ FullyConnected 784 256, Relu, FullyConnected 128 10 ]
                     '[ 'D1 784, 'D1 256, 'D1 256, 'D1 10 ]
-- hasktorch Torch.Typed: shape carried as a type-level [Nat]
linearForward :: Linear inF outF dtype dev
              -> Tensor dev dtype '[n, inF] -> Tensor dev dtype '[n, outF]
```
```text
• Couldn't match type ‘256’ with ‘128’                          # representative (GHC 9.x)
    arising from the second argument of ‘Network’
```

The catch — why people try this and give up: GHC's type-level `Nat` arithmetic isn't a
solver (`n + m ~ m + n` needs `-fplugin GHC.TypeLits.Normalise` or hand `:~:` proofs);
runtime-derived shapes need `singletons`/`SomeNat` reflection; there's no first-class
`Fin`; and errors read like `Could not deduce KnownNat (n + 1)`.

**idris-ml — compile error, native.** The shape is part of the type; the operation's
signature is the check. A `tadd` of mismatched shapes:

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

The same check threads through a whole `Seq` chain (`l1 ~~> reluA ~~> l2 ~~> Nil`): swap a
layer so the dims don't line up and elaboration fails with `Mismatch between: 10 and 5`.

And the part Haskell can't reach ergonomically today — **runtime values flowing into types
with no singleton ceremony**. `Gpt2.fromPretrained` reads `n_embd`/`n_head` from
`config.json` at runtime and `decEq`'s a proof straight into the model builder:

```idris
case decEq (hidden cfg) (numHeads cfg * headDim cfg) of
  No _     => pure (Left (ConfigError "n_embd not divisible by n_head"))
  Yes prfH => hfGpt2Model {hidden = hidden cfg} {numHeads = numHeads cfg} {prfH} ...
```

Plus `Fin`-indexed datasets (`item : Fin size -> IO sample`) and `rewrite`-based reshape
proofs you pass as ordinary terms.

---

## 2. Device mismatches

**PyTorch — runtime error** (captured here with MPS, since this is an Apple-silicon box):

```python
a = torch.randn(4, device="mps"); b = torch.randn(4)   # b on cpu
a + b
```
```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

**TensorFlow 1.x / JAX — runtime.** TF1 placement resolves at session-run; JAX arrays carry
a *committed* device and mixing two raises at execution:

```text
# TF 1.x:  InvalidArgumentError: Cannot assign a device for operation … (at session.run)   [representative]
# JAX:     ValueError: Received incompatible devices for jitted computation …              [representative]
```

**Haskell — compile error (hasktorch Torch.Typed).** Device is a type parameter
`'(DeviceType, Nat)`; a CPU tensor won't unify with a CUDA one (a phantom — no dependent
types needed; Grenade is CPU-only):

```haskell
addCuda :: Tensor '( 'CUDA, 0) dt sh -> Tensor '( 'CUDA, 0) dt sh -> Tensor '( 'CUDA, 0) dt sh
```
```text
• Couldn't match type ‘'( 'CPU, 0)’ with ‘'( 'CUDA, 0)’          # representative (GHC 9.x)
```

**idris-ml — compile error, three ways.** Since device-as-phantom is the easy part, the
value-add is what the phantom *enables*:

```idris
-- (a) cross-executor add doesn't typecheck — both operands share `ex`:
tadd : Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)

-- (c) Metal is F32-only — encoded as a MISSING Compatible instance:
bad = compatOK {ex=MlxExecutor MGpu} {dt = F64}
```
```text
Error: While processing right hand side of bad.
Can't find an implementation for Compatible (MlxExecutor MGpu) (Float 64).
```

For **(b) "CUDA on an M-series Mac"**, PyTorch tells you at runtime:

```python
torch.randn(4, device="cuda")
```
```text
AssertionError: Torch not compiled with CUDA enabled
```

idris-ml has a two-gate answer: `Linked` makes an un-built backend *unspellable*
(`Can't find an implementation for Linked (TorchExecutor (TCuda 0))`), and a
linked-but-absent device degrades to a typed `Left (DeviceUnavailable …)` via
`toExecutorChecked` rather than aborting deep in the backend. (Metal-F64 in PyTorch, for
contrast: `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64.` — captured — which idris-ml turns into the `Compatible` rejection
above.)

---

## 3. Multi-backend in a single program

The guarantee with **no precedent in any mainstream framework**: hold tensors from
*different* backends in one type-checked program, the compiler policing every transfer.

**PyTorch / TF1 / JAX / Haskell — none.** Each is a single runtime. The closest in PyTorch
is manual, untyped host copies to another array library — nothing tracks which backend a
value lives on, and there's no error to show because there's no check:

```python
import torch, mlx.core as mx
t = torch.randn(4)              # libtorch
a = mx.array(t.numpy())         # manual host round-trip; no type says "this is MLX now"
# t + a goes through numpy or errors at runtime — the type system is blind either way
```

JAX can't hold a torch tensor in a typed computation (you drop to NumPy); hasktorch is
libtorch-only with no open device kind to add a second backend.

**idris-ml — compile-tracked, with an explicit checked bridge.** `Executor` is an *open*
kind, and `BACKEND=tape,torch,mlx` links all three into **one** dylib. From the CI fixture
`Test/Transfer.idr`:

```idris
roundtripF64Smoke = do
  v0 <- makeVec4 {ex=TapeExecutor}        expected     -- pure-C tape
  v1 <- toExecutor (TorchExecutor TCpu) v0             -- → libtorch
  v2 <- toExecutor (MlxExecutor MCpu)   v1             -- → MLX
  v3 <- toExecutor TapeExecutor         v2             -- → back to tape
  check "F64 roundtrip Tape→Torch→Mlx→Tape preserves value" (matchesExpected (read4 v3))
```

You can't feed a tape tensor to an mlx op (the executors don't unify), and the only
backend crossing is the `toExecutor` you wrote. `Linked` keeps un-built backends
unspellable.

---

## 4. Grad-mode and single-owner model ownership

Where idris-ml uses **linear types**, and where every other setting falls short.

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

**TensorFlow 1.x / JAX — n/a or functional, but no ownership types.** TF1's static graph has
no in-place-mutation footgun, but no notion of a spent handle either. JAX is purely
functional — params are explicit arguments, so no mutation footgun — but equally no
compile-time tracking that you used the *current* params:

```python
grads  = jax.grad(loss)(params, batch)
params = sgd_step(params, grads)
grads2 = jax.grad(loss)(params_OLD, batch)   # stale pytree — no error, no warning
```

**Haskell — partial.** GHC 9 has the linear arrow `%1 ->`, but linear `do`-notation isn't in
`base` (you reach for experimental `linear-base`), multiplicity polymorphism is incomplete,
and no ML library threads models linearly. The capability exists; the idiom doesn't.

```haskell
{-# LANGUAGE LinearTypes #-}
eval :: Model %1 -> (Model, Output)   -- the arrow is real; the monadic linear plumbing isn't
```

**idris-ml — compile error.** A model is a **single-owner linear resource** threaded
through `L IO`; `forward`/`eval`/`freeze` consume the handle and return a fresh one. Reusing
a consumed handle is a compile error (fixture `Test/neg/ReuseAfterFreeze.idr`):

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
surface demands `WithGrad`, so the grad-mode mismatch is a compile error too
(`Test/neg/GateRejectsNoGrad.idr`). Tensors stay *unrestricted* — the linear discipline is
only at model granularity, exactly where the aliasing footgun lives.

---

## 5. Lossy dtype conversions must be explicit

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

**TensorFlow 1.x / JAX — silent.** `tf.cast` narrows freely; JAX's weak typing even demotes
float64 → float32 unless `jax_enable_x64` is set — captured here:

```python
jnp.array([1.0], dtype=jnp.float64).astype(jnp.bfloat16)
```
```text
UserWarning: Explicitly requested dtype float64 … is not available, and will be truncated
to dtype float32. …                                            # then silently → bfloat16
```

**Haskell — silent (no lossless order).** hasktorch carries dtype as a type parameter, but
`toDType` changes it in either direction with no partial-order gate:

```haskell
x :: Tensor dev 'Float sh
y = toDType @'Half x      -- compiles; silently narrows F32 → F16     [representative]
```

**idris-ml — compile error unless you opt in.** A *single* `LosslessTo` instance, with `LTE`
premises over the families' bit-widths, defines the whole lossless lattice; proof search
finds the witness or refuses. `F32 → F64` resolves; the lossy directions are CI fixtures
that must not compile:

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

A narrowing cast isn't forbidden — it just has to be *code-visible* (an explicit
`tcastUnsafe`), so the lossy edge shows up in review instead of hiding in an autocast
context.

---

## The ergonomics you keep

None of this costs you the dynamic-graph experience that made PyTorch win:

- **Standard control flow** — `if`/`for`/`while` are ordinary Idris; variable-length
  sequences and data-dependent architectures (RNN/LSTM/NTM/DNC) are natural.
- **Define-by-run autograd** — each forward builds a fresh tape; no `tf.cond`,
  `tf.while_loop`, sessions, or placeholders.
- **Normal debugging** — errors point at your code, not graph nodes.

The static-graph era conflated *shape safety* with *graph structure*. You don't need a
static graph to get static shape checking — you need a type system that can express
dimensional (and device, grad-mode, dtype) constraints.

## The uniformity argument

The deeper point: **one mechanism does all five jobs.** Shape needs genuinely dependent
types (type-level `Nat` arithmetic). Once the compiler is already computing on type
indices, device, grad-mode, and dtype reuse the *same* parameter machinery — no new
language features, no `|G| × |G|` overload tables, no plugins. Add linear resources for
model ownership and the last guarantee falls out of the same type system. In PyTorch four
of these are runtime-only (and one is unattainable); in idris-ml they're a single, uniform
compile-time discipline.

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

These gates regenerate the Python oracle and compare per-element in CI, so "matches
PyTorch" is verified on every publication push, not asserted. Fine-tuning is supported too:
`BertForSequenceClassification` heads, prefix-freeze (`freezeGroup`), subset warm-start
(`load {only := Just pfx}`), and LoRA / PEFT adapters (peft-compatible on disk). Full
guide: [**docs/users/idris-transformers.md**](users/idris-transformers.md).

---

## Go deeper

- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) — the dependent-types-for-shapes
  argument in full, with the NTM dimension-threading worked example.
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md) — phantom enums vs
  dependent types vs linear types: what each guarantee requires, and what it looks like in
  Python.
- [PyTorch Mapping](pytorch-mapping.md) — concept-by-concept translation for PyTorch users.
- [Getting Started](getting-started.md) / [Jupyter notebooks](../packages/jupyter/README.md)
  — run the compile-error demos live.
