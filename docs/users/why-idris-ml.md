# Why idris-ml

> Make illegal states unrepresentable.
>
> — [Yaron Minsky](https://www.youtube.com/watch?v=-J8YyfrSwTk)

idris-ml is a deep-learning framework with:
- the ergonomics of dynamic graphs (like PyTorch),
- the safety guarantees of static graphs (like TensorFlow v1),
- support for multiple (pluggable) backends.

The computation graph is dynamic, with all the usual benefits (define-by-run autograd, native control flow, transparent debugging). But the *constraints* — shapes, devices, grad-mode, dtype — live in the type system, meaning they're checked at compile time and erased at runtime.

To achieve this we make use of Idris 2's native support for dependent types and linear resource types, which is why other frameworks written in other languages are either unable to implement these features, or can only do so inelegantly (e.g. with substantial boilerplate).

This document guides you through each major feature of idris-ml, demonstrating the problem it solves with reference to other frameworks.[^prov]

## Assumed background

- [Static vs dynamic graphs](static-vs-dynamic-graphs.md)
- [Dependent types](https://en.wikipedia.org/wiki/Dependent_type)
- [Linear resource types](https://en.wikipedia.org/wiki/Substructural_type_system)

## At a glance

| Bug class | PyTorch (dynamic) | TF 1.x (static) | hasktorch (Torch.Typed) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| Shape mismatch | run time | graph build | **compile time** | **compile time** |
| Device mismatch | run time | run time | **compile time** | **compile time** |
| Grad-mode misuse | run time | n/a | not caught | **compile time** |
| Stale model handle after freeze | not caught | n/a | not caught | **compile time** |
| Lossy dtype cast | not caught | not caught | not caught | **compile time** (explicit opt-out) |
| Mixing multiple backends | unsupported | unsupported | unsupported | **compile time** |

> [!TIP]
> Every compile-error demo in this document can be run live: see [Getting Started](getting-started.md) and the [Jupyter notebooks](../../packages/jupyter/README.md).

---

## 1. Shape mismatches

Here's a common bug class in PyTorch:

```python
fc1 = nn.Linear(784, 256)  # this hidden layer size got increased from 128 to 256
fc2 = nn.Linear(128, 10)   # bug: this value didn't get updated
some_inputs = torch.randn(64, 784)
fc2(fc1(some_inputs))
```

The error only surfaces at runtime:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

If you have an expensive training or inference run, you can wait minutes or hours before that line runs and the whole thing falls over — and the bug only surfaces if that code path executes with that input at all. Both shapes are known the moment you write the layers, yet nothing checks they line up until data reaches that point.

> [!IMPORTANT]
> The primary goal of idris-ml is to make a program with bugs like this *unrepresentable* (rejected by the language's *compiler* as invalid).

If you're familiar with the debates over dynamic vs static graphs in machine learning frameworks, you might be thinking that this problem was already solved by frameworks that construct a static computation graph. For example in TensorFlow v1, when you run a program it first builds the computation graph, runs static checks such as whether all the tensor dimensions match, and only then performs the computation using the actual input data:

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
x = tf.placeholder(tf.float32, [None, 784])
h = tf.matmul(x, tf.Variable(tf.zeros([784, 256])))
y = tf.matmul(h, tf.Variable(tf.zeros([128, 10])))    # bug: 128 ≠ 256
```
```text
ValueError: Dimensions must be equal, but are 256 and 128 for '{{node MatMul_1}} =
  MatMul[T=DT_FLOAT, ...](MatMul, MatMul_1/ReadVariableOp)' with input shapes: [?,256], [128,10].
```

But the static graph approach comes with some serious downsides. With these libraries you are essentially *meta-programming*, for example writing Python code that outputs TensorFlow code that runs the computation you want. And this mismatch means (1) losing the ability to use native constructs for control flow (such as `if`/`else`, or `for`/`while` loops), and (2) debugging becomes a challenge because it isn't always straightforward to map the source of errors in the computation graph to its origin in the code you wrote. Hence the feeling of ["working with your computation graph through a keyhole"](https://www.georgeho.org/tensor-computation-libraries/), and the general drift of the machine learning community away from the static graph towards dynamic graphs.

idris-ml asks: can we have the benefits of static computation graphs without the indirection that comes from the meta-programming approach? The answer is yes: we move the correctness checks into the programming language itself, preserving native control flow and transparent debugging. To achieve this, we rely on a rich type system.

Python is unsuitable for this, even with [type hints](https://docs.python.org/3/library/typing.html) and a static type checker like [Pyright](https://github.com/microsoft/pyright). Imagine trying to create a tensor type that encodes its own dimensions to prevent the bug we introduced earlier. It might look something like this:

```python
from typing import Generic, Literal, TypeVar

M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)
N = TypeVar("N", bound=int)

class Tensor(Generic[M, N]):  # a 2-D tensor carrying its dims in the type
	# ...

def matmul(a: Tensor[M, K], b: Tensor[K, N]) -> Tensor[M, N]:
	# ...

fc1: Tensor[Literal[784], Literal[256]] = Tensor()  # hidden size increased from 128 to 256
fc2: Tensor[Literal[128], Literal[10]] = Tensor()   # bug: this value didn't get updated
some_inputs: Tensor[Literal[64], Literal[784]] = Tensor()

bad = matmul(matmul(some_inputs, fc1), fc2)  # fc2(fc1(some_inputs))
```

To be fair to Pyright, this actually catches our planted bug:

```text
error: Argument of type "Tensor[Literal[128], Literal[10]]" cannot be assigned to
    parameter "b" of type "Tensor[K@matmul, N@matmul]" in function "matmul"
  "Tensor[Literal[128], Literal[10]]" is not assignable to "Tensor[Literal[256], Literal[10]]"
    Type parameter "M@Tensor" is invariant, but "Literal[128]" is not the same as "Literal[256]"
```

But it only works because every dimension in the program is a literal written out in the source. The approach collapses as soon as a shape has to be *computed*. Concatenate two feature blocks, or flatten a `[28, 28]` matrix into a 784-vector, and the result type can't even be written down — there is no arithmetic in Python's type language:

```python
def concat_cols(a: Tensor[M, K], b: Tensor[M, N]) -> Tensor[M, K + N]:
	# ...
```
```text
error: Binary operator not allowed in type expression (reportInvalidTypeForm)
```

Nor can a dimension that arrives at *runtime* — a hidden size read from a config file, a vocab size taken from a checkpoint — ever become a `Literal[...]`: literal types are spelled in source text, not computed from values. This is why every PyTorch operation accepts and returns an untyped `Tensor`. Dimensions-in-types demands a type system where types can contain ordinary values and be computed with ordinary functions — which is precisely what dependent types are.

So how about a language with a rich static type system? Haskell is the natural candidate, and [hasktorch](https://github.com/hasktorch/hasktorch) is a serious attempt: its `Torch.Typed` API gives libtorch tensors types that carry their shapes. Note that *plain* Haskell can't express this either (standard Haskell has no way to put a number in a type); hasktorch builds on a stack of opt-in GHC extensions, each patching in one missing piece:

- `DataKinds` — lets values (numbers like `784`) be promoted to the type level, so they can appear in a type at all.
- `KindSignatures` — lets a type declaration state what kind of thing each parameter is (`rows :: Nat`, a type-level number rather than a type).
- `TypeOperators` — lets operators like `+` and `*` be used in type expressions.
- `NoStarIsType` — frees up the `*` symbol so it can mean multiplication (it historically meant "is a type" in kind syntax).

With those switched on, our two layers look like this (representative code[^prov]):

```haskell
{-# LANGUAGE DataKinds, KindSignatures #-}
import GHC.TypeLits (Nat)

data Tensor (rows :: Nat) (cols :: Nat) = Tensor
matmul :: Tensor m k -> Tensor k n -> Tensor m n   -- (m×k)·(k×n): inner dims must unify
fc1 :: Tensor 784 256      -- this hidden layer size got increased from 128 to 256
fc2 :: Tensor 128 10       -- bug: this value didn't get updated
someInputs :: Tensor 64 784
bad = matmul (matmul someInputs fc1) fc2           -- fc2(fc1(some_inputs))
```

The planted bug is a compile error before anything runs:

```text
error: [GHC-83865]
    • Couldn't match type ‘128’ with ‘256’
      Expected: Tensor 256 10
        Actual: Tensor 128 10
    • In the second argument of ‘matmul’, namely ‘fc2’
```

This gets further than Python. The `concat_cols` that Pyright rejected is writable here, and when the dimensions are literals GHC even evaluates the arithmetic: flattening a `[28, 28]` matrix produces a 784-vector *in the type*:

```haskell
{-# LANGUAGE DataKinds, KindSignatures, TypeOperators, NoStarIsType #-}
import GHC.TypeLits (Nat, type (+), type (*))

concatCols :: Tensor m k -> Tensor m n -> Tensor m (k + n)
flatten    :: Tensor r c -> Tensor 1 (r * c)

flat :: Tensor 1 784
flat = flatten (Tensor :: Tensor 28 28)   -- GHC computes 28 * 28 = 784: this compiles
```

The limit is *reasoning* about that arithmetic. GHC knows `k + n` is `k + n`, but not that it equals `n + k` (the `NB:` line below names the cause: type-level `+` is a *type family*, a function whose algebra the checker doesn't know), so the same concatenation in the opposite order is rejected:

```haskell
concatFlipped :: Tensor m k -> Tensor m n -> Tensor m (k + n)
concatFlipped a b = concatCols b a   -- same columns, opposite order
```
```text
error: [GHC-83865]
    • Couldn't match type: n + k
                     with: k + n
      Expected: Tensor m (k + n)
        Actual: Tensor m (n + k)
      NB: ‘+’ is a non-injective type family
```

Getting past this takes a compiler plugin (`GHC.TypeLits.Normalise`), and the pattern repeats from there. Division (splitting `hidden` across attention `heads`) needs a second plugin. Shape-polymorphic functions accumulate `KnownNat` constraints and the errors that come with them. A dimension read from a config file at runtime crosses into the types through the `singletons` encoding, at every crossing point. None of this stops an expert, but all of it adds work to the everyday path, and hasktorch's own examples show the result: the defaults (`mnist-mlp`, `xor-mlp`, `rnn`) use its *untyped* API, with the shape-checked versions as separately-maintained variants (`static-mnist-mlp`, `typed-transformer`).

These costs are properties of the language, so no amount of work on the library removes them; deleting the untyped API would just make the extra work mandatory. Taken together, the extensions, plugins, and encodings are hasktorch *simulating dependent types*, fighting the fact that Haskell doesn't have them. The real feature ([Dependent Haskell](https://gitlab.haskell.org/ghc/ghc/-/wikis/dependent-haskell)) has been on GHC's roadmap for years. A library can't add a language feature.

Which brings us to idris-ml, written in [Idris 2](https://www.idris-lang.org/), a language that has the feature. Types are ordinary expressions: they contain numbers, are computed by functions, and depend on values the program only learns at runtime. The checks therefore need no extension, plugin, or encoding, and an API where every tensor is typed becomes practical to work in — idris-ml has no untyped fallback because it doesn't need one. Each operation's signature states the shapes it accepts and produces; the dense-layer operation, for example, ties the weight, input, and output shapes together:

```idris
tlinear2d : Tensor [o, i] ex dt g -> Tensor [b, i] ex dt g -> Tensor [o] ex dt g
         -> IO (Tensor [b, o] ex dt g)
```

Here are our two layers again — `fc1` as weights `w1 : Tensor [256, 784]`, `fc2` as `w2 : Tensor [10, 128]` with the un-updated 128 — and the same batch of inputs:

```idris
h <- tlinear2d w1 x b1     -- x : Tensor [64, 784], so h : Tensor [64, 256]
y <- tlinear2d w2 h b2     -- fc2(fc1(some_inputs))
```
```text
Error: While processing right hand side of bad. When unifying:
    Tensor [64, 256] TapeExecutor F64 WithGrad
and:
    Tensor [?b, 128] TapeExecutor F64 WithGrad
Mismatch between: S (assert_total (integerToNat 127)) and 0.
```

The compiler reports the same shapes PyTorch reported — `[64, 256]` supplied where `[?b, 128]` is required — except at compile time, before any data exists. The same holds one level up: chaining the two layers into a model (`fc1 ~~> fc2 ~~> Nil`) is rejected when the model is constructed.

What about computed shapes? In Idris, type-level arithmetic is just arithmetic (the `+` in a type is the same `+` the program runs), so `concat_cols` and `flatten` need nothing special. And Idris can no more guess that `k + n` equals `n + k` than GHC could; the difference is that here the proof is an ordinary function from the standard library, passed like any other value. Mirroring the Haskell example:

```idris
data Tensor : (rows : Nat) -> (cols : Nat) -> Type where
  MkT : Tensor r c

concatCols : Tensor m k -> Tensor m n -> Tensor m (k + n)
flatten : Tensor r c -> Tensor 1 (r * c)

flat : Tensor 1 784
flat = flatten (MkT {r=28, c=28})   -- 28 * 28 = 784, computed by the checker

concatFlipped : {k, n : Nat} -> Tensor m k -> Tensor m n -> Tensor m (k + n)
concatFlipped a b = rewrite plusCommutative k n in concatCols b a
```

All of this compiles as-is. `plusCommutative` is regular library code and `rewrite` is part of the language — no plugin, no `unsafeCoerce`.

> [!NOTE]
> The lemma is only needed because `k` and `n` are unknowns; at concrete shapes the checker just computes. Idris also has automatic *proof search* (`{auto ...}` arguments, and the `%search` keyword you'll see later): the compiler finds routine evidence like bounds checks by itself, and several guarantees later in this document run on this mechanism. What search won't invent is an inductive argument like commutativity; those you name, as with `plusCommutative`.

And what about dimensions that only arrive at runtime? This is where idris-ml's dependent types take us a step beyond hasktorch. When you load a pretrained model, the hidden size, vocabulary size, and head count all come out of a `config.json` your compiled program has never seen. Pyright can't type such a dimension at all; Haskell has to reflect it through `singletons`. In idris-ml it's standard — `Gpt2.fromPretrained` for example reads the config at runtime and builds a model whose *type* carries those dimensions:

```idris
case decEq (hidden cfg) (numHeads cfg * headDim cfg) of
  No _     => pure (Left (ConfigError "n_embd not divisible by n_head"))
  Yes prfH => hfGpt2Model {hidden = hidden cfg} {numHeads = numHeads cfg} {prfH} ...
```

`decEq` is a runtime check that produces compile-time evidence: in the `Yes` branch the compiler knows `hidden = numHeads * headDim` and lets the model be built; the `No` branch is an ordinary typed error. The model-building code was type-checked once, for *every* possible value of these dimensions — the checkpoint just picks which one this run gets.

This is what makes it possible to apply the type-level guarantees to existing repositories of pretrained models: [idris-transformers](idris-transformers.md) loads real HuggingFace checkpoints (BERT, GPT-2, Llama) by name, with the same shape checking throughout — see the [Summary](#summary) below.

Run this section live: [`tutorials/01_tensors_and_types.ipynb`](../../packages/jupyter/notebooks/tutorials/01_tensors_and_types.ipynb), and [`models/bert.ipynb`](../../packages/jupyter/notebooks/models/bert.ipynb) for the checkpoint loading.

---

## 2. Device mismatches

Here's a different bug class. You move your model to the GPU to train faster — and forget to move another tensor along with it. In PyTorch:

```python
a = torch.randn(4, device="mps")   # mps is the GPU device on this Apple-silicon machine
b = torch.randn(4)                 # b defaults to cpu
a + b
```

Once again the error only surfaces at runtime:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

There's a second kind of device bug too: asking for hardware that isn't there. Request CUDA on that same Mac, and again the check only happens when the line runs:

```python
torch.randn(4, device="cuda")
```
```text
AssertionError: Torch not compiled with CUDA enabled
```

As with shapes, all the information needed to catch both bugs is already in the source. Every tensor's device is fixed at the point of creation (`device="mps"`), and which backends your installation supports is known before the program starts. Nothing checks either.

Does TensorFlow v1's static graph help this time? No. In TensorFlow v1 a device is a *placement directive*, resolved when the session runs, not when the graph is built. Pin an op to a missing device and construction succeeds; it fails only when executed:

```python
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
with tf.device('/device:GPU:0'):
    c = tf.constant([1.0]) + tf.constant([2.0])
tf.Session(config=tf.ConfigProto(allow_soft_placement=False)).run(c)
```
```text
InvalidArgumentError: Cannot assign a device for operation add: {{node add}} was explicitly
assigned to /device:GPU:0 but available devices are [ …/device:CPU:0 ]. … The requested device
appears to be a GPU, but CUDA is not enabled.
```

Could a static graph catch this in principle? Yes: TensorFlow's graph even records the device on each node (that's how the error above can name the assignment). It declines to check for reasons built into the graph design: a graph is portable, so which devices exist is a fact about whichever machine runs the session; and a cross-device edge isn't treated as a bug at all, because the runtime silently inserts the transfer, replacing PyTorch's error with a hidden copy. But suppose a different static-graph framework did check devices at construction. The check would still live in the generated graph, not in your program; you're still meta-programming through the keyhole. The fix is the same as before: move the check into the language.

hasktorch treats it as a type problem, and this time ordinary Haskell is enough: no arithmetic, just a phantom type parameter recording the device:

```haskell
data DeviceType = CPU | CUDA
data Tensor (dev :: (DeviceType, Nat)) = Tensor
addT :: Tensor d -> Tensor d -> Tensor d           -- both operands share the device
cpu  :: Tensor '( 'CPU, 0);  cuda :: Tensor '( 'CUDA, 0)
bad  = addT cpu cuda
```
```text
error: [GHC-83865]
    • Couldn't match type ‘CPU’ with ‘CUDA’
      Expected: Tensor '(CUDA, 0)
        Actual: Tensor '(CPU, 0)
    • In the first argument of ‘addT’, namely ‘cpu’
```

To be fair to hasktorch, this fully solves the first bug: mixing devices is a compile error, with no caveats about literals or arithmetic this time.

But nothing connects the phantom to the actual build. The type `'( 'CUDA, 0)` says nothing about whether this installation contains CUDA (our second bug). The tensor type-checks on any machine, and the assertion comes back at runtime, just from deeper inside libtorch. The same goes for combinations the hardware forbids (Apple's Metal GPU only supports 32-bit floats, for example): you could write such constraints by hand, but nothing ties them to the build you're actually running.

idris-ml starts from the same place. Both operands of an operation share the executor variable `ex`, which is all the first bug needs:

```idris
tadd : Tensor dims ex dt g -> Tensor dims ex dt g -> IO (Tensor dims ex dt g)
```

The second bug needs more: the type system has to be told which backends this build contains. When you build idris-ml you choose the backends (`make BACKEND=tape,torch,mlx`), and the build *generates* a `Linked` instance for each one, a type-level record of what's in the binary. Constructing a tensor requires `Linked`, so naming a backend you didn't build is a compile error: a missing instance, refused by the proof search we met in section 1:

```idris
cudaLinked : Linked (TorchExecutor (TCuda 0))
cudaLinked = %search
```
```text
Error: While processing right hand side of cudaLinked.
Can't find an implementation for Linked (TorchExecutor (TCuda 0)).
```

The same mechanism rules out (device, dtype) pairs the hardware can't support. Metal is F32-only, so there is simply no `Compatible` instance pairing it with F64. In PyTorch that's another runtime error (`TypeError: Cannot convert a MPS Tensor to float64 dtype …`); here the pair can't be constructed:

```idris
bad = compatOK {ex=MlxExecutor MGpu} {dt = F64}
```
```text
Error: While processing right hand side of bad.
Can't find an implementation for Compatible (MlxExecutor MGpu) (Float 64).
```

One case remains that no compiler can see: hardware that is linked into the build but absent on the machine where the binary eventually runs. That arrives as an ordinary typed error, `Left (DeviceUnavailable …)` from the explicit transfer functions, rather than an abort in the C layer. The rest is uniform: a mismatched device, a backend you didn't build, and a (device, dtype) pair the hardware can't support are all the same compile-time error, a missing instance.

Run this section live: [`tutorials/07_device_safety.ipynb`](../../packages/jupyter/notebooks/tutorials/07_device_safety.ipynb).

---

## 3. Grad-mode and model ownership

The next bug class is about *grad-mode*: whether a tensor is recording the operations that backpropagation will later walk. The simplest way to get it wrong in PyTorch is to call `backward` on a computation whose weights aren't marked trainable:

```python
w = torch.randn(4)         # requires_grad defaults to False
loss = (w * 2).sum()
loss.backward()
```
```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Building the loss inside a `no_grad` block (recording temporarily switched off) produces the identical error:

```python
with torch.no_grad():
    loss = (w * 2).sum()       # w requires_grad=True
loss.backward()
```

Grad-mode is a property of a value, so these two bugs are the same kind of problem as sections 1 and 2, and the other frameworks add nothing new (TensorFlow v1 has no grad-mode to toggle, since gradients are explicit graph construction; hasktorch's typed tensors don't carry grad-mode at all). In idris-ml, grad-mode is one more type parameter: a tensor is `WithGrad` or `NoGrad`, and `trainStep` only accepts a loss that is recording:

```idris
trainStep : ... => NativeOptimizer ex -> Tensor [] ex dt WithGrad -> IO Double
```

Pass it a `NoGrad` loss and it is rejected:

```idris
brokenStep : NativeOptimizer TapeExecutor -> IO Double
brokenStep opt = trainStep opt fakeNoGradLoss
```
```text
Error: While processing right hand side of brokenStep. When unifying:
    Tensor (the (Vect 0 Nat) []) TapeExecutor F64 NoGrad
and:
    Tensor [] TapeExecutor ?dt WithGrad
Mismatch between: NoGrad and WithGrad.
```

Once again a runtime error has become a compile-time one.

The third version of the bug is harder, because it is silent. When you fine-tune, you freeze part of a trained model to train just the classifier head, and then keep using the handles you already hold. For example in PyTorch:

```python
for p in model.parameters(): p.requires_grad = False
optimizer.step()               # references the same params: no error, and no training
```

The training loop runs, the loss is computed, `step()` returns — and the model never learns. There's no traceback because, as far as the runtime is concerned, nothing is wrong.

Setting `requires_grad = False` mutates state deep in the C++ runtime, while every Python reference to the model (including the parameter list inside the optimizer) looks exactly as it did before. The problem isn't a value with a wrong property; it's a *stale reference to something that changed*. Dependent types don't address that, but linear types do.

TensorFlow v1 doesn't have the stale-handle bug, but only because it doesn't have handles: parameters are variables inside the graph, updated by ops, so there is nothing to go stale. Nothing checks ownership here either. The bug disappears along with the eager values, and giving up direct values is part of the static graph trade-off already discussed.

Haskell is further along here than you might expect. GHC has linear types as an extension: `%1 ->` is a function arrow that requires its argument to be consumed exactly once, which is the check our bug needs. Consume a model twice and GHC rejects it:

```haskell
{-# LANGUAGE LinearTypes #-}

data Model = Model
data Output = Output

eval :: Model %1 -> (Model, Output)
eval m = (m, Output)

bad :: Model %1 -> ((Model, Output), (Model, Output))
bad m = (eval m, eval m)   -- uses the consumed model again
```
```text
error: [GHC-18872]
    • Couldn't match type ‘Many’ with ‘One’
        arising from multiplicity of ‘m’
    • In an equation for ‘bad’: bad m = (eval m, eval m)
```

To be fair to GHC, this catches the freeze-then-train bug at compile time, at least in a toy example. Scaling it to a real training loop is harder: the loop lives in `IO`, so the model must thread linearly through monadic code, and the linear `do`-notation and multiplicity polymorphism that requires are experimental or incomplete. And hasktorch itself uses none of this: its models are ordinary values, so the stale-handle bug compiles without complaint.

In idris-ml, a model is a linear resource: the multiplicity `1` is written into every operation's signature, and the whole training surface is built on it. `forward`, `eval`, and `freeze` each consume the handle and return a fresh one:

```idris
eval : ... => (1 _ : l i o ex dt WithGrad) -> L IO (l i o ex dt NoGrad)
```

Consume a model, then use the old handle (the PyTorch silent no-op), and the program is rejected:

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

Note `eval`'s return type: taking a model out of training also moves it to `NoGrad`, so its outputs can't reach `trainStep` either. The grad-mode gate from the start of this section and the linear discipline compose.

Individual *tensors* are deliberately not linear: reverse-mode autograd needs the same tensor to feed several branches of the graph, so use-exactly-once typing on tensors would reject correct programs. The linear rule applies only to the model handle, the value the stale-reference bug is about.

Run this section live: [`tutorials/05_model_ownership.ipynb`](../../packages/jupyter/notebooks/tutorials/05_model_ownership.ipynb).

> [!NOTE]
> Tensors have one ownership hazard of their own: a handle can outlive the memory scope that backs it. Counting uses is the wrong check for that; the planned mechanism is region types, a scope parameter in the style of Haskell's `runST` (see [the backlog](../../TODO.md)).

---

## 4. Lossy dtype casts must be explicit

The next bug class doesn't crash at all, and this time you don't even write the operation that causes it. Mix two dtypes in ordinary PyTorch arithmetic and *type promotion* silently picks a common dtype for the result. Usually that's fine. But the promotion rules have a lossy corner: an integer tensor meeting a float tensor promotes to float32, and float32 can only represent integers exactly up to 2²⁴:

```python
n = torch.tensor(16777217)   # int64; 2**24 + 1
x = torch.tensor([1.0])      # float32
print(n + x)
```
```text
tensor([16777216.])
```

The answer is wrong. There's no error and no warning; a conversion you never wrote sent a 64-bit integer through a 24-bit mantissa, and the value quietly changed. Deliberate casts have a milder version of the same problem: `x.half()` is at least explicit, but it's spelled exactly like the harmless `x.double()`, so nothing in the code marks which casts discard information.

Does the static graph help? Partly, this time. TensorFlow v1 refuses to promote implicitly (a mixed-dtype op is rejected when the graph is built), so every conversion goes through an explicit `tf.cast`. But `tf.cast` is one undiscriminating surface: the lossless and the lossy conversion are spelled identically, with no error or warning on the lossy one:

```python
import tensorflow.compat.v1 as tf
y = tf.cast(tf.constant([1.0], dtype=tf.float64), tf.float16)
print(y.dtype.name)
```
```text
float16          # no error
```

Whether a conversion is lossless is a relationship between the two dtypes: every F32 value fits in F64, a big int64 doesn't fit in F32, and you can compute all of this from the bit widths. A checker that wants to catch the lossy ones has to compare properties of the two types.

hasktorch carries the dtype in the tensor's type, and that alone kills the promotion bug: an int64 tensor and a float32 tensor have different types, so mixed arithmetic doesn't unify and every conversion is an explicit call. But nothing orders the dtypes, so the narrowing conversion type-checks exactly like the widening one. This compiles clean:

```haskell
data DType = F64 | F16
data Tensor (dt :: DType) = Tensor

toDType :: Tensor a -> Tensor b            -- no lossless-order premise
narrow :: Tensor 'F16
narrow = toDType (Tensor :: Tensor 'F64)   -- silently drops 42 bits of mantissa
```

The phantom records what the dtype is; it says nothing about what a conversion between two of them preserves. Expressing "lossless if neither the mantissa nor the exponent shrinks" means comparing bit widths at the type level, which is the same type-level arithmetic that section 1 showed Haskell struggling with.

idris-ml gets the first half the same way hasktorch does: dtype is a type parameter, mixed-dtype arithmetic doesn't unify, and there is no promotion to have a lossy corner. The second half is a rule written once, as an instance whose premises are ordinary comparisons on the dtypes' bit widths:

```idris
FloatPrecision from => FloatPrecision to =>
LTE (mantissaBits {t=from}) (mantissaBits {t=to}) =>
LTE (exponentBits {t=from}) (exponentBits {t=to}) =>
LosslessTo from to where
```

One instance covers all four float-family combinations (F→F, BF→BF, F→BF, BF→F), and the cast function `tcast` demands the resulting evidence. A widening cast compiles with nothing extra:

```idris
widen : Tensor [4] TapeExecutor F32 NoGrad -> IO (Tensor [4] TapeExecutor F64 NoGrad)
widen v = tcast v
```

The conversion that PyTorch's promotion performs silently is refused:

```idris
intToFloat : Tensor [4] TapeExecutor (IntN 64) NoGrad -> IO (Tensor [4] TapeExecutor F32 NoGrad)
intToFloat v = tcast v
```
```text
Error: While processing right hand side of intToFloat.
Can't find an implementation for UpcastableTo (IntN 64) (Float 32).
```

So is the deliberate drop to half precision:

```idris
halfIt : Tensor [4] TapeExecutor F64 NoGrad -> IO (Tensor [4] TapeExecutor F16 NoGrad)
halfIt v = tcast v
```
```text
Error: While processing right hand side of halfIt.
Can't find an implementation for UpcastableTo (Float 64) (Float 16).
```

That last one is a conversion you legitimately want (the model that doesn't fit in memory really should be halved), and it isn't forbidden: it goes through `tcastUnsafe`, a separate function whose name states that information is being discarded. That's the difference from `x.half()`: the lossy casts have their own distinguished surface, so every point of precision loss is spelled out in the program, and the lossless ones cost nothing.

Run this section live: [`tutorials/09_precision_devices.ipynb`](../../packages/jupyter/notebooks/tutorials/09_precision_devices.ipynb).

---

## 5. Multiple backends in one program

The previous sections harden properties the other frameworks also have, just checked later or not at all. This section is a capability they don't have. Suppose you prototype a model on a small dependency-free CPU backend where debugging is easy, train it on libtorch, and run inference on Apple's MLX. In every mainstream framework that's three programs, because a framework *is* its runtime: a PyTorch tensor is a libtorch value, and there is nothing else for it to be.

The closest PyTorch gets is a manual hop through numpy into a second array library, living alongside it in the same process:

```python
import torch, mlx.core as mx
t = torch.randn(4)         # a libtorch value
a = mx.array(t.numpy())    # copied out through numpy into MLX
```

Nothing records which library a value belongs to, so there is no check to fail; mixing `t` and `a` fails at runtime or silently detours through numpy, depending on the operation. TensorFlow has the same shape: one runtime, with conversion to anything else done by hand at the numpy boundary. And hasktorch is bound to libtorch by construction. Its device phantom from section 2 is a closed enumeration (`CPU | CUDA`); there is no way to say "a tensor belonging to a different tensor library" at all.

In idris-ml the backend is a type parameter like everything else, and the set of backends is *open*: `Executor` is an ordinary kind, and any type with the right interface implementations can inhabit it. That includes yours. Declare a tag type, bind your library's C symbols, implement the executor interfaces, and `Tensor [4] MyBackend` is a working type that dispatches every operation to your code; [`Example/BringYourOwn.idr`](../../packages/idris-ml-examples/src/Example/BringYourOwn.idr) walks through the whole recipe against a 100-line stub backend. The three that ship are tape (pure C), libtorch, and MLX, and one build links every backend you name into a single library:

```bash
make BACKEND=tape,torch,mlx backend     # tape, libtorch, and MLX in one libidrisml.{so,dylib}
```

One program then holds tensors from different backends at once. They simply have different types:

```idris
tapeVec  : Tensor [4] TapeExecutor        F64 g   -- pure-C tape
torchVec : Tensor [4] (TorchExecutor TMps) F32 g  -- libtorch on Metal
```

Mixing them is the by-now-familiar compile error:

```idris
mix : Tensor [4] TapeExecutor F64 NoGrad ->
      Tensor [4] (TorchExecutor TCpu) F64 NoGrad ->
      IO (Tensor [4] TapeExecutor F64 NoGrad)
mix a b = tadd a b
```
```text
Error: While processing right hand side of mix. When unifying:
    Tensor [4] (TorchExecutor TCpu) F64 NoGrad
and:
    Tensor [4] TapeExecutor F64 NoGrad
Mismatch between: TorchExecutor TCpu and TapeExecutor.
```

The only way across is the explicit transfer function. Here is a vector moving tape → libtorch → MLX → tape, its value checked at the end:

```idris
roundtripF64Smoke = do
  v0 <- makeVec4 {ex=TapeExecutor}        expected     -- pure-C tape
  v1 <- toExecutor (TorchExecutor TCpu) v0             -- → libtorch
  v2 <- toExecutor (MlxExecutor MCpu)   v1             -- → MLX
  v3 <- toExecutor TapeExecutor         v2             -- → back to tape
  check "F64 roundtrip Tape→Torch→Mlx→Tape preserves value" (matchesExpected (read4 v3))
```

Section 2's machinery applies per backend: `Linked` keeps un-built backends out of reach, `Compatible` rules on each backend's (device, dtype) pairs, and hardware absent at runtime is the same typed error. Prototype, train, and deploy across multiple runtimes, in one type-checked program.

Run this section live: the multi-backend cells of [`tutorials/07_device_safety.ipynb`](../../packages/jupyter/notebooks/tutorials/07_device_safety.ipynb).

---

## Summary

The static-graph era conflated *shape safety* with *graph structure*. You don't need a static graph to get static checking; idris-ml uses a type system that can express the constraints. And the five guarantees in this document come from just two language features: shapes, devices, grad-modes, and dtypes are ordinary type parameters checked by the same dependent-type machinery, and model ownership adds linear types. No compiler plugins, no reflection encodings. In PyTorch, four of these checks happen at runtime and the fifth can't be expressed; in idris-ml all five are the compiler's job.

The costs are real too. idris-ml is young: compile times are longer than you're used to, unification errors take practice to read, the ecosystem is one repository rather than PyTorch's universe of libraries and answers, and performance today trails PyTorch on many workloads (every example trains against a PyTorch reference implementation, which doubles as the benchmark). What you get in exchange is the subject of this document: whole classes of bugs caught before the program runs.

None of this is confined to synthetic demonstrations. [idris-transformers](../../packages/idris-transformers/) loads real HuggingFace checkpoints: each supported architecture is one Idris module whose parameters and shapes match the checkpoint on disk, so loading is a single `fromPretrained` call (parse `config.json`, fill the weights from `model.safetensors`) with no remapping layer, and the shapes flow into the model's type exactly as in section 1:

```idris
fromPretrained : Backend ex dt => KnownGrad g
              => (modelDir : String)
              -> IO (Either LoadError (cfg : Gpt2Config ** Gpt2Model cfg ex dt g))
```

idris-transformers is one of several packages in the repository alongside the core library:

- [`idris-transformers`](../../packages/idris-transformers/): the HuggingFace checkpoint loading above (BERT, GPT-2, Llama, BitNet).
- [`idris-ml-examples`](../../packages/idris-ml-examples/): runnable examples covering supervised training, recurrent and memory architectures (RNN/LSTM/NTM/DNC), reinforcement learning, and section 5's bring-your-own backend.
- [`idris-gym`](../../packages/idris-gym/): reinforcement-learning environments with a Gymnasium-style API, in pure Idris.
- [`idris-ml-notebook`](../../packages/idris-ml-notebook/): a prelude that re-exports the whole library for notebook use.
- [`jupyter`](../../packages/jupyter/): a Jupyter kernel for running Idris interactively, including every compile-error demo in this document.

---

## Next steps

- [Getting Started](getting-started.md) / [Jupyter notebooks](../../packages/jupyter/README.md): run the compile-error demos from this document live.
- [PyTorch Mapping](pytorch-mapping.md): concept-by-concept translation for PyTorch users.
- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md): the dependent-types-for-shapes argument in full, with a worked example threading dimensions through an NTM.
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md): what each guarantee requires from a type system, including the "Custom devices" section behind section 5's bring-your-own backend.

---

[^prov]: Every error message and every "this compiles" claim in this document is captured from a real toolchain run: PyTorch 2.11, TensorFlow 2.21 (graph mode via `tf.compat.v1`, the TF 1.x API), Pyright 1.1.410, GHC 9.10.3, and Idris 2 0.8, the non-PyTorch ones via disposable `nix shell` environments. The Haskell snippets are minimal stand-ins for the mechanism hasktorch's `Torch.Typed` uses (the same type-level `Nat` / DataKinds phantoms), since hasktorch itself needs a working libtorch to build. Only long internal node-attribute lists in TF errors are elided with `…`.
