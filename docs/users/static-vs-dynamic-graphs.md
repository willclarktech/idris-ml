# Static Graphs, Dynamic Graphs, and Dependent Types

How idris-ml combines the safety of static computation graphs with the flexibility of dynamic ones.

> This is the deep dive on the **shape** guarantee and the static/dynamic narrative. For
> the full five-guarantee comparison (shape, device, multi-backend, grad-mode, dtype)
> against PyTorch, TF1/JAX, and Haskell, start at [**Why idris-ml**](why-idris-ml.md).

## The two eras of deep learning frameworks

### Static graphs (TensorFlow 1.x, Theano)

Early frameworks used a **define-then-run** model. You built a symbolic computation graph — a data structure describing the operations — then compiled and executed it:

```python
# TensorFlow 1.x: build a graph, then run it in a session
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y = tf.matmul(x, W)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: data})
```

The framework could analyze the entire graph before execution: fuse operations, schedule GPU kernels, prune dead branches, distribute across devices. The graph was a first-class object that could be serialized, optimized, and deployed.

**Advantages:**
- Whole-graph optimization (operator fusion, memory planning, dead code elimination)
- Easy serialization and deployment (export the graph, run it anywhere)
- The framework knows the full computation structure upfront

**Disadvantages:**
- Painful debugging (errors reference graph nodes, not Python lines)
- No standard control flow (loops and conditionals require special graph ops like `tf.while_loop`)
- Two-language problem: Python describes the graph, C++ executes it. The two are disconnected
- Boilerplate-heavy (sessions, placeholders, feed dicts)

### Dynamic graphs (PyTorch, TensorFlow 2.x)

PyTorch popularized **define-by-run**: the graph is built implicitly as operations execute. Each forward pass constructs a fresh graph by recording operations on tensors with `requires_grad=True`:

```python
# PyTorch: just write Python
x = torch.randn(32, 784)
W = torch.randn(784, 10, requires_grad=True)
y = x @ W

y.sum().backward()  # graph was already built during forward pass
```

Standard Python control flow works naturally — `if`, `for`, `while` all just execute and the resulting operations are recorded. Debugging uses normal Python tools.

**Advantages:**
- Natural Python control flow (variable-length sequences, conditional architectures)
- Standard debugging (print, breakpoints, stack traces point to your code)
- Rapid prototyping (change the model, run it immediately)
- No two-language split

**Disadvantages:**
- Less opportunity for whole-graph optimization (though `torch.compile` partially recovers this)
- Graph must be rebuilt every forward pass
- Shape errors are runtime errors

## Why dynamic graphs won

Dynamic graphs won: TensorFlow 2.0 switched to eager execution by default, and every major framework adopted define-by-run.

The core reason: static graphs imposed a framework-specific programming model on top of the host language. Researchers had to learn `tf.cond` instead of `if`, `tf.while_loop` instead of `for`, `tf.print` instead of `print`. Every debugging session meant translating between two mental models — the Python code that built the graph and the graph nodes where the error occurred.

Dynamic graphs removed that translation layer: the model and the code are the same artifact. When something goes wrong, the error points to the line that caused it. When you need a conditional architecture, you write an `if` statement.

For architectures with variable-length or data-dependent structure — RNNs over sequences of different lengths, tree-structured networks, NTMs with dynamic memory access — dynamic graphs are natural. Static graphs require encoding these patterns as graph-level operations, which is awkward and error-prone.

## What we lost: the shape error problem

Dynamic graphs solved the usability problem but left a significant safety gap. **Shape mismatches are runtime errors**, and they can hide deep in a training pipeline:

```python
class MyModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # Bug: should be 256, not 128

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

This is a trivial example — you'll find it immediately. But consider:

**The NTM dimension problem.** A Neural Turing Machine has an LSTM controller, separate read/write head FC layers, an output FC, and a memory matrix. The dimensions interlock:

- LSTM input = memory width + data input width
- Read FC: hidden size → key width + shift kernel + 3 scalar params
- Write FC: hidden size → read params + add vector width
- Output FC: hidden size + memory width → data output width

Change the memory width and five layer dimensions must update in concert. In PyTorch, a typo in any one of them produces a runtime error — potentially thousands of epochs into training if the mismatch only triggers on certain sequence lengths or batch configurations. Worse, some mismatches don't crash at all: PyTorch may silently broadcast a `[1, 10]` tensor to match a `[5, 10]` tensor, producing plausible but wrong results.

**The broader pattern:**
- You wire up a complex architecture
- It type-checks (Python has no relevant types here)
- You start training
- 20 minutes in, a rare code path triggers a shape mismatch
- Or worse: shapes are "compatible" via broadcasting but semantically wrong, and you spend days debugging mysterious non-convergence

Static graphs at least caught some of these errors at graph construction time, before training began. Dynamic graphs pushed all shape validation to execution time.

## How idris-ml solves this

Idris 2 is a dependently typed language: types can contain *values*. A vector's length, a matrix's dimensions, a network's input/output sizes — these are part of the type, checked at compile time, and erased at runtime (zero overhead).

### Shapes in the type

The `Tensor` type is indexed by its shape (a `Vect rank Nat`), alongside its executor,
dtype, and grad-mode:

```idris
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr        -- backend handle (carries the autograd tape)
  paramId   : Maybe String
```

(The pure-Idris `Array` type *is* the Vect-of-Vect; `Tensor` is the autograd handle whose
type *carries* the shape while the data lives in a backend buffer.) A `Tensor [784] ex dt g`
and a `Tensor [256] ex dt g` are different types — you can't pass one where the other is
expected. The matrix-vector op makes the constraint explicit:

```idris
tmv : Tensor [m, n] ex dt g -> Tensor [n] ex dt g -> IO (Tensor [m] ex dt g)
```

If the matrix is `Tensor [10, 784]` and you pass a `Tensor [256]`, the compiler rejects
it: `n` can't unify `784` with `256`. No runtime check needed.

### Models enforce dimension threading

Models are records of `Nn` layers. A `Seq` chains `Module`s with compile-time dimension
threading; its type pins only the endpoints, and the hidden dimensions are existential:

```idris
data Seq : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  Nil  : Seq i i ex dt g
  (::) : Module l => {h : Nat} ->
         (1 _ : l i h ex dt g) -> (1 _ : Seq h o ex dt g) -> Seq i o ex dt g

(~~>) : ...   -- right-associative alias for (::)
```

The `(::)` / `~~>` constructor requires the first layer's output dimension `h` to equal
the next layer's input dimension. This is not a runtime assertion — it's a type
constraint. A dimension mismatch is a compile error:

```idris
mkModel : Init (Seq 2 3 ex dt WithGrad)
mkModel = do
  l1 <- linear {i=2} {o=10}
  l2 <- linear {i=10} {o=3}
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)   -- compiles: 10 (l1 out) unifies with 10 (l2 in)

-- Swap l2 for `linear {i=5} {o=3}` and the chain won't elaborate:
--   Mismatch between: 10 and 5
```

### NTM dimensions are computed at the type level

The dimension relationships that cause subtle bugs in PyTorch are type-level functions in idris-ml:

```idris
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m
```

The NTM cell record encodes every dimension dependency in its type:

```idris
record Ntm (n : Nat) (m : Nat) (h : Nat) (i : Nat) (o : Nat)
           (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkNtm
  controller : Lstm (m + i) h ex dt g              -- controller input = read output + data input
  readFc     : Linear h (ReadParamWidth m) ex dt g  -- hidden → addressing params
  writeFc    : Linear h (WriteParamWidth m) ex dt g  -- hidden → addressing + add vector
  outputFc   : Linear (h + m) o ex dt g             -- hidden + read output → data output
  memInitT   : TVec (m * n) ex dt g                 -- learned memory initialization
  -- ... per-sequence state (memT, read/write addresses, last read-out)
```

If you change `m` from 20 to 32, the compiler propagates the change through every dependent dimension. If any sub-layer constructor doesn't match, you get a compile error — not a crash 10,000 epochs into training.

### Device-dtype compatibility as compile-time constraint

The same dependent-types lever applies to **runtime errors that aren't about shapes at all**. Metal GPU dropped float64 support in mlx 0.31. PyTorch users find out at runtime:

```python
x = torch.tensor([1.0, 2.0], dtype=torch.float64).to("mps")
# RuntimeError: Cannot convert a MPS Tensor to float64 dtype as the
# MPS framework doesn't support float64. Please use float32 instead.
```

Same problem class as the shape error: a hardware/library limitation discovered when a particular code path runs. idris-ml lifts it to the type system via a `Compatible (0 ex : Executor) (0 t : DType)` empty marker interface — the instance head IS the proof:

```idris
public export Compatible (MlxExecutor MCpu) F64 where  -- ✓ mlx CPU supports f64
public export Compatible (MlxExecutor MCpu) F32 where  -- ✓ mlx CPU supports f32
public export Compatible (MlxExecutor MGpu) F32 where  -- ✓ Metal GPU supports f32
-- DELIBERATELY NO `Compatible (MlxExecutor MGpu) F64` instance
```

The `Tensor` record has a 0-quantity dtype slot, and every tensor-construction smart constructor carries `Compatible ex dt =>`. So (from the runnable demo `Example/DTypePitch.idr`):

```idris
okMlxGpuF32 : ()
okMlxGpuF32 = compatOK {ex=MlxExecutor MGpu} {dt = F32}   -- ✓ Compatible instance exists

-- Uncommenting this is a compile error:
-- badMlxGpuF64 = compatOK {ex=MlxExecutor MGpu} {dt = F64}
-- ✗ Can't find an implementation for Compatible (MlxExecutor MGpu) F64
```

The error fires at the construction site, with a name the user wrote (`MlxExecutor MGpu`, `F64`) and a concept the user can act on (`Compatible`).

A *derived* partial order extends the same machinery to lossless casts. The integer families (`IntN n`, `UInt n`) each declare an `UpcastableTo from to` instance requiring `LTE m n` on the bit-widths. Float upcasts are derived structurally: `FloatPrecision` records each float type's (mantissa, exponent) layout, `LosslessTo` admits exactly the conversions where every value stays exactly representable (both fields non-decreasing; integer→float when the mantissa covers the integer range), and a bridge instance turns every `LosslessTo` edge into an `UpcastableTo`. Idris's auto-search synthesizes the `LTE` proofs at the call site:

```idris
demoUpcast : UpcastableTo from to => ()

okF32ToF64 : ()
okF32ToF64 = demoUpcast {from = F32} {to = F64}    -- ✓ LTE 32 64 is provable

-- failF64ToF32 = demoUpcast {from = F64} {to = F32}  -- ✗ LTE 64 32 is not provable
```

Conversions that can lose information (`F64 → F32`; `F16 → BF16`, whose mantissa shrinks from 10 to 7 bits even though the width stays 16) have no instance and require an explicit `tcastUnsafe` — the compiler can't decide whether the loss is what the user wanted.

This is exactly the kind of guarantee a dynamic graph can't give you. PyTorch's `Tensor` is a single runtime type with a dtype field; the dtype isn't visible to the type system, so the "this can't run on Metal" check happens when the kernel launches. Static graphs in TensorFlow 1.x knew the dtype at graph-construction time but didn't enforce device-dtype admissibility either; you found out at session-run. Dependent types put the (device, dtype) pair into the tensor's type, and the `Compatible` table makes the per-pair check a one-line interface declaration.

### What you get

The `forwardSeq` function's type signature guarantees dimension correctness through the
entire model (batched-first; the model is a linear resource, consumed and threaded back):

```idris
forwardSeq : (1 _ : Seq i o ex dt g) -> Tensor [b, i] ex dt g
          -> L IO {use=1} (LPair (!* (Tensor [b, o] ex dt g)) (Seq i o ex dt g))
```

Input must be `Tensor [b, i]`. Output is guaranteed `Tensor [b, o]`. Every intermediate
dimension is checked at compile time by the `(~~>)` chain. The `Module` interface enforces
that each layer implementation respects its declared dimensions.

This gives you:

| Property | Static graphs | Dynamic graphs | idris-ml |
|----------|:---:|:---:|:---:|
| Shape errors caught before training | Partially | No | **Yes (all)** |
| Natural control flow | No | Yes | Yes |
| Standard debugging | No | Yes | Yes |
| Variable-length sequences | Awkward | Natural | Natural |
| Silent broadcasting bugs | Possible | Possible | **Impossible** |
| Dimension change propagation | Manual | Manual | **Automatic** |
| Illegal device-dtype combinations | Runtime | Runtime | **Compile time** |
| Silently lossy precision casts | Allowed | Allowed | **Require explicit `tcastUnsafe`** |

idris-ml's computation graph is dynamic — each forward pass builds a fresh autograd tape, control flow is standard Idris, variable-length sequences work naturally. But the *shape constraints, device-dtype admissibility, and lossless-upcast partial order* are static, verified at compile time by the type system. You get the ergonomics of PyTorch with stronger safety guarantees than TensorFlow 1.x ever provided.

Static graphs conflated two concerns — **shape safety** and **graph structure**. You don't need a static graph to get static shape checking. You need a type system that can express dimensional constraints. Dependent types provide exactly this: shapes (and devices, and dtypes, and grad-modes) live in types, checked at compile time, erased at runtime.
