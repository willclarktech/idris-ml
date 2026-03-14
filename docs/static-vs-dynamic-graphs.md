# Static Graphs, Dynamic Graphs, and Dependent Types

How idris-ml combines the safety of static computation graphs with the flexibility of dynamic ones.

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

Dynamic graphs won decisively. TensorFlow 2.0 switched to eager execution by default. Every major framework adopted define-by-run.

The core reason: **research moves faster when the framework gets out of the way.** Static graphs imposed a framework-specific programming model on top of the host language. Researchers had to learn `tf.cond` instead of `if`, `tf.while_loop` instead of `for`, `tf.print` instead of `print`. Every debugging session meant translating between two mental models — the Python code that built the graph and the graph nodes where errors actually occurred.

Dynamic graphs eliminated this friction. The model *is* the code. When something goes wrong, the error points to the line that caused it. When you need a conditional architecture, you write an `if` statement.

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

The `Tensor` type is indexed by its shape:

```idris
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty
```

A `Vector 784 Double` and a `Vector 256 Double` are different types. You cannot pass one where the other is expected:

```idris
matrixVectorMultiply : Matrix m n ty -> Vector n ty -> Vector m ty
```

If the matrix is `Matrix 10 784` and you pass a `Vector 256`, the compiler rejects it: `n` can't unify `784` with `256`. No runtime check needed.

### Networks enforce dimension threading

The `Network` type chains layers with compile-time dimension threading:

```idris
data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
  OutputLayer : AnyLayer i o ty -> Network i [] o ty
  (~>) : AnyLayer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty
```

The `(~>)` constructor requires the first layer's output dimension `h` to equal the next layer's input dimension. This is not a runtime assertion — it's a type constraint. A dimension mismatch is a compile error:

```idris
-- This compiles: Linear(2→10) feeds into Softmax(10→10)
ll <- linearLayer {i=2, o=10}
let model = ll ~> OutputLayer softmaxLayer

-- This would NOT compile: Linear(2→10) can't feed into Linear(5→3)
-- because 10 ≠ 5
```

### NTM dimensions are computed at the type level

The dimension relationships that cause subtle bugs in PyTorch are type-level functions in idris-ml:

```idris
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m
```

The NTM state record encodes every dimension dependency in its type:

```idris
record NtmState (n : Nat) (m : Nat) (h : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  lstm     : LstmState (m + inputSize) h ty          -- controller input = read output + data input
  readFc   : LinearState h (ReadParamWidth m) ty      -- hidden → addressing params
  writeFc  : LinearState h (WriteParamWidth m) ty     -- hidden → addressing + add vector
  outputFc : LinearState (h + m) outputSize ty        -- hidden + read output → data output
  memory   : Matrix n m ty                            -- n slots × m width
```

If you change `m` from 20 to 32, the compiler propagates the change through every dependent dimension. If any sub-layer constructor doesn't match, you get a compile error — not a crash 10,000 epochs into training.

### What you get

The `forward` function's type signature guarantees dimension correctness through the entire network:

```idris
forward : Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)
```

Input must be `Vector i`. Output is guaranteed `Vector o`. Every intermediate dimension is checked at compile time by the `(~>)` chain. The `LayerLike` interface enforces that each layer implementation respects its declared dimensions.

This gives you:

| Property | Static graphs | Dynamic graphs | idris-ml |
|----------|:---:|:---:|:---:|
| Shape errors caught before training | Partially | No | **Yes (all)** |
| Natural control flow | No | Yes | Yes |
| Standard debugging | No | Yes | Yes |
| Variable-length sequences | Awkward | Natural | Natural |
| Silent broadcasting bugs | Possible | Possible | **Impossible** |
| Dimension change propagation | Manual | Manual | **Automatic** |

idris-ml's computation graph is dynamic — each forward pass builds a fresh autograd tape, control flow is standard Idris, variable-length sequences work naturally. But the *shape constraints* are static, verified at compile time by the type system. You get the ergonomics of PyTorch with stronger safety guarantees than TensorFlow 1.x ever provided.

The key insight: static graphs conflated two concerns — **shape safety** and **graph structure**. You don't need a static graph to get static shape checking. You need a type system that can express dimensional constraints. Dependent types provide exactly this: shapes live in types, checked at compile time, erased at runtime.
