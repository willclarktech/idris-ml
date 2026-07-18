# Type-level guarantees: shape, device, grad-mode

idris-ml encodes properties of tensors at the type level: their
shape, the device they live on, and (optionally) whether they carry
gradients. The compiler rejects programs that mix CPU and CUDA
tensors, that try to multiply matrices with mismatched inner
dimensions, or that call `backward` on a tensor that wasn't being
tracked.

These three guarantees look similar on the surface. They aren't —
they sit at different levels of the type-system hierarchy. This doc
explains where each one fits, what Python's static type system could
in principle do here, and which guarantees genuinely need
dependent types.

## TL;DR

- **Shape safety** requires dependent types. Python can't express
  it (no type-level arithmetic over `Literal[N]`).
- **Device safety** and **grad-mode safety** are *phantom-type
  enums* — finite, closed sets carried as type parameters. They
  don't need dependent types. Python could express them with
  `Generic[…]` + `Literal[…]` + `@overload`, and pyright/mypy would
  enforce them. PyTorch chose not to for ergonomic reasons, not
  type-system reasons.

The interesting payoff of dependent types here isn't that they're
the *only* way to track grad-mode or device. It's that once the
compiler is already computing on shapes, the same machinery
applies uniformly to grad-mode, device, and precision — no extra
language features, no overload tables.

## Phantom-type enums vs. dependent types

**Phantom-type enum**: a type parameter whose value is drawn from a
fixed, finite set known at the type system level. The parameter
appears in types but isn't used in the runtime representation. The
classic example is a file handle parameterised over a finite set of
states:

```idris
data State = Open | Closed
record Handle (s : State) where
  ptr : AnyPtr

openFile  : String -> Handle Open
readFile  : Handle Open -> IO String     -- compile error on Closed
closeFile : Handle Open -> Handle Closed
```

This works in Haskell with `DataKinds`. It works in Rust with
zero-sized phantom types (`PhantomData<State>`). It works in
TypeScript with branded types. It works in Python with
`Generic[State]` + `Literal["open"] | Literal["closed"]`, modulo
some overload tedium.

**Dependent type**: a type parameter that depends on a *value* drawn
from an open, potentially-infinite set, *and* on which the type
system can compute (proofs, arithmetic, etc.).

```idris
matmul : Tensor [m, k] ex -> Tensor [k, n] ex -> Tensor [m, n] ex
```

Here `m`, `k`, `n` are arbitrary `Nat`s and the type system has to
unify `k` across the two arguments. The set of possible `Nat`s is
infinite, and the type checker computes on them (e.g. `Conv2D`'s
output dimension is `(w - k) `div` s + 1`).

The litmus test: can you write down all the possible type-level
parameter values on a single page? If yes, phantom enum. If no
(naturals, lists, trees, …), dependent type.

## How idris-ml uses each

The current tensor type is

```idris
Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode)
```

- `dims` is genuinely dependent — `Nat`-valued, arithmetic happens
  during type checking. This is where `matmul`-shape-safety and
  conv-output-dim safety come from.
- `0 ex : Executor` is the executor (backend) tag. `Executor` is a
  0-quantity alias for `Type`, so any type with a `UserExecutorCore`
  instance can be used. The built-ins `TapeExecutor`,
  `TorchExecutor d` (`d : TorchHwDev = TCpu | TMps | TCuda Nat`), and
  `MlxExecutor s` (`s : MlxStream = MCpu | MGpu`) forward to the
  linked C backends; users can declare their own (see "Custom
  backends" below).
- `0 dt : DType` is the dtype tag — also an open kind. `Float n`,
  `BFloat n`, `IntN n`, `UInt n`, `Bool` have built-in instances
  (`F32 = Float 32`, etc.). This is what makes the lossless-cast
  partial order and the `Compatible (ex, dt)` admissibility table
  possible.
- `0 g : GradMode` with `GradMode = WithGrad | NoGrad` is a real
  closed enum — the compiler rejects `backward` on tensors that
  weren't tracked.

The executor parameter started life as a closed sum
(`Device = CPU | CUDA Nat | MPS`) and got opened up to admit
user-supplied backends — see "Custom backends" below. The precision
parameter `dt` landed the same way: an open kind that reuses the
existing parameter machinery and additionally carries first-class
type-level arithmetic (bit-widths) for the lossless-upcast order.

## Custom backends: user-supplied executors

`Executor` is an open kind — any type with a `UserExecutorCore`
instance can sit in `Tensor`'s `ex` slot.

A worked example ships in the repo: `packages/backends/backend_byo.c`
is a small stub backend that exports `byo_tensor_add`,
`byo_tensor_item`, etc. and logs each call to stderr;
`packages/idris-ml-examples/src/Example/BringYourOwn.idr` is the
Idris-side recipe that wraps it. Run it with `make
example-bring-your-own` — you'll see the stderr `[byo] ...` lines
fire as ops dispatch through your instance, alongside the same
expression evaluated on the build's primary `ExampleExecutor` for
contrast.

A complete custom backend looks like (trimmed from that example):

```idris
import Executor
import Executor.Core

-- 1. Declare a type to tag tensors that live on this backend.
public export
data BYO : Type where MkBYO : BYO

-- 2. Bind your dylib's C symbols via %foreign.
%foreign "C:byo_tensor_add,libbyo"
prim__addBYO : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:byo_tensor_create_scalar,libbyo"
prim__createScalarBYO : Double -> Int -> AnyPtr

%foreign "C:byo_tensor_item,libbyo"
prim__itemBYO : AnyPtr -> Double
-- ... the rest of the lifecycle + arithmetic slice

-- 3. Implement the UserExecutorCore instance.
public export
UserExecutorCore BYO where
  deviceName       = "byo"
  primCreateScalar = prim__createScalarBYO
  primAdd          = prim__addBYO
  primItem         = prim__itemBYO
  -- ... all UserExecutorCore methods
```

Now `Tensor [4] BYO dt g` is a valid type, every op (`tadd`, `tmul`,
…) dispatches to your `BYO` instance, and you can transfer between
built-in and user-supplied backends via `toExecutor`.

### Parameterized backends

Your executor type can carry type parameters. CUDA's device index is
the canonical example — the built-in is `TorchExecutor (TCuda n)`; a
standalone illustration looks like:

```idris
data CUDA : Nat -> Type where MkCUDA : (n : Nat) -> CUDA n
```

`Tensor [4] (CUDA 0) dt g` and `Tensor [4] (CUDA 1) dt g` are
different types; the compiler will reject mixing them. To declare the
instance, bind the `Nat` parameter at the head of the instance
declaration:

```idris
{n : Nat} -> UserExecutorCore (CUDA n) where
  deviceName       = "cuda:" ++ show n
  primAdd          = prim__addCuda
  ...
```

The `{n : Nat} ->` prefix is the bit to notice. By default, type
parameters in an instance head are 0-quantity (erased): a body that
tried to `show n` would fail with "n is not accessible in this
context." Binding `n` non-erased makes its runtime value available
inside the method bodies.

### When you need runtime access to a parameter from the wrong context

There's one tripwire. `UserExecutorCore`'s `ex` parameter is declared
0-quantity (`interface UserExecutorCore (0 ex : Executor)`), and that
0-quantity propagates: a *caller* of `deviceName` working with
`ex = CUDA n` has `n` only at the type level. The instance body has
`n` at runtime (because the instance head binds it non-erased), but
generic library code that only sees `UserExecutorCore ex` doesn't.

If you need a separate operation that recovers the parameter, add a
helper interface:

```idris
public export
interface HasDeviceIndex (ex : Executor) where
  deviceIndex : Nat

public export
{n : Nat} -> HasDeviceIndex (CUDA n) where
  deviceIndex = n
```

`HasDeviceIndex`'s `ex` parameter is *unrestricted* (no `0`), so its
methods may observe the parameter, the same trick the built-in
`TorchExecutor (TCuda n)` uses to render its `deviceName`.

Type-only parameters (a phantom precision tag, a shape annotation
that's only there to keep types apart) need no such workaround —
just don't reference the parameter from the method body.

## What this would look like in Python

Pyright + recent typing extensions are expressive enough to model
both device and grad-mode.

### Grad-mode

```python
from typing import Generic, Literal, TypeVar, overload

WithGrad = Literal["with_grad"]
NoGrad   = Literal["no_grad"]
G = TypeVar("G", WithGrad, NoGrad)
S = TypeVar("S")  # shape
D = TypeVar("D")  # device

class Tensor(Generic[S, D, G]):
    ...

# Type-level join: WithGrad if either input is WithGrad.
@overload
def add(x: Tensor[S, D, WithGrad], y: Tensor[S, D, G]) -> Tensor[S, D, WithGrad]: ...
@overload
def add(x: Tensor[S, D, G], y: Tensor[S, D, WithGrad]) -> Tensor[S, D, WithGrad]: ...
@overload
def add(x: Tensor[S, D, NoGrad], y: Tensor[S, D, NoGrad]) -> Tensor[S, D, NoGrad]: ...

def backward(loss: Tensor[Literal[()], D, WithGrad]) -> None: ...
# Calling backward(t) where t: Tensor[..., NoGrad] is a type error.
```

This compiles in pyright and rejects `backward(no_grad_tensor)`. The
join across `(WithGrad, NoGrad)` is hand-encoded as overload rows
instead of a type-level function, but it works.

### Device

Same shape, different enum:

```python
CPU = Literal["cpu"]
MPS = Literal["mps"]
CUDA0 = Literal["cuda:0"]
CUDA1 = Literal["cuda:1"]
Dev = TypeVar("Dev", CPU, MPS, CUDA0, CUDA1)

class Tensor(Generic[S, Dev, G]): ...

def add(x: Tensor[S, Dev, G], y: Tensor[S, Dev, G]) -> Tensor[S, Dev, G]: ...
#                  ^^^         ^^^   same Dev required — cross-device add rejected
```

`Tensor[…, "cpu"] + Tensor[…, "cuda:0"]` is now a type error.
Phantom-type discipline; no dependent types involved.

## Why PyTorch doesn't do this

The capability is there in the type system. The reasons PyTorch
ships without it are practical:

1. **Overload explosion.** Each binary op needs `|G| × |G|`
   overloads for grad-mode alone. Add `|D| × |D|` for device,
   `|P| × |P|` for precision, and a top-level Tensor op needs
   ~64–256 overload rows. Idris collapses these to one signature
   via type-level functions (`Join`, type families). Python has no
   type families. PEP 696 (TypeVar defaults) doesn't help here.

2. **No higher-kinded types.** A polymorphic `join : G -> G -> G`
   over the grad-mode lattice can't be expressed in Python's type
   system. You inline it as overloads.

3. **`with`-block scoping doesn't narrow types.** `with no_grad():`
   in Python is a runtime context manager; pyright and mypy don't
   know that variables bound inside the block should be retyped
   from `Tensor[…, WithGrad]` to `Tensor[…, NoGrad]`. You'd need a
   different idiom — e.g. `ng = no_grad(); y = ng.lift(x).add(…)`
   where `ng.lift` returns a `Tensor[…, NoGrad]`. That's a real
   ergonomic regression.

4. **Backwards compatibility.** Adding type parameters to
   `torch.Tensor` is a 10+ year migration for the ecosystem. The
   library has 100k+ public-facing call sites in user code; any
   incremental rollout has to default the new params back to
   "anything", which defeats the point.

5. **`torchtyping` / PEP 646 already exist for shapes.** When
   PyTorch users want type safety, they reach for shape annotations
   first. Grad-mode and device statics are seen as a smaller win.

None of these are *type-system* obstacles. They're real but
contingent.

## Where dependent types are actually required

Three classes of guarantee that Python's type system can't reach:

1. **Shape arithmetic.** `matmul`, `conv2d` output dims, `reshape`
   `dims.prod` invariants, broadcast-shape unification. Python's
   `Literal[N]` doesn't support type-level addition or
   multiplication in mainstream checkers. PEP 646 `TypeVarTuple`
   handles *variadic shapes* but not arithmetic over them.

2. **Indexed values.** `Fin n` for in-bounds indices, `Vect n a`
   for length-indexed lists. Python falls back to `list[int]` with
   runtime bounds checks.

3. **Proofs as values.** "These two shapes are equal", "this tape
   has no requires-grad parameters", and similar evidence that ops
   can demand as a typed argument. Python has nothing here.

(1) is the load-bearing demo in idris-ml. (2) and (3) show up in
supporting machinery — `Fin` indices into a memory matrix, equality
proofs for `reshape`.

## Using grad-mode in practice

idris-ml ships two mechanisms that work together — one runtime, one
type-level:

**`withNoGrad : UserExecutorTraining ex => IO a -> IO a`** — runtime
block-scoped tape gating. Inside, tensor ops skip tape construction
(saves memory and allocation) and libtorch's `NoGradGuard` is active.
This is the direct analogue of PyTorch's `with torch.no_grad():` and
is what you want around inference / RL rollout / eval forward passes
for perf. The types of tensors created inside the block don't change.

**`weakenGrad : UserExecutorTraining ex => (1 _ : Tensor dims ex dt g) -> IO (Tensor dims ex dt NoGrad)`** —
a per-tensor type-level cast that also flips the C-side
`requires_grad` flag. After this, the tensor's *type* says `NoGrad`;
passing it to `trainStep` is a compile error. The `(1 _ : ...)`
quantity means the input is consumed linearly — the original
WithGrad-typed reference can't be used after the call (the runtime
flag has changed under it, so reuse would be a type lie).

The two are independent: `withNoGrad` is the perf knob, `weakenGrad`
is the static safety knob. The user-visible compile failure when you
accidentally feed a `NoGrad` loss back into training (from the
CI fixture `Test/neg/GateRejectsNoGrad.idr`):

```
Mismatch between: WithGrad and NoGrad.

  brokenStep opt = trainStep opt fakeNoGradLoss
                                 ^^^^^^^^^^^^^^^
```

PyTorch's equivalent of this mistake — computing a loss inside
`with torch.no_grad():` and then calling `loss.backward()` — fails
at runtime with `RuntimeError: element 0 of tensors does not require
grad and does not have a grad_fn`. Idris-ml fails at compile time
instead.

## Models as linear resources: eval, freeze, transfer learning

Grad-mode at the *model* level uses linear types. A model is a
**single-owner linear resource** threaded through `Control.Linear.LIO.L IO`;
`forward`, `eval`, `freeze` all *consume* the handle `(1 _ : …)` and
hand back a fresh one. This is what makes the PyTorch "freeze, then
keep training via the stale handle (silent no-op)" footgun a
compile-time error.

The grad-mode operations on models:

```idris
-- take the whole model out of training (genuinely tape-free inference):
eval      : (1 _ : l i o ex dt WithGrad) -> L IO {use=1} (l i o ex dt NoGrad)
trainable : (1 _ : l i o ex dt NoGrad)   -> L IO {use=1} (l i o ex dt WithGrad)  -- inverse

-- freeze a backbone INSIDE a trainable graph (grads still flow THROUGH
-- for downstream trainable layers — the fine-tune-backbone pattern):
freeze    : (1 _ : l i o ex dt g)            -> L IO {use=1} (Frozen (l i o ex dt g))
unfreeze  : (1 _ : Frozen (l i o ex dt g))   -> L IO {use=1} (l i o ex dt g)
```

`eval` flips every param's C-side `requires_grad` off and retypes the
model `WithGrad → NoGrad`. The optimizer can't accept the result (it
needs a `WithGrad` loss, which a `NoGrad` model can't produce), so
"eval a model then accidentally train it" doesn't typecheck. `freeze`
keeps the same grad-mode but wraps the model in `Frozen` (its field is
itself linear); the optimizer skips frozen params while gradients
still flow through to trainable layers downstream.

`forward` is grad-mode polymorphic, so an `eval`'d or frozen model is
fully usable for inference and the `NoGrad` propagates through the
output tensor automatically:

```idris
infer <- eval trained                           -- infer : Model … NoGrad
(MkBang pred # infer') <- forward infer batch   -- pred : Tensor [b,o] … NoGrad
discard infer'
-- trainStep opt (lossFrom pred)                -- ❌ COMPILE ERROR (NoGrad loss)
```

The linearity discipline is the part PyTorch (and Haskell today) can't
match. Reusing a consumed model handle is a compile error — from the
CI fixture `Test/neg/ReuseAfterFreeze.idr`:

```idris
badReuse m = do
  m1 <- eval m
  m2 <- eval m        -- ^ There are 2 uses of linear name m
  ...
```

### How this compares to PyTorch

PyTorch's equivalent is `for p in model.parameters(): p.requires_grad =
False`. The capability is the same; the safety guarantees differ:

| | PyTorch | idris-ml |
|---|---|---|
| Take model out of training | `requires_grad = False` loop | `eval` (retypes to `NoGrad`) |
| Freeze backbone, grads flow through | same loop on a subset | `freeze` → `Frozen`, or optimizer LR-0 group |
| Unfreeze / re-train | same with `True` | `unfreeze` / `trainable` |
| Compile-time rejection of "trained via the stale handle after freeze/eval" | no | **yes (linear types)** |
| Compile-time rejection of `NoGrad` loss in training | no | **yes** |

Tensors stay *unrestricted* (reverse-mode AD shares them freely) — the
linear discipline applies only at model granularity, exactly where the
aliasing footgun lives. Per-parameter / per-prefix freezing for
fancier fine-tuning (`freezeGroup`, optimizer LR-0 groups) composes on
top.

## Big picture

If you're coming from PyTorch, the user-visible difference is that
four properties — shape, device, grad-mode, precision — *could*
each be enforced by the compiler instead of checked at runtime. In
PyTorch only shape is checked, and only dynamically; in idris-ml
shape is checked statically, and grad-mode is checked statically
(device too, via the existing phantom parameter).

The one that *requires* dependent types is shape. The others come
along for free because the machinery is already there. That's the
honest distinction: dependent types aren't the only path to
grad-mode or device safety, but they are the only path to shape
safety, and they make uniform discipline across all four cheap
enough to be idiomatic.
