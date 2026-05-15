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
matmul : Tensor [m, k] d -> Tensor [k, n] d -> Tensor [m, n] d
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
Tensor (dims : Vect rank Nat) (0 d : Device) (0 g : GradMode)
```

- `dims` is genuinely dependent — `Nat`-valued, arithmetic happens
  during type checking. This is where `matmul`-shape-safety and
  conv-output-dim safety come from.
- `0 d : Device` is the device tag. `Device` is a 0-quantity alias
  for `Type`, so any type with a `UserDeviceCore` instance can be
  used. The built-ins `CPU`, `CUDA n`, `MPS` are types whose
  instances forward to the active C backend; users can declare
  their own device type and instance (see "Custom devices" below).
- `0 g : GradMode` with `GradMode = WithGrad | NoGrad` is a real
  closed enum — the compiler rejects `backward` on tensors that
  weren't tracked.

The device parameter started life as a closed sum
(`Device = CPU | CUDA Nat | MPS`) and got opened up to admit
user-supplied backends — see "Custom devices" below. A precision
parameter `p : Precision` with `Precision = F32 | F64 | BF16 | F16`
is a natural next extension in the same style; it's a phantom-type
enum that reuses the existing parameter machinery.

## Custom devices: user-supplied backends

`Device` is an open kind — any type with a `UserDeviceCore` instance
can sit in `Tensor`'s `d` slot.

A worked example ships in the repo: `packages/backends/backend_byo.c`
is a ~100-line stub backend that exports `byo_tensor_add`,
`byo_tensor_item`, etc. and logs each call to stderr;
`packages/idris-ml-examples/src/Example/BringYourOwn.idr` is the
Idris-side recipe that wraps it. Run it with `make
example-bring-your-own` — you'll see the stderr `[byo] ...` lines
fire as ops dispatch through your instance, alongside the same
expression evaluated on the built-in `CPU` instance for contrast.

A complete custom backend looks like:

```idris
module MyBackend

import Device.Core

-- 1. Declare a type to tag tensors that live on this backend.
public export
data MyDev : Type where MD : MyDev

-- 2. Bind your dylib's C symbols via %foreign.
%foreign "C:my_tensor_add,libmybackend"
prim__addMine : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:my_tensor_create_scalar,libmybackend"
prim__scalarMine : Double -> Int -> AnyPtr

%foreign "C:my_tensor_item,libmybackend"
prim__itemMine : AnyPtr -> Double
-- ... 17 more for the full lifecycle + arithmetic slice

-- 3. Implement the UserDeviceCore instance.
public export
UserDeviceCore MyDev where
  deviceName       = "mybackend"
  primCreateScalar = prim__scalarMine
  primAdd          = prim__addMine
  primItem         = prim__itemMine
  -- ... all UserDeviceCore methods
```

Now `Tensor [4] MyDev` is a valid type, every op (`tadd`, `tmul`,
`forwardVar` …) dispatches to your `MyDev` instance, and you can
transfer between built-in and user-supplied devices via `toDevice`.

### Parameterized devices

Your device type can carry type parameters. CUDA's device index is
the canonical example:

```idris
data CUDA : Nat -> Type where MkCUDA : (n : Nat) -> CUDA n
```

`Tensor [4] (CUDA 0)` and `Tensor [4] (CUDA 1)` are different types;
the compiler will reject mixing them. To declare the instance, bind
the `Nat` parameter at the head of the instance declaration:

```idris
{n : Nat} -> UserDeviceCore (CUDA n) where
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

There's one tripwire. `UserDeviceCore`'s `d` parameter is declared
0-quantity (`interface UserDeviceCore (0 d : Device)`), and that
0-quantity propagates: a *caller* of `deviceName` working with
`d = CUDA n` has `n` only at the type level. The instance body has
`n` at runtime (because the instance head binds it non-erased), but
generic library code that only sees `UserDeviceCore d` doesn't.

If you need a separate operation that recovers the parameter, add a
helper interface:

```idris
public export
interface HasDeviceIndex (d : Device) where
  deviceIndex : Nat

public export
{n : Nat} -> HasDeviceIndex (CUDA n) where
  deviceIndex = n
```

`HasDeviceIndex`'s `d` parameter is *unrestricted* (no `0`), so its
methods may observe the parameter. The built-in `CUDA n` ships
with this instance, and `deviceName` for `CUDA n` uses
`show n` directly (the instance head binds `n` non-erased, the same
trick).

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

**`withNoGrad : IO a -> IO a`** — runtime block-scoped tape gating.
Inside, tensor ops skip tape construction (saves memory and
allocation) and libtorch's `NoGradGuard` is active. This is the
direct analogue of PyTorch's `with torch.no_grad():` and is what you
want around inference / RL rollout / eval forward passes for perf.
The types of tensors created inside the block don't change.

**`weakenGrad : (1 _ : Tensor dims d g) -> IO (Tensor dims d NoGrad)`** —
type-level cast that also flips the C-side `requires_grad` flag.
After this, the tensor's *type* says `NoGrad`; passing it to
`runBackward` or `nativeTrainStep` is a compile error. The mechanism
is per-tensor (not block-scoped), so it survives across `IO`
boundaries. The `(1 _ : ...)` quantity means the input is consumed
linearly — the original WithGrad-typed reference can't be used
after the call (the runtime flag has changed under it, so reuse
would be a type lie).

The two are independent: `withNoGrad` is the perf knob,
`weakenGrad` is the static safety knob. Use both for the strongest
guarantee, or either alone where one fits the situation:

```idris
-- Inference: combine for runtime perf + static promise.
result <- withNoGrad $ do
  let (_, pred) = forwardVar net input
  predNG <- weakenGrad pred              -- predNG : Tensor [o] d NoGrad
  let probs = tsoftmax1d predNG          -- still NoGrad
  pure (tensorItem (telemSelect probs 0))

-- nativeTrainStep optimizer predNG       -- ❌ COMPILE ERROR

-- Eval that only reads scalars: runtime gating is enough. The
-- forward output stays WithGrad-typed but never flows to backward.
acc <- withNoGrad (pure (computeAccuracy trained testBatch))
```

The user-visible compile failure when you accidentally feed a
`NoGrad` loss back into training:

```
Mismatch between: NoGrad and WithGrad.

  brokenStep opt = nativeTrainStep opt fakeNoGradLoss
                                       ^^^^^^^^^^^^^^^
```

PyTorch's equivalent of this mistake — computing a loss inside
`with torch.no_grad():` and then calling `loss.backward()` — fails
at runtime with `RuntimeError: element 0 of tensors does not require
grad and does not have a grad_fn`. Idris-ml fails at compile time
instead.

## Freezing networks for transfer learning

For the "load pretrained backbone, train only the new head" workflow,
idris-ml provides two more linear operations on Networks:

**`freezeNetwork : (1 _ : Network i hs o d g) -> IO (Network i hs o d NoGrad)`** —
walks every parameter in the network, flips its C-side
`requires_grad` to false, and retypes the result as `NoGrad`. Frozen
params don't get updated by `optimizer.step()` (their gradient
buffers stay at zero) and the type system prevents accidentally
training the network end-to-end.

**`unfreezeNetwork : (1 _ : Network i hs o d NoGrad) -> IO (Network i hs o d WithGrad)`** —
the inverse. Used for *progressive fine-tuning*: train head first
with backbone frozen, then unfreeze the backbone for joint
fine-tuning at a lower learning rate.

```idris
-- Transfer learning sketch:
backbone <- buildPretrained ...                          -- WithGrad
backboneFrozen <- freezeNetwork backbone                 -- NoGrad
head     <- buildHead ...                                -- WithGrad
-- ... compose backboneFrozen and head into a combined network,
-- train only head's params (optimizer skips frozen ones)
...

-- Later: unfreeze for joint fine-tuning.
backbone'    <- unfreezeNetwork backboneFrozen           -- WithGrad
-- backboneFrozen is consumed; you can only use `backbone'` now.
```

Both ops are linear in their input. After calling `freezeNetwork
backbone`, the name `backbone` is consumed and any further use is
a compile error. This closes the aliasing footgun ("freeze the
network, then accidentally train via the original variable, which
silently no-ops because the C-side flags are flipped") that PyTorch
users have to remember to avoid.

### `forwardVar` works on frozen networks

The forward functions are polymorphic in the grad-mode parameter,
so a frozen network is fully usable for inference:

```idris
let (_, pred) = forwardVar backboneFrozen input
--                          ^^^^^^^^^^^^^^^
--                          Network ... NoGrad
-- pred : Tensor [o] d NoGrad — type-tracked through the forward
```

The `NoGrad` propagates naturally through the result, and `pred`
can't be fed to `nativeTrainStep` (Phase 4 gate). No need to
`weakenGrad` after the call.

### How this compares to PyTorch

PyTorch's equivalent is `for p in model.parameters(): p.requires_grad =
False`. The capability is the same; the safety guarantees differ:

| | PyTorch | idris-ml |
|---|---|---|
| Whole-model freeze | `for p in ...: p.requires_grad = False` | `freezeNetwork` |
| Unfreeze | same with `True` | `unfreezeNetwork` |
| Compile-time rejection of "trained via the original ref after freeze" | no | yes (linear types) |
| Compile-time rejection of `NoGrad` loss in training | no | yes (Phase 4 gate) |
| Per-parameter freeze (mixed trainable/frozen within one module) | yes | not yet — whole-network only |

idris-ml matches PyTorch's freeze/unfreeze ergonomics and adds two
compile-time safety nets PyTorch users live without. Per-parameter
freezing (for fancier fine-tuning strategies) is filed as future
work.

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
