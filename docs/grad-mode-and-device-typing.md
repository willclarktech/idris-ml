# Type-level grad-mode and device — what does it actually require?

## TL;DR

Both **grad-mode** and **device restriction** are closed-sum phantom
enumerations. Encoding them at the type level does **not** strictly
require dependent types. Python's static type system (`Generic[…]` +
`Literal[…]` + `@overload`) is, in principle, expressive enough to
reject `loss.backward()` after `with torch.no_grad():` and to reject
mixing CPU and CUDA tensors.

The reason PyTorch doesn't do this isn't a type-system limitation —
it's an ergonomics + ecosystem choice (overload explosion, no HKT, no
type families, `with`-block scoping doesn't narrow variable types in
mainstream Python checkers).

What **does** require dependent types in idris-ml is **shape
arithmetic**: `matmul : Tensor [m, k] -> Tensor [k, n] -> Tensor
[m, n]`, `Conv2D` output-dim formulas, length-indexed vectors / `Fin
n` bounds. Python's `Literal[N]` can't do arithmetic on type-level
naturals.

So the right framing for the library pitch:

- **Shape safety** — load-bearing dependent-types demo (Python can't).
- **Grad-mode safety**, **device safety**, **precision safety** —
  *idiomatic in Idris because the same machinery handles them
  uniformly*, but expressible in any sufficiently rich phantom-type
  system. Python could in principle; PyTorch didn't.

This doesn't weaken the idris-ml story — it sharpens it. Two of the
four type-parameter ideas (`(dims, device, grad_mode, precision)`)
ride on dependent types' *uniformity* rather than their *power*.

## What's a "phantom-type enum" vs. a "dependent type"?

**Phantom-type enum**: a type parameter whose value is drawn from a
fixed, finite set known at the type system level. The parameter
appears in types but isn't used in the runtime representation. The
classic example is `Maybe a` parameterised over a finite set of
states (`Open`, `Closed`) for a file handle.

```idris
data State = Open | Closed
record Handle (s : State) where
  ptr : AnyPtr

openFile  : String -> Handle Open
readFile  : Handle Open -> IO String     -- compile error on Closed
closeFile : Handle Open -> Handle Closed
```

This works in Haskell with `DataKinds`. It works in Rust with
zero-sized phantom types (`PhantomData<State>`). It works in TypeScript
with branded types. **It works in Python** with `Generic[State]` +
`Literal["open"] | Literal["closed"]`, modulo overload tedium.

**Dependent type**: a type parameter that depends on a *value* drawn
from an open, potentially-infinite set, *and* on which the type
system can compute (proofs, arithmetic, etc.).

```idris
matmul : Tensor [m, k] d -> Tensor [k, n] d -> Tensor [m, n] d
```

Here `m`, `k`, `n` are arbitrary `Nat`s and the type system has to
unify `k` across the two arguments. This is **not** a phantom enum —
the set of possible `Nat`s is infinite, and the type checker computes
on them (e.g. `Conv2D : Tensor [_, _, w, _] -> Tensor [_, _,
(w - k) `div` s + 1, _]`).

The litmus test: can you write down all the possible type-level
parameter values on a single page? If yes, phantom enum. If no
(naturals, lists, trees, …), dependent type.

## idris-ml's four `Tensor` type parameters, by this lens

Current and proposed:

| Parameter | Status | Phantom-enum? | Dependent? |
|---|---|---|---|
| `dims : Vect rank Nat` | shipped | no — infinite | **yes** |
| `d : Device` | shipped | **yes** — `CPU \| CUDA n \| MPS` * | mostly no |
| `g : GradMode` | proposed | **yes** — `WithGrad \| NoGrad` | no |
| `p : Precision` | proposed | **yes** — `F32 \| F64 \| BF16 \| F16` | no |

\* `CUDA n` carries a `Nat` device index, which is technically
value-indexed. But practically you target at most 8 devices, so this
is `Fin 8` in disguise — a finite enumeration, not arithmetic. Drop
the `Nat` for `CUDA0 \| CUDA1 \| …` and `Device` is a flat
4–12-element enum.

So out of the four, only `dims` is genuinely dependent. The rest are
phantom enums.

## What this looks like in Python

The bit the user asked about. Here are sketches showing each is
*expressible*, then the practical sticking points.

### Grad-mode in Python

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
# Calling backward(t) where t: Tensor[..., NoGrad] is now a type error.
```

This compiles in pyright and rejects `backward(no_grad_tensor)`. The
join across `(WithGrad, NoGrad)` is hand-encoded as overload rows
instead of a type-level function, but it works.

### Device in Python

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

`Tensor[…, "cpu"] + Tensor[…, "cuda:0"]` is now a type error. Again,
this is just phantom-type discipline; no dependent types.

## Why PyTorch doesn't do this

The capability is there in the type system. The reasons PyTorch
ships untyped (in this sense) are practical:

1. **Overload explosion.** Each binary op needs `|G| × |G|` overloads
   for grad-mode alone. Add `|D| × |D|` for device, `|P| × |P|` for
   precision, and a top-level Tensor op needs ~64–256 overload rows.
   Idris collapses these to one signature via type-level functions
   (`Join`, type families). Python has no type families. PEP 696
   (TypeVar defaults) doesn't help here.

2. **No higher-kinded types.** A polymorphic `join : G -> G -> G`
   over the grad-mode lattice can't be expressed in Python's type
   system. You inline it as overloads.

3. **`with`-block scoping doesn't narrow types.** `with no_grad():`
   in Python is a runtime context manager; pyright and mypy don't
   know that variables bound inside the block should be retyped from
   `Tensor[…, WithGrad]` to `Tensor[…, NoGrad]`. You'd need a
   different idiom — e.g. `ng = no_grad(); y = ng.lift(x).add(…)`
   where `ng.lift` returns a `Tensor[…, NoGrad]`. That's a real
   ergonomic regression for end users.

4. **Backwards compatibility.** Adding two type parameters to
   `torch.Tensor` is a 10+ year migration for the ecosystem. The
   library has 100k+ public-facing call sites in user code; any
   incremental rollout has to default the new params back to
   "anything", which defeats the point.

5. **`torchtyping` / PEP 646 already exist for shapes.** When PyTorch
   users want type safety, they reach for shape annotations first.
   Grad-mode + device statics are seen as a smaller win.

None of these are *type-system* obstacles. They're real but
contingent.

## Where dependent types actually buy us something Python can't have

Three classes of guarantee:

1. **Shape arithmetic.** `matmul`, `conv2d` output dims, `reshape`
   `dims.prod` invariants, broadcast-shape unification. Python's
   `Literal[N]` doesn't support type-level addition or multiplication
   in mainstream checkers. PEP 646 `TypeVarTuple` handles
   *variadic shapes* but not arithmetic over them.

2. **Indexed values.** `Fin n` for in-bounds indices, `Vect n a` for
   length-indexed lists. Python has no equivalent — you fall back to
   `list[int]` with runtime bounds checks.

3. **Proofs as values.** "These two shapes are equal", "this tape
   has no requires-grad parameters", etc., as first-class evidence
   that ops can demand. Python has nothing here.

In idris-ml, (1) is the load-bearing demo. (2) and (3) are present
but secondary.

## How the four-parameter Tensor pitch should be framed

Today: `Tensor (dims : Vect rank Nat) (0 d : Device)`.
After all four TODOs: `Tensor (dims : Vect rank Nat) (0 d : Device)
(0 g : GradMode) (0 p : Precision)`.

The pitch isn't "you need dependent types to track grad-mode" —
that's false, Python could. The honest pitch is:

> **Dependent types make uniform type-system discipline cheap.** Once
> the compiler is computing on shapes, it's free to also compute on
> grad-mode, device, and precision. The same `Join` machinery handles
> all four. In Python you'd need overload tables, branded literals,
> and giving up `with`-block scoping. In Idris it's four type
> parameters with no extra machinery — and the runtime tag fields
> (`requires_grad`, `device_str`, `dtype`) become statically redundant.

This is also the right way to motivate the device-as-interface TODO
(open the closed `Device` sum into a typeclass for user-supplied
backends). That one *does* lean harder on dependent types — "what
operations are available depends on which device value you pick" is
a textbook value-determines-type story — but it's value-determines-
*interface-instance*, which Haskell could also do with type classes.
The dependent-types value is still mostly uniformity, not raw power.

## Bottom line

Grad-mode and device tracking at the type level are
**medium-effort uniformity wins**, not proof-of-concept
dependent-types showcases. They're worth doing because:

- the runtime `requires_grad` flag and the runtime `device` string
  become statically redundant;
- accidental cross-device / post-no-grad bugs become unrepresentable;
- the API is uniform with the shape parameter that's already there.

But they shouldn't be sold as "this is what dependent types unlock."
The shape-safety story is the load-bearing one. Keep that distinct.
