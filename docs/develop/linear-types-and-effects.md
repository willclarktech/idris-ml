# Linear types & effects: making stale model aliases a compile error

This is a design + learnings doc for the linear-resource model migration
(the `ModuleL`/`SeqL`/`evalL` surface). It is written to outlive the
migration: half of it is reusable knowledge for a future project that builds
its own linear-typed ML language, so it spells out the *why* behind each
mechanic, not just the recipe.

## The bug class we are killing

Freezing a model — or `eval`-ing it for inference — flips the C-side
`requires_grad` switch off on its *shared* parameter tensors and hands back a
wrapper. But in plain `IO` the original Idris handle is still in scope, still
typed trainable, and now **stale**. Reuse it to train and the optimizer
silently updates nothing (every param is frozen): a quiet, afternoon-wasting
no-op with no crash, no warning, no wrong number — just a loss that won't move.

This is the textbook "stale alias to a mutated shared resource" bug, and
linear types are the textbook tool: a linear value must be used **exactly
once**, so "consume the model (freeze/eval), then use the old handle again"
becomes unrepresentable. The fix matches the codebase ethos ("make invalid
programs unrepresentable" — the same principle behind indexing models by
`GradMode`).

## Idris-2 QTT in one screen

Idris 2 is built on Quantitative Type Theory: every binder carries a
**multiplicity** drawn from `{0, 1, ω}`.

- **`0`** — *erased*. The value exists only at type-check time; it is gone at
  runtime. Phantom indices (`{0 ex : Executor}`, `{0 dt : DType}`,
  `{0 g : GradMode}`) are all `0`.
- **`1`** — *linear*. Must be used **exactly once** along every code path.
- **`ω`** (written by omission) — *unrestricted*. Use zero or more times. This
  is the default for `let`/`<-`/lambda binders and ordinary record fields.

Two facts that are easy to get wrong and cost real time:

1. **A linear *argument* does not constrain an ω-bound *caller* variable.**
   If `freeze : (1 _ : Model) -> IO (Frozen Model)` and you call it as
   `f <- freeze model` where `model` was bound by `<-` (hence ω), nothing
   stops you using `model` again afterwards. The `(1 _)` annotation is a
   promise *freeze* makes about its own body, not a restriction it imposes on
   the caller. **Consequence:** a merely-`(1 _)`-annotated `freeze` in plain
   `IO` is *cosmetic* — the guard is real only when the model is linear *from
   creation* and threaded in a monad whose bind preserves linearity. The
   legacy `freezeNetwork` "linearity guard" was illusory for exactly this
   reason.

2. **Pattern-matching a linearly-bound record binds each field at the
   field's *constructor* multiplicity, not the scrutinee's.** If `m` is bound
   `(1 _ : Linear …)` and you write `forwardL (MkLinear w b) x = …`, then `w`
   and `b` are bound at **ω** (their declared field multiplicity), free to use
   in the matmul *and* to rebuild `MkLinear w b`. The whole-`m` linear
   obligation is discharged by the single match. **Precedent already in the
   tree:** `weakenGrad (MkTensor ptr pid) = do { primSetRequiresGrad ptr 0;
   pure (MkTensor ptr pid) }` uses `ptr` twice.
   - **Corollary — never `.field`-project a linear value.** Projection
     (`m.weightT`) consumes the scrutinee uncleanly; always pattern-match the
     constructor.

## Why plain `IO` can't do it, and `L IO` can

Plain `IO`'s `(>>=) : IO a -> (a -> IO b) -> IO b` binds its result at ω (the
continuation `a -> IO b` is an unrestricted function). So you literally cannot
thread a linear value through an `IO` do-block — the bind throws it away into
an ω context.

`Control.Linear.LIO` (the `linear` package, 0.8.0) provides `L io`, whose bind
is **quantity-polymorphic**: it can thread a `1`-use value from one statement
to the next. The surface we use:

- `L io {use : Quantity} a` — the monad. `use` is the multiplicity of the
  *result*; a linear-returning function must be typed `L io {use=1} a`, **not**
  the default `L io a` (which is `{use=Unrestricted}`). Getting this wrong
  yields "Mismatch between Linear and Unrestricted".
- `pure1 : (1 _ : a) -> L io {use=1} a` — linear `pure`.
- `liftIO1 : (1 _ : IO a) -> L io a` — run an ordinary `IO` action; **result
  is unrestricted**. This is the bridge that let the model surface land
  *before* `Tensor.idr` is converted: `forwardL` calls today's IO tensor ops
  via `liftIO1 (tlinear2d …)`. The seam drops when the tensor ops become
  `L IO` natively.
- `run : L io a -@ io a` — re-enter ordinary `io` at the top level
  (`main = run $ do …`). `-@` is the linear function arrow.

### `LPair` makes *both* components linear — hence the bang

`LPair a b` (written `a # b`, constructor `(#)`) is the linear pair: **both**
components are linear. So a `forwardL` that wants to return *(output tensor,
rebuilt model)* hits a wall — the output tensor is unrestricted (reverse-mode
AD shares it; see below), but `LPair`'s first slot demands linear.

The fix is the **bang modality** `(!*)` from `Data.Linear`, constructor
`MkBang`. `!* a` wraps an unrestricted `a` so it can ride through a linear
context; you unwrap by matching `MkBang x`. So every `forwardL` returns:

```idris
L IO {use=1} (LPair (!* (Tensor [b,o] ex dt g)) (l i o ex dt g))
--                    ^ unrestricted output, banged   ^ linear model
```

and the caller destructures `(MkBang y # m') <- forwardL m x`. This bang/
unbang ceremony is pervasive and is the main ergonomic tax of the design.

## Why tensors stay unrestricted (the principled boundary)

We deliberately keep **tensors `ω`** and make **only the model** linear. This
is not a compromise — it is where the tool fits:

- Reverse-mode autodiff **inherently shares** tensors. A forward activation is
  read by the next layer *and* captured by the backward tape; a value feeding
  two downstream ops is a multi-consumer DAG node. "Use exactly once" is
  simply false for tensors.
- The high-value *tensor* lifetime bugs (use-after-free across an mlx
  generation boundary, double-free) are **region / scope** problems, not
  single-owner problems. The right tool there is regions or borrowing, not
  strict linearity. Strict linear ≠ affine (≤1) ≠ uniqueness ≠ borrowing;
  conflating them leads to fighting the checker for no guarantee.
- A **model**, by contrast, *is* a single-owner resource: there is one current
  version, and freeze/eval/step produce the next version. Linearity is exactly
  right at that granularity.

For a future language: this argues for **per-kind multiplicity policy** —
tensors unrestricted (with a separate region discipline for buffers), models
linear, phantoms erased — rather than one global answer.

## The interface shape and why it is what it is

`ModuleL` (in `Nn/Module.idr`) has four model-consuming methods. A generic
function over an *abstract* layer `l` **cannot pattern-match** `l` (it doesn't
know the constructor), so anything polymorphic must go through a method:

- `forwardL : (1 _ : l …) -> Tensor → L IO {use=1} (LPair (!* Tensor) (l …))`
  — consume, return banged output + rebuilt model.
- `reflectL : (1 _ : l …) -> LPair (!* (List SomeParam)) (l …)` — expose the
  param list **without consuming** (returns the model alongside). This is what
  lets the *generic* `evalL`/`freezeL` flip C `requires_grad` over the params
  yet still hand the model back. Per-layer impl mirrors `castGrad`.
- `castGradL : (1 _ : l … g) -> l … g'` — the linear `castGrad` (phantom
  `g → g'` retype; pure).
- `discardL : (1 _ : l …) -> L IO ()` — the **explicit linear consumer**. A
  use-once value cannot fall out of scope, so any model not threaded onward
  (or returned) must be explicitly discarded. This is pervasive — every
  not-fully-consumed model, including the final trained one if it isn't
  returned.

Generic lifecycle ops are then `reflect → flip flags → rebuild`:

```idris
evalL m = do
  let (MkBang ps # m') = reflectL m
  traverse_ (\p => liftIO1 (primIO (primSetRequiresGrad p.paramPtr 0))) ps
  pure1 (castGradL m')
```

Two signature gotchas, both load-bearing:

- **`l` must be a *relevant* implicit** (`{l : Nat -> … -> Type}`, no `0`) in
  the generic functions — method dispatch needs it at runtime. Auto-binding it
  via `ModuleL l =>` alone leaves it erased and you get "l is not accessible
  in this context". Mirror the IO `eval`/`freeze` signatures.
- **`Frozen`'s ω field can't hold a linear model.** Putting a `1`-bound value
  into an ω field would license duplication, so the checker rejects it
  ("Trying to use linear name m' in non-linear context"). Hence `FrozenL` with
  a **linear** field for the `L IO` surface.

### Mixed field multiplicity by role

The decisive structural rule of the whole migration:

- **Leaf** layer param fields stay **ω** (reused in the matmul *and* the
  rebuild — see QTT fact 2).
- **Composite** fields that thread + re-pack sub-models (`SeqL`'s `(::)`,
  `ResidualL`'s sublayer, and later `AttentionL`/`TransformerBlockL`) must be
  **linear `(1 _)`**, so the rebuilt linear sub-models returned by `forwardL`
  are accepted.

Multiplicity is chosen **per field by its role**, not uniformly per type.

### The existential-under-linearity result

The make-or-break risk was threading a linear value through `SeqL`'s
existential `(::)` (v0.8's linearity checker is weakest under existential /
dictionary-dispatched matches). It **works**: `forwardSeqL (l :: rest) x`
destructures the banged output of `forwardL l x`, recurses on `rest`, and
re-packs `(l' :: rest')` with both fields linear. Proven first in a spike,
then on the real `Nn.Seq`/`Nn.Linear` types.

## Writing a gate test that fails for the *right* reason

The negative test (`Test/neg/ReuseAfterFreeze.idr`) must fail with a
**linearity** error, not an unrelated one. Two traps cost time:

- **Constraint resolution races inference.** `evalL m` with `m`'s type only
  partially pinned reports "Can't find implementation for `UserExecutorTraining
  ?ex`" / "Can't find implementation for `ModuleL ?l`" — the constraints are
  attempted before the argument pins `ex`/`l`. Fix: give the test function a
  **concrete, fully-pinned signature** (concrete `TapeExecutor`, `F64`, and a
  pinned *return type* so inference flows backward), e.g.
  `badReuse : (1 _ : Linear 2 3 TapeExecutor F64 WithGrad) -> L IO {use=1}
  (Linear 2 3 TapeExecutor F64 NoGrad)`. Then the *only* possible error is the
  double-use.
- **`do`-block bind ambiguity.** With both `Prelude.(>>=)` and
  `Control.Linear.LIO.(>>=)` in scope the elaborator can report "Ambiguous
  elaboration". In practice this was a *cascade* from an earlier "Undefined
  name `WithGrad`" (missing `import GradMode`); fixing the import resolved both.

With those, the gate reports:

```
Error: While processing right hand side of badReuse.
There are 2 uses of linear name m.
```

The v0.8 message is **either** "There are N uses of linear name …" **or** "…
is not accessible in this context" depending on the shape, so the gate greps
for `uses of linear name|not accessible in this context|linearly bounded`. The
positive companion (`Test/pos/SingleUseCompiles.idr`) must compile, proving the
negative fails for the linearity reason and not because the surface is broken.
Wired as `make test-integration-typegate-linear-model`.

## The File-API mental model

The pattern is exactly the standard linear file API: `openFile : … -> L IO
(Either … File)`, `fGetLine : (1 _ : File) -> L IO (… # File)`, `closeFile :
(1 _ : File) -> L IO ()`. Each operation **consumes and returns** the handle;
forgetting to thread it, or using a stale one, is a type error. A model is a
file: `forwardL`/`recurStep` are `fGetLine`, `freezeL`/`evalL` transform it,
`discardL` is `closeFile`.

## Notes for a from-scratch linear-typed ML language

- **Per-kind multiplicity policy** beats one global rule: erased phantoms,
  linear models, unrestricted tensors + a separate region discipline for
  device buffers.
- **The bang tax is real.** `LPair`-makes-both-linear forces banging every
  unrestricted payload that rides beside a linear one. A language designed for
  this should make "linear pair with one unrestricted side" a first-class,
  syntax-light thing (or infer the bang).
- **Linearity wants to be born at construction**, not bolted on at a seam — an
  after-the-fact `(1 _)` annotation on an ω value guarantees nothing (QTT fact
  1). Resource-producing constructors should return linear by default.
- **Generic-over-abstract-layer code can't match**, so the interface must
  expose `reflect`-style "observe without consuming" methods. Plan for this in
  the interface vocabulary from day one.
- **Pattern-match, never project**, linear values — design the surface so
  projection of a linear field is simply unavailable.

## Migrating the op surface bottom-up (the `L IO` Tensor ops)

The autograd op surface (`tadd`/`tlinear`/`ttanh`/… in `Tensor.idr`) is the
root of the dependency DAG: ~125 files call these ops in `IO` do-blocks.
Retyping them in place from `IO (Tensor …)` to `L IO (Tensor …)` would break
all 125 at once — incompatible with build-green-per-commit. So the migration
is **additive**: an `L IO` op surface lands beside the `IO` one, consumers
move onto it bottom-up (layers → fit → examples → transformers), and the `IO`
ops are deleted last (rename `*L` → base). Standard shape for a monad change
at the bottom of the stack.

- **One lifting primitive.** `ioRerunL : (() -> a) -> L IO a`, defined
  `ioRerunL f = liftIO1 (ioRerun f)`. Every `L IO` op is `ioRerunL (\_ =>
  MkTensor (prim… ) Nothing)` — the *same* pure FFI-thunk body as its `IO`
  twin, only the wrapper differs. The `%foreign` prims stay `PrimIO`/`IO`; the
  single lift is centralized in `ioRerunL`, so no `liftIO1` is scattered in
  layer code. Tensors are **unrestricted**, so the ops return `L IO` at the
  default `use = Unrestricted` (a tensor binds with `<-` and is freely reused
  — only the *model* is linear).
- **Import the type qualified, the lift unqualified.** `L` lives in
  `Control.Linear.LIO` but `liftIO1` is in `Prelude.IO`. In a file full of
  `IO` do-blocks (`Tensor.idr`), `import Control.Linear.LIO as LIO` and write
  `LIO.L IO τ` for the type — a qualified import keeps LIO's names out of the
  unqualified namespace so it can't make the existing `IO` `>>=`/`pure`
  ambiguous. `liftIO1` is then just `Prelude.IO.liftIO1`, unqualified.

### The `the (L IO τ)` pin (a real elaboration trap)

`Applicative (L io)` exists (`Applicative io => Applicative (L io)`), but a
bare `pure x`, or a `case`/`if-then-else` whose branches feed a `<-` bind,
fails with **"Can't find an implementation for Applicative (L IO)"**. The
cause: the `L` type's `use` parameter is an unsolved metavariable during that
sub-elaboration, so `Applicative (L IO {use = ?u})` can't match the
`use = Unrestricted` instance head. Pin the monad explicitly:

```idris
p <- the (L IO (TVec o ex dt WithGrad)) $ case prev of
       Just po => pure po               -- now resolves: use = Unrestricted
       Nothing => tzeroState1dL {n = o}
```

Needed wherever a branch is a *value lifted with `pure`* (the `Just`/identity
arms of recurrent state init and eval-mode dropout) and wherever a `case`
result is the subject of a do-bind (the `Activation` kind dispatch). A branch
that is itself an op call (`ttanhL x`) needs no pin — its return type is
already concrete `L IO τ`.

## Status / file map

- `packages/idris-ml/src/Nn/Module.idr` — `ModuleL`, `FrozenL`, generic
  `evalL`/`freezeL`/`unfreezeL`/`trainableL`.
- `packages/idris-ml/src/Nn/Linear.idr` — leaf exemplar (`ModuleL Linear`).
- `packages/idris-ml/src/Nn/{Activation,Dropout,LayerNorm}.idr` — leaf Modules.
- `packages/idris-ml/src/Nn/SeqL.idr` — list composite (existential threading).
- `packages/idris-ml/src/Nn/Residual.idr` — `ResidualL`, one-sublayer composite.
- `packages/idris-ml/src/Test/{neg/ReuseAfterFreeze,pos/SingleUseCompiles}.idr`
  + `scripts/check-linear-model-gate.sh` — the gate.

- `packages/idris-ml/src/Tensor.idr` — the additive `L IO` op surface
  (`ioRerunL` + `taddL`/`tlinearL`/`tlinear2dL`/`tzeroState1dL`/
  `tlstmGatesPairL`/`tgruCellL` + the six activation twins). The `IO` ops are
  unchanged beside them.
- `packages/idris-ml/src/Nn/{Recurrent,Lstm,Gru}.idr` — `RecurrentL` bodies on
  the `L IO` ops.

Coexists with the IO `Module`/`Params`/`Seq`/`Frozen` surface. No `forwardL`/
`recurStepL` body still uses `liftIO1` for tensor math — the only remaining
lifts are principled: `Module.idr`'s `evalL`/`freeze`/`unfreeze` flip C
`requires_grad` via `liftIO1 (primIO …)` (a param-flag side effect, not a
tensor op), and `Rnn` lifts its user-supplied IO activation field. The IO op
surface is deleted once every caller (layers → fit → examples → transformers)
is on `L IO`; that collapse renames `*L` → base.
