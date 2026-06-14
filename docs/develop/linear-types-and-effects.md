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
- **Composite** fields that thread + re-pack sub-models *via a linear
  `forwardL`* (`SeqL`'s `(::)`, `ResidualL`'s sublayer) must be **linear
  `(1 _)`**, so the rebuilt linear sub-models returned by `forwardL` are
  accepted.
- **Read-only composites** (`TransformerBlock`, the `Ntm`/`Dnc` controllers)
  keep their sub-model fields **ω** and delegate the forward/step to IO at the
  linear boundary (see "consume-match-rebuild-delegate" above) — they never
  re-pack a *linear* sub-result, so no linear field is needed.

Multiplicity is chosen **per field by its role**, not uniformly per type.

### The existential-under-linearity result

The make-or-break risk was threading a linear value through `SeqL`'s
existential `(::)` (v0.8's linearity checker is weakest under existential /
dictionary-dispatched matches). It **works**: `forwardSeqL (l :: rest) x`
destructures the banged output of `forwardL l x`, recurses on `rest`, and
re-packs `(l' :: rest')` with both fields linear. Proven first in a spike,
then on the real `Nn.Seq`/`Nn.Linear` types.

### Consume-match-rebuild-delegate (the composite recurrent pattern)

A leaf's `forwardL`/`recurStepL` can be written as a native `L IO` body (mirror
the IO body, swap the wrapper for `ioRerunL`, return `MkBang y # rebuilt`). But
the big composite recurrent cells (`Ntm`, `Dnc`) have large, raw-prim-heavy
bodies that thread an LSTM controller **stored in an ω record field**. Two facts
collide:

- The controller field is **ω** (the IO `Recurrent`/`Params` instances project
  it ~15×, so it can't be made linear while the IO surface still coexists).
- A `forwardL`/`recurStepL` on the controller via the *linear* `recurStepL`
  returns a `1`-bound updated controller — which **cannot go back into the ω
  field** (the `Frozen`-ω rule above).

So for now these delegate at the linear boundary:

```idris
recurStepL (MkNtm ctrl …state…) input = do
  (updSt, out) <- liftIO1 (recurStep (MkNtm ctrl …state…) input)  -- IO step, ω
  pure1 (MkBang out # updSt)
```

The **pattern-match discharges the scrutinee's linearity** (fields bind at ω);
we rebuild an ω cell, run the IO `recurStep` (which threads the controller in ω
internally), and return the fresh ω cell as the linear component of the pair.
The handle-level guarantee — *you cannot reuse a stale cell after a step* —
holds; only the internal threading stays ω. The inline `L IO` body (with a
linear controller field) lands at the IO-surface collapse.

**Trap that forces this shape:** you cannot pass the linear scrutinee `st`
*directly* to the IO step — `recurStep st input` errors with "Trying to use
linear name st in non-linear context", because `recurStep`'s argument is ω and
applying an ω function **scales the argument's usage by ω** (1 ≠ ω). You must
match-and-rebuild first. (This is the *dual* of the `Frozen`-ω rule: there a
linear value can't *enter* an ω field; here a linear value can't *feed* an ω
parameter.) The same applies to `recurResetL` / any delegation to an ω-arg IO
function. `TransformerBlock`'s `forwardL` uses the identical shape — its forward
is read-only on every sub-layer, so it delegates the multi-step pre-norm body to
the IO `forward` and rebuilds the unchanged block.

### Kind-mismatched layers get plain-function linear analogues

`Attention` carries three config Nats (`dModel`/`numHeads`/`headDim`), so it
doesn't fit the `(i, o, …)` 2-Nat kind of `Params`/`Module`; the IO surface
exposes it as plain functions (`attentionParams`/`attentionCastGrad`/
`attentionForward`) that the enclosing `TransformerBlock` splices into *its*
interface impls. The linear surface mirrors that exactly: `attentionReflectL` /
`attentionCastGradL` / `attentionDiscardL` / `attentionForwardL` are plain
linear functions (same signatures as the interface methods would have, minus the
dispatch), and the block's `ParamsL`/`ModuleL` splice them over the ω-bound
`attn` field. Parameter-free free functions (`PosEncoding`, `RoPE`) need no
linear surface at all — they hold no model resource.

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

## Migrating an example to the linear surface (the recipe)

Phase 6 turns each example's `main` into a linear pipeline. The mechanic is
identical across examples; the recipe (verified on `Supervised` + `Rnn`):

1. **Imports.** Add `import Control.Linear.LIO` (`L`, `run`, `pure1`,
   `liftIO1`) and `import Data.Linear.Notation` (`MkBang`, `!*`) and
   `import FitL` (`fitL`/`fitSupervisedL`/`fitCustomL`). `LPair`/`#` are
   `Builtin`. Never `import Data.Linear` (Copies — see above). The examples
   build with `-p linear` (added to `IDRIS_FLAGS`).
2. **Fully-qualify `Control.Linear.LIO.run`.** `import System` (getArgs)
   brings `System.run` / `System.Escaped.run`; an unqualified `run $ do …`
   makes the whole block ambiguous and blows the elaborator's depth-3
   ambiguity limit (the symptom is *"Maximum ambiguity depth exceeded"* with
   a `System.run --> (>>) --> (>>=)` trace). Qualifying pins it.
3. **Lift inline `runInitL $ do {…}` to a top-level `Init` value.** A nested
   `do`-block under the linear `run $ do …` also trips the ambiguity limit;
   move the model derivation to a top-level `mkModel : Init Model` and call
   `runInitL mkModel`.
4. **The loss / step function goes linear.** For `fitSupervisedL`, the loss
   consumes the model, runs `forwardL`, returns `MkBang loss # model'`. For a
   custom `fitL` step over a **user record** model that's read many times per
   epoch (recurrent/RL), use **consume-match-rebuild-delegate**: match the
   record (fields bind ω), reuse them freely inside an IO body via `liftIO1`,
   rebuild the record beside the banged result. (Same pattern as Ntm/Dnc.)
5. **Consume the final model.** `fitL`/`fitSupervisedL` return
   `LPair (!* (Nat, Double)) m`; unwrap `(MkBang (epochs, loss) # trained)`,
   then `evalL`/`forwardL` it for inference (each consumes + returns), and
   discard the last handle — `discardL` for a `ParamsL` layer, or a 3-line
   `discardModel (MkRec _ _) = pure ()` for a user record (the handles are
   C-managed; dropping the matched ω fields is a no-op discharge).
6. **`Seq` models become `SeqL`.** The linear sequence is a *distinct type*
   `Nn.SeqL` (linear `::` fields), not `Nn.Seq` — and both export
   `Nil`/`(::)`/`(~~>)`. So a Seq-based example sets `Model = SeqL …`, builds
   the chain with the same `~~>`/`Nil` syntax, and `%hide`s the three
   `Nn.Seq.{Nil,(::),(~~>)}` constructors so the builder resolves to `SeqL`.
   `forwardSeqL` / `discardL` (`ParamsL SeqL`) then apply. (`Nn.SeqL` is
   re-exported from the `Nn` umbrella alongside `Nn.Seq`.)
7. **Loss/eval ops not yet on the `L IO` surface** get an `*L` twin added to
   `Tensor.idr` as needed (e.g. `tnllLossMeanL`), same `ioRerunL (\_ => …)`
   shape as the rest.
7. **Mixed-precision paths stay on IO.** `ModuleMixed`/`ParamsMixed` have no
   linear surface yet (the `*MixedL` follow-up), so a mixed branch keeps using
   the IO `forwardMixed`/`fitSupervisedMixed`. `main : IO` calls `run` for the
   linear branch and plain IO for the mixed one — they coexist.

## Born-linear construction (`runInitL`)

A model must enter the linear world *linear* — if construction handed back an
ω value, the single-owner discipline would only start at the first `forwardL`,
and nothing would force the caller to thread it. So the construction seam
confers linearity at its exit: `runInitL : Init a -> L IO {use = 1} a`, the
`L IO` twin of `runInit`. The `Init` name-derivation monad is unchanged
internally (still plain `IO` state-threading); `runInitL` lifts the whole
derivation once with `liftIO1` and re-emits through `pure1`, so the result is
born at `use = 1`. `use = 1` (not the default `Unrestricted`) is the load-bearing
bit — it makes the bound model linear at the `<-` site, so the caller *must*
thread it onward. (Same `{use = 1}` discipline as every `forwardL`.)

## Threading the model through training (the fit loop)

The fine-grained guarantee is only real if the model stays linear *through
training*, not just at a single `forwardL`. The training stack threads it end
to end:

- **`EpochStepL m batch = (1 _ : m) -> batch -> L IO {use=1} (LPair (!* Double) m)`**
  — a step consumes the model and returns it beside the banged loss. The
  recursive batch pass (`runPassL`) and epoch loop (`runEpochLoopL` /
  `epochLoopGoL`) thread the linear `m` through every recursive call; the
  result carries the model out (`LPair (!* (Nat, Double)) m`). This compiled
  first try — the v0.8 checker handles a linear var threaded through deep
  recursion + `if`/`case` branches (each branch uses `m'` exactly once) just
  as it did the `Seq` existential. The fit-threading risk gate is **passed**.
- **Metrics never touch the model (risk #3 dissolved).** Every real metrics
  callback ignores its model argument (`\_ => readRLMetrics …`; default
  `const (pure [])`) — metrics read C-registry / IORef state. So the linear
  loop uses a **model-free** `MetricsFnL = IO (List (String,String))` and never
  has to hand the linear model to metrics (which it *couldn't*, since the engine
  is generic in `m` with no reflect method). `TrainConfig` grows a `metricsL`
  field beside `metrics`; the plan's feared "metrics peeking at the linear `m`"
  was a non-problem once we checked what callers actually do.
- **Only the model is linear; everything else is lifted.** `DataStream.next`
  stays `IO` (data is not a single-owner resource) and is lifted with
  `liftIO1`. The optimizer/checkpoint step fns (`nativeTrainStep`,
  `trainStepScaled`, `applyScale`, `tick`, `postEpoch`, …) touch only the C
  registry, never the model *value*, so they stay `IO` and are lifted at the
  call site — no linear threading there. Their `L IO` recast (decision 1's
  "whole library on `L IO`") buys no extra guarantee and is deferred to the
  Phase-9 collapse.

### The `Data.Linear` import perturbs `Nat`-literal defaulting (a real trap)

Importing `Data.Linear` into a file with delicate `Nat` arithmetic *breaks
previously-compiling code*:

- **`Copies.Nil` shadows `[]`.** `Data.Linear` re-exports `Data.Linear.Copies`,
  whose `Nil` makes the `[]` in scalar `Tensor []` (the loss type) ambiguous
  (`Copies` vs `Vect` vs `List`). Fix: `%hide Data.Linear.Copies.Nil` **and**
  ensure `import Data.Vect` is present (so `Vect.Nil` is in scope to win against
  `List.Nil` under the expected `Vect rank Nat`).
- **Bare numeric literals re-default to `Integer`.** With those modules in
  scope, `if improved then 0 else stale + 1` and `accCount + 1` (plain `Nat`
  before) start failing with *"Mismatch between Integer and Nat"* — an Idris
  elaboration-order fragility, not a new `Num` instance (the linear modules
  define none). Dodge it by avoiding bare literals in `Nat` arithmetic: `S n`
  instead of `n + 1`, `Z`/`case` instead of `== 0`, `the Nat 1` where unavoidable.

Because the existing `Train.Engine`/`Fit` are dense with such arithmetic, the
linear loop and driver live in **sibling modules** (`Train.EngineL`, `FitL`)
that import the linear stack, reusing every model-agnostic piece from their IO
twins (early-stop machines, checkpoint resume/keep-best, `shouldLog`,
`isDiverged`, `fmtMetrics`). The IO files stay linear-import-free and unbroken.
This is the same additive-then-collapse shape used for the layers (`*L` beside,
merge at Phase 9).

## Status / file map

- `packages/idris-ml/src/Nn/Module.idr` — `ModuleL`, `FrozenL`, generic
  `evalL`/`freezeL`/`unfreezeL`/`trainableL`.
- `packages/idris-ml/src/Nn/Linear.idr` — leaf exemplar (`ModuleL Linear`).
- `packages/idris-ml/src/Nn/{Activation,Dropout,LayerNorm}.idr` — leaf Modules.
- `packages/idris-ml/src/Nn/{Conv,Pool}.idr` — batched 4-D Modules (`ParamsL` +
  `ModuleL`, inline `ioRerunL` bodies).
- `packages/idris-ml/src/Nn/{BatchNorm,BitLinear,Embedding,LoraLinear,SwiGLU,RmsNorm}.idr`
  — leaf `ParamsL`-only layers (1-D forwards, not batched Modules).
- `packages/idris-ml/src/Nn/{Ntm,Dnc}.idr` — composite recurrent (`ParamsL` +
  `RecurrentL`, consume-match-rebuild-delegate).
- `packages/idris-ml/src/Nn/{Attention,Transformer}.idr` — the composites
  (Attention's plain linear fns spliced by `TransformerBlock`'s `ParamsL`/
  `ModuleL`).
- `packages/idris-ml/src/Nn/SeqL.idr` — list composite (existential threading).
- `packages/idris-ml/src/Nn/Residual.idr` — `ResidualL`, one-sublayer composite.
- `packages/idris-ml/src/Test/{neg/ReuseAfterFreeze,pos/SingleUseCompiles}.idr`
  + `scripts/check-linear-model-gate.sh` — the gate.

Not yet on the linear surface: `LinearMixed` (the mixed-precision
`ModuleMixed`/`ParamsMixed` family — a distinct kind with the `computeDt` slot;
a `*MixedL` surface is a follow-up). Parameter-free `PosEncoding`/`RoPE` need
none.

- `packages/idris-ml/src/Tensor.idr` — the additive `L IO` op surface
  (`ioRerunL` + `taddL`/`tlinearL`/`tlinear2dL`/`tzeroState1dL`/
  `tlstmGatesPairL`/`tgruCellL` + the six activation twins). The `IO` ops are
  unchanged beside them.
- `packages/idris-ml/src/Nn/{Recurrent,Lstm,Gru}.idr` — `RecurrentL` bodies on
  the `L IO` ops (the small recurrent cells; `Ntm`/`Dnc` delegate instead).
- `packages/idris-ml/src/Nn/Init.idr` — `runInitL` (born-linear construction
  seam) beside `runInit`.
- `packages/idris-ml/src/Train/Engine.idr` — `MetricsFnL` (model-free) added;
  `fmtMetrics`/`forceMetrics` exported for the linear loop to reuse.
- `packages/idris-ml/src/Train/EngineL.idr` — the `L IO` epoch loop
  (`runEpochLoopL`/`epochLoopGoL`/`withEpochL`/`logEpochL`/`divergedL`); sibling
  to `Train.Engine` (linear-import isolation).
- `packages/idris-ml/src/FitL.idr` — the `L IO` fit driver
  (`fitSupervisedL`/`fitSupervisedMixedL`/`fitL`/`fitCustomL`/`runPassL`,
  `EpochStepL`); sibling to `Fit.idr`. Hides `Copies.Nil`; `Z`/`S` accumulators.
- `packages/idris-ml/src/Train.idr` — `TrainConfig` gains a `metricsL` field
  (model-free) beside `metrics`.
- `packages/idris-ml-examples/src/Example/{Supervised,Rnn}.idr` — first two
  example families on the linear surface (supervised `fitSupervisedL`; custom
  record + `fitL` recurrent step). The recipe above; the rest of Phase 6 is
  mechanical. `mk/config.mk` adds `-p linear` to the example `IDRIS_FLAGS`.

Coexists with the IO `Module`/`Params`/`Seq`/`Frozen` surface. No `forwardL`/
`recurStepL` body still uses `liftIO1` for tensor math — the only remaining
lifts are principled: `Module.idr`'s `evalL`/`freeze`/`unfreeze` flip C
`requires_grad` via `liftIO1 (primIO …)` (a param-flag side effect, not a
tensor op), and `Rnn` lifts its user-supplied IO activation field. The IO op
surface is deleted once every caller (layers → fit → examples → transformers)
is on `L IO`; that collapse renames `*L` → base.
