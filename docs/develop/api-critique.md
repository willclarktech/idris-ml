# API Critique — the user-facing surface of `packages/idris-ml`

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

**Date**: 2026-06-11 · **Commit**: `de5dc612` · **Status**: findings recorded; refactors not yet started.
**Decision record**: the north-star items were walked through with the user 2026-06-11 and three were
adjusted (§N3 — no `Num`-on-IO/`share`, operator aliases on plain tensors instead; §N5 — `scoped` +
`groupOf`, leaf naming gated on an elab-reflection derivation spike (the initial "Idris can't reflect
record field names" rejection was disproven by probe 2026-06-11); §N6 — engine pieces exported
alongside `fit`).
[`roadmap.md`](roadmap.md) "Decisions taken" is authoritative where it and this doc differ.

**Scope**: publish-readiness critique of what an `import Tensor` / `import Layer` / `import Train` user
sees, ahead of package-manager publication. There are no users yet — **backwards compatibility is a
non-goal**; every verdict assumes APIs can break freely. The yardstick is twofold: (a) what tensor-library
users expect (PyTorch/JAX ergonomics), (b) what a from-scratch dependently-typed design would look like
(the [North star](#north-star-the-target-v1-api) section).

**Boundaries vs related backlog rows**:
- *Layer/module naming rework* — pure rename candidates (`*LayerAny`, `Network` → `Sequential`, …) are
  delegated to that row; this doc judges whether the **composition abstraction** is right at all (§S2).
- *PyTorch design survey* — this doc does targeted per-surface PyTorch comparisons (loss dual API,
  DataLoader trio, optimizer param groups, `.to(device)`); it does not survey hooks/profiler/compile.
  Items covered here can be struck from that row's list.
- *DRY-up idris-transformers* — that row is about sharing implementation; §S8 here is about the
  user-facing package shape.

Constraint rationales live in [`design-decisions.md`](design-decisions.md); this doc cites them by ID
from the [constraint ledger](#constraint-ledger) and never re-litigates them.

---

## Method

- **Evidence standard**: every verdict cites call-site counts (grep commands reproducible from repo
  root, `-w` word-boundary matching, occurrence counts not line counts), one canonical exhibit
  (`file:line`), and a paired-PyTorch comparison where a pair exists. Counts are split
  **lib** (`packages/idris-ml/src`) vs **user** (`idris-ml-examples`, `idris-transformers`,
  `idris-gym`, `idris-ml-notebook` sources) — the *user* column is what matters for API judgment.
- **Verdict vocabulary**: `rename` / `merge` / `split` / `remove` / `add-facade` / `rationale stands`.
- **Buckets**: **[A]** fixable API debt · **[B]** inherent to Idris 2 today · **[C]** load-bearing
  design (must cite a constraint ID or design-decisions.md section).
- **Scoring**: Severity ∈ {blocks-publish, embarrassing, cosmetic} × Effort ∈ {S, M, L, XL}.
- **Discipline**: *a constraint excuses only what it forces.* C1 forces losses to be `IO`-typed; it
  does not excuse `Math.idr` duplicating every loss over `Array`.

### Correcting the folklore numbers

Two numbers circulating in earlier discussion don't survive measurement, and the critique is built on
the corrected ones:

1. **"12 epoch variants"** → `Backprop.idr` exports **8**: `epochVar`, `epochVarMixed`,
   `epochVarTensor`, `epochVarTensorMixed`, `epochVarTensorBatch`, `epochVarTensorBatchMixed`,
   `epochRecurrentVar`, `epochTwoPhaseVar`. The Recurrent/TwoPhase × Tensor/Mixed crosses were never
   written — itself evidence that the cross-product design doesn't scale (§S5).
2. **"Examples are ~3× their PyTorch refs"** → per-pair, raw line counts are near parity:

   | Pair | Idris | PyTorch (models/ + scripts/) | Ratio |
   |---|---|---|---|
   | Mnist | `Example/Mnist.idr` 329 | `mnist_cnn.py` 137 + `mnist.py` 136 = 273 | 1.2× |
   | Dqn | `Example/Dqn.idr` 466 | `dqn.py` 333 + `dqn.py` 143 = 476 | 1.0× |
   | Supervised | `Example/Supervised.idr` 237 | 67 + 104 = 171 | 1.4× |
   | Gpt | `Example/Gpt.idr` 454 | 343 + 243 = 586 | 0.8× |

   The verbosity problem is **concentrated, not uniform**: PyTorch spends its lines on docstrings,
   argparse, and convergence harnesses; Idris spends its lines on *ceremony* — data marshalling,
   handle wrapping, IO threading, shape-constant declarations. The per-construct ratios are where it
   bites: a DQN batched TD loss is ~37 lines of Idris (`Dqn.idr:152-188`) vs ~3 lines of PyTorch
   (`q.gather(1, a)` + `F.mse_loss`). Verdicts below target those concentrations, because that is
   what a reader evaluating the library from its examples will actually notice.

---

## Constraint ledger

| ID | Constraint | Source | What it forces |
|---|---|---|---|
| **C1** | Tensor ops are `IO`-typed: FFI must fire on `<-` sequencing so `withNoGrad` brackets correctly under strict evaluation | design-decisions.md (forward-pass IO typing); `Layer/Core.idr` comments | Ops return `IO (Tensor …)`. Does **not** force the absence of operators/combinators over those IO values. |
| **C2** | Wrapped-handle Chez ABI: `tensorPtr` is a Chez vector; retain/release via guardian; `KeepAlive` exists for values escaping `withNoGrad` | design-decisions.md "Tensor lifecycle"; `docs/develop/tensor-lifecycle.md` | The lifecycle machinery exists. Does **not** force it to appear in example-level user code. |
| **C3** | Idris 2 Peano `Nat` hangs the elaborator on multiplicative shape literals and dims ≳ 2300 | gotchas.md "Large Nat type-level reduction" | `TVec`/`TMat` aliases; factored shapes. Does **not** force users to hand-name every intermediate product. |
| **C4** | Open `Device`/`DType` kinds (`Device = Type` + instances) are the headline polymorphism story | design-decisions.md "Open `d` parameter", "Open `dt`" | Per-(ex,dt) constraints exist on every signature. Does **not** force *six separate* constraints per signature. |
| **C5** | Gradient flow requires param registration under a stable string name in the C-side registry | CLAUDE.md "paramId is required for gradient flow" | Names must exist and be stable. Does **not** force users to invent them by hand. |
| **B1** | No `vmap`/broadcasting tracer — batching a computation means writing the batched form | inherent (no tracing infra in Idris 2) | Single-sample and batched code paths diverge unless signatures are batched-first. |
| **B2** | Shape-on-the-value means *some* implicit/constraint overhead is the price of admission | inherent | Signatures will always be longer than Python's. The question is how much longer (§S9). |

---

## S1 · Tensor construction + data marshalling

**Current shape.** Getting host data into a `Tensor` goes through a wall of per-rank, per-init,
per-purpose names: `tinput1d`, `bulkToTensor`, `bulkToTensor2d`, `dtCreate1d/2d`, `tparam1dConst`,
`tparam1dNormal`, `tparam2dNormal`, plus the raw buffer route `prim__allocDoubles` →
`prim__setDouble` → wrap. Several names the original critique row lists (`tinputN`, `treadVec`,
`tstate1d/2d`) **do not exist** — the row itself had drifted from the surface, which says something
about the surface's memorability.

**Evidence** (`grep -rwo <sym> packages/... | wc -l`):

| Symbol | lib | user | Note |
|---|---|---|---|
| `bulkToTensor` / `bulkToTensor2d` | 20 / 4 | 42 / 20 | the workhorse |
| `tinput1d` | 12 | 14 | |
| `tparam1dConst` / `tparam2dNormal` / `tparam1dNormal` | 43 / 16 / 5 | 17 / 12 / 0 | |
| `dtCreate1d` / `dtCreate2d` | 11 / 4 | 5 / 0 | |
| `packDoubleBuf` | 5 | 7 | hand-rolled, then copied between examples |
| `prim__allocDoubles` | 28 | **17** | raw FFI in *user* code |
| `tvecToVector` | 4 | 0 | defined in `Backprop.idr` of all places |

Canonical exhibit: `Example/Dqn.idr:173-174` — the user builds a batch by calling `bulkToTensor2d`,
then **hand-wraps the raw pointer**: `MkTensor obsBT Nothing`. The constructor + a raw `AnyPtr` + the
magic `Nothing` paramId is the library's internal ABI leaking straight into example code. PyTorch
equivalent: `torch.tensor(batch)`.

**Comparison.** PyTorch/JAX have exactly two entry points users remember: `torch.tensor(data)` and
factory functions (`zeros`, `randn`, …) — rank is inferred from the data, init is the function name,
param-ness is a wrapper (`nn.Parameter`). Three orthogonal axes, three orthogonal API elements.
idris-ml currently multiplies the axes into the *name*.

**Constraints.** C1 applies (construction is IO) — fine. C3 caps full rank-polymorphism in places
(shape `Vect` literals are fine; it's *arithmetic over* dims that hangs). C5 explains `tparam*` taking
name strings, not the per-rank × per-init naming.

**Verdict — merge (add-facade + demote).** [A] · Severity: blocks-publish · Effort: M.
Target shape (north star §N2): one `fromVect : Vect (prod dims) Double -> IO (Tensor dims ex dt NoGrad)`-style
constructor family with an `Init` *value* (`Zeros | Const x | Normal μ σ | FromVect xs`) carrying the
initialization, rank-polymorphic where C3 allows, per-rank workers kept `private` behind it. All
`prim__alloc*`/`pack*Buf` marshalling moves behind the facade; `MkTensor` stops being the documented
way to get a batch tensor.

**Tradeoffs.**
- The per-rank workers don't disappear (C3 makes truly rank-generic packing hard); they just stop
  being the *public* surface. Cost: a thin dispatch layer.
- Do-nothing cost: the very first thing every new user writes (load my data) is the worst part of the
  API, and examples teach `MkTensor`-wrapping as normal.

**Follow-up row**: yes — clustered with §S10 export hygiene (same module surgery).

---

## S2 · Model composition: `Network` / `OutputLayer` / `AnyLayer` / `LayerLikeMixed`

**Current shape.** A model is a `Network i hs o ex dt g` — a typed chain of existentially-wrapped
layers (`AnyLayer` via `MkAnyLayer`), built with `(~~>)` and terminated by `OutputLayer`. Every layer
ships dual constructors (`linearLayer` + `linearLayerAny`). Mixed precision duplicates the whole
tree: `LayerLikeMixed`, `NetworkMixed`, `forwardVarMixed`. Batched forward is an interface method
with a default body of `idris_crash`.

**Evidence:**

| Symbol | lib | user |
|---|---|---|
| `~~>` | 29 | 62 |
| `OutputLayer` | 34 | 40 |
| `*LayerAny` ctors in examples | — | ~100 (44 `linearLayerAny`, 16 `reluLayerAny`, 10 `tanhLayerAny`, 9 `ntmLayerAny`, …) |
| `MkAnyLayer` | 68 | 0 |
| `LayerLike` / `LayerLikeMixed` | 90 / 23 | 0 / 0 |
| `forwardVar` / `forwardVarBatch` | 57 / 11 | 57 / 23 |

Canonical exhibits: `Example/Mnist.idr:42-112` — the **71-line shape-constant block**
(`Conv1OutH`…`AfterPool2`) exists because `Network` is indexed by *flattened vector widths*, so every
conv/pool boundary needs its `C*(H*W)` product pre-named (the per-axis helpers `ConvOutDim`/`PoolOutDim`
already exist in `Layer/Conv.idr:16-22` — the flattening is what forces the naming, not missing
type-level arithmetic). `Layer/Core.idr` — `applyVarBatch` defaulting to `idris_crash` means a
recurrent layer in a batched driver fails *at runtime*, in a library whose pitch is compile-time
shape safety. This is the single deepest inelegance in the codebase.

**Comparison.** PyTorch's actual lesson is not `nn.Sequential` — it's that *a model is a function
plus a parameter container*, and the chain combinator is a convenience for the linear case only.
idris-ml inverted this: the chain is the primary abstraction, and every non-chain architecture
(actor-critic, two-tower, residual-heavy) falls off it and hand-rolls records anyway (see
`Sac.idr`, `Ppo.idr`).

**Constraints.** None force this shape. C1 forces `IO` in forward; C5 forces params to be registered.
Neither forces existential wrapping, the `hs : List Nat` index, the `OutputLayer` terminator, or a
crash-defaulted method.

**Verdict — split + remove.** [A] · Severity: blocks-publish (the crash hole) / embarrassing (the
rest) · Effort: XL (the largest single refactor in this doc).
Target shape (north star §N4): models are plain records of layers + a forward function; a `Params`
interface (params/freeze/unfreeze traversal) replaces `LayerLike` for the operations that must be
generic; a small heterogeneous `Seq i o ex dt` (indexed by endpoints only, hidden dims existential)
replaces `Network`/`OutputLayer`/`AnyLayer` for the chain case; batched-first signatures (B1) make
`applyVar`-vs-`applyVarBatch` one function and delete the `idris_crash` default — layers that can't
batch simply don't offer the batched type. `LayerLikeMixed` dies with the interface; precision
becomes driver config (§S5).

**Tradeoffs.**
- Records + functions lose the "swap a layer by editing one chain line" demo. The `Seq` convenience
  keeps it for tutorials.
- `Params` needs per-record instances until generic deriving lands — 3-line boilerplate per model
  record, far cheaper than today's per-layer dual constructors.
- Do-nothing cost: `idris_crash` in the flagship abstraction, ~100 `*Any` incantations in examples,
  and a type index (`hs`) that leaks architecture into every helper signature
  (`Example/Mnist.idr:140-211` repeats the full 7-arg `Network` type six times).

**Follow-up rows**: yes — (a) crash-hole fix folded into the §S5 driver row (cheap, urgent);
(b) records+`Params`+`Seq` as its own row. Renames delegated to the layer-naming row.

---

## S3 · Losses: function vs class, the IO chain, and the `Math.idr` double

**Current shape.** Tensor-side losses are functions — `tmseLoss`, `tnllLoss`, `tbceLoss` — fed to
drivers as `LossFn`. Meanwhile `Math.idr` exports a *second*, `Array`-based pure surface:
`meanSquaredError`, `binaryCrossEntropy`, `crossEntropy`, `nllLoss`, `l1Loss`, `huberLoss`,
`klDivLoss`, `klDivLossLog` (`Math.idr:76-135`), plus activations and matrix helpers.

**Evidence.** Tensor losses are used: `tbceLoss` 27, `tnllLoss` 20, `tmseLoss` 8 user call sites.
The `Math.idr` loss surface has **zero example call sites** (`crossEntropy` user = 0; the others
similarly — they serve only `idris-gym` internals and pre-Tensor code paths). Two parallel loss
vocabularies, one of them dead weight in the public import.

The sharper pain is *composing* a custom loss. Canonical exhibit `Example/Dqn.idr:152-188`: the
batched TD loss is 37 lines — `trowSelect`/`telemSelect`/`tconstScalar`/`tsub`/`tmul` chained
through `<-`, a hand-written `go` recursion over rows, and a `meanScalarLoss` that reaches for
**`primAdd` + `MkTensor` directly** because folding `IO` actions felt worse. PyTorch:
`F.mse_loss(q.gather(1, a).squeeze(), target)`.

**Comparison.** PyTorch's dual API (`nn.MSELoss` class + `F.mse_loss` function) is widely regarded
as redundant; JAX/optax settled on functions. idris-ml already picked functions — correct, keep it.

**Constraints.** C1 forces `IO`-typed losses — bucket [C], rationale stands for the *type*. C1 does
**not** excuse the absence of expression-level combinators (§N3) or the `Math.idr` double.

**Verdict — keep functions (rationale stands, C1); remove the double; add-facade for expressions.**
[A] for the double and ergonomics · Severity: embarrassing · Effort: S (Math demotion) + M (TensorM
combinators, shared with §N3).
Demote `Math.idr`'s loss/activation exports to an internal namespace (or delete where `idris-gym`
doesn't need them); one loss vocabulary, Tensor-typed, batched-first, targets typed `NoGrad`. Add
gather-style indexing (`tgather` over a row-index tensor) so the §S3 exhibit's 37 lines become ~4.

**Tradeoffs.** `idris-gym` uses a few `Math.idr` helpers — those move with it or stay non-exported.
Do-nothing cost: a published library where `crossEntropy` (the name every user reaches for first)
resolves to the dead Array surface instead of the real one.

**Follow-up row**: yes — Math demotion is its own small row; expression combinators land with §N3.

---

## S4 · Optimizers: `native*` proliferation + stringly-typed scopes

**Current shape.** Five constructors — `nativeSgd`, `nativeRmsprop`, `nativeAdamGlobalClip`,
`nativeAdamGroup`, `nativeAdamW` — differing in which *options* (clip mode, param-group filter,
weight decay) are baked into the *name*. Param groups and per-param LR are string-prefix matched:
`nativeAdamGroup "actor_" …`, `setParamLR opt "ll0_weights" lr`.

**Evidence:** user call sites — `nativeSgd` 14, `nativeRmsprop` 11, `nativeAdamGroup` 8,
`nativeAdamGlobalClip` 6, `nativeAdamW` 6, `setParamLR` 1, `setLearningRate` 0 (user).
Multi-network RL examples each carry the prefix discipline by hand (`actor_`/`critic_`/`q1_`/…,
see CLAUDE.md gotcha "ParamId scoping for multi-network examples" — a documented *bug class*).

**Comparison.** PyTorch: three things — an optimizer class, a `lr`/options dict, and param groups as
*data* (`[{params: actor.parameters(), lr: …}]`). Param groups reference parameter *objects*, not
name strings, so a typo is impossible. JAX/optax: composable gradient transformations.

**Constraints.** C5 forces names to exist in the registry. It does not force *users* to type them:
with auto-generated scoped names (§N5) and a `Params`-traversal (§S2), groups can be expressed as
`groupOf actorModel` rather than `"actor_"`.

**Verdict — merge.** [A] · Severity: embarrassing · Effort: M.
Drop the `native` prefix (there is no other optimizer; the prefix advertises an implementation
detail). Collapse to `sgd lr` / `adam cfg` / `adamW cfg` with an options record (`clip : Maybe
ClipSpec`, `weightDecay`, `groups : List (Scope, Overrides)`); `Group`/`GlobalClip` become fields.
Schedules attach to the optimizer (`withSchedule`), not via `TrainConfig.beforeEpoch` hooks.

**Tradeoffs.** The C ABI underneath (one handle per optimizer kind) doesn't change — this is purely
an Idris-side facade, low risk. Do-nothing cost: five names where one concept lives, and a silent
gradient-leak bug class (string prefix typo) shipping as the documented multi-network pattern.

**Follow-up row**: yes — clustered with typed scopes (depends on §N5 naming).

---

## S5 · Training drivers: 8 `epoch*` variants × 2 runners

**Current shape.** `Backprop.idr` exports 8 epoch functions = a cross-product of data representation
{`DataPoint`, `TensorDataPoint`, batched-`TensorDataPoint`} × precision {single, mixed} + two
task shapes {recurrent, two-phase}. Above them, `Train.idr` has `runTraining` / `runTrainingIO`.
RL examples bypass the epoch layer entirely and hand-roll loops inside `runTrainingIO`.

**Evidence** (user call sites): `epochTwoPhaseVar` 22, `epochVarTensorBatch` 14, `epochVar` 10,
`epochRecurrentVar` 5, `epochVarMixed` 1, and **three variants with zero users**: `epochVarTensor`,
`epochVarTensorMixed`, `epochVarTensorBatchMixed`. The Recurrent/TwoPhase × Tensor/Mixed crosses
were never even implemented. A 16-cell matrix with 8 cells filled, 3 of those unused — the
cross-product is failing on both axes (unneeded cells exist; needed cells don't).

Second exhibit: every RL example writes its episode loop twice — single-env and batched
(`Example/Dqn.idr` `runEpisode` + `runEpisodeBatched`) — a direct cost of B1 plus non-batched-first
layer signatures (§S2).

**Comparison.** PyTorch has *zero* epoch functions — users write the two-line loop, and the
ecosystem's repeated reinvention of trainers (Lightning, HF Trainer, fastai) shows both that the
loop wants absorbing *and* that the absorbed version must take a user **step function**, not
enumerate task shapes. Recurrent vs two-phase vs supervised is the *step's* business.

**Constraints.** C1 applies inside the step. Nothing forces the cross-product: precision is config
(GradScaler wiring is mechanical), data representation unifies if samples are tensors from the start
(§S7), and task shape belongs to the user-supplied step.

**Verdict — merge.** [A] · Severity: blocks-publish (it's the first API page any user reads) ·
Effort: L.
Target (north star §N6): one driver — `fit : Step m batch ex dt -> Optimizer ex -> Stream batch ->
FitConfig -> m -> IO (FitResult m)` where `Step m batch ex dt = m -> batch -> IO (Tensor [] ex dt
WithGrad)`. `runTrainingIO`'s engine (NaN detection, early stop, checkpointing, mlx generation
hygiene) survives as `fit`'s internals; the `Backprop.idr` layer above it is deleted. Mixed
precision = `FitConfig` field. The three zero-user variants can be deleted *today* with no
migration at all.

**Tradeoffs.** The 22 `epochTwoPhaseVar` call sites (NTM/DNC family) move their encode/decode fold
into their step functions — mechanical but touches many examples. Do-nothing cost: the API page
that should read `fit model data` instead reads like a flag matrix, and three exported functions
are provably dead.

**Follow-up row**: yes — includes the §S2 `idris_crash` batched-forward hole (the driver decides
what "batched" means; fix them together).

---

## S6 · Checkpointing

**Current shape.** Save/load: `saveModel`, `loadModel`, `loadModelAllowCast`, `loadModelPrefix`,
`loadModelPrefixAllowCast` — Boolean-blind variants returning `IO Bool` with detail on stderr; all
operate on the *global* C-side registry, not a model value. Loop integration: `CheckpointPolicy` +
`fileCheckpoint` + `withCheckpoint` (recently landed, well-shaped).

**Evidence** (user): `loadModelAllowCast` 23, `loadModel` 16, `loadModelPrefix` 6,
`loadModelPrefixAllowCast` 3, `saveModel` 5, `withCheckpoint`/`fileCheckpoint` 7/7. Note the most
used variant is the *cast-permissive* one — the "safe" default lost.

**Comparison.** PyTorch: `model.state_dict()` / `load_state_dict(strict=…)` — options as arguments,
errors as exceptions, scoped to a model object. safetensors-only as the format is a *good* idris-ml
decision (design-decisions.md "Training-loop checkpointing" — rationale stands).

**Constraints.** C5 (registry) explains the global-registry implementation; with `Params` (§S2)
save/load can scope to a model's own params. Nothing forces 4 load names or `IO Bool`.

**Verdict — merge + re-type.** [A] · Severity: embarrassing · Effort: S–M.
`save : Params m => m -> String -> IO ()`; `load : Params m => m -> String -> LoadOpts -> IO
(Either LoadError ())` with `LoadOpts = { allowCast : Bool, only : Maybe Prefix, remap : String ->
String }`. `CheckpointPolicy`/`withCheckpoint`: **rationale stands** (cite: design-decisions.md
checkpointing section) — keep as is.

**Tradeoffs.** Global-registry save is also what makes "save everything, no model in scope" work in
notebooks; keep one explicitly-named escape hatch (`saveAll`). Do-nothing cost: `IO Bool` error
reporting in a typed library, published.

**Follow-up row**: yes (small).

---

## S7 · Data: `DataPoint` / `TensorDataPoint` / `RecurrentDataPoint` / `TwoPhaseDataPoint` / `DataLoader` / `Sampler`

**Current shape.** Four sample types (boxed-`Double` `DataPoint`, pre-tensored `TensorDataPoint`
— which stores raw `AnyPtr`s and *loses the shape index it claims*, `DataPoint.idr:49-53` — plus
recurrent and two-phase task shapes), two loader factories, a `Sampler` module.

**Evidence** (user): `DataPoint` 34, `TensorDataPoint` 20, `TwoPhaseDataPoint` 15,
`RecurrentDataPoint` 8, `Sampler` 6, `DataLoader` **1**, `mkGeneratorLoader` 0, `mkIndexedLoader` 1.
The loader abstraction is essentially unused — examples marshal by hand (§S1) and pass `Vect`s.

**Comparison.** PyTorch carves at `Dataset` (indexed access) / `Sampler` (order) / `DataLoader`
(batching+collation) — three orthogonal joints. idris-ml's split is instead by *element
representation* and *task shape*, neither of which is a data-loading concern.

**Constraints.** The boxed/tensored split exists because samples aren't tensors from the start;
it alone forces half the §S5 matrix. Nothing load-bearing here.

**Verdict — merge (redesign).** [A] · Severity: embarrassing · Effort: L (coupled to §S5).
Target (north star §N7): samples are ordinary (tuples of) tensors; `Dataset sample = { size : Nat,
item : Fin size -> IO sample }` + `Stream` combinators (`shuffle`, `batched`, `generate` for
synthetic tasks). `RecurrentDataPoint`/`TwoPhaseDataPoint` become plain sample types owned by their
examples' step functions, not library exports. The Fisher–Yates C shuffle and idx readers survive
as engines behind `Dataset.mnist`-style provided datasets.

**Tradeoffs.** `Fin size` indexing is the dependent-types win PyTorch can't have — no OOB at
runtime; cheap to provide, good for the pitch. Do-nothing cost: four exported sample types whose
distinction users must learn before their first training run, plus a loader API nobody — including
our own examples — uses.

**Follow-up row**: yes — same workstream as §S5 (they only make sense together).

---

## S8 · HF adapter package (`idris-transformers`)

**Current shape.** One package, one ipkg (not per-arch — the critique row's premise is stale), one
module per architecture (`HfBert` 35K, `HfBitNet` 43K, `HfLlama` 36K, `HfGpt2` 24K) + support
modules (`HfCommon`, `Tokenizer`, `KVCache`, `HfDataset`, `HfLoraIO`). Examples import modules
directly (`import HfBert`, 5 sites; `import HfLlama`, 1; …).

**Comparison.** This mirrors HF transformers' own one-module-per-arch layout, deliberately
("the module IS the adapter, expressed as type-checked code" — CLAUDE.md). That argument is sound.

**Constraints.** C5 + safetensors naming make per-arch typed records the honest design. Bucket [C]
for the *split*; [A] for the *naming*.

**Verdict — split: rationale stands; rename.** Severity: cosmetic · Effort: S (mechanical rename).
The per-arch module split stays. Two naming-level changes for publication: (1) `Hf*` prefix →
namespace (`Transformers.Bert`, `Transformers.Llama`) — flat `import HfBert` into the global
namespace is un-Idris-like and the `Hf` prefix describes provenance, not function; (2) loading
should converge on a `fromPretrained : String -> IO (Either LoadError (cfg ** Model cfg ex dt))`
shape — the dependent pair is the honest type for "shapes determined by a file at runtime" and is
this package's best showcase of why Idris, not its appendix. Code-sharing questions stay with the
DRY row.

**Follow-up row**: yes (rename + fromPretrained shape), small, coordinated with the DRY row.

---

## S9 · Signature noise: implicit args + constraint saturation

**Current shape.** The canonical exported signature is
`{0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex
=> Compatible ex dt => IsFloating dt => …` — six constraints before the first real argument
(`Backprop.idr:114` and dozens of siblings).

**Evidence:** occurrence counts — `UserExecutorTraining` 274 lib / 113 user, `Compatible` 229/87,
`RuntimeDType` 183/71, `Linked` 164/84, `UserExecutorCore` 101/53. Users *also* pay: examples write
the full prefix on every helper they define (`Example/Mnist.idr:140-211`), and 43 of 51
`withNoGrad` calls in examples need an explicit `{ex=ExampleExecutor}`.

**Comparison.** Nothing comparable exists in PyTorch (B2 — this is partly the price of
shape-on-the-value). The fair comparison is Haskell's `ConstraintKinds`: bundle once, write one.

**Constraints.** B2 says *some* overhead is inherent. C4 says the open kinds must survive. Neither
forces six separate constraints: Idris 2 supports constraint-bundling interfaces with a blanket
implementation.

**Verdict — merge (bundle).** [A] · Severity: embarrassing (it's every line of the published docs) ·
Effort: M (mechanical but wide).
`interface (UserExecutorTraining ex, UserExecutorCore ex, RuntimeDType dt, Linked ex, Compatible ex
dt) => Backend ex dt where {}` + one blanket implementation; signatures become `Backend ex dt => …`.
Tiered bundles (`BackendCore`, `Backend`, `BackendStreamed`) mirror the existing executor-interface
tiers. BYO-backend users implement the same underlying interfaces as today (C4 intact) and get the
bundle for free. The `withNoGrad {ex=…}` noise is a separate inference gap — worth a targeted look
at making `ex` solvable from the bracket body (likely fixable; if not, document the idiom once).

**Tradeoffs.** Idris error messages will sometimes name the bundle instead of the missing leaf —
acceptable; the leaf appears in the "can't find instance" chain. Do-nothing cost: every signature in
the published API reference leads with 120 characters of plumbing.

**Follow-up row**: yes — and it should land **first**; it's mechanical, touches everything, and
every other refactor's diffs get smaller after it.

---

## S10 · Cross-cutting hygiene: naming conventions, export boundary, lifecycle leakage

**Current shape & evidence.**
- **Three competing prefixes** on the same conceptual surface: `t*` (user ops), `dt*`
  (dtype-dispatched), `prim*`/`prim__*` (FFI). All exported; nothing marks which is public.
  `Tensor.idr` alone: 117 `export` + 53 `public export`, including 16 `export`-ed `prim__*`.
- **`prim__*` leaks into examples 97 times** (`prim__setDouble` 14, `prim__allocDoubles` 11,
  `prim__allocInts`/`prim__setInt` 9 each, a long `prim__*BYO` tail). Examples are the de-facto
  docs; they currently teach raw FFI as the normal idiom.
- **Lifecycle ceremony**: `withNoGrad` 51× in examples (43 with explicit `{ex=}`); `KeepAlive`
  appears in example code (1×) at all, which is one time more than a lifecycle ABI detail should.
- **The dead `Num (Tensor …)` instance**: elementwise `(*)` exists but real code can't use it
  (ops are IO), so examples chain `tmul`/`tadd` — the instance is a false promise in its current
  home (honest home: `IO (Tensor …)`, §N3).

**Constraints.** C2 forces the lifecycle machinery to *exist* — not to be exported next to `tadd`.

**Verdict — split (public/internal boundary) + rename pass.** [A] · Severity: blocks-publish
(publishing a namespace with 16 exported `prim__` functions and raw-pointer ABI in examples) ·
Effort: M.
Move `prim__*`/`dt*`/raw-pointer surface into `Tensor.Unsafe` / `Internal.*` modules (still
importable — BYO-backend authors need them — but visibly fenced). After the §S1 facade lands,
examples should contain **zero** `prim__` references; enforce with a lint gate
(`scripts/`-level grep, wired into CI per the test-gates rule). Naming: settle `T.add`-style
namespacing vs `t*` prefixes in the layer-naming row; the decision here is only that *one*
convention survives publication.

**Follow-up row**: yes — clustered with §S1.

---

## North star: the target v1 API

The verdicts above each point at a piece of the same destination. This section sketches the whole,
so follow-up rows aim at one coherent design rather than ten local fixes. Everything here is
**reachable in Idris 2 today** except where flagged; signatures are sketches, not commitments.

### N1 · One constraint, not six (→ S9)

```idris
interface ( UserExecutorTraining ex, UserExecutorCore ex
          , RuntimeDType dt, Linked ex, Compatible ex dt
          ) => Backend (0 ex : Executor) (0 dt : DType) where

-- blanket implementation; BYO backends get it for free
linear : Backend ex dt => {i, o : Nat} -> Init (Linear i o ex dt)
```

### N2 · Construction: two constructors × an `Init` value (→ S1)

```idris
data InitSpec : Type where
  Zeros : InitSpec ; Const : Double -> InitSpec
  Normal : (mu, sd : Double) -> InitSpec ; Uniform : (lo, hi : Double) -> InitSpec
  FromVect : Vect n Double -> InitSpec          -- subsumes packDoubleBuf et al.

tensor : Backend ex dt => {dims : Vect rank Nat} -> InitSpec -> IO (Tensor dims ex dt NoGrad)
param  : Backend ex dt => {dims : Vect rank Nat} -> InitSpec -> Init (Tensor dims ex dt WithGrad)
```

Per-rank workers stay, `private`, behind these. `MkTensor` leaves the documented surface.

### N3 · Expressions over IO, not instead of it (→ S3, C1 intact)

Purity is unreachable (C1). But `IO (Tensor …)` is a perfectly good carrier for operators:

```idris
0 TensorM : Vect rank Nat -> Executor -> DType -> GradMode -> Type
TensorM dims ex dt g = IO (Tensor dims ex dt g)

Backend ex dt => Num (TensorM dims ex dt g) where
  a + b = do x <- a; y <- b; T.add x y
  -- …

share : TensorM dims ex dt g -> IO (TensorM dims ex dt g)   -- evaluate once, reuse
val   : Tensor dims ex dt g -> TensorM dims ex dt g          -- lift evaluated tensor
```

With this plus bang notation (already in the language: `mean !(T.mul d d)`), the 37-line DQN loss
becomes a handful of lines, and the `Num` instance moves to the type where it's honest. Document
`share` as the `let` of the DSL (un-shared subexpressions re-run — correct gradients, wasted
compute). Add `tgather`-style indexing ops so select-by-index stops being a hand recursion.

### N4 · Models are records + functions; `Seq` for chains (→ S2)

```idris
interface Params m where           -- replaces LayerLike for the generic operations
  params   : m -> List SomeParam
  freeze   : m -> IO (Frozen m)
  unfreeze : Frozen m -> IO m

record MLP (i, h, o : Nat) (0 ex : Executor) (0 dt : DType) where
  constructor MkMLP
  l1 : Linear i h ex dt
  l2 : Linear h o ex dt

forward : Backend ex dt => MLP i h o ex dt -> Tensor [b, i] ex dt g -> TensorM [b, o] ex dt g

data Seq : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> Type where
  Nil  : Seq i i ex dt
  (::) : (Layer l, Params (l i h ex dt)) => l i h ex dt -> Seq h o ex dt -> Seq i o ex dt
```

Batched-first signatures throughout (single sample = `b 1`); no `applyVarBatch`, no `idris_crash`,
no `MkAnyLayer`, no `OutputLayer`. `GradMode` comes off the model type (a `Linear` owns `WithGrad`
params by construction; `g` lives on activations). Mixed precision is not a parallel interface tree
— it's `fit` config. Keeping shapes *factored* (`[b, c, h, w]`, conv output dims via the existing
`ConvOutDim` per axis) instead of flattened products eliminates the Mnist shape-constant wall
within C3; `flatten` defers its product to an opaque `prod ds` that unifies without reducing.

### N5 · Names are derived, not invented (→ S4, C5 intact)

```idris
Init : Type -> Type                 -- IO + scope-path/counter state
scoped    : String -> Init a -> Init a
runInit   : Init a -> IO a

mlp : Init (MLP 4 64 2 ex dt)
ac  : Init (ActorCritic ex dt)
ac  = [| MkAC (scoped "actor" mlp) (scoped "critic" mlp) |]
-- registers actor.linear_0.weight, critic.linear_0.bias, … (PyTorch-convention names)
```

The C registry (C5) is unchanged; users stop typing `"actor_ll0"`. Param groups become
`groupOf model.actor` (via `Params`), killing the string-prefix bug class. Bonus: names match HF
`state_dict` conventions, shrinking the idris-transformers mapping surface. `named` stays as the
explicit escape hatch for checkpoint pinning.

### N6 · One driver (→ S5)

```idris
0 Step : Type -> Type -> Executor -> DType -> Type
Step m batch ex dt = m -> batch -> TensorM [] ex dt WithGrad

fit : (Backend ex dt, Params m) => Step m batch ex dt -> Optimizer ex
      -> Stream batch -> FitConfig -> m -> IO (FitResult m)
```

`fit` owns: zero-grad/backward/step, clipping, GradScaler when `precision := Mixed`, NaN guard,
early stop, checkpoint policy, the mlx generation/GC hygiene (exactly the code users must never
see), eval-bracket metrics. Recurrent/two-phase are step-function folds, not driver variants.
`runTrainingIO`'s engine survives as the internals. CLI arg parsing (`ArgSpec`, in `Train.idr`
today) moves to the examples package — it is not tensor-library surface.

### N7 · Data (→ S7)

```idris
record Dataset (sample : Type) where
  size : Nat
  item : Fin size -> IO sample     -- Fin: out-of-bounds is unrepresentable

stream  : ShuffleSpec -> Dataset a -> Stream a
batched : {b : Nat} -> Stream (Tensor [i] ex dt g) -> Stream (Tensor [b, i] ex dt g)
generate : IO a -> Stream a        -- synthetic tasks (copy/recall/RL rollout feeds)
```

### N8 · Checkpoints (→ S6)

```idris
save : Params m => m -> (path : String) -> IO ()
load : Params m => m -> (path : String) -> LoadOpts -> IO (Either LoadError ())
```

### N9 · The prelude

```idris
import ML          -- Tensor, TensorM + operators, T.*, NN.*, Loss.*, sgd/adam/adamW,
                   -- Dataset/Stream, fit, save/load, Backend, Init/runInit/scoped
import ML.Simple   -- additionally pins (Ex, F) to the build's BuildConfig cell;
                   -- what every README/tutorial example imports — zero {ex=} anywhere
```

`ML.Simple` generalizes the `ExampleDevice`/`ExampleDType` BuildConfig trick into the library
itself instead of leaving it re-rolled in the examples package.

### Explicitly unreachable (document, don't chase)

- **Purity / lazy graphs** — C1. The `TensorM` algebra is the approximation.
- **`vmap` / broadcasting tracer** — B1, no tracing infra. Batched-first signatures are the
  approximation; say so in the README before a JAX user asks.
- **Arbitrary type-level shape arithmetic** — C3. Factored shapes + opaque products are the
  idiom; large flattened literals stay a documented limitation.

---

## Verdict summary

| § | Surface | Verdict | Bucket | Severity | Effort | Follow-up |
|---|---|---|---|---|---|---|
| S1 | Constructor wall + marshalling | merge behind `tensor`/`param` + `InitSpec`; demote prims | A | blocks-publish | M | row 4 (w/ S10) |
| S2 | Network/AnyLayer/LayerLikeMixed | split: records+`Params`+`Seq`; remove existential chain + crash default | A | blocks-publish | XL | rows 3, 6 |
| S3 | Losses | functions: **rationale stands** (C1); remove Math.idr double; add TensorM combinators + gather | A/C | embarrassing | S+M | rows 5, 7 |
| S4 | Optimizers | merge to 3 ctors × config record; typed scopes | A | embarrassing | M | row 2 |
| S5 | 8 epoch variants + runners | merge into `fit` + user step fn; delete 3 dead variants now | A | blocks-publish | L | row 3 |
| S6 | Checkpoint | merge loads into `LoadOpts`; typed errors; policy **rationale stands** | A/C | embarrassing | S–M | row 7 |
| S7 | Data cluster | merge: `Dataset`/`Stream`; task-shape types leave the library | A | embarrassing | L | row 3 |
| S8 | idris-transformers shape | per-arch split **rationale stands**; rename `Hf*` → `Transformers.*`; `fromPretrained` | A/C | cosmetic | S–M | row 8 |
| S9 | Constraint saturation | merge: `Backend ex dt` bundle | A (B2 residual) | embarrassing | M | row 1 |
| S10 | Hygiene / export boundary | split: `Unsafe`/`Internal` modules; zero `prim__` in examples; one naming convention | A | blocks-publish | M | row 4 |

## Boilerplate ledger (where the example pain actually lives)

| Pain | Exhibit | Cost today | Bought back by |
|---|---|---|---|
| Custom-loss expression chains | `Dqn.idr:152-188`, 37 lines vs ~3 PyTorch | every RL example | N3 (TensorM ops + gather) |
| Shape-constant walls | `Mnist.idr:42-112`, 71 lines | conv/structured examples | N4 (factored shapes) |
| Data marshalling + `MkTensor` wrapping | `Dqn.idr:173-174`; 97 `prim__` uses in examples | every example | N2 (+S10 fence) |
| Param-prefix strings | `"actor_*"` discipline, documented bug class | all multi-net RL | N5 (scope monad) |
| Single-vs-batched loop duplication | `runEpisode`/`runEpisodeBatched` pairs | every RL example | N4 (batched-first) + N6 |
| Driver-variant selection | 8 `epoch*`, 3 unused | every supervised example | N6 (`fit`) |
| `withNoGrad {ex=…}` ceremony | 43/51 explicit `{ex=}` in examples | every eval loop | N6 internalizes; S9 inference fix |
| Constraint walls on helpers | `Mnist.idr:140-211` | every example with helpers | N1 (`Backend`) |
| CLI parsing | ~20 lines × 33 examples | all examples | move to examples-pkg helper (not library surface) |

## Draft follow-up rows (copy-paste into TODO.md; not yet filed)

Ordered: row 1 first (shrinks every later diff); rows 2/4/5/7 are independent after 1; row 3 is the
big one; 6 depends on 3's `Params`; 8 anytime.

| # | Row | Tag | Size | Blocks publish? |
|---|---|---|---|---|
| 1 | **Constraint bundle `Backend ex dt`** — add bundling interface + blanket impl (tiered: `BackendCore`/`Backend`/`BackendStreamed` mirroring executor tiers); sweep all exported signatures. Evidence: 6 constraints × ~270 lib + ~110 user sites (api-critique.md §S9). | refactor | M | yes |
| 2 | **Optimizer collapse** — drop `native*` prefix; `sgd`/`adam`/`adamW` × options record (clip/weightDecay/groups); schedules attach to optimizer; typed scopes once row 6 lands (interim: keep string scopes inside the record). §S4. | refactor | M | yes |
| 3 | **Single `fit` driver + data redesign** — replace 8 `epoch*` (3 already dead → delete immediately) + `runTraining*` with `fit` over user `Step`; samples become tensors (`Dataset`/`Stream`); `RecurrentDataPoint`/`TwoPhaseDataPoint` leave the library; fix `applyVarBatch` `idris_crash` hole via batched-first signatures. §S5+§S7+§S2(crash). | refactor | XL | yes |
| 4 | **Construction facade + export boundary** — `tensor`/`param` × `InitSpec` (incl. `FromVect` replacing `packDoubleBuf`/`prim__alloc*`); move `prim__*`/`dt*` to `Internal`/`Unsafe` modules; CI lint: zero `prim__` in examples. §S1+§S10. | refactor | M | yes |
| 5 | **TensorM expression layer** — `Num`/`Neg`/ops on `IO (Tensor …)`, `share`/`val`, `tgather`-style indexing; retire dead `Num (Tensor …)`; rewrite the Dqn loss exhibit as the acceptance test. §S3/N3. | feat | M | yes |
| 6 | **Models-as-records: `Params` + `Seq`, delete `Network`/`AnyLayer`/`LayerLikeMixed`** — precision becomes `fit` config; `GradMode` off model types; factored-shape conv path kills the Mnist constant wall. Depends on rows 1,3. §S2/N4. Coordinates with the layer-naming row. | refactor | XL | yes |
| 7 | **Checkpoint surface** — `save`/`load` over `Params m`, `LoadOpts`, `Either LoadError`; demote `Math.idr` loss/activation exports to internal. §S6+§S3. | refactor | S | no |
| 8 | **Transformers naming + `fromPretrained`** — `Hf*` → `Transformers.*` namespaces; converge loaders on `fromPretrained : … -> IO (Either LoadError (cfg ** Model cfg ex dt))`. Coordinates with the DRY row. §S8. | refactor | S–M | no |
| 9 | **`ML` umbrella + `ML.Simple` prelude** — single-import surface; generalize the BuildConfig device/dtype pin into the library; examples and README migrate to it. §N9. Depends on rows 1–6 for what it exports. | feat | S | yes |

## Reproducing the evidence

All counts: `grep -rwo '<sym>' packages/idris-ml/src | wc -l` (lib) vs
`grep -rwo '<sym>' packages/idris-ml-examples/src packages/idris-transformers/src packages/idris-gym/src packages/idris-ml-notebook/src | wc -l` (user),
run at `de5dc612`. Exceptions: `~~>`/`OutputLayer` etc. use `grep -rFo`; `prim__` leakage uses
`grep -rho 'prim__[a-zA-Z0-9]*' packages/idris-ml-examples/src | sort | uniq -c`.
