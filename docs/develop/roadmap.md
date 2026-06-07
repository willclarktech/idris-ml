# Roadmap — publishable v1

**Filed**: 2026-06-11 · **Synthesizes**: [`api-critique.md`](api-critique.md) (the evidence base —
verdicts, call-site counts, north-star API §N1–N9), [`glaive-survey.md`](glaive-survey.md) (naming
verdict + the named-axis borrow), and the pre-existing High rows (user docs, publish). TODO.md rows
remain the canonical backlog entries; this doc records the *sequence and dependencies* between them,
which a flat backlog can't.

**Target**: a publishable v1 — the api-critique north-star API, then the user-facing docs overhaul
written against it, then pack publication.

## Decisions taken 2026-06-11

1. **Publish timing**: API rework → docs overhaul → publish. Publication is the moment the
   "no users → no backwards compatibility" free ride ends, so it comes last, against the polished
   surface, and APIs break freely until then.
2. **Example migration**: one sweep at the end. All new library surfaces land first (old + new
   coexist, CI stays green throughout), then each of the ~33 examples migrates once, straight to the
   final API. One multi-seed convergence campaign instead of two — example migration is where the
   re-validation cost lives, so examples are touched exactly once.
3. **License**: choice deferred (see the LICENSE TODO row, which blocks the publish row). The repo
   currently has no LICENSE file; the AGPL adjacency from the TensorType survey makes the gap newly
   visible.
4. **Composition**: records + `Params` + `Seq` confirmed. `~~>` survives as `Seq`'s chaining
   operator — list-literal sugar and operator chains coexist (both desugar to the same
   constructors); no `OutputLayer`-style terminator (the endpoints-only index needs none).
5. **Expression layer** (deviates from api-critique §N3): operator aliases on *plain evaluated
   tensors* returning IO (`(+.)`, `(*.)`, …) used with bang notation, plus `tgather`-style indexing.
   **No `Num` instances on `IO (Tensor …)` and no `share`** — ops consume already-evaluated tensors,
   so nothing can silently re-execute and there is no extra sharing semantics to teach. Given up:
   integer-literal promotion. The dead `Num (Tensor …)` instance retires with nothing claiming its
   place.
6. **Param naming** (adjusts api-critique §N5; leaf-name half amended 2026-06-11): hierarchical
   `scoped` combinators + `groupOf` for optimizer scoping are settled. Leaf names: **explicit for
   now, with a time-boxed derivation spike gated into the models-as-records row.** The original
   rejection of auto-derivation rested on a false premise — "Idris can't reflect record field
   names". It can: a ~25-line `%runElab` script over `getCons`/`getType` reads them from base's
   `Language.Reflection` (verified on idris2 0.8.0). Derived names would be real field names —
   reorder-safe; the honest tradeoff is field-rename hazard + unmeasured elaborator cost
   (`%runElab` per model record, in a codebase with documented elaboration blowups) versus the
   leaf-typo / silent-gradient-non-flow class that explicit strings carry. Spike outcome decides:
   if deriving `Params` field names elaborates cheaply at real model size,
   derived-with-explicit-override becomes the design; explicit leaves are the fallback. Prefixes
   compose structurally either way (can't desync between networks); the C registry is unchanged.
7. **Driver** (adjusts api-critique §N6): `fit` is the primary documented path, and the engine
   pieces it composes (NaN guard, eval bracket, checkpoint tick, mlx generation hygiene) are
   **exported** so RL/custom loops compose them rather than reimplement the subtle parts.
8. **Batching**: batched-first where it earns it — dense/conv/transformer modules ship `[b, …]`
   forms (single sample = `b=1`); recurrent/memory modules (RNN/LSTM/NTM/DNC) keep their natural
   per-sequence forms. Under records there is no uniform interface forcing fake batched
   implementations; the type says what each module supports.

## Workstream sequence

Each workstream is a TODO.md row; sizes and evidence live there and in api-critique.md. Dependencies
are the arrows that matter — within a tier, rows are independent and can interleave.

**First — the constraint bundle** (`Backend ex dt`, api-critique §S9/N1). Mechanical but wide;
every later refactor's diffs shrink after it, so nothing else starts first. Its opening commit also
deletes the three zero-user epoch variants (`epochVarTensor`, `epochVarTensorMixed`,
`epochVarTensorBatchMixed`) — no call sites, no migration.

**Then, independently (any order, parallelizable)**:
- Optimizer collapse (§S4): `sgd`/`adam`/`adamW` × options record, schedules attach to the optimizer.
- Construction facade + export boundary (§S1+§S10): `tensor`/`param` × `InitSpec`; `prim__*`/`dt*`
  fenced into `Internal`/`Unsafe`; CI lint gate "zero `prim__` in examples".
- Tensor expression ops (§S3/N3 as adjusted by decision 5): `tgather`-style indexing + operator
  aliases on plain tensors with bang notation; the 37-line DQN loss exhibit is the acceptance test.
- Checkpoint surface + Math.idr demotion (§S6+§S3): `save`/`load` over the coming `Params`
  interface, `LoadOpts`, typed errors; one loss vocabulary.

**Then the two XL rows, in order, library surface only** (old surfaces coexist until the sweep):
- Single `fit` driver + data redesign (§S5+§S7/N6+N7, engine pieces exported per decision 7): `fit`
  over a user `Step`, `Dataset`/`Stream`; recurrent/two-phase become step-function folds.
- Models-as-records (§S2/N4+N5 as adjusted by decisions 4/6/8; depends on the bundle and `fit`):
  `Params` + `Seq` (with `~~>`), batched-first where it earns it (closes the `applyVarBatch`
  `idris_crash` hole structurally), `LayerLikeMixed` tree dies, `scoped` + explicit leaf names +
  `groupOf` for params. **Absorbs the former "Reconsider layer / module naming" row** —
  the new abstraction is born with the glaive-survey vocabulary (`Module`, `Sequential`, bare
  constructor names) rather than renaming `LayerLike` first and deleting it later. Before the
  interface locks, skim the two PyTorch-design-survey items that inform it (`__call__`-vs-`forward`
  + hooks; `register_buffer` vs `register_parameter`).

**Then the surface finish**:
- Transformers naming + `fromPretrained` (§S8): `Hf*` → `Transformers.*`; dependent-pair loader.
- `ML` umbrella + `ML.Simple` prelude (§N9; depends on everything above for what it exports).

**Then the example migration sweep** (decision 2): every example once, to the final API, with the
multi-seed convergence campaign per the alignment policy (PyTorch refs unchanged — hyperparameters
and architectures are identical, only Idris code shape changes, so paired-side alignment is not
triggered). The sweep ends by **deleting the old surfaces** (`Network`/`AnyLayer`/`LayerLike(Mixed)`/
`OutputLayer`/remaining `epoch*`/`runTraining*`/superseded constructors) — the concrete collapse step
that transitional scaffolding requires.

**Then docs, then publish**:
- User-facing documentation redesign (existing High row): written once, against the final API.
- Publish to package managers (existing High row): blocked by the docs overhaul and the LICENSE row.

## Post-v1 / parallel — not blocking

- **Named-axis experiment** (Medium row, glaive verdict D): layers on the Tensor surface that v1
  stabilizes; sequenced after the sweep so the experiment targets the surviving API.
- **PyTorch design survey remainder** (Medium row): the critique already covered the loss dual API,
  the DataLoader/Dataset/Sampler split, and param groups/schedulers/`state_dict`; the row continues
  with what's left (hooks, profiler, compile/fx, functional mirror).
- **Perf / backend / CI tracks** (fused-op epics, Idris-side per-op overhead, nix CI, CUDA-when-
  hardware, build/link backend model): orthogonal to the API rework; continue on their own cadence.

## Design-decision reconsideration ledger

Standing decisions are in principle open to reconsideration; most were re-tested by the critique and
survey rather than assumed. Where this roadmap *keeps* a decision, the evidence was re-examined; where
it *changes* one, the change is deliberate and recorded here.

| Decision | Status | Where re-tested |
|---|---|---|
| `IO`-typed tensor ops (C1) | **kept** — forced by FFI sequencing under strict eval; operator aliases + bang notation are the ergonomic answer, not purity | api-critique constraint ledger + §N3 |
| Wrapped-handle Chez ABI (C2) | **kept** — but fenced out of user-visible surface (§S10) | api-critique §S10 |
| Peano-Nat limits → factored shapes (C3) | **kept** — factored `[b, c, h, w]` + opaque products; large flattened literals stay documented limitation | api-critique §N4 + "explicitly unreachable" |
| String-named C param registry (C5) | **kept** — independently confirmed load-bearing by the survey (safetensors/HF/freeze/groups all key on names); but *usage* changes: prefixes compose via `scoped`, optimizer scoping goes through `groupOf`, leaf names explicit pending the derivation spike (decision 6 as amended) | glaive-survey verdicts A/B; api-critique §N5 |
| Autodiff lives C-side | **kept** — TensorType's own state (composition undefined, `%hint` dead-ends, `train` commented out) is the strongest evidence the pure-Idris path isn't ready | glaive-survey verdicts A/G |
| safetensors-only checkpoint format | **kept** | api-critique §S6 |
| Per-arch HF adapter modules | **kept** (naming changes only) | api-critique §S8 |
| Losses as functions (not classes) | **kept** | api-critique §S3 |
| Existential layer chain (`Network`/`AnyLayer`/`LayerLike`) | **changed** → records + `Params` + `Seq` | api-critique §S2/N4 |
| Single-sample-first signatures | **changed** → batched-first where it earns it: dense/conv/transformer modules ship `[b, …]`; recurrent/memory modules keep per-sequence forms (decision 8) | api-critique §S2/S5 |
| Mixed precision as parallel interface tree | **changed** → `fit` config | api-critique §S5 |
| `Num (Tensor …)` instance | **changed** → deleted, nothing claims its place; operator aliases on plain tensors + bang replace it (decision 5 rejected `Num` on IO carriers) | api-critique §S10/N3 |
| CLI arg parsing in `Train.idr` | **changed** → moves to the examples package | api-critique §N6 |
| `Math.idr` parallel loss surface | **changed** → demoted/internal | api-critique §S3 |
| Flat `Hf*` module names | **changed** → `Transformers.*` namespaces | api-critique §S8 |
| LICENSE | **deferred** (blocks publish) | decision 3 above |
| Build/link backend model (multi-link vs dlopen/static) | **deferred** — its own future design session per its TODO row | TODO Medium row |
| Opaque type-level Nats | **deferred** — external/upstream row | TODO Medium row |
