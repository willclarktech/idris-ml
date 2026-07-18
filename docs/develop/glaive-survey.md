# Glaive Research / TensorType survey

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

Surveyed 2026-06-11. Closes the TODO row "Survey Glaive Research for usable ideas"
and unblocks the "Reconsider layer / module naming" row (see the recommendation at
the end).

[Glaive Research](https://glaive-research.org/) is Bruno Gavranović's nonprofit
applying category theory to AI verification, publishing working code in Idris 2.
The overlap with idris-ml — shape-indexed tensors, typed NN composition, categorical
autodiff — is real, so the survey question was: which of their formalisms are
directly adoptable, which are interface inspiration, and which are out of scope
given idris-ml's architecture (typed Idris surface over C-side autograd backends)?

## License constraint

[bgavran/TensorType](https://github.com/bgavran/TensorType) is **AGPL-3.0**
(idris-ml currently carries no LICENSE file at all). Every verdict below is about
**ideas and type-level interface shapes only — no code may be copied** from
TensorType or any AGPL-adjacent repo. zanzix/idris-neural-net has no license file
(all rights reserved by default), so the same rule applies there.

## Sources read

| Source | What it is |
|--------|------------|
| [bgavran/TensorType](https://github.com/bgavran/TensorType) | Idris 2 library (active, pushed 2026-05): container-based tensors with **named axes**, Para machinery, chart/lens autodiff scaffolding, NN architectures (MLP, RNN, self-attention incl. over trees), lens-shaped optimisers, partial training loop. Read in full at the module level. |
| [zanzix/idris-neural-net](https://github.com/zanzix/idris-neural-net) | 6-file companion to the 2024 post: `GPath` free graded path, layers as pure lens pairs, end-to-end toy training. Read in full. |
| [Building a Neural Network from First Principles using Free Categories and Para(Optic)](https://glaive-research.org/2024/04/15/neural-network-first-principles.html) (2024-04) | Conceptual walkthrough of the zanzix repo. |
| [Generalised Tensors in Idris](https://glaive-research.org/2026/01/21/Generalised-tensors.html) (2026-01) | Containers (`shapes`/`position`) as generalised tensor dimensions; the theory behind TensorType. |
| [Generalized Transformers from Applicative Functors](https://glaive-research.org/2025/02/11/Generalized-Transformers-from-Applicative-Functors.html) (2025-02) | Attention/MLP generic over any Applicative; Chebyshev-basis "Funcformer" for operator learning ([tlaakkonen/funcformer](https://github.com/tlaakkonen/funcformer), Haskell). |
| [Autodiff through function types](https://glaive-research.org/2026/02/20/categorical-semantics-ultimate-backpropagator.html) (2026-02) | Additive lenses are cartesian closed → AD through higher-order functions. Author's own conclusion: limited practical use (function-valued params memorise; would need a new compute kernel). |
| [Q4 2025 report](https://glaive-research.org/2025/12/08/q4-report.html), [Q1 2026 report](https://glaive-research.org/2026/04/07/progress-report-q1-2026.html) | Polylang (tactics as natural transformations between polynomial functors), structured encoding/decoding for LLMs, Algebraic Positional Encoding extended to inductive types. |
| [Types and Neural Networks](https://glaive-research.org/2026/04/20/types-and-neural-networks.html) (2026-04) | Differentiating *with respect to* program structure vs *through* it; no code. |

## Verdict table

| # | Idea | Verdict | One-line rationale |
|---|------|---------|--------------------|
| A | Para/Optic as the compositional base for layers | **inspiration only** (mechanism); **informs naming** (see recommendation) | idris-ml's params/grads live in the C registry behind `IO`-typed FFI; Para puts them in the type. Adopting the mechanism = rewriting autograd in Idris. Glaive's own code shows the road is hard even pure (see A below). |
| B | Free-category `GPath` typed parameter accumulation | **inspiration only** | Elegant (kills the "param without paramId is invisible" footgun at the type level) but incompatible with the string-named C registry that safetensors/HF alignment, `freezeByPrefix`, and per-group optimisers all depend on; type-level shape-list accumulation also stresses the known Idris-2 elaborator limits. |
| C | Container-based generalised (non-rectangular) tensors | **out of scope** | C backends (tape/torch/mlx) are dense rectangular buffers; tree-shaped axes cannot reach the kernels. Revisit only if structured-data examples ever become a goal. |
| D | **Named axes** with compile-time consistency | **borrow interface** → follow-up TODO row filed | The one genuinely adoptable idea: a name layer over `Vect rank Nat` catches transposed-same-size-dims bugs the bare shape vector can't. TensorType's `ConsistentWith` proof-carrying shape vector is the reference design (details below). |
| E | Applicative-generalized attention (transformers over trees) | **inspiration only** | Impressive demo (same params run attention on a matrix and a binary tree) but fails the PyTorch-precedent test for kernels and serves no current library user. |
| F | Naperian/representable functors for transpose-like ops | **inspiration only** | Elegant for a pure surface; `Array`/Math.idr already has shape-typed transpose, and the abstraction adds no user-visible capability. |
| G | Idris-level autodiff core (charts forward / lenses backward) | **out of scope** (by architecture); **borrowed their negative results** into gotchas.md | idris-ml deliberately keeps autodiff C-side for performance. TensorType's own `SearchIssues.idr` documents why `%hint`-driven derivative discovery dead-ends in Idris 2 — that finding is recorded in our gotchas. |
| H | Optimiser-as-dependent-lens, typed training loop | **inspiration only** | Their `train` is currently commented out; idris-ml's `Train.idr` + native optimizer is strictly more complete (checkpointing, NaN detection, schedules). The `composeParallel` optimiser-product idea is what `nativeAdamGroup` prefixes already deliver. |
| I | Algebraic Positional Encoding over inductive types | **inspiration only / revisit later** | Research-grade (Q4 2025–Q1 2026 reports), no published code surveyed; a long-horizon idea if structured inputs ever matter. |

## Notes per verdict

### A — Para/Optic (and what it means for naming)

The Para view: a layer is a morphism `(P × A) → B` with parameters `P` carried in
the type; composition accumulates parameter types; backprop is the lens/optic
structure on top. zanzix/idris-neural-net demonstrates it end-to-end at toy scale
and it is genuinely beautiful — `learningRate` and `crossEntropyLoss` are just more
lenses in the chain, and `train` is a fold over `eval`.

Why not adopt the mechanism:

- idris-ml's parameters are **registry state addressed by string `paramId`**, not
  type-level data. That choice is load-bearing: safetensors round-trips, HF name
  alignment (`HfBert` et al.), `freezeByPrefix`, `nativeAdamGroup` prefix filtering,
  and the C-side optimizer all key on names. Para-style typed accumulation would
  have to be *translated back* into names at every boundary.
- Forward passes are `IO`-typed because the FFI sequencing demands it (the
  `withNoGrad` strictness trap). Pure lens pairs assume pure forwards.
- TensorType's own state is the strongest evidence on difficulty: `DepParaMor`
  composition is left undefined ("not necessary for us"), the `%hint`-based
  derivative search "doesn't work" (their words, `Core/Forward.idr`), composition
  needs `believe_me` (18 `believe_me`/`unsafePerformIO` sites in the repo vs
  idris-ml's zero-tolerance policy), and the categorical `train` function is
  commented out. The pure-Idris Para path has not yet been made to work in anger
  by its own authors.

What Para composition *does* expose as a real idris-ml weakness: parameter
registration is invisible to the types (the `paramId` footgun). Verdict B notes
why the type-level fix doesn't fit; the practical mitigations (construction-time
prefix discipline, registry assertions in tests) stay as-is.

### B — `GPath` typed parameter accumulation

`GPath g ps a b` (zanzix `Path.idr`) is a snoc-list-indexed free path: each edge
carries its parameter shape in the index, and `eval` folds the lens pairs into one
`(All Tensor ps, Tensor s) → Tensor t` pass. ~25 lines, fully typed, no existential.
The idris-ml equivalent (`Network` + `AnyLayer` existential + string registry)
erases all of that. The trade was deliberate — see A — and the elaborator cost of
type-level shape-list accumulation at real network sizes (cf. the documented
Nat-reduction hangs) makes a retrofit unattractive. No follow-up filed.

### D — Named axes (the adoptable idea)

TensorType's `Axis` is a `(name : String, cont : Container)` pair; `TensorShape`
is a proof-carrying vector where every cons demands `a ConsistentWith as` — the
axis name is either new, or already present *with the same container* (dictionary
semantics, `Data/Tensor/Shape/Shape.idr`). Cubical axes are built with
`"seqLen" ~~> 3`. The payoff: `Tensor [SeqLen, NumTokens]` instead of
`Tensor [3, 4]`, and a transposed argument is a type error even when the
dimensions happen to be equal.

For idris-ml the container half is out of scope (verdict C), but the *name* half
layers cleanly onto `Tensor (dims : Vect rank Nat)`: an optional aliasing scheme
where examples define `SeqLen = Named "seq" 32` style axes and ops propagate
names. Open design questions (name propagation through matmul/reshape; whether
names live in a parallel index or a richer shape entry; elaborator cost) are real,
so this is a follow-up TODO row, not a quick win. Their string-name +
`decEq`-driven `NotElem`/`Elem` proofs are the reference pattern. Note their
`~~>` collides with idris-ml's network-chaining `~~>` — a rename would be needed
regardless.

### G — Autodiff negative results worth keeping

`Data/Autodiff/Core/SearchIssues.idr` is a self-contained catalogue of Idris-2
proof-search behaviour: monomorphic `%hint`s resolve; polymorphic hints *without*
constraints resolve; polymorphic hints *with* constraints (`Num a => Diff (F {a})`)
fail; and `%search` can never decompose `Diff (g . f)` into hints for `g` and `f`.
This rules out the "annotate every primitive with a `%hint` derivative and let
search assemble the chain rule" design in today's Idris 2 — relevant to any future
idris-ml feature hoping to dispatch on function values. Recorded in
`docs/develop/gotchas.md` ("%search / %hint limitations").

## Naming-row recommendation (unblocks "Reconsider layer / module naming")

**Align to PyTorch's `nn.Module` framing. Do not adopt `Para` or pure-CT
`Morphism` naming.**

- *Naming honesty*: the row's own motivating principle. `LayerLike` values are not
  Para morphisms — their parameters are registry state, not type-level data, and
  their forwards are `IO` actions. Calling the interface `Para` (or `Lens`/
  `Morphism`) would claim categorical structure the code measurably does not have.
  `Module` claims exactly what it is: a named, composable, stateful NN component.
- *Oracle alignment*: the repo's whole methodology treats PyTorch as the
  correctness oracle and aligns examples/HF adapters to it. `Module`/`Sequential`
  vocabulary is what every user arriving from PyTorch already holds.
- *Glaive's naming is itself unsettled*: `DepParaMor` / `Para` / `DPara` /
  `-\->` vs `-\-->` — adopting it would chase a moving target.

Concretely for the naming row's axes: `LayerLike` → `Module`, `AnyLayer` →
`AnyModule` (or collapse), `Network` → `Sequential`, `Layer/` dir → `Module/` or
`Nn/`, `*LayerAny` constructor suffix → bare names. The Para reading stays as a
documentation footnote (this survey), not as API vocabulary.

## Follow-ups filed

- TODO row (Medium): **Named-axis experiment on the typed tensor surface** —
  verdict D.
- `docs/develop/gotchas.md` entry: %search / %hint limitations — verdict G.
- Naming row unblocked with the `Module` recommendation above.
