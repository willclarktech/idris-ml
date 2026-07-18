# Idris-2 JIT / JAX backend investigation

> **Historical record.** Identifiers and paths reflect the tree at the time of
> writing; not updated for later renames (Executor spellings 2026-06-06, `Ml.*`
> module nesting 2026-07-27). Name decoder: [path-c-migration.md](path-c-migration.md).

Date: 2026-06-08. Spike commit: pending.

Two questions, both originally parked under one TODO row ("Investigate
Idris-2 JIT / JAX backend (separate question marks)") while chasing the
GPU-friendly story. The 2026-05-14 Phase 3 measurements established that
Idris-side wallclock dominates every backend at GptLarge (8.4 s/ep on
torch, 8.5 s/ep on mlx-cpu, with C-time ≤ 1 s) — see `perf-changes.md`
"Where the 9000 ms/ep GptLarge wall actually goes". This doc captures the
spike conclusions for each question.

----------------------------------------------------------------------
## Question 1: Does Chez Scheme have a JIT lever we aren't using?
----------------------------------------------------------------------

**Verdict: No.** Chez has no dynamic JIT; idris-2 already uses Chez's
maximum static-optimization knob (`optimize-level 3`) + AOT
`compile-program` to native machine code. The dominant 7.6 ms/op cost is
in the *Idris-emitted Scheme* (existential dispatch, closures, GC
pressure, typeclass dispatch), not in Chez's interpreter overhead — fixing
it means changing what Idris emits, not changing how Chez runs it. The
right home for that work is the existing sibling row "Idris-side per-op
overhead — partial".

### Evidence

**Idris-2's emitted build script uses `optimize-level 3` + `compile-program`
and nothing else.** From `idris2-0.8.0/src/Compiler/Scheme/Chez.idr:525-529`:

```idris
let build = "(parameterize ([optimize-level 3] "
            ++ (if prof then "[compile-profile #t] "
                       else "") ++
            "[compile-file-message #f]) (compile-program "
            ++ show outSsAbs ++ "))"
```

The exact `compileChez` script generated for an example in this codebase's
build tree confirms the same — no cp0-*, no `optimize-procedure`, no
`generate-allocation-counts`:

```
(parameterize ([optimize-level 3] [compile-file-message #f])
  (compile-program "/Users/.../build/.../<example>.ss"))
```

**Chez `optimize-level 3` is already the maximum.** Per the Chez Scheme
User's Guide §6: levels 0-2 generate "safe" code (full type/bounds
checking); level 3 generates "unsafe" code (checks disabled, faster). 3
is the ceiling; there's no level 4.

**Chez compiles to native machine code via `compile-program`, not to
bytecode.** This was a load-bearing assumption to verify — the spike-time
hypothesis "switch to Chez native code" would have been wrong because we
already get it. Sources: Chez Scheme docs + community confirmation
(hn:16406391: *"Chez is not JITted. Chez compiles to native binaries that
run in the chez runtime."*).

**Chez has no runtime JIT / tier-up.** No `compile-on-hot-loop`, no
adaptive optimization, no equivalent to V8's TurboFan or PyPy's tracing
JIT. All optimization happens at AOT compile time. (Confirmed by the Chez
docs section on system parameters: every compile control parameter is
read once at `compile-*` time; there is no runtime-tier flag.)

**Other Chez compile knobs exist but are unlikely to deliver meaningful
wall.** `cp0-effort-limit` / `cp0-score-limit` / `cp0-outer-unroll-limit`
tune inlining aggressiveness (defaults set by Chez authors). Raising them
trades compile time for runtime; there's no public report of a Scheme
project landing a measurable runtime win by tuning them beyond defaults.
Worth trying as a one-line probe in a future Idris-2 fork; not worth a
multi-day spike here.

**Idris-2 explicitly is not an optimizing compiler.** Per the official
FAQ: "Idris 2 has significantly better type checking performance...and
generates significantly better code" (vs Idris 1) but the design goal is
"compile dependently typed functional code in a timely manner" — i.e. the
codegen is for correctness, not for hot-path optimization. Any hot-path
optimization is delegated to Chez.

**The cost is *between* FFI calls, in Idris-emitted Scheme.** Per
`perf-changes.md` 2026-05-14:

  - GptLarge mlx-cpu wall: 8600 ms/ep
  - C-time (mlx compute): ≤ 140 ms/ep
  - FFI dispatch wall: 0.46 µs/call × 1136 calls = 0.5 ms/ep
  - **Idris-side wall between FFI calls: ~7960 ms/ep = 7.0 ms/op**

So 99.4% of wall is the Idris-emitted Scheme. The lazy-cache
`foreign-procedure` fix (commit `2385e3f`, 2026-05-27) closed the only
known Chez-side leak in the FFI path. Further wins live in suspects
catalogued by the sibling "Idris-side per-op overhead — partial" row:
existential `AnyLayer` dispatch, typeclass dictionary resolution, Tensor
record packing, `Vect` shape arithmetic, GC pressure.

### Decision criteria that would flip this verdict

  - Someone publishes a Chez-side tuning recipe (cp0-* knobs, alternative
    code generators, profile-guided rebuild) that delivers ≥ 10% wall on
    a Scheme-heavy ML-style workload. File the recipe + a follow-up row.
  - Idris-2 itself acquires a different default backend (Racket, Gambit,
    native via `compile-whole-program` patches) that beats Chez on tight
    loops by ≥ 2×. Re-evaluate cross-backend.
  - The sibling "Idris-side per-op overhead" row's targeted measurement
    surfaces a Chez-codegen specific hotspot (rather than an
    Idris-emitted Scheme hotspot). At that point Chez tuning becomes
    leverage.

----------------------------------------------------------------------
## Question 2: Should we add a JAX / XLA backend alongside tape/torch/mlx?
----------------------------------------------------------------------

**Verdict: No.** Three reasons compound:

  1. **JAX itself is Python.** There is no "JAX C++ API" that an Idris
     backend could embed. The C++ surface is **PJRT** (the stable plugin
     ABI) + **StableHLO** (the IR) + **XLA FFI** (custom-call API). A
     "JAX backend" would actually be a PJRT-client-emitting-StableHLO
     backend, parallel to PyTorch/XLA.

  2. **XLA FFI is currently experimental.** Per OpenXLA docs: *"the
     custom-call API/ABI uses PJRT-style versioning, however at this
     point it is still experimental and can be broken at any time"*. PJRT
     itself is more stable, but the application would still drag in the
     XLA C++ runtime as a dependency.

  3. **mx::compile() solves the same problem in our existing mlx backend
     with zero new instance code.** The "compile-the-whole-step" win that
     XLA delivers — trace a multi-op function once, replay as a fused
     kernel — is exactly what `mx::compile()` does on Apple Silicon. The
     `mx::compile()` integration is already a tracked follow-up (see
     `project_mlx_gpu_environment.md` and the mlx-survey doc). Adding
     XLA-via-PJRT to do the same job costs an estimated 3-8 KLOC C/C++ +
     1.5 KLOC Idris (per the `UserExecutorCore` 22-method surface + 10
     sister interfaces, with rename-header machinery auto-handled) — a
     much larger surface than `mx::compile()` for ~the same win on the
     same hardware.

The unique advantage JAX/XLA has over mlx+torch is **TPU + cloud
accelerator coverage**. idris-ml's roadmap does not include TPU. Until
that changes, the integration cost is not justified.

### Evidence per probe

**Probe 1 — JAX C++ API stability.** JAX is Python-first. The
stable C++ integration story is PJRT (plugin ABI for ML hardware/runtimes)
+ StableHLO (the IR JAX emits, designed for backward compatibility).
PyTorch/XLA exists as a precedent for "non-JAX framework consuming
XLA via PJRT". The XLA FFI custom-call API is **experimental, can break**.
A from-scratch PJRT client is the stable path; embedding JAX-the-Python-
library is not.

**Probe 2 — XLA HLO emitter ergonomics.** The win is "trace once, replay
many" on a multi-op function. To capture that win from idris-ml, we'd
have to emit StableHLO modules at coarser granularity than per-tensor-op
— ideally a whole `epochVar` would become one StableHLO module compiled
once and replayed N times. This is essentially the same shape as a
deeply-traced `mx::compile()` call. If the traced graph is per-op, we
re-pay XLA compile cost on every op and lose the JIT advantage entirely.

**Probe 3 — `mx::compile()` comparison.** Per MLX docs (deepwiki +
ml-explore.github.io): *"MLX's JIT compilation system optimizes
computation graphs by fusing operations, simplifying expressions, and
generating optimized kernels for different hardware backends ... transforms
sequences of array operations into efficient, fused primitives that reduce
memory overhead and improve execution performance."* First call compiles +
caches; subsequent calls replay. This is the same value proposition as
`jax.jit`, with two advantages over an XLA integration: (a) no new
backend — it's an extension to the existing mlx backend; (b) handles
varying shapes with a single compiled function (sglang issue #19146
contrasts this favourably with CUDA graphs).

**Probe 4 — Hardware coverage delta.** Torch covers CPU + CUDA + ROCm +
MPS. MLX covers Apple Silicon GPU. JAX/XLA uniquely covers TPU + some
cloud accelerators. idris-ml's current hardware roadmap (per CLAUDE.md +
`docs/develop/llm-inference.md`) targets Apple Silicon + commodity CUDA
boxes. No TPU. Until that changes, "unique JAX-only hardware" is empty.

### Decision criteria that would flip this verdict

  - idris-ml acquires a "we need TPU" goal (TPU pod inference for
    LLM-scale workloads, research collaboration with a TPU-resident lab,
    etc.). At that point, file a new row "PJRT-client backend emitting
    StableHLO" — not "JAX backend".
  - `mx::compile()` lands and underdelivers vs in-tree XLA benchmarks on
    the same Apple Silicon hardware (current MLX JIT docs suggest this is
    unlikely — they target the same fusion + replay shape).
  - XLA FFI stabilizes (drops the "experimental, can break at any time"
    note) AND a non-Python framework lands a maintained PJRT client we
    could template off (PyTorch/XLA is the obvious candidate; track its
    state).

----------------------------------------------------------------------
## Summary
----------------------------------------------------------------------

| Question | Verdict | Next step |
|---|---|---|
| Idris-2 JIT lever? | **No** — Chez has no JIT, idris-2 already uses optimize-level 3 + native AOT | Defer to sibling row "Idris-side per-op overhead — partial" (TODO.md) |
| Add JAX/XLA backend? | **No** — the right framing is "PJRT/StableHLO client", and `mx::compile()` already targets the same compile-the-whole-step win | Pursue `mx::compile()` integration (already tracked in mlx project memory) |

Neither outcome is a surprise relative to the spike's draft conclusions;
the value of the spike is the documented evidence + flip-criteria so the
questions don't get re-asked from scratch.

### Cross-references

  - `docs/develop/perf-changes.md` 2026-05-14 "Where the 9000 ms/ep
    GptLarge wall actually goes" — the per-op-cost decomposition.
  - `docs/develop/perf-changes.md` 2026-05-27 / 2026-06-03 — Chez
    `foreign-procedure` lazy-cache fix (commit `2385e3f`), the one known
    Chez-side win to date.
  - `docs/develop/chez-profiling.md` — recipe for source-level Chez
    profiles, the methodology used to find the Nat-arithmetic and
    foreign-procedure hotspots.
  - `idris2-0.8.0/src/Compiler/Scheme/Chez.idr:519-535` — the idris-2
    function that emits the `compile-program` invocation; canonical
    reference for what Chez flags idris-2 sets.
  - Chez Scheme User's Guide §6 (optimize-level), §11 (system parameters
    including cp0-*).
  - OpenXLA project: PJRT integration guide, StableHLO spec, XLA FFI
    custom-call docs.
  - MLX `mx::compile()` — ml-explore.github.io/mlx + deepwiki MLX
    compilation page.
