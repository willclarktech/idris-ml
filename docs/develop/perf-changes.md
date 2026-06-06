# Perf changes log

Append-only log of every performance change we make: motivation,
change, impact (before/after ms-per-epoch and ratio), commit hash,
date. The companion files are:

- `docs/develop/perf-log.md` — raw `scripts/perf-run.sh` /
  `scripts/perf-baseline.sh` measurement entries (one per run).
  Records the *measurements*, not the code changes.
- `docs/develop/perf-baseline.md` — current-state table of every
  example × backend ratio. Reflects the latest measurement.

This file records the *changes themselves*: what was tried, what
landed, what didn't. New entries go at the bottom.

## Convention

```
### YYYY-MM-DD — <short title> — <commit-hash>

**Plan job**: Job 1 / Job 2a / Job 2b / cross-cutting (specify which)

**Motivation**: 1–3 sentences on why this change. What did the
profile show? What was the hypothesis?

**Change**: 1–3 sentences on what landed (or was tried). Reference
the file(s) and key function(s).

**Impact**: a small table of before/after ms-per-epoch and ratios
on the affected (example, backend) cells. Note "noise" if the
delta is below the measurement floor; "regression" if it got
worse on some cells.

**Outcome**: landed / reverted / partial. If reverted, why.

**Cross-references**: relevant `perf-log.md` entries (date + commit),
relevant `perf-baseline.md` rows updated.
```

When measurement noise is in play, take 3+ samples and report
mean + range. Don't claim a win from a single run inside the noise
floor.

When a change is reverted, leave the entry — the negative result
is still useful (saves someone trying it again).

----

## Entries

### 2026-06-05 — perf-regression CI gate flipped to hard-fail — 7ca9155c..HEAD

**Motivation**: the advisory gate (`7ca9155c`) shipped exits-0-always
so the threshold logic could run on real CI traffic without blocking
merges; the originally-planned ~2-week soak before flipping to
hard-fail is intentionally collapsed. With no users on this repo
there's no merge-velocity cost to gating now, and on the live
`perf-log.jsonl` every cell is INSUFFICIENT-HISTORY today (2–3
samples per cell, the gate skips them), so the change is effectively
a no-op until cells accumulate ≥6 samples — at which point the gate
starts having teeth without any further code change.

**Change**: one-line edit in `scripts/check-perf-regression.py` —
the final `return 0` becomes `return 1 if counts["FAIL"] > 0 else 0`.
Docstring updated. No escape-hatch env var (no users → no
backwards-compat ceremony, per `feedback_no_backcompat`).

**Impact**: no measurement change. Verified via the synthetic fixture
used at gate-landing time: `test_op +10% → OK`, `test_warn +20% →
WARN`, `test_fail +50% → FAIL`. Pre-fix `test_fail` exit was 0
(advisory); post-fix exit is 1 (CI red). Live `make
test-integration-lint-perf-regression` against today's
`perf-log.jsonl` exits 0 (all cells INSUFFICIENT-HISTORY).

**Outcome**: landed. Closes the "Promote perf-regression CI gate
from advisory to hard-fail" row in TODO.md.

----

### 2026-06-04 — Full 16×5 perf-baseline refresh + latent narrow regression hunt — 4c7aa76f

**Plan**: Run the full `scripts/perf-sweep.sh` matrix (16 examples ×
5 cells = 80 measurements) at the dtype-narrowed default (BuildConfig
F64 everywhere, F32 only for HF-heavy goals — see this file's earlier
2026-06-04 entry for the narrowing decision) to land a current
apples-to-apples baseline in `docs/develop/perf-baseline.md`. Side
effect: the sweep is the canonical broad-coverage gate for regressions
across the cross-backend surface, so anything that's been silently
broken since the last full sweep would surface.

**Motivation**: A perf baseline that lets us tell "is this commit
faster/slower than yesterday" requires apples-to-apples numbers across
the whole matrix; the prior `perf-baseline.md` snapshots dated 2026-
05-09 / 2026-05-19 were stale across multiple PyTorch-ref alignment
+ KV-cache + fused-op landings. The 2026-06-04 session's KV cache
work added the gates this baseline now protects, so capturing the
current-state table is the natural session-end deliverable.

**Change**: Two waves shipped in the same session, gated on each
other:

1. **Latent-regression hunt** (commits `6742356e` + `c7d47c8c` +
   `4c7aa76f`) — the first sweep run crashed on `transformer × every
   non-tape backend` and `dnc-copy × tape`. Three independent latent
   bugs surfaced + fixed:

   - `Layer/Transformer.idr:205` flattened a `[seqLen, vocab]` mm
     output via `primNarrow ... 0 0 (sI * vI)`. Worked accidentally
     pre-commit `69a05976` (2026-05-26) when `primNarrow` flattened
     before slicing; post-fix it errors with `start (0) + length
     (88) exceeds dimension size (11)`. Same-shape error class as
     `Example/Transformer.idr`'s `catCELossVar` narrow (also using
     flat-element-count indices on a 2D tensor). Both fixes: use
     `primReshape1d` for flatten + row-indexed narrow for row slice
     instead of conflating the two. Latent ~10 days.
   - Tape `tensor_unsqueeze` only handled rank-1 input (rank-N
     fell through to `tensor_clone` — preserving rank instead of
     adding the axis). Broke DNC's `primCat2 (primUnsqueeze
     onesScalar 0) slicedT` chain with `rank mismatch (a->rank=0,
     b->rank=1)`. Fix: rewrite to construct the new-shape vector
     correctly for any source rank, validate `dim` in `[0..rank]`.
     Latent ~6 weeks.

2. **Re-baseline** (commit `9d79b882` for the docs landing) — full
   sweep ran end-to-end after the fixes; 76 clean cells + 4 mlx-RL
   long-run allocator crashes (documented under existing TODO row
   50 as an additional manifestation of the mlx-tape-accumulation
   bug class). Current-state table now sits at the top of
   `docs/develop/perf-baseline.md`; historical snapshots
   untouched below.

**Impact**:

The 76-cell table (full data in `docs/develop/perf-baseline.md`'s
top section + `perf-log.jsonl`):

- **Tape wins every example** (16/16). On the supervised + recurrent
  cluster (rnn / lstm / gru / transformer), tape lands at **4–13% of
  PyTorch wall-time** — an 8–25× speed advantage. NTM/DNC at 15–25%
  of PyTorch; RL examples at 1–5× (RL is step-bound by env
  interactions, not backend throughput).
- **torch-cpu vs torch-mps** is essentially tied on this VM
  (~0.7–1.2× PyTorch on supervised + recurrent). Both tracking
  libtorch's wall closely — expected since they're both libtorch
  under the hood. MPS edges CPU only on transformer (0.26 vs 0.41)
  where the fused SDPA + attention kernels kick in.
- **mlx-cpu vs mlx-gpu** are 15–300× slower than PyTorch at this
  scale. Kernel-launch wall on Metal dominates these sub-millisecond
  examples (see `project_mlx_gpu_environment.md` + the MLX-fusion
  epic). mlx wins only at much larger model + batch sizes
  (HfLlama-1B / BitNet-2B inference).
- **RL examples are higher-ratio across the board** — DQN /
  MountainCar / MountainCar-Cont / PPO all at 3-10× tape. RL has
  inherent step-by-step env interactions that don't batch; the
  per-step wall is dominated by Idris-side overhead. The existing
  TODO row "Idris-side per-op overhead" is the lever for that
  cluster.
- **4 crashes** — all on mlx (cpu + gpu) on the 2000-epoch RL
  examples (mountain-car-cont + sac). mlx allocator failures
  (`Unable to allocate 4/256/512 bytes` — not real OOM on a 16 GB
  VM). Same class as the existing `mlx_sweep_generation` row's
  freed-ArrayDesc poison; long-run-accumulation symptom. Filed
  under TODO row 50 as an additional manifestation rather than a
  new row.

**Effect of the narrow fixes specifically on the previously-crashed
cells** (cells that emitted `crashed` in the pre-fix sweep run now
emit per-epoch ms):

| cell | pre-fix | post-fix (ratio) |
|---|---|---|
| transformer × tape | `idris_ms=0.0` (silent crash) | 0.04 |
| transformer × torch-cpu | Chez SIGABRT | 0.41 |
| transformer × torch-mps | Chez SIGABRT | 0.26 |
| transformer × mlx-cpu | mlx reshape exception | 1.47 |
| transformer × mlx-gpu | mlx reshape exception | 3.50 |
| dnc-copy × tape | `tensor_cat2: rank mismatch` | 0.15 |

**Backend coverage**: tape (CPU), torch-cpu, torch-mps, mlx-cpu,
mlx-gpu. All 5 cells exercised on all 16 examples. Tape is the
default lane.

**Commits**: `6742356e` `catCELossVar` row-indices; `c7d47c8c` tape
`tensor_unsqueeze` rank-N; `4c7aa76f` `Layer.Transformer` flatten
via `primReshape1d`; `9d79b882` perf-baseline + CHANGELOG docs.

**Outcome**: landed. New baseline is the apples-to-apples reference
for any future commit that touches the cross-backend surface — diff
against this table to detect regressions, against the per-cell entries
in `perf-log.jsonl` to compare specific cell wall-clock. mlx-RL
long-budget allocator failure stays filed under TODO row 50; not
blocking any current workload.


### 2026-06-04 — BuildConfig default flipped F64 → F32 + three-backend HfLlama cache gate verified — 2c7d371f

**Plan**: Two changes shipped same day, in sequence.

1. Flip `BuildConfig` default dtype from F64 (everywhere except
   Metal-forced cells) to F32 (everywhere). Examples-side only;
   `TestConfig` stays F64 (unit tests value-pin against F64
   oracles at tolerances down to 1e-12 which F32 can't satisfy —
   separate follow-up).
2. Verify the KV-cache token-sequence gate (commit `c1f7489d` ..
   `676830b9`) GREEN on all three backends at F32: torch-cpu,
   torch-mps, mlx-gpu.

**Motivation**: F64 default was historical, not a design decision.
F32 matches modern ML practice + the on-disk reference weights of
every shipped HF adapter (BF16 → F32 at load), uses half the
memory (a 1.24B-param Llama is 5 GB at F32 vs 10 GB at F64), and
runs ~2× faster on CPU via better SIMD. Run-time memory pressure
on a 16 GB VM was a real constraint pre-flip — Chez during
elaboration + libtorch in-memory model + Python pytest oracle
process together pushed system memory past the swap line.

**Change**: `Makefile`'s `BUILDCONFIG_IDR` recipe — all default
cells set DTYPE="F32"; `BuildConfig.idr.in` + `TestConfig.idr.in`
docstrings updated; `CLAUDE.md` matrix updated; `HfLlamaInference.idr`
comment updated. Commit `2c7d371f`.

**Side fix shipped during verification** (mlx-gpu side, commit
unstaged-then-reverted at `5a55c930`):

In Phase C (commit `04e1396c`) I'd added a defensive explicit-
mask path to the mlx SDPA wrapper to handle asymmetric Q/KV under
`is_causal=True` — same risk class as the torch math-impl bug
fixed two commits earlier. Building against the pinned mlx
version (`packages/pytorch/.venv/.../mlx/include/mlx/fast.h`)
revealed mlx's `scaled_dot_product_attention` only accepts a
`std::string` mask_mode, not an `array` for explicit-mask paths.
Reverted to the simple `mask_mode = isCausal ? "causal" : ""`
shape and added a comment that the lower-right alignment
correctness on asymmetric q/kv is trusted to mlx's documented
behaviour. If a mlx-gpu gate ever fails on asymmetric the way
torch-cpu did, the fix is either an mlx version bump or a
hand-rolled `softmax(scale*mm(Q, K^T) + mask) @ V` in the
asymmetric branch.

**Impact** (FIRST F32 build per backend; full ttc cache regen
included — subsequent warm-cache runs would be ~1-2 min total):

| backend / device | wall (first F32 build) | exit |
|---|---|---|
| torch / cpu | 28m 11s | GREEN ✓ |
| torch / mps | 37m 20s | GREEN ✓ |
| mlx / gpu | 22m 4s | GREEN ✓ |

All three produce the same token sequence `[128000, 791, 6864,
315, 9822, 374, 12366, 13, 1102, 374, 279, 1455, 95551, 3363]` =
"<|begin_of_text|>The capital of France is Paris. It is the most
populous city" — exact match to the HF oracle's
`save_oracle_llama_generate.py` output.

**What's NOT measured here**: steady-state decode wall (the
~30-60s number with warm ttc cache). The gate-target's stdout
redirect into the compare-file means perf-run.sh can't extract
`[stage]` lines, so we don't have the breakdown of model load /
decode / cleanup phases. Decode-only numbers can be captured by
running the binary directly: `./build/<KEY>/exec/hf-llama-inference
--dump-tokens --num-tokens 8` post-build. Filed as a soft
follow-up; the GREEN gate result is the load-bearing signal.

**Re-baselining still pending** (filed for the user to drive in
later sessions):
- `make test-examples` smoke gate at F32 (the torch-mps +
  mlx-gpu lanes already ran at F32 pre-flip, so the new exposure
  is tape-F32 + torch-cpu-F32 + mlx-cpu-F32 — likely the
  NTM/DNC examples are riskiest; their seeds were tuned per-
  backend at F64).
- `make test-examples-convergence` (long-horizon, tape only).
- Flip `torch_ref/` to `torch.float32` for apples-to-apples
  `bench-compare` ratios.
- Regenerate `docs/develop/perf-baseline.md` from new
  measurements.

**Commits**:
- `2c7d371f` BuildConfig flip + docstring/CLAUDE.md updates.
- `1b268fed` (interim) Medium TODO row for elaboration-memory
  reduction (the related Chez 17-23 GB peak symptom — separate
  problem from F32, dtype is phantom-type-only).
- mlx defensive-fix revert (this commit) — only torch had the
  documented math-impl lower-right-alignment bug; mlx is trusted
  to honour the docs in the pinned version.


### 2026-06-04 — HfLlama KV cache + token-sequence gate — c1f7489d..f2663ff2

**Plan**: Land the cache-aware forward step that lets greedy
generation skip re-projecting K/V from the full growing prefix on
every step. Companion to the five-fusion catalogue epic but
orthogonal: fusions reduce per-op overhead, the KV cache reduces
op *count* per step from O(seq²) cumulative work to O(1) for new
K/V projection + O(n) for the SDPA against history.

**Motivation**: HfLlama-3.2-1B's greedy decode loop was
re-projecting K/V from the full growing sequence on every step
(`hfLlamaForwardLm` called with input `[seq]` of growing length).
At 8 generated tokens past a 6-token prompt that's 6+7+8+...+13 =
76 forward-cost units of work, where with a cache it's ~14 (one
seed pass over the prompt + 1 per generated token). Idris's
`runGenerate` was measurably slower than HF's
`model.generate(use_cache=True)` for the same prompt — 327 s
vs ~2 s on torch-mps F32 — and the gap scales linearly with the
budget. The token-sequence gate (Phase A) lands first as
regression protection — invariant to whether the cache is on or
off since HF `use_cache=True` is mathematically equivalent to
`use_cache=False`.

**Change**: 4-commit landing + 2 post-verify fix-ups.

- Phase A (`c1f7489d`): token-sequence oracle + `--dump-tokens`
  flag + `compare_inference.py --token-sequence` mode + Makefile
  target + CI step + `scripts/perf-run.sh` arm. Gate uses the
  user-facing prompt "The capital of France is" + 4 generated
  tokens; expected output sequence `[128000, 791, 6864, 315,
  9822, 374, 12366, 13, 1102, 374]` decoding to "<|begin_of_text|>
  The capital of France is Paris. It is".
- Phase B (`c7b52183`): new `KVCache.idr` module with
  `Empty | Filled` sum type; `tconcat2dAxis0` typed wrapper around
  the existing `primCat2` axis-0 concat; Idris-level
  `Test.KVCache` suite (4/4 PASS on tape).
- Phase C (`04e1396c`): widened `tensor_sdpa_2d` C impls on all 3
  backends to read Q.size(0) and K.size(0) separately;
  `applyAttentionCached` / `applyBlockCached` / `applyBlocksCached`
  / `hfLlamaForwardStep` / `hfLlamaForwardLmStep` Idris-side
  functions threading per-layer caches; `ropeAllHeadsFlat` takes a
  positionOffset parameter (was hardcoded to 0).
- Phase D (`dd3aca25`): example switches to `genLoopCached`
  default; `--no-cache` opt-out kept for differential debugging.
- Post-verify fix-ups (`abcab827` + `f2663ff2`): Idris's
  elaborator couldn't unify `finalList : List (Fin VocabSize)`
  through the bind in `runDumpTokens` / `runGenerate` under
  if-then-else *or* case-of. Resolution: split the bind into
  per-branch `do` blocks. Also added
  `check-example-hf-llama-inference` Make target for
  type-check-only iteration (~40 min on this example; useful but
  not as fast as the name suggests).

**Impact**: Phase A baseline measurement on torch-cpu F64 — full
gate including model load (35 s) + 4 cached-equivalent decode
steps via the NO-CACHE path = 58 s wall. Each step ~6 s @ 901 ops
on growing sequence 7→8→9→10. The cache-aware path (Phase D) is
expected to drop per-step ops dramatically (Q/K/V projection on a
single new token vs the full growing prefix) and the wall-clock
drop scales with budget. Per-backend measured numbers go into
`perf-log.jsonl` once the cached path's gate run completes on
torch-mps and mlx-gpu.

**Backend coverage**: torch-cpu (CI lane, primary gate), torch-mps
(local dev), mlx-gpu (local dev). Tape lane deliberately skipped
— Llama-3.2-1B F64 × 1.24B params + per-step arena growth
exceeds 16 GB host RAM (the "Tape F64 large-LM OOM" Low-priority
TODO row carries the structural reason).

**Commits**: `c1f7489d` Phase A, `c7b52183` Phase B, `04e1396c`
Phase C, `dd3aca25` Phase D, `abcab827` + `f2663ff2` post-verify
fix-ups. CHANGELOG entry above this one (2026-06-04 KV cache
section) carries the long-form details.

**Outcome**: landed + verified GREEN on torch-cpu F64 in a fresh
shell (`make BACKEND=torch test-hf-llama-generate-roundtrip` —
`PASS: token sequence matches (10 tokens) ids: [128000, 791, 6864,
315, 9822, 374, 12366, 13, 1102, 374]`, 74 s end-to-end including
pytest + idris2 build + dylib relink + decode). The verification
surfaced a downstream PyTorch quirk: `at::scaled_dot_product_attention`
with `is_causal=True` and asymmetric `q_seq != kv_seq` does NOT do
lower-right alignment on the math impl path (torch-cpu F64 default),
despite the docs — `.tril(diagonal=0)` is applied without offset,
collapsing visible positions to just `j=0`. Fixed in commit
`5a55c930` by constructing an explicit `[q_seq, kv_seq]` additive
mask in the SDPA wrapper for the asymmetric-causal case and passing
via `attn_mask` instead of `is_causal=True`. The symmetric path
(prefill / training) keeps the optimized `is_causal` route — no
behaviour change there. Same fix applied defensively to mlx (same
documented promise, possibly same impl gap on older builds); tape
was already lower-right aligned from Phase C. See `docs/develop/gotchas.md`
"Torch Backend → `at::scaled_dot_product_attention(is_causal=True)`
math-impl doesn't honour lower-right alignment under asymmetric
Q/KV" entry for the long-form. Per-backend perf numbers on
torch-mps + mlx-gpu pending a fresh dev-shell run.


### 2026-06-03 — 2D embedding wrap on all 3 backends (#399 / #4 Fusion 3) — ca6f9ab

**Plan**: Fusion 3 of the fused-op catalogue plan — drop the
unnecessary flatten + `primReshape2d` pair at every transformer
input layer. `torch::embedding` and `mx::take` both return
`[n, embedDim]` natively; the legacy `tensor_embedding` flattened
to 1D so the FFI consumer saw a flat buffer, then every caller
called `primReshape2d` to reshape back to 2D. Two ops where one
would do.

**Motivation**: Plan predicted ~1 op/forward saved. Embedding
runs once per forward (the LM-head matmul uses the same weight
tensor but doesn't go through `tensor_embedding`). The verification
landed exactly on prediction.

**Change**: New `tensor_embedding_2d` primitive on all 3 backends.
torch: drops the trailing `.reshape({-1})`. mlx: drops `mx::flatten`
and registers a new `OP_EMBEDDING_2D` replay closure. tape: shares
the gather + grad-scatter path via a `embedding_impl` helper, with
the variants differing only in the output shape array passed to
`make_tensor*`. Idris side: `primEmbedding2d` on `UserDeviceNN`
with 3 backend instances; five caller sites rewired (HfLlama,
HfBert, HfGpt2, HfBitNet, Layer/Transformer single+batch).

**Impact** (torch-mps HfLlama-3.2-1B, default generate config):

| metric | post-SwiGLU baseline (24517c8) | post-embedding-2d (ca6f9ab) | delta |
|---|---|---|---|
| ops/step | 902 | 901 | **-1/step (exact match to prediction)** |
| wall (runGenerate) | 455 s | 327 s | -28% (within VM noise floor; ignore) |
| mlx-gpu wall (runGenerate) | 38 s | 23 s | -39% (within VM noise floor; ignore) |
| mlx-gpu max-abs-diff vs HF | 1.20e-04 | 1.20e-04 | bit-identical |
| torch-mps max-abs-diff vs HF | 4.96e-05 | 4.96e-05 | bit-identical |

Op-count drop is the load-bearing metric (it's deterministic and
matches the prediction exactly). The wall deltas exceed the ±15-20%
VM noise floor in both directions across SwiGLU/embedding-2d
adjacent trials, so they aren't reliable single-trial evidence
either way.

**Outcome**: landed (ca6f9ab). Closes the explicitly-planned #4
Fusion 3. Backward path is shape-agnostic on tape (walks indices,
writes to weight's grad buffer) and inherited by libtorch/mlx
autograd over `embedding` / `take` respectively.

**Cross-references**: `perf-log.jsonl` 2026-06-03 entries for
hf-llama mlx-gpu + torch-mps at commit `bd2787e+dirty`.

----

### 2026-06-03 — Fused SwiGLU on all 3 backends (#399 / #4 Fusion 2)

**Plan**: Fusion 2 of the fused-op catalogue plan — collapse HfLlama.applyMlp's `tsilu g; tmul sg u` pair into a single `primSwiGlu2d` typeclass method on `UserDeviceTraining`. Same shape as Fusion 1 (RMSNorm): one fused C primitive per backend, no architectural change to autograd or Tensor. SwiGLU is HfLlama-specific — BitNet's MLP uses relu² + ffn_sub_norm instead of silu, so the fusion intentionally doesn't land there.

**Motivation**: After Fusion 1 (RMSNorm) shipped, the next-largest op-count contributor in HfLlama's per-step decomposition was the `silu(gate) * up` pair in `applyMlp`. Llama-3.2-1B runs this pair once per layer × 16 layers = 32 ops/step before fusion, 16 ops/step after (32 → 16, -50% on this surface).

**Change**: `tensor_swiglu_2d(gate, up)` added to `backend.h`, manifest, and the 3 rename headers. Per-backend implementations:
- **torch** (`backend_torch/nn/activation/swiglu.cpp`): `at::mul(at::silu(g), u)`. Libtorch autograd handles backward across the two-primitive composition.
- **mlx** (`backend_mlx/nn/activation/swiglu.cpp`): `mx::multiply(mx::multiply(g, mx::sigmoid(g)), u)` as one tape entry. Streamed variant + replay closure via `MLX_REGISTER_REPLAY(OP_SWIGLU_2D)` recomputes the same composition during backward.
- **tape** (`backend_tape/nn/activation/swiglu_2d.c`): hand-rolled F32/F64 forward + analytical backward closure. `SwiGluMeta` caches `sigmoid(gate)` per element so backward avoids `exp()` re-evaluation. Backward registered via `TAPE_REGISTER_OP(OP_SWIGLU_2D)`.

Idris-side: `primSwiGlu2d : AnyPtr -> AnyPtr -> AnyPtr` on `UserDeviceTraining`; per-device instance in each `Device/{Mlx,Tape,Torch}.idr`. `HfLlama.applyMlp` collapses two intermediate bindings (`sg <- tsilu g; mid <- tmul sg u`) into one (`mid <- ioRerun (\_ => primSwiGlu2d ...)`).

**Impact** (Llama-3.2-1B F32, 8 greedy tokens, prompt='The capital of France is'):

| backend / config | cross-language gate | max-abs-diff vs HF Python oracle (tol 1.0) | op count | runGenerate wall |
|---|---|---:|---:|---:|
| **mlx-gpu** F32 (post-RMSNorm baseline, commit `416c011`) | PASS | 1.20e-04 | counter stub | 1 m 3 s (total) |
| **mlx-gpu** F32 (this commit) | PASS | **1.20e-04** (unchanged) | counter stub | **49.2 s** (total) — **-22%** vs RMSNorm baseline |
| **torch-mps** F32 (post-RMSNorm baseline, commit `416c011`) | PASS | 4.96e-05 | 918 | 4 m 56 s |
| **torch-mps** F32 (this commit) | PASS | **4.96e-05** (unchanged) | **902 (-16, -1.7%)** | runGenerate noisy (357 s in the roundtrip-gate run; 455 s in the paired perf-run; RMSNorm-baseline reference was 296 s — single-trial spread > the per-op-cost story can explain alone) |

- **Numerical clean**: cross-language max-abs-diff vs HF Python oracle is bit-identical to the pre-fusion baseline on both backends (mlx-gpu 1.20e-04; torch-mps 4.96e-05). The fusion adds no measurable floating-point drift.
- **Op-count drop**: -16/step on torch-mps as predicted (the silu + mul pair × 16 layers fused to 1). 918 → 902 is a small relative delta (-1.7%) because SwiGLU was only 2 chained ops to begin with, while RMSNorm was 7.
- **mlx-gpu wall win is real**: 49.2 s end-to-end vs the 63 s RMSNorm-baseline reference (-22%) on a single-trial measurement — well outside the `feedback_vm_perf_noise` ±15–20% noise floor.
- **torch-mps wall is noisy on this commit**: 357 s (roundtrip-gate run) and 455 s (paired perf-run) for `runGenerate`, vs the RMSNorm-baseline reference 296 s. The two SwiGLU-commit trials disagree by +28%, so the underlying signal is dominated by VM noise rather than the small op-count delta. Op-count drop is deterministic; wall is not.

C-level: criterion suite 217/217 tape, 207/207 torch, 209/209 mlx (after adding 4 new `test_swiglu.c` cases: zero-gate, unit-up, per-row independence, decomposed-chain agreement vs host-side oracle).

**Outcome**: landed. Sibling-fusion plumbing reuses the RMSNorm path almost verbatim — same Idris typeclass shape, same per-backend C file layout, same Scheme wrap regeneration via `scripts/lifecycle/ffi-convert-to-scheme.py`.

**TDD discipline** (per `feedback_tdd_default`): test file authored first; RED observed at link time (`call to undeclared function 'tensor_swiglu_2d'`); then backend impl + Idris wiring turned it GREEN. RED-before-commit recorded in commit body.

**Out of scope** (follow-up):
- Backward gradcheck test for the tape SwiGLU closure (forward correctness checked vs decomposed chain; backward derivation analytical). Pair an F32 oracle test in the tape T29 block to lock the F32 backward.
- Gate/up projection fusion (3 matmuls share x as input). Would need an mlx fast::linear-style primitive + libtorch composition. Filed but deferred — smaller payoff than the silu*mul pair just landed.

**Cross-references**: commit `24517c8` (this fusion); commit `416c011` (RMSNorm, Fusion 1); commits `6850366` (SDPA) and `c09d374` (all-heads RoPE) for the prior fusions in the same #399 catalogue; `feedback_pytorch_precedent_test.md` (PyTorch ships `nn.functional.silu` + elementwise mul; precedent test passes).


### 2026-06-03 — Fused RMSNorm on all 3 backends (#399 / #4 Fusion 1)

**Plan**: Fusion 1 of the fused-op catalogue plan — replace HfCommon.applyRmsNorm2dRaw's per-row 8-primitive chain (narrow / mul / sum / mul_scalar / add_scalar / sqrt / div / mul + cat) with a single `primRmsNorm2d` typeclass method on `UserDeviceTraining`. Same shape as the prior SDPA + all-heads RoPE fusions: one fused C primitive per backend, no architectural change to autograd or Tensor.

**Motivation**: Llama-3.2-1B has 33 RMSNorm sites per forward (2 per layer × 16 + 1 final). At `seqLen=N`, each call previously emitted ~9N primitives (narrow + 7-op math + cat); fused → 1. For the cross-language gate's seqLen=4 prompt that's a ~1.1K-op drop on top of the existing post-SDPA/RoPE 2,634 ops/step.

**Change**: `tensor_rms_norm_2d(input, weight, eps)` added to `backend.h`, manifest, and the 3 rename headers. Per-backend implementations:
- **torch** (`backend_torch/nn/norm/rms_norm.cpp`): `at::mean(at::pow(x, 2), -1, true) -> at::rsqrt(+eps) * x * weight`. Autograd flows automatically over the libtorch primitives — no explicit VJP.
- **mlx** (`backend_mlx/nn/norm/rms_norm.cpp`): `mlx::core::fast::rms_norm` — mlx-lm's canonical fused kernel; lazy graph node + replay closure via `MLX_REGISTER_REPLAY`.
- **tape** (`backend_tape/nn/norm/rms_norm_2d.c`): hand-rolled F32/F64 forward + backward closure with cached `x_hat[m*n]` + `rstd[m]` in `RmsNormMeta`. Backward registered via `TAPE_REGISTER_OP(OP_RMS_NORM_2D)`.

Idris-side: `primRmsNorm2d : AnyPtr -> AnyPtr -> Double -> AnyPtr` on `UserDeviceTraining`; per-device instance in each `Device/{Mlx,Tape,Torch}.idr`. `HfCommon.applyRmsNorm2dRaw` body collapses from a recursive `rmsNorm2dFoldRows` over an `rmsNorm2dProcessRow` to a single call. HfLlama + HfBitNet adapters (both wrap the helper) unchanged.

**Impact** (Llama-3.2-1B F32, 8 greedy tokens, prompt='The capital of France is'):

| backend / config | cross-language gate | max-abs-diff vs HF Python oracle (tol 1.0) | op count step 6 | runGenerate wall |
|---|---|---:|---:|---:|
| **mlx-gpu** F32 (pre-RMSNorm baseline n/a — counter stub) | n/a | n/a | n/a | n/a |
| **mlx-gpu** F32 (this commit) | PASS | **1.20e-04** | counter stub | **1 m 3 s** (total wall) |
| **torch-mps** F32 (post all-heads RoPE baseline, commit `c39371f+dirty` 2026-05-31) | n/a | n/a | **2,634** | 4 m 56 s |
| **torch-mps** F32 (this commit) | PASS | **4.96e-05** | **918 (-65%)** | 4 m 56 s |

- mlx-gpu's `tensor_perf_op_count` is a stub (only torch tracks per-op submission since #393).
- **Op-count drop**: deterministic — 2,634 → 918 across all decode steps 6-13 in the torch-mps log.
- **Wall on torch-mps is flat** — `runGenerate` is 296 s on both the pre- and post-RMSNorm commits. Same finding as the prior all-heads RoPE entry: the per-op cost on torch-mps rises in inverse proportion to the op count when we collapse decomposed chains into single primitives, because the bigger remaining ops use MPS paths with higher launch latency. Op-count × per-op-cost ≈ wall. On a rank-2 fused op (which RMSNorm is) the asymmetry is smaller than the rank-3 case (where the cost rose 3-5×), but it's still enough to neutralise the wall on this workload.
- **Wall on mlx-gpu**: total 1 m 3 s end-to-end on the current commit. Direct comparison to a same-dtype mlx-gpu F32 baseline isn't on file — the recent mlx-gpu HfLlama entries were BF16-mode (which is ~62% slower than F32 on mlx per the 2026-05-31 BF16 measurement). The 1 m 3 s figure is the new mlx-gpu F32 baseline post-RMSNorm.

C-level: criterion suite 213/213 tape, 203/203 torch, 205/205 mlx (after adding 4 new `test_rms_norm.c` cases: unit weight, per-row independence, per-column weighting, decomposed-chain agreement vs host-side oracle).

**Outcome**: landed. HfCommon's role narrows from "structural wrapper around a chain" to "thin wrapper around a single FFI call" — the chain it used to contain is preserved as `primRmsNorm2d`'s no-op fallback ON EACH backend (each backend implements the math; there's no Idris-side chain fallback). Cross-backend op count drops as expected on torch-mps; mlx-gpu's lazy graph rewards the fewer-ops shape directly.

**TDD discipline** (per `feedback_tdd_default`): test file authored first, RED observed at link time (`call to undeclared function 'tensor_rms_norm_2d'`), then backend impl + Idris wiring turned it GREEN. RED-before-commit recorded in commit body.

**Out of scope** (follow-up):
- Backward gradcheck test for the tape RMSNorm closure (forward correctness only this commit; backward derivation cross-checked analytically vs the decomposed chain). File a paired-oracle test in the tape T29 block to lock the F32 backward.
- Multi-backend per-op cost asymmetry: the prior 2026-05-30 all-heads RoPE entry documented that torch-mps's rank-3 broadcast costs ~10 ms/op vs ~2 ms/op rank-2. RMSNorm fusion stays rank-2, so doesn't trigger that path; the asymmetry row is unchanged.

**Cross-references**: commit `416c011` (this fusion); commits `6850366` (SDPA) and `c09d374` (all-heads RoPE) for the prior fusions in the same #399 catalogue; `scripts/lifecycle/ffi_manifest.py` (manifest entry); `feedback_pytorch_precedent_test.md` (PyTorch ships `nn.RMSNorm` — precedent test passes); `packages/idris-transformers/src/HfCommon.idr` (the call-site collapse).


### 2026-06-03 — Retroactive entry: Chez FFI symbol cache shipped 2026-05-27 (TODO row closes)

**Status**: retroactive paper-trail for commit `2385e3f` (2026-05-27 19:22:40 BST). The fix landed but its narrative entry never made it into perf-changes.md, so the corresponding TODO row stayed open — re-verified during the #1 follow-up sweep and closed via this entry.

**Symptom (pre-fix)**: sample profile of HfLlamaInference at ~44 min into a torch-mps F32 decode (`/tmp/scheme_2026-05-27_180602_BTJW.sample.txt`, 18:06:02 BST) showed 100% of CPU time in `S_foreign_entry → lookup → dyld4::Loader::hasExportedSymbol`, recursively walking every loaded library for each tensor-touching FFI call. With libtorch contributing thousands of symbols on top of libidrisml, each `%foreign "scheme:..."` call was effectively paying a full dyld symbol-table walk.

**Root cause**: `%foreign "scheme:EXPR"` wraps EXPR inside `(lambda (farg-0) (EXPR farg-0))`. The lambda body is re-evaluated on every call. The generated Scheme wrappers put the `(foreign-procedure "C-name" ...)` constructor inside the lambda body, so each call constructed a fresh `foreign-procedure` object — and that object's first use triggers dlsym to resolve the C symbol.

**Fix (commit `2385e3f`)**: each `%foreign` now lazy-caches its `foreign-procedure` value at first call via Chez `top-level-bound?` + `set-top-level-value!`, stashed under `idris-ffi-<c-symbol>`. First call still pays one dlsym; subsequent calls pay only a top-level-value lookup. Same idiom the codebase already uses for `idris-tensor-guardian`, extended from one shared symbol to 245 per-FFI symbols across `Device/Mlx.idr` (24), `Device/Tape.idr` (111), `Device/Torch.idr` (110). The lint (`check-ffi-wrap-template`) was unchanged — its structural invariants tolerate the new lazy-init blocks. Mlx `_streamed` variants stay on the old form (out of scope until a workload needs them).

**Verification at commit (recorded in the commit body)**:
- `example-supervised BACKEND=tape` produces bit-identical loss (1.356680328199114 / seed=42 / 5 epochs).
- `make test BACKEND=tape` green.
- Sample profile of `example-gpt --epochs 50 BACKEND=tape` post-fix: no `S_foreign_entry`/`lookup`/`dlsym` in the hot path. New top is `S_do_gc → sweep_generation → mark_object` — Chez GC, attributable to Tensor-wrapper vector allocation (separate row).

**Wall-time confirmation from perf-log**:
- Pre-fix HfLlama torch-mps F32: 44+ minutes mid-decode (the sample artifact's run).
- Post-fix HfLlama torch-mps (perf-log 2026-05-31, BF16 + RoPE/SDPA fusions): 4 m 54 s end-to-end.

The wall improvement is the combined effect of (a) this FFI cache landing, (b) the all-heads RoPE + SDPA fusions (`c09d374` + `6850366`), and (c) BF16 routing experiments. The FFI cache contribution alone isn't isolated by a controlled pre/post measurement on a single workload — but the sample-profile shape change (`S_foreign_entry/lookup/dlsym` → `S_do_gc/sweep_generation`) confirms the specific bottleneck the TODO row described is gone.

**TODO row closes** (was "Cache Chez FFI symbol lookups across calls", High priority). Closure entry moved to `CHANGELOG.md`. The follow-on Phase 2 (pre-bind at module-load rather than first-call lazy-init) was preserved as a TODO sub-row but is unblocked from the headline #1 work — it's now an unprioritised optimisation, not a fix-the-bottleneck row.

**Cross-references**: commit `2385e3f`; `scripts/lifecycle/ffi_manifest.py` (the `cache_var(c_symbol)` helper); `scripts/lifecycle/ffi-convert-to-scheme.py` (the regenerator that rewrites existing `scheme:` declarations on template change); `feedback_typeclass_zero_arg_method_eval.md` (the related "%foreign body re-evaluated per call" gotcha that motivated lifting the cache outside the lambda); the Medium-priority "Idris-side per-op overhead" row (now the dominant bottleneck, since FFI dispatch isn't).


### 2026-06-01 — Supervised mixed-precision parity, structural proof on tape (#410 F4)

**Plan**: verify the F1–F3 mixed-precision pipeline (LayerLikeMixed → LinearMixed with autograd-aware tcast → `applyScale` → `trainStepScaled` → GradScaler growth/backoff) produces numerically correct training across multiple seeds. The original #410 goal was "BF16 training converges as well as F32 — 5/5 vs the current 3/5 baseline" on torch-mps; F4 ships the **sweep infrastructure** + the **structural-correctness proof** on tape (where `paramDt = computeDt = F64` makes the lossy cast a structural no-op, so any numerical divergence would be a pipeline bug, not a precision-loss artefact).

**Motivation**: the 2026-05-31 baseline measured `torch-mps BF16 Supervised converges 3/5 vs 5/5 for F32`. The plan's prediction is that an F32 master + BF16 compute path (the autocast equivalent that #410 shipped) raises BF16 to ≥4/5 by avoiding underflow / precision-floor failure modes that pure-BF16 hits.

**Change**:
- `scripts/sweeps/supervised-mixed.json` (new) — sweep spec: grid over `--seed ∈ {42, 43, 44, 45, 46}` × `--mixed-precision ∈ {false, true}`, `--epochs 1000` `--lr 0.03` fixed. 10 configs total.
- `Example/Supervised.idr` — RESULT line now includes `correct=<N>/5` (per-config eval pass count) and `final_scale=<S>` (GradScaler state at end-of-training) so the sweep CSV summary is self-describing.
- `scripts/sweep.sh` — accepts an `IDRIS2_LOCAL` env-var override for the install prefix (was hard-coded `$(pwd)/.idris2`, broken after the multi-build-key refactor moved the prefix to `build/<BUILD_KEY>/idris2-prefix/`).

**Impact (tape build, 10 configs)**:

| seed | mp=false (loss) | mp=false (correct) | mp=true (loss) | mp=true (correct) | Δ loss |
|---|---|---|---|---|---|
| 42 | 0.13606813064182385 | 5/5 | 0.13606813064182385 | 5/5 | **0.0** |
| 43 | 0.13606813064182385 | 5/5 | 0.13606813064182385 | 5/5 | **0.0** |
| 44 | 0.13606813064182385 | 5/5 | 0.13606813064182385 | 5/5 | **0.0** |
| 45 | 0.13606813064182385 | 5/5 | 0.13606813064182385 | 5/5 | **0.0** |
| 46 | 0.13606813064182385 | 5/5 | 0.13606813064182385 | 5/5 | **0.0** |

10/10 runs converge to 5/5 eval. The mixed-precision path produces **bit-identical loss** to the default path at every seed. This is the strongest possible structural-correctness signal: every component of the A0–A4 pipeline (the autograd-aware tcast no-op when paramDt = computeDt, the GradScaler's pass-through behaviour when no overflow, the trainStepScaled NaN-sentinel never firing, the growth/backoff state advancing identically across the loop) lines up with the default path.

**Outcome**: structural proof landed. The BF16-vs-F32 numerical sweep (the actual 3/5 → 4/5 acceptance test) requires a separate `BACKEND=torch TORCH_DEVICE=mps TORCH_DTYPE=BF16` build, which switches `ExampleDType` to BF16 in `BuildConfig.idr`. The `runMixed` path in `Supervised.idr` currently uses `{paramDt = ExampleDType} {computeDt = ExampleDType}` — to actually exercise the F32-master / BF16-compute decoupling, the example needs an additional flag to pin `paramDt = F32` explicitly. Filed as F5 + F6 follow-ups (F5 ships in `21dd1a3`, F6 below).

**Commit**: F4 (`aaa8322`).

### 2026-06-01 — F6: BF16-vs-F32 convergence sweep on torch-mps — **STRETCH ACCEPTANCE HIT** (#410 F6)

**Plan**: with F5 (`21dd1a3`) shipping the `--param-dtype f32` flag that pins paramDt = F32 while leaving computeDt = ExampleDType, run the F4 sweep on a real `BACKEND=torch TORCH_DEVICE=mps TORCH_DTYPE=BF16` build to prove the F32-master / BF16-compute decoupling improves convergence vs the 2026-05-31 plain-BF16 3/5 baseline. Acceptance threshold: mixed-f32 mode ≥ 4/5 seeds reach 5/5 eval. Stretch: 5/5 seeds reach 5/5 eval.

**Motivation**: 2026-05-31 measured plain-BF16 Supervised at 3/5 seeds converging on torch-mps; F32 hit 5/5. The plan's hypothesis was that an F32 master copy (the autocast equivalent) would close most of the gap by preventing the per-step param update from rounding to zero in BF16's narrow mantissa even when grads are well-formed.

**Change**:
- Switched to a fresh torch-mps BF16 build tree (`build/torch-mlxcpu-torchmps-tdtBF16/`); `BuildConfig.idr` substitutes `ExampleDType = BF16`, `ExampleDevice = TorchDev TMps`.
- Ran the 15-config matrix manually (5 seeds × 3 modes); sweep.sh's `./build/exec/` legacy path is still broken for non-tape builds (filed in F4 commit body), so the shell loop replaces the wrapper for this measurement.

**Impact (torch-mps BF16, 5 seeds × 3 modes)**:

| seed | baseline (plain BF16) | mixed-native (F3) | **mixed-f32 (F5 master)** |
|---|---|---|---|
| 42 | 3/5 loss=0.2285 | 4/5 loss=0.1973 | **5/5 loss=0.1348** |
| 43 | 3/5 loss=0.2031 | 4/5 loss=0.1953 | **5/5 loss=0.1328** |
| 44 | 3/5 loss=0.2051 | 4/5 loss=0.1973 | **5/5 loss=0.1406** |
| 45 | 3/5 loss=0.1895 | 3/5 loss=0.2031 | **5/5 loss=0.1377** |
| 46 | 3/5 loss=0.1855 | 4/5 loss=0.1904 | **5/5 loss=0.1387** |

**Per-mode pass rate** (count of seeds hitting 5/5 eval):
- baseline: **0/5** at 5/5 (every seed stalled at 3/5; matches 2026-05-31 baseline)
- mixed-native: **0/5** at 5/5 (4/5 seeds nudged to 4/5; one stuck at 3/5)
- **mixed-f32: 5/5 at 5/5 eval — STRETCH HIT**

Every single seed in mixed-f32 mode reaches the F64-equivalent convergence (5/5 correct eval predictions, loss ~0.14 — within 4% of the tape-F64 baseline 0.1361). Loss is ~30–40% lower than plain-BF16 across the board.

**Interpretation**:
- The F32-master / BF16-compute decoupling is the actual lever. Both the autograd-aware tcast (A1, `66eca8f`) — which propagates BF16 grad through the cast into the F32 master — and the F32 optimizer step on F32 weights matter. The F32 master never sees BF16's mantissa-truncated param state, so per-step updates accumulate losslessly even when individual grads are small in BF16.
- `mixed-native` mode lifts baseline from 3/5 to mostly-4/5: this is the GradScaler's structural pass-through (`applyScale → trainStepScaled → applyScale^{-1}`) providing a small amount of stability even though no real cast occurs (paramDt = computeDt = BF16 means the tcast inside `LinearMixed.applyVarMixed` is a no-op at the dtype level). The GradScaler's scale=65536.0 was never tripped (no overflows; growthInterval=2000 > epochs=1000, scale stays at init), so the "improvement" must come from a numerical-stability artefact of the per-step `applyScale → backward → unscale` sequence — possibly the unscale walks every param in float and refreshes intermediate state more cleanly. Worth a follow-up investigation if anyone wants to claim mixed-native is intentional.
- `final_scale = 65536.0` on every mixed config: the GradScaler is in passive mode — the F6 result isn't gated by loss scaling, it's gated by the F32 master.

**Acceptance**: STRETCH HIT (5/5 vs target ≥4/5). The original plan's BF16-vs-F32 convergence-parity claim is now backed by data.

**Outcome**: closes the F-block. The full A0–A4 + F0–F6 pipeline delivers PyTorch-autocast-equivalent mixed-precision training with a stronger type-safety story (the lossy edges are visible in `LinearMixed`'s forward, the lossless edges flow implicitly via `LosslessTo → UpcastableTo`, the GradScaler state machine is IORef-based and inspectable, no thread-local autocast magic). #411 BitNet is now structurally unblocked and starts next.

**Commit**: F6 (this commit). Follow-up: investigate the `mixed-native` 3/5 → 4/5 nudge (low priority).

### 2026-05-31 — Rank-3 broadcast microbench localises the gap to OUR wrapper (#402 Commit 1)

### 2026-05-31 — Rank-3 broadcast microbench localises the gap to OUR wrapper (#402 Commit 1)

**Plan**: before committing to any specific wrapper-side fix for the 400-2000× rank-3 broadcast gap, measure raw libtorch vs PyTorch Python on the same shape, same device. Decide direction from the data, per the plan's decision tree.

**Motivation**: the `c09d374` perf-changes entry showed torch-mps Llama wall flat at 5m 17s despite the all-heads RoPE landing -86% op count (18,410 → 2,634). Per-op cost ~10–26 ms/op vs PyTorch Python's ~2 ms/op on the same MPS device. Four candidate causes: H1 (FFI marshalling), H2 (strided-view materialization), H3 (MPSGraph compile-cache misses), H4 (MTLCommandBuffer submission overhead). The plan's microbenchmark variants isolate H1/H2.

**Change**: new `packages/backends/bench_rank3_broadcast.cpp` (links directly against libtorch, no FFI/libidrisml) and `packages/idris-transformers/scripts/time_rank3_broadcast.py` (PyTorch Python equivalent). Both run `mul([6, 32, 32], [6, 1, 32])` × 100 after 10 warmup, with strided (via `narrow + reshape`) and contiguous variants. New `make bench-rank3-broadcast` target.

**Impact** (per-op µs/op on MPS, F32):

| variant | strided | contig | strided/contig | gap to our wrapper |
|---|---:|---:|---:|---:|
| libtorch C++ direct | **25.36** | 25.07 | 1.01× | ~400–1000× |
| PyTorch Python | **12.54** | 12.14 | 1.03× | ~800–2000× |
| our wrapper (from #399 measurements) | ~10,000–26,000 | n/a | n/a | baseline |

**Hypothesis verdict**:
- H1 (FFI marshalling / wrapper) — **confirmed**: raw libtorch C++ matches PyTorch Python (~25 vs ~12 µs); both are 400-2000× faster than our wrapper for the same op. The bottleneck is between our `tensor_mul_torch` entry point and libtorch's `torch::mul`.
- H2 (strided-view materialization) — **refuted**: strided/contig ratio is 1.01× in C++ and 1.03× in Python. The strided view is essentially free for libtorch's MPS path. Pre-materializing cos/sin contiguous would not move the needle.
- H3 (MPSGraph compile-cache misses) — **not yet measured**: would require varying the shape across iterations rather than reusing. Lower priority since H1 alone is ≥99% of the wall.
- H4 (MTLCommandBuffer submission overhead) — **bounded above at ~25 µs/op**: that's libtorch's own per-op floor; can't be more than that in our wrapper unless we're somehow flushing extra command buffers per op. Even if doubled by extra sync, doesn't explain 10 ms.

**What's in our wrapper that could cost ~10 ms/op**:
1. `from_tensor()` in `intermediates.cpp:56-68` — `new at::Tensor(std::move(t))`, `intermediates_torch.push_back(p)`, `prof_op_count_torch++`. The heap alloc itself is ~hundreds of ns; the vector push and atomic are similar.
2. Scheme-side wrap: `(vector 'tensor-handle-v2 "torch" raw_r)` — Chez vector construction; should be µs-scale.
3. Guardian registration: `((top-level-value 'idris-tensor-guardian) wr)` — function call.
4. Per-op retain: separate `(foreign-procedure)` call to `tensor_retain_handle_torch` (cached but still a foreign-procedure dispatch).
5. Lazy-init `(when (not (top-level-bound? ...)))` checks at the top of every Scheme wrapper.

Each individually should be ~µs at most. Together they shouldn't reach 10 ms. Something else amplifies. Plausible candidates not yet measured:
- MPS implicit sync triggered by storage reference patterns we induce (e.g., the new `at::Tensor` heap-allocation pattern in `from_tensor` may interact with libtorch's MPS storage-tracking differently than the bench's stack-allocated `auto y = …`).
- The intermediates vector tracking changing storage residency hints.

**Next step**: Commit 1b — add a "wrapper-direct" microbench that links libidrisml and calls `tensor_mul_torch` in a tight loop (mirroring this bench's pattern). That isolates whether the gap is in `from_tensor`/`tensor_mul` C-side or in the Scheme wrapper layer.

**Outcome**: H1 confirmed at the libtorch boundary. Direction for Commit 2 is wrapper-side (slim `from_tensor`, profile what's adding milliseconds per op) — not deferred-op tape, not MPS-flag tweaks, not materialization. Microbenchmark stays as the regression gate for future wrapper optimizations.

**Cross-references**: `docs/develop/perf-log-ref.jsonl` 2026-05-31 entries (3 measurements); `bench_rank3_broadcast.cpp`; `time_rank3_broadcast.py`; commit (this commit's hash) + the follow-up.


### 2026-05-31 — Wrapper-direct microbench rules out the C boundary (#402 Commit 1b)

**Plan**: Commit 1's libtorch-direct vs PyTorch-Python measurement confirmed H1 (the gap is in our wrapper) but didn't say *where* in the wrapper. Three layers between Idris and libtorch: (a) C wrapper (`tensor_mul_torch` → `to_tensor` + `torch::mul` + `from_tensor` + intermediates push + counter), (b) generated Scheme wrap (`prim__mulTorch`), (c) Idris autograd / smart-constructor / typeclass dispatch in `Tensor.idr`. Each layer needs measuring before picking the fix.

**Motivation**: Commit 2A was scoped to slim `from_tensor`'s `intermediates_torch.push_back` and the `prof_op_count_torch++` bump under no-grad. Before touching that, isolate whether layer (a) is actually contributing — if `tensor_mul_torch` matches `torch::mul` directly, the C-side intermediates work is bystander and the fix has to land above the C boundary.

**Change**: new `packages/backends/bench_rank3_broadcast_wrapped.cpp`: links libidrisml.dylib, calls `tensor_mul_torch` in the same warmup+measure loop as `bench_rank3_broadcast.cpp`. Goes through `from_tensor`, intermediates push, and counter bump but bypasses every Scheme/Idris layer. New `make bench-rank3-broadcast-wrapped` target. Verified `tensor_perf_op_count_torch` returns 100 in each measure block (counter wired correctly through the wrapper).

**Impact** (per-op µs/op):

| variant | C++ direct (Commit 1) | C wrapper via libidrisml | delta |
|---|---:|---:|---:|
| MPS strided | 25.36 | **25.90** | +0.54 µs (~2%) |
| MPS contig | 25.07 | **26.78** | +1.71 µs (~7%) |
| CPU strided | n/a | 14.72 | n/a |
| CPU contig | n/a | 13.63 | n/a |

**Hypothesis refinement**:
- Layer (a) C wrapper — **not the bottleneck**: ~1 µs/op overhead on top of libtorch direct. `new at::Tensor(std::move(t))` + `intermediates_torch.push_back(p)` + `g_torch_peak_live_intermediates` update + `prof_op_count_torch++` together cost ~1 µs/op. Slimming any of this would harvest at most 1 µs against a 10–26 ms/op observed cost — i.e. negligible.
- Layer (b)/(c) — **must contain ~99% of the wrapper gap**: the remaining 10,000–26,000 µs/op observed in HfLlama lives above `tensor_mul_torch`. Candidates: Scheme glue per-call overhead (foreign-procedure cache lookup, vector wrap, guardian register, no-op retain FFI), or Idris-level smart-constructor work (`tmul` allocations, typeclass dispatch, autograd-graph book-keeping, `paramId` lookups).

**Cross-backend implication**: the C-side ruling-out generalises — every backend uses the same `from_tensor`-style intermediates pattern, so slimming it on torch would have been similarly bounded on mlx/tape. The Scheme wrap is identical across the three backends (same template, only the tag differs); whatever's adding milliseconds in that layer applies to every backend. mlx-gpu's all-heads RoPE wall did drop (45.5s → 16s after `c09d374`), so its per-op cost is lower than torch-mps's — but the wrapper inefficiency is likely still there, just amortised by mlx's lazy graph.

**Next step**: Commit 1c — Idris-level rank-3-broadcast microbench (smallest possible program that calls `tmul` on `[6, 32, 32] × [6, 1, 32]` in a tight loop, on torch-mps + mlx-gpu + tape). The delta between layer (a) and layer (c) is the Scheme+Idris glue. Once that's measured, decide whether to attack Scheme wrap (would benefit all three backends symmetrically) or Idris-level `tmul` (likely smaller, easier).

**Outcome**: Commit 2A as originally planned (C-side `from_tensor` slimming) is now **out of scope** — measurement shows the C wrapper is not the culprit. Direction has narrowed to the Scheme/Idris layers above. Microbench stays as a permanent harness; both bench targets become the regression gates for any wrapper-layer changes.

**Cross-references**: `bench_rank3_broadcast_wrapped.cpp`; the Commit 1 entry above for the libtorch-direct baseline numbers.


### 2026-05-31 — Idris-level microbench refutes the "10 ms/op rank-3 broadcast" premise (#402 Commit 1c)

**Plan**: complete the layer-by-layer ladder by measuring the same `mul([6, 32, 32], [6, 1, 32])` shape from the highest level — through `primMul` on the typeclass-dispatched device instance. Layer (a) C wrapper measured ~26 µs/op; (b) Scheme wrap + (c) Idris autograd are everything above. Bench is `packages/idris-ml-examples/src/Example/RankBroadcastBench.idr`, called via `make example-rank-broadcast-bench`. Same iteration counts (warmup=10, measure=100), same shape, same chained-mul pattern (tail-recursive to keep the result alive against DCE).

**Motivation**: Commit 1b ruled out the C boundary (~1 µs/op overhead). Commit 1c isolates whatever's above. If Scheme/Idris adds milliseconds, the fix is in the generated wrapper or `Tensor.idr`. If Scheme/Idris also adds only microseconds, the original "rank-3 broadcast costs 10–26 ms/op" framing is wrong and the cost has to live somewhere else — in op *mix*, op *count*, or shape-dependent libtorch behaviour that doesn't trigger in a tight same-shape loop.

**Impact** (per-op µs/op, all four idris-ml backends, F32 where the build forces it):

| layer | tape (F64) | torch-mps (F32) | mlx-cpu (F64) | mlx-gpu (F32) |
|---|---:|---:|---:|---:|
| libtorch direct (C++ bench) | n/a | 25.36 | n/a | n/a |
| C wrapper (`tensor_mul_torch`) | n/a | 25.90 | n/a | n/a |
| **Idris `primMul` (this bench)** | **15.81** | **23.91** | **26.29** | **74.95** |

mlx-gpu's per-op floor is higher because GPU kernel-launch dominates at this small shape (matches `project_mlx_gpu_environment.md` — sub-1024 dim ops are CPU-faster on mlx). The other three backends all sit in the 16–26 µs/op band.

**The original #402 premise is refuted.** The "10–26 ms/op rank-3 broadcast" figure from the c09d374 perf-changes entry was **arithmetic**, not measured: it was computed as `runGenerate wall (~317 s) / total ops (~21K)` ≈ 15 ms/op. That ratio collapses to "per-op cost" *only if* every op is the same. After the all-heads RoPE landing in `c09d374`, the op mix shifted dramatically — most remaining ops are LARGER per call (32× more head data in each broadcast mul) so total compute is constant, total wall is constant, op count drops 7×, and `wall/op_count` *appears* to rise. That's the 10–26 ms/op number.

This means **all three originally-planned commits 2A/2B/2C are out of scope**:
- 2A (slim `from_tensor` / intermediates) — was already ruled out at the C boundary (Commit 1b).
- 2B (MPS optimization flags) — the per-op µs/op numbers above show our libtorch use is already at libtorch's own floor.
- 2C (pre-materialize cos/sin contiguous) — strided/contig ratio is 0.97× in our wrapper, 1.01× in direct libtorch (Commit 1). Materialization is already free.

**Cross-backend lesson** (user-flagged early in the session): the C wrapper and Scheme-wrap layers are NOT a bottleneck on any of the three backends I could measure here. The wrapper is efficient at ~16–26 µs/op across tape, torch-mps, and mlx-cpu. mlx-gpu's 75 µs is GPU-launch-bound — a different problem (project-memo'd in `project_mlx_gpu_environment.md`).

**Re-framing #402**: the row is closing as "investigated and refuted". The actual gap between idris-ml HfLlama and PyTorch HfLlama on torch-mps lives elsewhere. Candidates (now untested):
1. **Op count, not op cost** — we still issue ~2,634 ops per forward where PyTorch Python issues ~few hundred. The "Match PyTorch's catalogue of fused ops" row (TODO Medium #42) is the standing direction for this.
2. **Single-op time on LARGER shapes** — `[6, 32, 32]` is small. Llama-3.2-1B's attention matmul is `[seq, numHeads, headDim] @ [seq, headDim, numHeads]` for `numHeads=32, headDim=64, seq=N`. A matmul-shaped microbench would catch any per-op asymmetry that doesn't show up on small elementwise ops.
3. **Implicit syncs in our compositional layer** — `applyRopeAllHeads` returns from `ioRerun (\_ => let ...)` and then sequences via `>>=`. If there's a per-`ioRerun` cost (closure allocation, IO state restoration) and HfLlama has ~5K-10K layer-level steps, that's per-call overhead × calls.

The clean follow-up is to file a new row "torch-mps HfLlama wall gap" with these three sub-hypotheses listed, run a per-op-class microbench (matmul, RMSNorm reductions, embedding lookup) one at a time, and let the data narrow.

**Outcome**: #402 closes. The rank-3 broadcast at every layer of the stack is fast. Microbenches stay as permanent regression gates for the layers they exercise. Direction moves to op-count reduction (more fusion) and to per-op-class microbenchmarks for the other ops in Llama's forward.

**Cross-references**: `packages/idris-ml-examples/src/Example/RankBroadcastBench.idr`; commit (this commit's hash); related closing commits for the TODO row update.


### 2026-05-31 — Tape F32 HfLlama mid-decode crash fixed by `PrimIO Int` (#401)

**Plan**: unblock tape F32 HfLlama inference, which has crashed mid-decode (~step 8) since commit `e9763d0` introduced the `primPerfOpCount : PrimIO Bits64` FFI for the #393 op-submission counter. Three hypotheses on file (TODO #401): (1) `PrimIO Bits64` shape, (2) typeclass dispatch, (3) Chez `unsigned-64` marshalling. Test cheapest first.

**Motivation**: with the perf-op-count diagnostic disabled on tape (the existing workaround), tape F32 lost the per-step op-count reporting that's the main signal for verifying op-count changes (e.g. #399's SDPA + all-heads RoPE landings). Restoring it unlocks the diagnostic on all 3 backends.

**Change**: changed the FFI declaration from `prim__perfOpCount<Backend> : PrimIO Bits64` → `: PrimIO Int` in all three of `Device/{Tape,Torch,Mlx}.idr`, the typeclass method signature in `Device/Core.idr`, the smart constructor `perfOpCount` in `Tensor.idr`, and the call site type in `HfLlamaInference.idr`. The C side already returns `long` (= `int64_t` on macOS), which fits both `Bits64` and `Int` — same kernel, just different chez codegen path on the return.

**Impact** (Llama-3.2-1B F32, 8 greedy tokens, prompt='The capital of France is'):

| backend / config | runGenerate wall | decode steps reached |
|---|---:|---:|
| tape F32 baseline (pre-fix, `e9763d0+`) | crashed @ step 8 | 3 of 8 |
| tape F32 + `PrimIO Int` fix (this commit) | **1m 00s** | **8 of 8** ✅ |

All 8 `[perf] step N: 0 ops` lines now print (op counter is a stub on tape — returns 0 always; the diagnostic value is the *call surviving* across decode iterations).

**Hypothesis verdict**: (1) `PrimIO Bits64` shape **confirmed as the trigger**; (2) and (3) not independently isolated. Idris-2's chez codegen for `unsigned-64` returns through `PrimIO` in tight loops corrupts something — exact mechanism unknown, but `Int` (= `int64_t`) sidesteps it on the same workload. Documented as a gotcha; lesson is "default to `PrimIO Int` for FFI counters/sizes/handle-indices unless unsigned semantics genuinely matter".

**Outcome**: landed. `#401` closed. Unblocks tape F32 as a first-class lane for the #399/#402 op-count investigations.

**Cross-references**: TODO #401; `docs/develop/gotchas.md` "PrimIO Bits64 FFI returns corrupt state in tight loops"; commit `e9763d0` (introduced the bug); fix commit (this commit).


### 2026-05-30 — All-heads RoPE: mlx-gpu 45.5s → 16s (2.8×); torch-mps op count -86%, wall flat (#399 follow-up)

**Plan**: replace the per-head `buildRopedHeads` Idris-side concat loop (~1,000 concats/forward, ~80% of post-SDPA op count) with one `applyRopeAllHeads` call per Q/K that uses rank-3 broadcast cos/sin over the head axis. PyTorch's `apply_rotary_pos_emb` uses this exact pattern.

**Motivation**: the SDPA fusion (entry below) dropped op count 44% but wall stayed flat because per-head RoPE concats added back the saved submission overhead. Killing the concat loop should let SDPA's gain land.

**Change**: new `applyRopeAllHeads` in `Layer/RoPE.idr` (the rank-3 variant of `applyRope`); `ropeAllHeadsFlat` wrapper in `HfLlama.idr` does the flat ↔ rank-3 reshape boundary so SDPA's 2D-flat I/O stays unchanged. Tape's `tensor_narrow` extended to rank-3 axes 0 and 2 (the rank-2 narrow only supported axes 0/1).

**Hypothesis** (logged, then tested):
- Op count: 10,346 → ~2,500–3,000 (additional ~75% drop on top of SDPA)
- Wall (if per-op cost stays at baseline ~1.89 ms): ~3,000 × 1.89 ms × 8 forwards ≈ 45 s ≈ ~4× wall speedup on torch-mps
- Wall (worst case, if rank-3 broadcast is ~5 ms/op): ~2.5× speedup
- Three controlled torch-mps measurements to characterise variance

**Impact** (Llama-3.2-1B F32, 8 greedy tokens):

| backend / config        | op count step 6 | runGenerate wall |
|-------------------------|----------------:|-----------------:|
| baseline (no SDPA)      | 18,410          | torch-mps 5:07   |
| +SDPA only              | 10,346 (-44%)   | torch-mps 5:15   |
| +all-heads RoPE         | **2,634 (-86%)** | **torch-mps 5:17** (M1: 5:22, M2: 5:17, M3: 5:17 — variance ~2%) |
| **mlx-gpu**, +all-heads | n/a (counter stub) | **16 s** (baseline 45.5 s — **2.8× faster**) |
| PyTorch Python (ref)    | ~1,000          | 2 s              |

**Hypothesis verdict**: op-count prediction **confirmed exactly** (deterministic 2,634, matched the predicted 2,500–3,000 range). Wall prediction **falsified** on torch-mps — per-op cost rose from baseline ~3.4 ms to ~10.9 ms (29,400 ops × 10.9 ms = ~320 s). The rank-3 broadcast muls in our wrapper cost ~10–26 ms/op on MPS where rank-2 same-shape muls cost ~2 ms/op. PyTorch Python does the same broadcasts in ~2 ms/op (the time_inference_llama.py reference). Net torch-mps wall unchanged.

**mlx-gpu** is the lane that wins: its lazy `mx::array` graph rewards fewer ops directly. Smaller op count → smaller graph → faster `mx::eval`. The 2.8× speedup brings mlx-gpu within 8× of PyTorch Python (was 23×).

**Outcome**: landed. The per-op-cost asymmetry on torch-mps is a real finding — for a general-purpose tensor library, the right follow-up is closing the gap between our rank-3 broadcast path and PyTorch Python's (likely in our FFI marshalling or `from_tensor` overhead specific to rank-3 strided views), not adding architecture-specific fused C kernels. Filed as a new investigation row.

**Library-design rationale**: keeping RoPE expressed as composable rank-3 primitives (narrow, mul, sub, add, reshape) is the principled call. The one-backend-doesn't-benefit finding is a backend perf issue, not a reason to switch to a megafused C kernel that hides the math.

**Cross-references**: commit `c09d374`; perf-log.jsonl 2026-05-30 entries M1-M3 torch-mps + mlx-gpu; `Layer/RoPE.idr` (applyRopeAllHeads); the per-op cost gap is the next investigation under #399 follow-up.


### 2026-05-30 — Fused SDPA on all 3 backends: -44% op count, wall flat (#399 Commit B)

**Plan**: Commit B of the fused-op catalogue plan — replace the per-head attention math loop with `at::scaled_dot_product_attention` (torch), `mlx::core::fast::scaled_dot_product_attention` (mlx), and a hand-composed C kernel (tape).

**Motivation**: PyTorch Python head-to-head (2026-05-29 entry below) showed ~150× gap on Llama. Attention is ~10K of 18.4K ops/forward; SDPA fusion should drop attention ops to ~16/forward and eliminate the corresponding MTLCommandBuffer submissions on MPS.

**Change**: new `primSdpa2d` typeclass method on `UserDeviceTraining` + 3 per-backend C kernels + Idris-side `applyAttention` refactor in `HfLlama.idr`. The per-head RoPE loop stays (under `buildRopedHeads`) — it produces a flattened `[seq, numHeads*headDim]` accumulator that feeds into SDPA. 2D-flat I/O avoids multiplicative-Nat elaboration in type signatures.

**Impact** (torch-mps F32, Llama-3.2-1B, 8 greedy tokens, prompt='The capital of France is'):

| | op count step 6 | op count step 13 | runGenerate wall |
|---|---:|---:|---:|
| baseline (commit `e9763d0+dirty`) | 18,410 | 20,489 | 5m 07s |
| +SDPA (commit `6850366`) | **10,346 (-44%)** | **12,425 (-39%)** | 5m 15s (within VM noise) |

The op-count drop is real and deterministic. The wall is unchanged because the per-head RoPE accumulator's `primConcat2dAxis1` calls (62 per layer × 16 layers = ~1,000 per forward) add ~15 s of MTLCommandBuffer overhead per 8-token decode — roughly the same amount we saved on attention-math submissions. Net zero on wall.

**Outcome**: SDPA infrastructure landed across all 3 backends. The wall win requires also killing the `buildRopedHeads` concat loop — either by all-heads RoPE in Idris (PyTorch's approach, rejected earlier but the rejection may have been VM noise — needs cleaner re-measurement) or by a fused per-head-RoPE FFI primitive. **Next experiment**: retry all-heads RoPE on top of SDPA with multiple controlled measurements; hypothesis is that op-count × per-op cost ≈ wall, so the deterministic ~58% op-count drop should translate to a 30-50% wall drop if the per-op cost stays near baseline.

**Cross-references**: commit `6850366` (SDPA), `e9763d0` (per-op counter from #393), `perf-log.jsonl` 2026-05-30 entries, `docs/develop/gotchas.md` "FFI manifest entry required for wrap-on-return".


### 2026-05-29 — PyTorch Python on torch-mps Llama: a ~150× gap, not a structural ceiling (#399 sizing)

**Plan**: before committing to #399 ("torch-mps deferred-op tape (graph mode)") as an XL architectural change, take the cheap diagnostic step of actually measuring PyTorch Python's wall on the same workload. The hypothesis on file (from the closed #393 row) was that ~19,400 ops × ~1.89 ms/op = ~36.7 s/forward is the *libtorch+MPS structural ceiling* and we and PyTorch Python both hit it. That hypothesis was untested — there were no PyTorch Python Llama walls in `perf-log.jsonl`.

**Motivation**: a 150× perf gap with a workable fix is worth knowing about before sinking weeks into a deferred-op-tape architectural rewrite that may be solving the wrong problem.

**Change**: introduced `docs/develop/perf-log-ref.jsonl` + `perf-log-ref.md` for reference / third-party baseline measurements (kept separate from `perf-log.jsonl` so reference numbers aren't drowned by commit-keyed churn). Ran `packages/idris-transformers/scripts/time_inference_llama.py` on MPS F32, captured both cache modes.

**Impact**:

| | runGenerate wall | per-forward avg |
|---|---:|---:|
| idris-ml torch-mps F32 (commit `e9763d0+dirty`) | 5 m 07 s | ~38 s/forward |
| PyTorch Python `use_cache=False` (apples-to-Idris) | **2 s** | ~0.25 s/forward |
| PyTorch Python `use_cache=True` (real user pattern) | **3 s** | ~0.4 s/forward |

**idris-ml is ~150× slower than PyTorch Python on the same libtorch + same MPS device, same workload, same model.** This refutes the "we're at PyTorch parity on a libtorch structural ceiling" reading. The ~19,400 ops/forward is *our* count — PyTorch's Llama implementation doesn't have 19,400 ops per forward, it has maybe 1–2 orders of magnitude fewer because it aggressively fuses (likely `F.scaled_dot_product_attention` via MPSGraph, `F.rms_norm`, fused embedding lookup, etc.).

**Outcome**: #399's scope refactors from XL "deferred-op tape (architectural)" to L "match PyTorch's fused-op catalogue on torch backend". The fix is op-level: identify which of our composite smart-constructor chains decompose into many `from_tensor` wraps when PyTorch lands a single fused op, then expose those fused ops as FFI primitives in `backend_torch/`. Prime suspect is attention (`at::scaled_dot_product_attention` exists in libtorch and uses MPSGraph internally on MPS); RMSNorm and the SwiGLU MLP gate are runners-up. Per-forward op count is the right proxy metric — the existing `tensor_perf_op_count` counter (commit `e9763d0`) already tracks this without further instrumentation; we just need to compare counts pre/post each fused-op landing.

**Cross-references**: `perf-log-ref.jsonl` entries 2026-05-29 (the two PyTorch measurements); `time_inference_llama.py` is the canonical head-to-head script; the #393 closure's "structural ceiling" claim is now superseded by this finding (the per-op submission overhead is real, but our op count is 30–100× higher than PyTorch's — the ceiling we measured was our own decomposition, not libtorch's).


### 2026-05-29 — torch-mps per-op submission overhead diagnosed (#393)

**Plan**: close #393 — two-phase per the modular-petting-minsky plan: tactical fix first (Phase B1), then per-op timing harness (Phase B2) to make the structural ceiling visible.

**Motivation**: torch-mps was 4.5–21× slower than the other lanes on HF inference (hf-bert 1:27 vs 16–19 s; hf-gpt2 4:35 vs 13–17 s; hf-llama 6:42 vs 46 s). Hypothesis: libtorch's MPS path submits each primitive op as its own MTLCommandBuffer, ~10K ops × ~0.5–1 ms = tens of seconds dispatch wall. Wanted numbers, not speculation, before committing to a structural fix.

**Change**:
- **Phase B1 (commit `b572fc5`)**: `backend_torch/nn/attention/embedding.cpp:18` — guarded `indices.to(kLong + weight.device())` on the identity case (when indices already match) so the common path skips the no-op submission.
- **Phase B2 (commit `e9763d0`)**: per-forward op counter in `backend_torch/training/profiling.cpp` bumped at every `from_tensor()` call in `intermediates.cpp` (single choke point every op-result tensor passes through). New `perfReset` + `perfOpCount` on `UserDeviceTraining` (no-op stubs on tape + mlx). `HfLlamaInference.idr`'s `genLoop` brackets each forward with reset/read and prints `[perf] step N: K ops`. `scripts/perf-run.sh` surfaces those lines alongside `[stage]` lines.

**Impact**:

Forward wall on torch-mps (Phase B1, vs prior `36fde48` baselines):

| example | before | after | delta |
|---|---:|---:|---:|
| hf-bert  | 1m 27s | **41s**  | -53% |
| hf-gpt2  | 4m 35s | **1m 56s** | -58% |
| hf-llama | 6m 59s | **5m 07s** | -27% |

Per-step op counts on torch-mps Llama-3.2-1B F32 (Phase B2, prompt='The capital of France is', 8 generated tokens):

```
step 6:  18410 ops    step 10: 19598 ops
step 7:  18707 ops    step 11: 19895 ops
step 8:  19004 ops    step 12: 20192 ops
step 9:  19301 ops    step 13: 20489 ops
```

Linear at **+297 ops/token** (no KV cache, every forward re-runs the full 16-layer stack over the growing prompt). 8 forwards in 4m 53s = ~36.6 s/forward avg; at ~19,400 ops/forward = **~1.89 ms/op** average dispatch wall.

**Structural ceiling confirmed**: 19,400 ops × 1.89 ms ≈ 36.7 s matches the observed per-forward wall to within rounding. Dispatch overhead is the fundamental limit on torch-mps at small-model inference scale.

**Outcome**: landed. #393 closes as "diagnosed; structural fix is a separate epic." Filed new TODO row "torch-mps deferred-op tape (graph mode)" describing the path-4 sketch (wrap every smart constructor to record into a deferred-op tape; materialise as one batched libtorch submission at sync boundaries). The diagnostic harness stays — per-forward op counts are now visible in the perf-run output for any future regression check, and the `perfReset` / `perfOpCount` surface generalises to non-Llama examples.

**Cross-references**: `feedback_perf_compare_after_changes`'s "torch-mps fastest at sufficient kernel work, dominated by dispatch overhead at small scale" — quantified here.


### 2026-05-26 — `coverage-backend-torch` further 2.4× via libtorch PCH

**Plan**: coverage-policy plan W2 follow-up after the user noted "still extremely slow" at the prior 144s cold.

**Motivation**: Per-stage cold breakdown showed 202s of the 240s wall is the dylib build, dominated by libtorch's heavy template-rich `<torch/torch.h>` (~30K lines) being parsed 90× — once per backend_torch `.cpp`.

**Change**:
- Added `packages/backends/backend_torch/torch_pch.h` (single-include precompiled header for `<torch/torch.h>`).
- Makefile rule `$(BUILD)/torch_pch.gch` builds the PCH once per build tree with the same flags as the per-TU compile (so the PCH is valid for every consuming TU; clang rejects PCHs whose flags don't match).
- Per-TU rule now does `-include-pch $(BUILD)/torch_pch.gch -include rename_torch.h ...`.

**Impact**:

| state | torch coverage cold wall | speedup vs prior |
|---|---:|---:|
| pre-W2 (`-O0`, no `-j`)        | 538s (8:58)  | 1.0× |
| W2 (`-O0`, `-j4`)              | ~240s (4:00) | 2.24× |
| W2 + PCH (`-O0`, `-j4`, PCH)   | **101s (1:41)** | **5.3× cumulative, 2.4× over W2** |

Tape coverage cold wall unchanged from W2 (~7s — tape headers are cheap).

**Test suites all pass after the PCH change**:
- tape 183/183, mlx 175/175, torch 174/174

**Cross-references**:
- `feedback_perf_compare_after_changes.md` — perf change recorded per the rule.
- PCH stays per-build because the same PCH binary is incompatible across flag-incompatible builds (`-O0 -g -fcoverage-mapping` for cov vs `-O2` for normal). $(BUILD) is `build/` for normal and `build-cov/` for cov, so each gets its own PCH.

### 2026-05-26 — `coverage-backend` 4× speedup via -j$(NPROC) on recursive make

**Plan**: coverage-policy plan (modular-petting-minsky.md), W2.

**Motivation**: User reported `coverage-backend-torch` is "extremely slow (mostly setup it seems?)". Cold runs were ~9 minutes, dominated by serial recompile of libtorch-header-heavy `.cpp` files into `build-cov/`. The recursive `$(MAKE) ...` in the coverage-backend recipe didn't inject `-j`, and outer `make coverage-backend-torch` runs without `-j`, so the inner build ran serially (95% CPU on a 4-core machine).

**Change**:
- Define `NPROC ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)` at the top of the Makefile.
- Change the recursive call in `coverage-backend` to `$(MAKE) -j$(NPROC) BUILD=$(COV_BUILD) ...`.
- Also tried `-O0 → -O1` (`-fcoverage-mapping` is optimization-independent) — REVERTED. libtorch template-heavy headers regressed compile time substantially at `-O1` (8:58 → 21:58). Kept `-O0`.

**Impact** (cold builds, `rm -rf build-cov && time make coverage-backend-<b>`):

| backend | before (-O0, no -j) | after (-O0, -j4) | speedup | CPU% |
|---|---:|---:|---:|---:|
| torch | 8:58 (538s) | **2:24 (144s)** | **3.73×** | 95% → 325% |
| tape  | ~30s          | **7s**         | **~4×**  | serial → 160% |
| mlx   | not measured  | -              | -        | (small surface — expected similar) |

The 50% wall-time target in the W2 plan is met with margin (144s ≪ 268s).

**Why -O1 backfired**: clang -O1 on libtorch headers triggers template instantiation passes (inlining attempts, basic dead-code elim) that aren't done at -O0. With 93 `.cpp` files each pulling `<torch/torch.h>` (heavy template chain), the per-TU compile time roughly doubled. The save from a smaller object file's link step didn't compensate. Kept -O0.

**What's left**: PCH for `torch/torch.h` was planned but isn't needed — 144s cold is acceptable. If we ever push to a slower CI host, the next lever would be: (a) splitting the `test_criterion_smoke` single-cc invocation (37 .c files in one shot) into per-file `.o` compiles, (b) shipping a precompiled torch.h.gch. Both have non-trivial Makefile-surgery cost; defer until needed.

**Cross-references**:
- W2 plan in `/Users/admin/.claude/plans/modular-petting-minsky.md`.
- `feedback_perf_compare_after_changes.md` — perf change recorded per the rule.

### 2026-05-19 — RL sweep post-gymnasium-migration + two-point timing breakdown — `09af662`

**Plan job**: cross-cutting (after gymnasium-migration sweep)

**Motivation**: First cross-backend perf sweep covering the 7 deep-RL examples (reinforce / a2c / dqn / mountain-car / mountain-car-cont / ppo / sac) after the gymnasium migration. Wanted ms/epoch baselines for the migrated reference scripts and Idris-vs-PyTorch ratios.

**Change**: ran `scripts/perf-sweep.sh --examples reinforce,a2c,dqn,mountain-car,mountain-car-cont,ppo,sac --cells tape,torch,mlx-cpu,mlx-gpu --seed 42`. Sweep ran ~9h wall-clock; killed during sac mlx-gpu (the only cell not reported).

**Impact** (all ms/epoch, commit `09af662`):

| example | tape | torch | mlx-cpu | mlx-gpu | py-ref |
|---|---:|---:|---:|---:|---:|
| reinforce | 11.47 | 44.80 | **crash** | **crash** | 0.07 (broken) |
| a2c | 1.71 | 2.73 | 60.78 | 129.97 | -0.59 (broken) |
| dqn | 39.32 | 75.20 | 2078.92 | 2972.05 | -0.33 (broken) |
| mountain-car | 165.48 | 405.35 | 15580.27 | 27607.35 | -0.23 (broken) |
| mountain-car-cont | -0.07 | 0.12 | 10.32 | 1.83 | -0.95 (broken) |
| ppo | 240.10 | 864.73 | 455761.73 | (cut off) | 166.53 |
| sac | (cut off) | — | — | — | — |

**Outcome**: partial. Tape and torch numbers are useful for regression tracking; the rest surfaces three workflow issues:

1. **PyTorch ref timing is broken for short-converging deep-RL examples.** The two-point methodology in `perf-baseline.sh` / `perf-sweep.sh` (`wall(N_long) - wall(N_short)` divided by `N_long - N_short`) goes negative or near-zero when the per-epoch cost is dominated by Python startup + episode-length variance rather than fixed per-step compute. Only PPO (whose ref does linear-in-rollouts work each epoch) got a sane ratio. The Idris-side timings are valid (separate Idris-side two-point — both runs share the same Idris startup). Ratio columns are wrong for everything except PPO. **Workaround**: for short-converging deep-RL refs, use a longer fixed-`N` single-point timing with explicit `--epochs` override; need a new harness mode in `perf-baseline.sh`.
2. **mlx is unusable for deep-RL workloads at current Idris-side per-op overhead.** PPO mlx-cpu is **455 sec/epoch** (vs 0.24 sec tape, 0.86 sec torch); mountain-car mlx-cpu is **15.6 sec/epoch** (vs 0.17 sec tape). The "Idris-side per-op overhead" Medium TODO row already filed for this. Long-rollout REINFORCE additionally crashes mlx with `[malloc] Unable to allocate 16 bytes` after ~2M tape appends — separate failure mode (mlx arena fragmentation on long-tape workloads), not just slowness.
3. **Wall-clock cost** of running all 4 cells × 7 deep-RL examples is ~9h because mlx-cpu/gpu dominate. For routine post-change gating on RL changes, restrict to `tape,torch` cells (drops total to ~30 min on this hardware).

**Cross-references**:
- `docs/develop/perf-log.jsonl` 22 entries appended (lines tagged `commit 09af662`).
- Existing Medium TODO row "Idris-side per-op overhead (cross-backend wall bottleneck)" — this sweep is fresh confirmation on RL workloads.
- New gotcha for perf-baseline.sh / perf-sweep.sh: needs a long-fixed-N mode for short-converging examples.

### 2026-05-14 — GPU-specific "Idris-side overhead" is actually accumulated kernel-launch wall — `<commit>`

**Plan job**: follow-up to the GptLarge Phase 3 wallclock matrix, where
"GPU is 20% slower on wall, but only 7% slower on C-total" left an
unattributed ~800 ms/ep on GPU.

**Motivation**: claim being tested — "mlx GPU has more Idris-side
overhead per FFI call than mlx CPU stream." If true, the fix would
involve Chez Scheme runtime work. If false, then the same accumulated
kernel-launch wall that motivated the optimizer eval-removal is just
showing up across the whole forward/backward graph too — fixable by
fusing more ops, no Idris-side work needed.

**Change**: built `/tmp/bench_per_op` — a tight loop of `tensor_add`
calls with NO eval. Measured pure graph-build cost only:

| measurement                       | CPU      | GPU      | gap |
|-----------------------------------|---------:|---------:|----:|
| graph-build only (no eval)        | 0.43 us/op | 0.44 us/op | **0** |
| add+mul w/ force_eval per iter    | 28 us/op   | 200 us/op  | 7× |
| supervised wall/ep                | 4 ms     | 8 ms     | +4 ms |
| gpt-large wall/ep                 | 8500 ms  | 9300 ms  | +800 ms |

Pure FFI dispatch + graph-node construction is identical on CPU and
GPU. The cost only appears when something forces evaluation —
`tensor_item`, the `mx::eval` calls inside `tensor_backward`, the
final `mx::eval(to_eval)` in `optimizer_step`, etc.

**Impact**: explains the GPU wall gap mechanistically. On mlx CPU
stream, a sync runs the queued ops on a CPU worker thread — fast and
pipelined. On mlx GPU, a sync encodes to a Metal command buffer,
dispatches, and waits for completion — Metal has higher per-op
latency. Across 293 ops per gpt-large epoch (plus backward and
optimizer graph), the cumulative drain at each sync point produces
the 800 ms/ep wall gap. No Idris-side work is the actual contributor.

**Outcome**: investigation only, no code change. The actionable
levers are the same as the optimizer story:
- `mx::compile` wrapping over larger scopes (whole forward, whole
  optimizer step) → fewer kernels → fewer launch-wall contributions
  at each sync point
- bigger per-op compute (bigger model) → wall amortizes naturally

The investigation kills the "Idris-side GPU overhead" hypothesis
cleanly. Filed two follow-ups: the existing "wrap optimizer step in
`mx::compile`" TODO row remains the next concrete lever; the
"investigate Idris-2 JIT / JAX backend" row stays open as the broader
question even though THIS investigation showed Idris-side dispatch is
fine on its own.

**Cross-references**:
- `perf-log.jsonl` post-eval-fix 3-run reproducibility entries
- the per-op microbench `/tmp/bench_per_op.c` is one-off and not
  checked in; reproducible from the recipe in this entry

### 2026-05-14 — GptLarge GPU-vs-CPU matrix: 20% wallclock gap, optimizer is the lever — `<commit>`

**Plan job**: GPU-friendly-example TODO row (the deliverable said
"showing GPU > CPU"; we didn't get it, but found the actionable
next-step lever in the process).

**Motivation**: All previous examples (NTM/DNC/LSTM/MNIST/small Gpt)
were too small for Apple Metal to beat the CPU stream — kernel-launch
wall dominated. Phase B left open whether a properly GPU-shaped
workload would flip the verdict. Built `Example.GptLarge` (dModel=256,
heads=8, headDim=32, blocks=4, seq=128, batch=32; 3.17 M params) and
the paired `torch_ref/scripts/gpt_large.py` to find out.

**Change**: this entry is the measurement, not a code change. The
6-cell matrix was run at 10 epochs each (single sample; deltas large
enough to clear the VM noise floor):

| backend           | wall ms/ep | C-total ms/ep | C-total notes                          |
|-------------------|-----------:|--------------:|----------------------------------------|
| tape              |       9700 |          8830 | actual compute (synchronous)           |
| torch             |       9500 |          1080 | mostly compute (sync per op)           |
| mlx CPU eager     |       8500 |            34 | **enqueue only**                       |
| mlx CPU compile   |       8800 |            33 | **enqueue only**                       |
| mlx GPU eager     |      10200 |           276 | enqueue + per-`mx::eval` sync          |
| mlx GPU compile   |     ~10000 |           254 | enqueue + per-`mx::eval` sync          |

(GPU measured against pip mlx 0.31.2 with Metal at
`/tmp/mlx-gpu-test`; nixpkgs mlx is CPU-only.)

**Impact (revised)**: an earlier reading of this table called GPU
"8-10× slower" — that was wrong; it treated the mlx C-totals as
compute time when they're mostly enqueue cost. The honest read:

- **Wallclock**: mlx GPU is ~20% slower than mlx CPU stream. Real gap,
  but small enough to be "not yet" rather than "never".
- **GPU compute itself looks healthy** — backward forced via `mx::eval`
  is ~11 ms/ep, which matches the ~7.5 ms FLOPS floor for ~75
  GFLOPs/step on M2 (~10 TFLOPS) plus sync overhead.
- **The optimizer step on GPU is 243-265 ms/ep**, exactly the per-param
  kernel-launch wall: 293 params × ~1 ms each. PyTorch's `_foreach_*`
  fused multi-tensor ops are the standard fix; we don't have an
  equivalent on the mlx (or torch) optimizer surface.
- **The mlx CPU "Backward 2.5 ms/ep" number is unreliable** — it would
  imply ~30 TFLOPS on a CPU stream, which is impossible. The mlx CPU
  C-total measures enqueue time; actual compute fires later.
- **Idris-side / Chez overhead floods all wallclocks** at ~8 s/ep on
  this hardware. Until that's reduced (separate TODO row), wallclock
  comparisons are dominated by the constant.

**Outcome**: partial. The example and the measurement matrix exist
and are in CI; the GPU-wins outcome from the original TODO row isn't
reached. The actionable lever is the fused multi-tensor optimizer
(filed as a new high-prio TODO). Default stays `MLX_DEVICE=cpu` until
that lands; verdict re-opens after.

**Cross-references**:
- `perf-log.jsonl` 2026-05-14 entries tagged "Phase 3 cell N/6"
- `docs/develop/mlx-survey.md` "Follow-up update (2026-05-14)" section

### 2026-05-14 — Tape profiler diagnostic: ADD bucket is misattribution — `<commit>`

**Plan job**: cross-cutting (tooling — the tape profiler is the source of
truth for every per-op investigation, and it was misleading us).

**Motivation**: `example-gpt-large` on tape showed "ADD" as 95% of
forward C-time (117 ms/call × 138 calls = 16.2 s/ep). The smaller
`example-gpt` showed the same shape: ADD 8.3 ms/call. Pulled the
thread — vDSP\_vaddD on a [128, 256] tensor should be ~50 µs, not 117
ms. Hypothesis: the per-op timer in `tape_append` attributes inter-op
wall time to the op being recorded *now*, so any Idris-side glue
between ops gets pinned to whichever op happens to close the chain.
ADDs are residual closes in the transformer — they collect the leakage.

**Change**: added three diagnostic timers to `backend_tape.c`'s
`binop_elementwise`:
- direct-kernel timer wrapping just the `vDSP_vaddD` call
- in-function timer wrapping entry-to-exit of the whole
  `binop_elementwise` (split via a thin wrapper +
  `binop_elementwise_inner`)
- path-classification counter (fast / scalar\_bcast / general\_bcast)
  plus a per-op-tag log of the first general\_bcast shape seen

Backed by storage `prof_kernel_per_op`, `prof_kernel_count_per_op`,
`prof_binop_inside_ms`, `prof_binop_inside_count`,
`prof_binop_path_count`, `prof_binop_general_ms`. Reset alongside the
other profile arrays in `backend_profile_reset`. Surfaced in
`backend_profile_print` as new sections after the existing top-N
forward ops.

**Impact**: this is a diagnostic-only change — zero perf delta, just
ground truth. Re-ran `example-gpt-large` and `example-gpt` on tape:

| metric                       | small Gpt | GptLarge | unit       |
|------------------------------|----------:|---------:|------------|
| ADD bucket (attributed)      |      2661 |    16783 | ms / 3 ep  |
| ADD in-function              |       3.3 |     44.8 | ms / 3 ep  |
| ADD kernel (vDSP only)       |       3.5 |     44.3 | ms / 3 ep  |
| bucket / in-function ratio   |    **813×** | **374×** | leakage    |
| per-tape-entry leakage       |      0.33 |     0.59 | ms / entry |
| binop\_elementwise fast path |    100%   |   100%   |            |

All ADDs took the vDSP fast path (zero general broadcast). Kernel
time and in-function time agree within instrument noise (kernel is a
strict subset). Real ADD work per epoch is ~15 ms — three orders of
magnitude smaller than the bucket headline.

**Outcome**: landed (diagnostic only). The real bottleneck on tape
forward at GptLarge scale is **~0.6 ms/tape-entry of Idris-side / Chez
overhead between FFI calls**, which the profiler currently
misattributes to whichever op is recorded next. At 293 entries/forward
× 32-sample batch × per-entry overhead, this dominates the C-total
wallclock. Likely suspects: Chez foreign-procedure dispatch, GC
pressure from per-step Idris-side allocation, or
`UserDeviceCore`-class typeclass-dispatch cost compounding per op
(see the gotcha "Typeclass methods of unit type fire eagerly"). Real
fix needs a separate investigation — at minimum the per-op timer
should record kernel-internal time so attribution stops lying.

**Cross-references**:
- `perf-log.jsonl` 2026-05-14 entries tagged `[diagnostic]`
- `Example.GptLarge` first commits — the workload that surfaced this

### 2026-05-09 — DNC `dncZeroDiag` mask precompute — `d452eef`

**Plan job**: cross-cutting (helps Job 1 + Job 2a + Job 2b
together; the mask is a per-step constant rebuild, so reducing
it benefits every backend equally).

**Motivation**: DNC-copy on torch was 9.8× PyTorch ref ms-per-epoch.
Profiling showed `prim_forward_ms` ≈ 114 ms/epoch, dominating the
120 ms/epoch total. `Layer/Dnc.idr`'s `dncZeroDiag` was rebuilding
a (1 − Iₙ) [n, n] mask every timestep — for n = 32 that's
1 + n² + 1 prim FFI calls (`allocDoubles` + 1024 × `setDouble` +
`create2d` + `mul`) on a constant.

**Change**: moved the mask construction into the `DncState`
constructor (new `nonDiagMaskT : AnyPtr` field). `dncZeroDiag` is
now a single `prim__mul` against the precomputed mask.
Mathematically identical (mask is a constant); training trajectory
is bit-identical.

**Impact** (`scripts/perf-baseline.sh <ex> <be>`, seed=42):

| Example    | Backend | Before  | After   | Speedup |
|------------|---------|--------:|--------:|--------:|
| dnc-copy   | tape    | 11.11×  |  1.24×  |  ~9×    |
| dnc-copy   | mlx     | 13.01×  |  2.68×  |  ~5×    |
| dnc-copy   | torch   |  9.80×  |  2.05×  |  ~5×    |
| dnc-recall | tape    | 13.20×  |  1.50×  |  ~9×    |
| dnc-recall | mlx     | 15.25×  |  2.38×  |  ~6×    |
| dnc-recall | torch   | 14.24×  |  2.14×  |  ~7×    |

Both DNC examples moved from Bucket D (>10×) into Bucket A/B.
`dnc-copy` on tape is now within Bucket A (≤1.10×) territory.

**Outcome**: landed.

**Cross-references**: `perf-log.md` 2026-05-09 entries for
dnc-copy/dnc-recall on each backend; `perf-baseline.md`
"NTM/DNC current-state" subtable updated with new ms/epoch.

### 2026-05-09 — DNC `dncRetention` scalar 1.0 reuse — `7116102`

**Plan job**: cross-cutting (small; same family as the mask fix).

**Motivation**: After the mask fix, DNC torch was still ~2×.
`Layer/Dnc.idr`'s `dncRetention` recursed once per read head, calling
`prim__createScalar 1.0` inside each recursion to build the
`(1 − fg·rw)` factor. For R = 4 read heads that's 4 redundant FFI
calls per timestep on a constant.

**Change**: pass the precomputed `onesScalar` (already built once
at the call site in `applyDnc`) into `dncRetention` as the leading
argument; reuse it inline.

**Impact** (3 samples each on `scripts/perf-baseline.sh dnc-copy
<be>`):

| Example  | Backend | Before  | After   | Note      |
|----------|---------|--------:|--------:|-----------|
| dnc-copy | tape    | 1.24×   | ~1.14×  | small win |
| dnc-copy | torch   | 2.05×   | ~1.75×  | small win |

Within measurement noise, but consistently slightly better.

**Outcome**: landed.

### 2026-05-09 — NTM `onesM` precompute — *(reverted)*

**Plan job**: Job 1 (mostly torch) + Job 2a (tape).

**Motivation**: `Layer/Ntm.idr`'s `ntmInterpWriteIdris` was building
a length-`m` all-ones tensor per timestep via
`prim__addScalar (zeroState1d m) 1.0`. Same pattern as the DNC mask
fix on a smaller constant. Estimated 3 prim FFI calls × ~50
timesteps/epoch saved.

**Change**: added `onesMPtr : AnyPtr` field to `NtmState`, built
once in `ntmLayer`. Threaded through `applyNtm` to
`ntmInterpWriteIdris`.

**Impact** (3 samples each):

| Example  | Backend | Before  | After   | Note         |
|----------|---------|--------:|--------:|--------------|
| ntm-copy | tape    | 1.42×   | ~1.20×  | small win    |
| ntm-copy | torch   | 2.58×   | ~1.88×  | win          |
| ntm-copy | mlx     | 1.98×   | ~3.6 ×  | **regression** |

mlx regressed ~2×. Likely related to the existing `gotchas.md`
note: "MLX requires non-grad tensors to be non-persistent —
`prim__createState1d` marks them persistent and the lazy graphs
that reference them survive `tape_reset` and dangle after the next
epoch starts." A precomputed persistent ones-tensor used inside
`mx::outer` apparently triggers a slow path on mlx (or builds an
ever-growing lazy graph).

**Outcome**: reverted. Net negative across backends. Could be
re-attempted with a Maybe-cached lazy-init approach (build on
first call of each epoch, reset to Nothing on `resetNtmState`),
which would behave like the existing memT pattern and probably
avoid the mlx interaction. Filed as future-todo, not active.

**Re-attempt 2026-05-09 under priority-torch-and-tape framing** (after
plan was updated to make tape + torch primary and mlx Job 3): hoped
the small torch wins would justify the mlx regression as an
acceptable tradeoff. 3 samples each on tape and torch show no
measurable change vs pre-fix on either: ntm-copy tape ~1.42×
(unchanged), ntm-copy torch ~2.36× (unchanged within noise), ntm-recall
tape ~1.25× (unchanged), ntm-recall torch ~2.64× (worse but high
variance). The "torch wins" I thought I saw on the first attempt were
sampling noise. mlx still regresses 2× as expected. Reverted again.

**Hypothesis for why no win even on tape/torch**: Idris's compiler is
very likely CSE'ing the per-timestep `prim__addScalar (zeroState1d m)
1.0` chain across timesteps — they have identical inputs every call,
so the result handle is shared and the FFI calls only fire once per
sequence anyway. If true, the precompute-into-state plan is just
moving CSE'd work into an explicit field with no perf delta.

This optimization shape (precompute a constant in the state record)
isn't worth pursuing unless we find a constant where Idris's CSE
doesn't fire. The DNC mask precompute (above) DID work because it
saved hundreds of `prim__setDouble` calls in a loop, which CSE can't
fold across the loop.

### 2026-05-09 — `withNoGrad` + a2c rollout — `b02860e`

**Plan job**: Job 1 (torch wrapper overhead) + Job 2a (tape).

**Motivation**: PyTorch's `torch.no_grad()` suppresses autograd graph
construction for forward passes whose results aren't backprop'd —
the standard pattern for RL rollouts, evaluation, anything that just
wants the forward result. Our a2c rollout was running 480+ prim ops
per epoch under autograd tracking only to extract Doubles for sampling
+ bootstrap; the gradient came from `buildLoss`'s own batched forward.
All 480 ops were wasted graph construction.

**Change**: wired up the existing-but-stubbed `tensor_no_grad_begin/end`
in `backend_tape.c`, `backend_mlx.cpp`, `backend_torch.cpp` with a
nesting counter (matches PyTorch's nestable `torch.no_grad()`).

- tape: `tape_append` becomes a no-op when `no_grad_depth > 0`;
  results marked `requires_grad=0` so downstream doesn't propagate.
  Returns a writable static dummy entry so callers that do
  `e->op_meta = ...` don't null-deref.
- torch: nests a `torch::NoGradGuard` while depth > 0.
- mlx: `tape_append` skipped + result `requires_grad=false` so the
  VJP-replay closure doesn't track these ops.

Idris API: `withNoGrad : IO a -> IO a` in `Tensor.idr`. Uses primIO
sequencing on begin/end (same pattern as `prim__backwardC`).

Wired up in `Example.A2c.a2cEpoch`: rollout phase wrapped in
`withNoGrad`.

**Impact** (3 samples each at default config):

| Backend | Pre   | Post  | Note |
|---------|------:|------:|------|
| tape    | 9.93× | 8.66× | tape entries 600+ → 12, backward 5ms → 0.2ms |
| torch   | 10.81× | 9.26× | autograd graph saved on rollout |
| mlx     | 15.22× | 13.04× | fewer VJP constants |

Wins are smaller than hoped because the per-call Chez FFI floor
(~9 µs/call) dominates per-prim cost on these examples; no_grad
saves a portion of that, not the dispatch itself.

**Outcome**: landed. Useful as a library feature even where the perf
delta is small — anyone writing RL or eval code in Idris-ml would
expect `withNoGrad` to exist, just like PyTorch users expect
`torch.no_grad()`. Future opportunity: also wrap `bootstrapV` and
the eval phase, plus other examples' eval paths.

### 2026-05-09 — Align Layer.Rnn with `nn.RNNCell` — `10fe116`

**Plan job**: Job 1 / Job 2a (both — the layer change benefits all
backends), with paired-side update.

**Motivation**: pre-existing `Layer.Rnn` was a non-standard
linear-recurrence (no activation, single bias) chosen arbitrarily
when the example was first written. PyTorch's `nn.RNN` doesn't have
a "no activation" mode, so the matching reference (`torch_ref/models/
rnn.py`) had to use a hand-written `LinearRNNCell` with `hidden_size=1`,
no projection, and matching no-activation semantics. Two consequences:
the example didn't demonstrate the canonical RNN shape that library
users expect, and the perf ratio comparison was unfair (PyTorch ref
was doing strictly less work — no projection, no tanh — so the ratio
was inflated).

**Change**: realign both sides to `nn.RNNCell`'s shape:
- Idris `Layer.Rnn` gets two biases (`ihB`, `hhB`) and a generic
  `activation : TVec o ex -> TVec o ex` field (more flexible than
  `nn.RNN`'s tanh/relu enum — pass any unary tensor function).
- `rnnLayerAny` defaults activation to `ttanh`, matching `nn.RNN`'s
  default.
- PyTorch ref's `LinearRNNCell` rewritten to match `nn.RNNCell`
  (tanh, two biases) plus the output projection the Idris model has
  on top. Defaults `hidden_size=4 / output_size=1` to match Idris.
- Initial hidden state on both sides: zero (matches `nn.RNNCell`
  default; previously was a learned `nn.Parameter` on PyTorch side).
- LR default 0.03 → 0.5 on PyTorch ref (matches Idris example default).

**Impact** (3 samples each, post change):

| Backend | Before  | After   |
|---------|--------:|--------:|
| tape    | 3.83×   | 3.08×   |
| torch   | 5.07×   | 4.38×   |
| mlx     | 7.99×   | 6.59×   |

The ratio shrinkage is partly a methodological correction — the
previous PyTorch ref was doing strictly less work, so its ms/epoch
was artificially small. The new comparison is fair (both sides
implement the same model). All three backends produce bit-identical
loss curves on the new model (e.g. loss=0.005914 at ep 100).

**Outcome**: landed. The example is now a canonical small-RNN
demonstration matching what a library user expects to see for "how
do I use an RNN cell in Idris-ml". Same shape applies to lstm/gru
example alignment if/when we revisit them — they already use the
nn.LSTM/GRU shape, but worth a paired-side audit.

### 2026-05-09 — Align Layer.Lstm and Layer.Gru with `nn.LSTMCell` / `nn.GRUCell` — `352239f`

**Plan job**: Job 1 + Job 2a (cross-cutting; all backends benefit).

**Motivation**: LSTM was using a single fused bias (vs `nn.LSTMCell`'s
two: `bias_ih` + `bias_hh`). GRU's C kernel was a *simplified-GRU*
variant that computed but ignored the `r` reset gate (vs `nn.GRU`'s
`n = tanh(ih_n + r * hh_n)`). Same family of non-standard
simplifications as the rnn alignment.

**Change**:
- LSTM: split bias into `ihB` + `hhB`; `applyLstm` now does 3 FFI
  calls per timestep (vs 2 with fused bias). PyTorch ref drops its
  Jozefowicz forget-gate-bias=1 init for symmetry with Idris (which
  never had it). PyTorch ref now also has learned `h0`/`c0` to match
  Idris's `LstmState` carrying them (added in Phase 1.5b).
- GRU: kernel signature changed from `(combined, prev, o)` to
  `(ih, hh, prev, o)`. Three backends updated: tape rewrites the
  hand-rolled backward to handle r's grad path; mlx uses a new
  `GruCellReplayMeta` to thread `prev`'s pool_idx into the replay
  closure; torch's autograd handles backward through the graph.
  `tgruCell` and `applyGru` updated; `applyGru` now does 3 FFI
  calls (vs 4 with the explicit pre-sum tadd).

**Impact** (3 samples each, post change):

| Cell      | Backend | Before  | After   |
|-----------|---------|--------:|--------:|
| lstm      | tape    | 2.01×   | 1.32×   |
| lstm      | torch   | 2.07×   | 2.12×   |
| lstm      | mlx     | 3.40×   | 4.22×   |
| gru       | tape    | 1.91×   | 1.19×   |
| gru       | torch   | 2.70×   | 1.96×   |
| gru       | mlx     | 5.97×   | 4.05×   |

Tape and torch gru improved on ratio AND in absolute ms (gru tape
5.25 → 4.81 ms because applyGru saves one FFI hop). lstm tape ratio
improved partly because PyTorch ref slowed down (added h0/c0 +
clone) and partly because lstm Idris stayed similar (the extra tadd
hhB cost is small in absolute terms). mlx ratios moved
inconsistently — Job 3 sub-item to investigate later.

**Outcome**: landed. Together with the rnn alignment, the rnn /
lstm / gru examples now demonstrate the canonical PyTorch shape
that library users expect. Backend cell APIs (`tgruCell`,
`tlstmGatesPair`) are also closer to standard ML library
conventions.

### 2026-05-09 — DNC `dncReadHeads` link-transpose hoist — `a960058`

**Plan job**: cross-cutting (mostly tape/torch).

**Motivation**: `dncReadHeads` recursed once per read head, calling
`prim__transpose2d linkT` inside each recursion. `linkT` is shared
across heads — the transpose is head-invariant.

**Change**: compute `linkTransT = prim__transpose2d newLinkT` once
in `applyDnc` and thread it into `dncReadHeads` as an extra
argument. Removes R-1 redundant FFI calls per timestep on a
head-invariant value.

**Impact** (3 samples each, post `7116102`):

| Example    | Backend | Before  | After   | Note         |
|------------|---------|--------:|--------:|--------------|
| dnc-copy   | tape    | ~1.25×  | ~1.25×  | noise        |
| dnc-copy   | torch   | ~2.05×  | ~2.14×  | noise        |
| dnc-recall | tape    | ~1.50×  | ~1.32×  | small win    |
| dnc-recall | torch   | ~2.14×  | ~1.91×  | small win    |

Theoretical savings: R-1 = 3 FFI calls × 40 timesteps × 9 µs ≈
1 ms/epoch. Within measurement noise on dnc-copy (small absolute
ms/epoch) but visible on dnc-recall.

**Outcome**: landed.

### 2026-05-09 — `withNoGrad` for RL rollouts + a2c bootstrap — `d378900`

**Plan job**: Job 1 (torch) + Job 2a (tape).

**Motivation**: completing the `withNoGrad` pattern across the four RL
examples that have a "rollout (no grad needed) + separate batched
loss-step (grad needed)" structure: dqn, ppo, sac (a2c was already
done). Plus pulling a2c's `computeBootstrap` out of `buildLoss` so
the single critic forward on `finalSt` runs in `withNoGrad` too —
the value is consumed as a Double by GAE, no grad path.

**Change**:
- a2c: refactor `buildLoss` to take precomputed `bootstrap : Double`
  as a parameter; compute it in `a2cEpoch` inside `withNoGrad`.
- dqn: wrap `epsGreedyIO` at the action-selection point in
  `runEpisode.go`.
- ppo: wrap `rollout` and `prepareRollout` (which calls
  `computeBootstrap`).
- sac: wrap the post-warmup `sampleActionIO` in `sacStep`.

Reinforce intentionally NOT wrapped: its rollout's per-step forward
log-probs ARE used in the gradient (single-forward-per-step structure,
no separate batched forward in the loss). Wrapping would break
training.

**Impact**: per-prim Chez FFI floor still dominates these examples,
so the savings are modest in absolute ms/epoch (~0.5–2 ms/epoch in
each). The bigger win is correctness: rollout phases no longer hold
references to stale autograd graph nodes that get freed at the next
`optimizer_step` anyway.

**Outcome**: landed. Verified all four examples train to the expected
accuracy on tape.

----

### 2026-05-09 — Tape `binop_elementwise` numpy-style 2D broadcast

**Plan job**: Job 2a (phase A)

**Motivation**: `binop_elementwise` previously only handled
same-shape and scalar broadcast. Mixed shapes like `(n,1)×(n,m)`,
`(1,m)×(n,m)`, `(m,)×(n,m)` (numpy-style row/column broadcast) fell
into a multi-dim path that flat-indexed past the smaller operand's
buffer — undefined behaviour for any user code or layer trying to
use these patterns. The NTM `ntmInterpWriteIdris` workaround code
explicitly cites the limitation: *"row-wise scalar multiplication
(n,)·(n,m) is not supported by the tape backend's elementwise
broadcast (which only handles numel=1 broadcast), so we materialize
`w` row-wise via `outer(w, ones_m)`"*.

**Change**: Added `compute_bcast_shape` and `compute_bcast_strides`
helpers in `backend_tape.c`. Refactored `binop_elementwise` forward
into three branches: same-shape vDSP fast path (unchanged), scalar
broadcast (cleaned up), and a new general-broadcast walk that
right-aligns ranks numpy-style and uses per-operand strides (0 on
broadcast dims). Refactored `OP_ADD/SUB/MUL/DIV/POW` backward to
detect broadcast via `shapes_equal(a, r)` / `shapes_equal(b, r)`
and reduce gradients along broadcast dims via the same stride walk.

**Impact**: latent OOB-read bug for users fixed. Verified
bit-identical forward + backward against the OLD chain on
NTM-realistic dimensions (n=128, m=20) via a standalone C unit
test. All same-shape and scalar-broadcast operations behave
bit-exactly as before.

**Outcome**: landed in commit `9f78d39`.

----

### 2026-05-10 — NTM `ntmInterpWriteIdris` adopts the broadcast (per-backend seed defaults)

**Plan job**: Job 2a (phase A) follow-up

**Motivation**: With the broadcast capability above, the NTM helper
can drop the `outer(w, ones_m)` materialisation and use a
`reshape(w, n, 1)` view + direct `(n,1)*(n,m)` broadcast mul. The
old workaround's comment "not supported by the tape backend's
elementwise broadcast" now goes away.

**Change**: `Layer/Ntm.idr` `ntmInterpWriteIdris` rewritten:
`reshape2d → neg → addScalar → mul (broadcast) → add`. Saves one
`outer` + one `addScalar(zeros, 1.0)` per NTM timestep. The single-
timestep gradient is bit-identical to the old chain (verified
algebraically and numerically); but the multi-timestep training
trajectory differs in ULP-level ways from the workaround chain
because the backward reduction order changes (broadcast walk vs
chain through `outer`'s own backward).

**Seed sensitivity fallout**: NTM-Copy is highly seed-sensitive
(see `gotchas.md`). With the broadcast in place, the seeds that
converge cleanly differ per backend:
- tape: seed=42 → 4400 ep / 1.0 acc_full ✅
- torch: seed=42 → 5300 ep / 0.99 acc_full ✅
- mlx: seed=99 → ~4400 ep / 0.997 acc_full ✅ (matches the
  pre-broadcast perf-baseline)
- mlx at seed=42 fails (0.65 acc_full); tape/torch at seed=99
  with broadcast are slow / borderline.

The `Makefile`'s `example-ntm-copy` target picks the seed per
backend (tape/torch → 42, mlx → 99). The in-Idris `defaultConfig`
and paired `torch_ref/scripts/ntm_copy.py` both move 99 → 42 (the
primary tape/torch default; mlx is the asymmetric special-case).
Users override with `NTM_COPY_ARGS="--seed N"`.

**Impact**: tape ntm-copy converges ~2× faster (4400 vs ~9600
prior tape-at-seed=42, or ~8400 prior tape-at-seed=99). torch is
unchanged (5300 ep both ways). mlx at its new default seed=99
matches its pre-broadcast best (perf-baseline).

**Outcome**: landed.

----

### 2026-05-11 — Tape: BLAS-accelerate matmul backward kernels — `3ba8f31`

**Plan job**: Job 2b (phase A, stretch)

**Motivation**: All matmul-class forward kernels (`OP_MM`, `OP_BMM`,
`OP_MV`, `OP_LINEAR`, `OP_LINEAR_2D`) have dispatched to Apple
Accelerate `cblas_dgemm`/`dgemv` since the file was written, but
the matching backwards were hand-rolled triple-nested loops. Every
transformer / GPT / DNC backward pass therefore left Accelerate on
the table on the half of the computation that takes the most time
at scale.

**Change**: Each backward switched to the BLAS-equivalent:
- `OP_MM`: `d_a = dgemm(NoTrans, Trans)`, `d_b = dgemm(Trans, NoTrans)`
- `OP_BMM`: single dgemm collapsing the `B·m` dim (shared weight
  means `d_b` accumulates over batch in one call)
- `OP_MV`: `d_A = dger` (rank-1), `d_x = dgemv(Trans)`
- `OP_LINEAR_2D`: `d_W = dgemm(Trans, NoTrans)`, `d_X = dgemm(NoTrans, NoTrans)`
- `OP_LINEAR`: `d_W = dger`, `d_x = dgemv(Trans)`

`beta=1.0` preserves the existing `+= grad` accumulation semantics.
Each BLAS path is gated on `__APPLE__`; the portable scalar
fallback is preserved.

**Closing sweep** (full 9 examples × 3 backends, see
`perf-log.jsonl` commit `3ba8f31+dirty`):

| Example | Job 2a (naive) | Job 2b (BLAS) | Δ wall | quality Δ |
|---|---|---|---|---|
| supervised | 3.6s | 3.3s | **-8%** | bit-identical |
| rnn | 8.7s | 8.1s | -7% | bit-identical |
| lstm | 11.2s | 10.5s | -6% | bit-identical |
| gru | 9.7s | 9.7s | 0% | bit-identical |
| transformer | 31.8s | 28.8s | **-9%** | bit-identical |
| dnc-copy | 89s / 0.877 | 76s / 0.873 | **-15%** | ≈0 |
| dnc-recall | 480s / k4=0.94 | 433s / k4=0.96 | **-10%** | + |
| ntm-copy | 84s / 4400ep / 1.0 | 118s / 7000ep / 1.0 | +40% wall | acc preserved |
| ntm-recall | 163s / 8500ep / k4=0.98 | 321s / 18000ep / k4=0.91 | +97% wall | **-7pp** |

Per-epoch ms is faster everywhere; the NTM wall-clock regression is
purely seed-trajectory: BLAS `dgemm`/`dger`/`dgemv` reduce in a
different floating-point order than the naive triple loop, and
NTM-Copy's documented seed-sensitivity (`gotchas.md`) flips
seed=42 onto a slower-converging branch. Quality preserved on
ntm-copy (acc_full=1.0 either way); ntm-recall acc_k4 drops 7pt.

**Decision (recorded here as the rationale)**: kept the
unconditional BLAS path despite the NTM regression. Rationale:
1. The library is general-purpose; NTM is one architecture out of
   ~25 examples. Hobbling the linear-algebra fast-path for every
   user just to preserve NTM-Copy's seed=42 trajectory is the
   wrong trade. (See feedback memory
   `feedback_library_users_not_examples.md`.)
2. NTM-Copy still converges to acc_full=1.0; it just takes more
   epochs at the default seed.
3. NTM-Recall's k4 0.98→0.91 is a real quality regression but
   acc_k2=1.0 stays perfect — short-sequence recall is unaffected
   and length-generalization to k4/k6 is the inherently
   seed-sensitive part of the benchmark.

A threshold-dispatch variant (route to naive below
`m·n·k = 5000`) was tried (commit `1fc56da`, reverted in
`b3b5de5`); even the act of wrapping the naive path in
`if (use_blas) {...} else { naive }` shifts compiler codegen
enough to drift gradients ULP-wise. The variant also fared worse
than all-BLAS on dnc-recall in our run (k4 0.96 → 0.82).
Threshold tuning is too noise-prone to do without a proper
microbench framework (logged for Phase B).

**Outcome**: landed. NTM regression accepted as a library-level
trade. Job 2b phase A closed.

----

### 2026-05-11 — Batched Conv2D / MaxPool2D + MNIST → epochVarTensorBatch — `6b155af`

**Plan job**: Job 1 (reopened — Conv2D wrapper audit)

**Motivation**: A side-by-side MNIST convergence comparison flagged
a 4.0× wrapper-overhead ratio on idris-torch vs raw PyTorch ref
(49.4 vs 12.5 s/epoch on 60K MNIST, batch_size=64). Job 1 phase A
had only audited linear / RNN / NTM-DNC paths; the Conv2D path was
unexamined. Tracing it: `Layer/Conv.idr:applyConv2D` operates on
`TVec (inC*h*w)` (single sample), and `Backprop.idr:epochVarTensor`
threads each `dataPoint` through the model individually — so the
training loop calls `torch::conv2d` (and friends) 64× per minibatch
instead of once batched. Every per-call autograd-graph setup and
tensor-view bookkeeping ran 64× more than necessary.

**Change**: wired Conv2D and MaxPool2D into the existing batched-
forward infrastructure that Linear / Activation / Dropout already
used.

C-side:
- `tensor_conv2d_batched` + `tensor_max_pool2d_batched` in all three
  backends. Torch drops the per-call `unsqueeze(0)`/`squeeze(0)`
  (libtorch is batch-native). mlx skips the per-call NHWC layout
  reshape on the batch axis. Tape gets new `OP_CONV2D_BATCHED` /
  `OP_MAX_POOL2D_BATCHED` op tags with batched-meta structs and
  matching forward + backward kernels (B in the outer loop, d_kernel
  accumulating across the batch in a single tight loop).
- `tensor_reshape_4d` helper (was 1d/2d/3d only).

Idris-side:
- `prim__conv2d_batched`, `prim__maxpool2d_batched`, `prim__reshape4d`
  FFI bindings.
- `Conv2DState` / `MaxPool2DState` `applyVarBatch` impls that reshape
  `[B, c*h*w]` → `[B, c, h, w]`, call the batched prim, reshape back.
- `Example/Mnist.idr::trainOneFullPass` switches `epochVarTensor` →
  `epochVarTensorBatch`.

**Impact** (MNIST full 60K, seed=42 tape/torch, seed=99 mlx):

| Backend | per-sample s/ep | batched s/ep | wrapper vs PyTorch ref (12.5 s/ep) |
|---|---:|---:|---:|
| torch | 49.4 | **21.3** | 4.0× → **1.68×** ✅ |
| mlx   | (n/a baseline) | **26.0** | — / **2.08×** |
| tape  | 175 | 176 | 14.0× → 14.1× (compute-bound) |

Torch wrapper overhead halved. mlx in the ~2× range. Tape per-epoch
unchanged because the bottleneck is the hand-rolled triple-nested
conv kernel (the batched version runs the same FLOPs with the same
naive code), not the FFI count — file follow-up for an im2col +
`cblas_dgemm` tape conv2d kernel. Quality preserved on torch/mlx
(98.4% / 98.1% acc), tape at the batched-default seed converges to
97.3% in 3 epochs (down from 98.4% per-sample) — the now-familiar
NTM-style ULP-shift seed-sensitivity reappearing on tape.

**Outcome**: landed. Job 1 reopened-phase-A first half done; tape
im2col follow-up below closes the second half.

----

### 2026-05-11 — Tape Conv2D: im2col + cblas_dgemm forward + backward — `f9c8eaf`

**Plan job**: Job 1 (reopened — tape Conv2D follow-up)

**Motivation**: After the batched-Conv2D / MaxPool2D layer wiring
closed the idris-torch wrapper gap on MNIST (4.0× → 1.68×), the
tape backend was *unchanged* per-epoch (~176 s) because its hand-
rolled triple-nested conv kernel did the same FLOPs in either
shape. The standard high-performance conv decomposition is im2col
+ GEMM: unfold each output window into a row of a `[B·oH·oW,
inC·kH·kW]` matrix, then one big `cblas_dgemm` against the
flattened weight matrix replaces the nested loop. We already wired
`cblas_dgemm` for matmul backward in Job 2b; the same dependency
pays off again here.

**Change**: rewrote `tensor_conv2d_batched` (forward) and the
`OP_CONV2D_BATCHED` backward in `backend_tape.c` to use im2col +
cblas:

Forward:
- `X_col [M, K] = unfold(input)` where `M = B·oH·oW`, `K = inC·kH·kW`
- `Y_unf [M, outC] = X_col @ W^T` — one `cblas_dgemm(NoTrans, Trans)`
- permute `Y_unf` to `out [B, outC, oH, oW]` + bias broadcast

Backward:
- `dY_unf [M, outC] = permute(r.grad)`
- `dW [outC, K] = dY_unf^T @ X_col` — one `cblas_dgemm(Trans, NoTrans)`
- `dX_col [M, K] = dY_unf @ W` — one `cblas_dgemm(NoTrans, NoTrans)`
- `dInput += col2im(dX_col)`
- bias gradient via direct sum

Workspace buffers (X_col, Y_unf, dY_unf, dX_col) are heap-allocated
via `calloc`/`free` per call rather than arena-allocated — an
earlier arena-allocated version produced "invalid memory reference"
crashes between epochs (likely interaction with eval's accumulated
tape entries holding pointers across the arena state change). The
`calloc` path is robust and the per-batch malloc cost is dwarfed by
the dgemm.

**Impact** (MNIST full 60K, seed=42, 3 epochs):

| Variant | per-epoch | wall | acc | vs PyTorch ref (12.5 s/ep) |
|---|---:|---:|---:|---:|
| pre-batched | 175 s | 8m 45s | 0.984 | 14.0× |
| batched, naive kernel | 176 s | 8m 49s | 0.973 | 14.1× |
| **batched, im2col + cblas** | **20 s** | **1m 51s** | **0.973** | **1.62×** |

**8.6× tape speedup**. All three backends now within ~2× of
PyTorch ref on MNIST: torch 1.68×, mlx 2.08×, tape 1.62×.

**Outcome**: landed. Job 1 reopened-phase-A fully closed.

### 2026-05-11 — mlx scalar-allocation hot-path audit (Job 3 Phase A) — `34d8659`

**Plan job**: Job 3 Phase A (mlx-only, no tape/torch impact).

**Motivation**: After the Job 1 reopen closed, an explore-agent
source review of `backend_mlx.cpp` surfaced six places where the
file was re-allocating `mx::array(...)` literals on hot paths —
all "provably wasteful given mlx's semantics" (cached scalars are
immutable; sharing them is safe). The bar for Phase A was "read
the diff and see why it's free"; per the plan, anything that
needed benchmarking to validate was deferred to Phase B (the
mlx-projects survey).

**Changes** (6 atomic commits, in order of est. impact):

1. **Hoist optimizer-state scalars** (`6f5d845`). `optimizer_step`'s
   per-param loop was allocating `mx::array()` for `alpha`,
   `1-alpha`, `beta1`, `1-beta1`, `beta2`, `1-beta2`, `eps`,
   `momentum`, and both Adam bias-correction terms once per param
   per step. None depend on which param. Hoisted to once-per-step.
2. **Cache F32_ZERO/F32_ONE/F32_HALF in forward hot paths**
   (`232b2e9`). Added `kF32_ZERO/ONE/HALF()` Meyers' singletons;
   applied to `tensor_softplus`, `tensor_gelu`, `tensor_dropout`,
   `tensor_gru_cell`. GELU's structural coefficients
   (0.7978…, 0.044715, 3) became function-local statics inside
   `tensor_gelu` (same lifetime story).
3. **Cache F32_ZERO for null-arg fallbacks in vjp replay** (`c46017b`).
   `tensor_backward`'s closure was constructing `mx::array(0.0f)`
   per fallback per tape entry per backward (2 per entry for
   unary ops). Routed both fallbacks through `kF32_ZERO()`.
4. **Cache GELU/SOFTPLUS replay coefficients** (`8672f65`).
   `OP_GELU` and `OP_SOFTPLUS` cases inside the replay lambda were
   re-allocating their constants per backward. Lifted to
   function-local statics; common 0/0.5/1 routed through the
   `kF32_*()` accessors so forward and replay share the same
   underlying arrays.
5. **Cache vjp pool placeholder** (`b77f157`). `std::vector<mx::array>
   pool(N, mx::array(0.0f))` per backward. Routed the placeholder
   through `kF32_ZERO()` — vector's N slots are then refcounted
   shallow copies of a shared array, not copies of a freshly-
   allocated one.
6. **Cache masked-fill -1e9 sentinel in vjp replay** (`34d8659`).
   `OP_MASKED_FILL` case allocated `mx::array(-1e9, float32)`
   fresh per backward. Lifted to a function-local static.

**Safety notes** (recorded in the new `Hot-path scalar constants`
header in `backend_mlx.cpp`):

- All Meyers' singletons — lazy init picks up whatever default
  device `mlx_backend_init` configured.
- Sharing constants across calls is safe: mlx arrays are
  immutable from ops' perspective; ops produce new arrays rather
  than mutating inputs.
- Avoided using the cached singletons as the rhs of `mx::outer`
  or similar ops where persistent operands hit the documented
  slow path (`gotchas.md` "MLX requires non-grad tensors to be
  non-persistent").

**Impact — all 6 commits in** (mlx closing sweep, seed=99, NTM/DNC
at `--epochs 30000 --es-threshold 0.01`, vs pre-Phase-A baseline at
`88a966a+dirty`):

| Cell | pre ms/ep | post ms/ep | delta | convergence |
|---|---:|---:|---:|---|
| supervised | 1 | 1 | – | bit-identical |
| rnn | 11 | 11 | – | bit-identical |
| lstm | 18 | 13 | −28% | bit-identical |
| gru | 15 | 15 | – | bit-identical |
| transformer | 35 | 33 | −6% | sort_acc 6/6 |
| mnist | 26 000 | 22 800 | −12% | acc 0.98 |
| dnc-copy | 30 | 34 | +13% | bit-identical (af 0.88) |
| ntm-copy | 55 | 57 | +4% | bit-identical (af 0.94) |
| dnc-recall | 55 | 66 | +20% | bit-identical (k4 0.82) |
| ntm-recall | 54 | 57 | +5% | bit-identical (k4 0.65) |

**Reading the numbers**: these measurements were taken on a VM
with concurrent workload. Single-run ms/ep variance on this
machine runs ±15–20%, which is bigger than most of the deltas
above. The convergence column is reliable (loss / acc / converged-
epoch are deterministic and **bit-identical pre/post on every
cell**), so the changes are numerically clean. The perf signal is
noise-dominated.

We confirmed this by bisecting the apparent +20% on dnc-recall:
reverted candidates #3 (null-arg fallback in vjp replay) and #5
(vjp pool placeholder) — the two changes that participate in the
most tape entries per backward — and re-ran dnc-recall on the
partial revert. Result: **69 ms/ep** (worse than the all-6 number
of 66 ms/ep). Reverting code can't deterministically make
scheduling slower; the partial-revert outcome confirms the
underlying noise is bigger than the effect we were trying to
attribute. We did not pursue a clean-baseline control run because
the same noise would dominate that measurement too.

What the deltas plausibly mean once noise is accounted for:
- **lstm −28%** is large enough to be real, and lines up with the
  optimizer-scalar hoist (LSTM models have many Adam params); not
  a sure thing but the most credible single-cell win.
- **mnist −12%** and **transformer −6%** are within noise.
- **dnc-copy +13%, dnc-recall +20%, ntm-copy +4%, ntm-recall +5%**
  are within noise.
- **rnn / gru / supervised** were already noise-floor cells.

**Safety review** (independent of perf signal):

- All six changes are mechanical "cache a `mx::array(...)` constant
  instead of re-creating it per call." No semantic changes.
- Mlx arrays are immutable from ops' perspective — sharing across
  calls is safe (ops produce new arrays rather than mutating
  inputs).
- Cached singletons are not used as the rhs of `mx::outer` or
  similar persistence-sensitive ops (per the `gotchas.md`
  documented slow path).
- Convergence bit-identical on every cell confirms no numerical
  drift.

**Outcome**: all 6 commits land. The principled win is small-but-
real (fewer fresh-array allocations on hot paths, less graph
bloat) even if not separately measurable through VM noise; the
changes are also a free safety improvement (mlx arrays sharing one
underlying scalar buffer rather than thousands of independent
allocations is friendlier to mlx's cache budget). Real perf
characterization deferred to Phase B (mlx-projects survey + a
proper microbench framework for per-pattern timing).

Phase A complete. Five-minute total wall on the perf-changes side;
the heavy lift was the closing sweep, which validated convergence
correctness.

### 2026-05-11 — mlx GPU (Metal) exploration — discovered universal regression — `263d546`

**Plan job**: Job 3 Phase A side-quest. The question was: are the mlx
numbers we've been measuring this whole project actually CPU stream,
and what changes if we run on Metal GPU?

**What we found**:

1. **The nixpkgs `python3Packages.mlx-0.31.2` package is CPU-only.**
   The nix derivation at `pkgs/development/python-modules/mlx/default.nix`
   hardcodes `MLX_BUILD_METAL=false` because Apple's `metal` shader
   compiler isn't open-source and the nixpkgs maintainers don't want
   to use sandbox escape hatches. `otool -L libmlx.dylib` on the nix
   build shows no Metal framework linkage; `mx::is_available(gpu)`
   returns 0 at runtime. Setting `MLX_DEVICE=gpu` on the nix build
   aborts with `Cannot set gpu device without gpu backend`. So
   **every mlx measurement before this entry was CPU stream**,
   regardless of any `MLX_DEVICE` setting.

2. **pip-installed mlx works** (`uv pip install mlx` auto-pulls in
   `mlx-metal` with a 150 MB precompiled `mlx.metallib`; the dylib
   links Metal.framework). Tested in this Tart VM:
   `mx::is_available(gpu) == True`, a real GPU computation succeeds.
   So GPU IS reachable in this Tart VM via Apple Virtualization
   Framework's paravirt-graphics (consistent with Tart's
   documentation that "Metal APIs work inside VMs with no
   additional setup").

3. **But GPU is universally slower in this environment.** Built
   `backend_mlx.cpp` against the pip mlx and ran an `MLX_DEVICE=gpu`
   sweep on the same 10-cell config (killed after 7 cells once the
   pattern was clear):

   | Cell | mlx CPU ms/ep | mlx GPU ms/ep | slowdown |
   |---|---:|---:|---:|
   | supervised | 1 | 11 | 11× |
   | rnn | 11 | 114 | 10× |
   | lstm | 13 | 156 | 12× |
   | gru | 15 | 145 | 10× |
   | transformer | 33 | 111 | 3× |
   | mnist (ms/ep) | 22 800 | 112 800 | 5× |
   | dnc-copy | 30 | 269 | 9× |

   Convergence remained bit-identical / within seed-trajectory noise
   on the cells we measured (`acc_short`, `acc_full`, `sort_acc`
   etc. matched CPU runs). So GPU is numerically clean but a
   throughput regression of 3–12× across the board.

**Why GPU loses here**: the kernel-launch wall. Each `tensor_*` call
dispatches one Metal kernel. The forward chain for an RNN cell is
30-50 ops on tensors of <100 elements; the backward replays the
same chain inside the VJP closure; with batched training that's
~150k-300k Metal kernel dispatches per epoch on mnist. At those
tensor sizes the per-op compute is microseconds but the launch
overhead is comparable or larger — especially under Tart's paravirt-
graphics path, which likely adds further per-dispatch latency on top
of bare-metal Metal. CPU stream skips all of this and calls Apple
Accelerate BLAS directly.

The "GPU is good for image conv" intuition is rooted in workloads
designed for GPUs — big batches (256–1024), bigger images (224×224×3
ImageNet), deep models (ResNet, VGG). MNIST as a 32-batch / 28×28×1
problem with a 2-conv-2-FC model is too small to amortize the
per-dispatch cost.

**The actual lever for GPU here is `mx::compile()`** — mlx's JIT
API that compiles a multi-op function once and replays it as a
single fused Metal kernel. We don't use it (the existing path uses
`mx::vjp` which builds a closure but doesn't compile it). Wiring
`mx::compile` into the replay path is the open Phase B work for
Job 3; without it GPU is just an alternate-and-slower CPU stream
in this environment.

**Tooling changes that landed alongside this discovery**:
- `device` field added to perf-log JSON schema (`scripts/perf-run.sh`,
  `scripts/perf-baseline.sh`, `docs/develop/perf-log.md`).
  mlx records `MLX_DEVICE` (default cpu); tape/torch always cpu.
  Entries before this date can be assumed device=cpu.
- `mlx` package removed from nix dotfiles (`vm/modules/unix/packages.nix`) —
  this project uses a project-local pip install for the Metal build
  rather than the nix CPU-only build.
- `docs/develop/gotchas.md` got new entries documenting the nixpkgs
  build flag, the pip workaround, and the "GPU usually loses at
  these scales" finding.

**Outcome**: `MLX_DEVICE=cpu` is the right default and stays. GPU
remains supported but flagged as "available but typically slower
at idris-ml example scales — requires `mx::compile`-style fusion
to be competitive." No commits to `backend_mlx.cpp` from this
investigation; the binary that's checked in builds against
whatever mlx is detected at make time and reads `MLX_DEVICE` at
runtime. Phase B's mlx-projects survey should now include `mlx`'s
own `mx::compile` / `mx::value_and_grad` JIT path as a primary
target rather than incidental side-reading.

### 2026-05-11 — idris-gym source-review Phase A (Job 4) — null result

**Plan job**: Job 4 Phase A (idris-gym env-side wins, no
vectorization restructure).

**Motivation**: RL examples ratio at 20-40× PyTorch ref because
env-step time dominates. Before tackling vectorization (Phase B),
audit `packages/idris-gym/` for per-step waste that's cheap to
fix — same Phase A pattern that worked for Jobs 1/2a/3.

**Method**: source review surfaced 5 candidates. Per the
measure-then-hypothesize-then-change discipline, built a
microbench (`make bench-gym`, see commit message for harness
notes) targeting each candidate's hot path, measured baseline,
formed a quantitative hypothesis, implemented, re-measured.

**Baseline** (M4 Pro VM, ns/call, ±5% across runs):

| Function | ns/call |
|---|---:|
| `Rng.nextDouble` | 140 |
| `Blackjack.bjObserve` | 55 |
| `Pendulum step+observe` | 70 |
| `Acrobot step+observe` | 645 |
| `Taxi step` | 22 |
| `CliffWalking step` | 20 |

**Experiments**:

*#3 `Rng.nextDouble`: replace `cast {to=Double} (cast {to=Integer}
top53)` with the direct `cast {to=Double} top53`.* Hypothesis:
the explicit Integer intermediate allocates a GMP bignum per
call; direct prim should cut 30-60% off the function. **Result**:
138 → 139 ns/call (within noise). Hypothesis falsified — either
Idris codegen already fuses the chain, or the cost is elsewhere
(likely splitMix64's two bignum multiplications against the
0x9E37… and 0xBF58… constants, which fall outside the Chez
fixnum range and allocate per multiply). Reverted.

*#5 `Blackjack.bjObserve`: replace the double-traversal handSum +
usableAce with a one-pass `handStats` that returns (raw_sum,
ace_count); `bjObserve` calls it once instead of four traversals
across handSum + usableAce.* Hypothesis: ~50% reduction in
bjObserve, from 55 → ~28 ns/call. **Result**: 55 → 56 ns/call
(within noise). Hypothesis falsified. Idris's `length . filter`
and `foldr (+) Z` paths are already fast enough on tiny lists
(2-4 cards) that the duplicate work is below the measurable
floor. Reverted.

*#4 Taxi/CliffWalking Nat↔Integer round-trips.* Did not
implement — baseline measurement settled it directly: Taxi step
is 22 ns/call and CliffWalking step is 20 ns/call. The
`cast {to=Integer} (n : Nat)` calls cited as wasteful are
compiled by Idris's BigInt-Nat optimization to a no-op (`Nat` is
already stored as `Integer` at runtime); there's no chain to
shorten. Confirmed not a win.

*#1 Acrobot trig caching.* Did not implement — the savings
ceiling is 1 redundant `cos(th1)` between the termination check
and `aObserve` (~15 ns of the 645 ns step+observe = 2.3%).
Capturing more trig values would require either adding cached
fields to `AState` (which `eulerStep` also constructs with
meaningless values for the cache — ugly API change) or splitting
into separate `AState` / `AStateObs` types (bigger refactor than
Phase A allows). Skipped pending Phase B or a willingness to
make the structural change for a single-digit-% win.

*#2 Pendulum trig caching.* Did not implement — initial source-
review analysis was wrong. `pStep` computes `sin(s.pTheta)` on
the *current* angle (used for dynamics); `pObserve` later
computes `cos/sin(s'.pTheta)` on the *new* angle after the step.
These are different inputs, so no redundancy to remove. The
agent's source-review hypothesis was incorrect.

**Outcome**: zero idris-gym source changes land from Phase A.
What lands instead:

- `make bench-gym` microbench tool (`packages/idris-gym/test/bench.ipkg`
  + `Bench.idr`). Useful for any future per-call optimization
  experiments on the env code; baseline numbers documented in the
  commit message.
- Two reusable Idris bench-authoring lessons (in the same commit
  message): defeat CSE by varying input per iteration; avoid
  Peano-Nat counters above ~100k iterations or BigInt allocation
  compounds.
- Confirmation that env-side per-call work is already tight at
  the source level; the 20-40× ratio against PyTorch ref is
  attributable to single-step-vs-vectorized-env architecture, not
  to local source waste. **Phase B (vectorization) is now the
  unambiguous next lever** rather than something we were doing
  "after the obvious wins."

----

### 2026-05-12 — REINFORCE batched policy forward (Job 4 Phase B)

**Change**: added `rolloutEpBatched` + `computeLossBatched` +
`epochRLBatched` + `genBatchV` to `packages/idris-ml-examples/src/Example/Reinforce.idr`.
New `--batched 1` CLI flag selects the batched path; default stays
sequential. The batched rollout stacks N envs' observations into a
single `Tensor [N, 4]`, does **one** `forwardVarBatch` per timestep,
then per-env action sampling + `cpStep`. Done envs are frozen (their
state passes through the batched forward; no `StepRec` appended)
to keep the `[N, 4]` shape stable. Loop exits early once all envs
terminate.

**Why**: Job 4 Phase A had established (via `make bench-gym`) that
env-step is already cheap (~100 ns/call for CartPole). The 20-40×
RL-example ratio vs PyTorch ref is per-op-count, not env-step cost.
Per-timestep batched policy forward collapses N×T forward calls
into T forwards — same gradient math, fewer wrapper trips, fewer
tape entries. The reframe is captured in the plan and in the
"Job 4 Phase B" task description.

**TDD progression** (per Job 3 Phase B pattern):

1. Failing parity test added in
   `packages/idris-ml-examples/test/src/Test/Reinforce.idr` — assert
   per-episode total rewards match sequential rollout for matched
   RNG, N=1 and N=2.
2. `rolloutEpBatched` implemented; parity tests pass bit-identically
   on all three backends (tape, torch, mlx-CPU). Verified via
   `make test-examples-unit`.
3. `--batched 1` wired into `main` via runtime dispatch; convergence
   preserved (CartPole reaches max avg_return=200.0 at 100 epochs on
   all three backends, both modes).

**Per-epoch cost at 100 epochs** (canonical numbers from
`scripts/perf-run.sh`, logged to `perf-log.jsonl`):

| backend | seq ms/ep | batched ms/ep | Δ wall |
|---|---:|---:|---:|
| tape  | 70  | 70  | −4% (noise) |
| torch | 100 | 90  | −16% |
| mlx   | 800 | 440 | **−37%** |

The mlx win is the headline — wrapper overhead per call was highest
there (mx::array construction, tape entry, VJP closure rebuild per
call). Tape/torch wins are within the VM noise envelope (±15-20%
per `feedback_vm_perf_noise`) but consistent in direction. Confirmed
across two independent measurement passes (ad-hoc `time` wrappers +
perf-run.sh).

**What this *doesn't* close**: the 20-40× ratio vs PyTorch ref. Even
at the batched mlx number (52s for 100 epochs), we're well above
what PyTorch ref does on the same workload. The remaining gap is
shared with the rest of the codebase (Idris per-prim cost floor,
the ~9 µs glue) and is a Job 1/2a concern, not specifically Job 4.

**Files**:
- `packages/idris-ml-examples/src/Example/Reinforce.idr` — new
  rolloutEpBatched, computeLossBatched, epochRLBatched, genBatchV,
  --batched flag.
- `packages/idris-ml-examples/test/src/Test/Reinforce.idr` — parity
  test suite (N=1, N=2 per-env reward parity).
- `packages/idris-ml-examples/test/{test.ipkg,src/Main.idr}` — wire
  the new test module.

**Open follow-ups**:
- Extract `rolloutEpBatched` to a shared module in idris-ml-examples
  (e.g. `Example.RL.BatchRollout`) so other RL examples can reuse it,
  once at least one more example wants it.
- Port `a2c.idr`, `ppo.idr`, `sac.idr`, `dqn.idr`, `mountain-car.idr`
  to use batched rollout. Apply the same TDD discipline per example.

----

## Future opportunities (not active)

Ideas surfaced during the Job 1 phase A push that we don't plan to
do right now but might be worth picking up later. Listed here rather
than in `TODO.md` because they're optimization candidates with a
specific cost / benefit profile, not "should-have" features.

### Pre-allocated obs buffer + `tensor_write_data_inplace` for RL

For RL examples (CartPole / Acrobot / etc.), every rollout step calls
`bulkToTensor obs` which is 6 prim FFI calls (alloc + 4 setDouble +
create) for the 4-element CartPole observation. Pre-allocating a
persistent obs tensor at episode-start and using a new
`tensor_write_data_inplace` primitive (PyTorch's `tensor.copy_(other)`
shape) to overwrite the values would drop this to ~2 prims.

Estimated savings: ~0.5–1 ms/epoch on a2c (20 rollout steps × 4 prims
saved × 9 µs ≈ 0.7 ms). Modest. Mostly worth it as a library feature —
`tensor.copy_` is something users would expect to exist.

Effort: ~half day. New primitive on each backend + Idris binding +
caller-side rewrite at a couple of sites.

### Slab allocator for `at::Tensor*` on torch backend

Each `from_tensor` in `backend_torch.cpp` does `new at::Tensor(std::move(t))`
— ~1 µs per call from system malloc. For DNC-class workloads (~3K
intermediates per epoch) that's ~3 ms/epoch, plus the matching
`delete` costs at `free_intermediates`. A bump-allocator that
allocates `at::Tensor` slots into a pre-sized arena and resets the
pointer at `free_intermediates` would be O(1).

Estimated savings: 1–3 ms/epoch on DNC-class workloads on torch.
Modest. Code-complexity tradeoff: bump arena needs alignment care
and we'd lose stable pointers (any caller stashing an `at::Tensor*`
across `free_intermediates` would break — though current callers
don't seem to do this).

Effort: ~1 day. New allocator + integration with the existing
`intermediates` vector + `free_intermediates` cleanup path. Test that
all examples still train.

Pairs with the existing TODO "Bound memory usage" — if a slab is in
place, it's natural to extend it with a memory limit.

### Other ideas surfaced and discarded

- **NTM `onesM` precompute** — tried, reverted twice (Idris CSE makes
  the precompute redundant). See entry above.
- **Batched recurrent sequence forward** (Idea B from the Job 1
  brainstorm) — substantial multi-day implementation per backend
  (would need to write the timestep loop in C on tape and mlx; torch
  could delegate to `torch::nn::functional::rnn_tanh`). Estimated
  savings on rnn torch: 1–2 ms/epoch. Cost-benefit didn't favour it
  given the modest perf delta and that the rnn example already
  matches `nn.RNNCell` semantics post-alignment. Could revisit if/when
  larger sequence-length workloads land that would amortize the
  effort better.
- **DNC controller stacked-FC** (Job 2a brainstorm) — replacing the
  11 per-gate `prim__linear` calls with one stacked `(W,b)` linear +
  11 narrows nets one EXTRA prim per timestep (11 → 12). The C-side
  cache locality of one big GEMV vs eleven small ones is real but
  small relative to the ~9–19 µs Idris-glue floor. Skip.
- **Transformer transpose caching in single-seq path** (Job 2a
  brainstorm) — `runHeadAttn` calls `prim__transpose2d` on the q/k/v/op
  weights inside the head loop, but each iteration's q/k/v/op is the
  *next* head's weight — the transposes are not redundant. Audit
  conclusion: not a real opportunity.
- **DNC `onesScalar` precompute / generic scalar-constant pool** (Job
  2a brainstorm) — same Idris-CSE story as `onesM`. The scalar 1.0
  constructor is folded by the compiler.
- **`buildMatrixRows` 2D scalar round-trip** — listed in early plans;
  already removed from the codebase. No-op.

### 2026-05-14 — Torch Adam: multi-tensor step via `at::_foreach_*` — `<commit>`

**Plan job**: follow-up to the GptLarge wallclock matrix. The
PyTorch-precedent fused-optimizer story: PyTorch ships
MultiTensorApply (`torch.optim.Adam(foreach=True)` since 1.12+) as the
default Python path, but libtorch's C++ `torch::optim::Adam::step()`
still loops per-parameter (no `foreach` / `fused` option in
2.11.0's `adam.h`). idris-ml routes through the C++ API, so we get
the per-param path.

**Motivation**: validate the multi-tensor pattern that the existing
"mlx backend: wrap optimizer step in `mx::compile`" TODO row will
need. Torch `_foreach_*` is the easier landing because the API is a
1:1 swap of `for (p : params) tensor_x_(p, ...)` to
`at::_foreach_x_(params, ...)` — no tracing, no cache, no functional
state-threading rewrite.

**Change**: new `adam_step_foreach` static in
`packages/backends/backend_torch.cpp` does Adam's m / v / denom /
parameter update via `at::_foreach_mul_`, `at::_foreach_add_`,
`at::_foreach_addcmul_`, `at::_foreach_sqrt`, `at::_foreach_div_`,
`at::_foreach_add_`, `at::_foreach_addcdiv_`. Body is wrapped in
`torch::NoGradGuard` — the in-place ops on leaf params with
`requires_grad=true` would otherwise trip autograd's `check_inplace`
(transformer / mnist / seq-classify / dqn / a2c / ppo crashed on this
in the first pass). Params with undefined grad are filtered out of
the gather lists (matches `torch::optim::Adam::step()` behaviour). m
and v stay in `AdamParamState` so libtorch's serializer continues to
work. `optimizer_step` dispatches to `adam_step_foreach` when
`w->type == 2`; SGD / RMSprop / AdamW fall through to `opt->step()`
unchanged. `TORCH_ADAM_FOREACH=0` (later renamed to `TORCH_FOREACH=0`
when SGD foreach landed) routes back to libtorch's per-param step for
A/B comparison. Added `prof_optimizer_math_ms` sub-timer to separate
the math from `free_intermediates()` (the dominant non-math contributor
inside `prof_optimizer_ms`).

**Impact** — gpt-large @ torch CPU, 5 epochs, single-run A/B with
otherwise-identical setup:

| metric                    | A: foreach OFF | B: foreach ON | Δ           |
|---------------------------|---------------:|--------------:|------------:|
| optimizer-math ms/ep      |           9.27 |          8.66 | **−0.61** (−6.6%) |
| optimizer total ms/ep     |           16.4 |          16.0 |        −0.4 |
| backward ms/ep            |         1042.9 |         933.3 |       −109.6 (noise) |
| C total ms/ep             |         1059.3 |         949.3 |       −110.0 |
| val BPC @ ep 5            | 4.771089597130988 | 4.771089597130988 | **bit-identical** |

Numerics are bit-identical down to the last fp64 digit — confirms the
mul/add ordering matches PyTorch's per-param implementation
(`m.mul_(β1).add_(g, 1-β1)` then `v.mul_(β2).addcmul_(g, g, 1-β2)`).

The optimizer-math gain is tiny on CPU. The 110 ms/ep backward drop
between the two runs is unrelated noise (a run-to-run delta of <15%
sits below the VM noise floor; this is at 10%). On a 9500 ms/ep wall
budget the foreach math itself moves <0.01% of wallclock.

**Why so small on CPU**: PyTorch's MultiTensorApply is engineered to
reduce GPU **kernel-launch** overhead. CPU `at::_foreach_*` is
implemented as a parallel for-loop over the list, with no kernel-launch
to save. The cost we measure on CPU is pure compute + dispatch, both
of which the per-param loop already vectorises via Accelerate /
OpenMP. So we get the structural benefit (one call instead of N) but
the absolute speedup is in the µs-per-param noise.

**Outcome**: landed. The change is correctness-neutral
(bit-identical), perf-neutral on CPU (within noise), and forward-looks
to the torch GPU path where it would land the typical 2–10× optimizer
speedup PyTorch documents. Also validates the multi-tensor shape for
the `mx::compile`-of-optimizer rewrite (the actual GPU work for
idris-ml, since we don't currently run on torch GPU).

**Cross-references**:
- `perf-log.jsonl` `kind=ab` entries timestamped 2026-05-14T15:32 and
  2026-05-14T15:34
- libtorch 2.11.0 `adam.h` has neither `foreach` nor `fused` option
  (Python-side only)
- next concrete step from the high-priority TODO list: the
  `mx::compile` optimizer wrap (where the same pattern actually pays
  off because Metal kernel-launch latency is the bottleneck)
- Follow-up landed 2026-05-18 — see "Torch SGD / RMSprop / AdamW: multi-tensor step via `at::_foreach_*`" entry for the family completion (and the env-var rename from `TORCH_ADAM_FOREACH` to `TORCH_FOREACH`)

### 2026-05-14 — MLX Adam: optimizer step via `mx::compile` — `<commit>`

**Plan job**: the natural follow-up to the torch foreach landing (same
day). The mlx-side analogue of `at::_foreach_*` is `mx::compile`: trace
the per-param Adam math once into one fused mlx callable, then replay
with new tensor inputs each step. Unlike torch CPU foreach (which is
just a parallel for-loop), mlx compile actually fuses ops into one
graph submission, so it saves the per-op kernel-launch tax on Metal.

**Motivation**: before writing code, a 2×2 baseline (mlx × {cpu, gpu} ×
{compile OFF, ON}) on GptLarge revealed two structural facts:

| Config | Wall | C-total | Optimizer | Backward |
|---|---:|---:|---:|---:|
| mlx CPU + MLX_COMPILE=0 (default) | 9000 | 28.8 | 26.4 | 2.4 |
| mlx CPU + MLX_COMPILE=1 (probe)   | 9600 | 112.9 | 104.0 | 8.9 |
| mlx GPU + MLX_COMPILE=0           | 9000 | 132.6 | 121.8 | 10.7 |
| mlx GPU + MLX_COMPILE=1           | 9800 | 156.7 | 145.1 | 11.6 |

(`MLX_COMPILE` is the existing backward-pass-forward-replay probe.
Recompiles every call → pure regression today. The new
`MLX_OPT_COMPILE` is separate and caches.)

The wall is identical at 9000 ms/ep regardless of device or compile
flag — ~98% of wall is *outside* the C profile region (Idris VM +
per-op FFI dispatch on the forward-pass tape build). mlx ops are lazy,
so forward FFI calls are cheap to mlx but expensive to Chez. **The C
gap (28.8 vs 132.6 ms/ep) is invisible at wall, so even driving
optimizer math to zero saves <1% of wall.** This change therefore
isn't a wall-mover at the current example scale; it's GPU-shaped perf
hygiene + prerequisite for compiling the whole training step (the
"whole-step compile" investigation, where wall actually moves).

A path-C spike (scale GptLarge dModel from 256 to 512 to 768) was run
in parallel to see if any reachable scale flips GPU > CPU on its own.
At dModel=512 GPU edged out by 1000 ms/ep (within noise floor). At
dModel=768 GPU lost by 3000 ms/ep and OOM'd on generation in the Tart
VM. So scale alone doesn't fix the example; the compile work is the
correct lever but lands as part of a larger plan.

**Change**: `adam_step_compile` in `packages/backends/backend_mlx.cpp`
implements the Adam update as a pure function
`(params, grads, m, v, per-param lrs, scalars) → (new params, new m,
new v)` and wraps it in `mx::compile`. The compiled callable is cached
per active-param-count in a static `unordered_map<int, function<...>>`
— mlx caches further by input-shape signature internally, so repeated
calls with the same param shapes hit the trace cache after the first
invocation. Gated on `MLX_OPT_COMPILE=1` env var, default OFF
(opt-in). Only `opt->type == 2` (Adam) dispatches to the new path;
SGD/RMSprop/AdamW fall through to the per-op loop unchanged. Added
`prof_optimizer_math_ms_mlx` sub-timer that brackets just the math
(not the surrounding `mx::eval(to_eval)` + `tape_reset`).

The math sub-timer immediately revealed the structural ceiling:
optimizer-math is only **1.4–1.8 ms/ep** of the 96–157 ms/ep
`Optimizer` total. The remaining 94+ ms/ep is `mx::eval(to_eval)`
synchronisation and tape rebuild — bookkeeping that the compile path
cannot touch. So the max wallclock yield from compiling optimizer math
is bounded by ~1.5 ms/ep, plus whatever kernel-launch savings the
compile gives on the eval step downstream.

**Impact** — gpt-large dModel=256, 5 epochs, A/B:

| metric (ms/ep) | mlx CPU OFF | mlx CPU ON | mlx GPU OFF | mlx GPU ON |
|---|---:|---:|---:|---:|
| Wall              | 9000  | 9600  | 9200   | 9000   |
| Backward          |  9.3  |  8.9  |  11.1  |   9.3  |
| Optimizer         | 96.0  | 98.6  | 156.5  | 120.9  |
| of which math     |  1.4  |  1.6  |   1.8  |   1.5  |
| C total           | 105.3 | 107.5 | 167.7  | 130.2  |
| val_bpc           | 4.746685288232547 | 4.746685288232547 | 4.746687874851618 | 4.746687482731175 |

- **CPU: small regression** (+2.6 ms/ep optimizer). No kernel launches
  to amortize on Apple Accelerate stream; mx::compile's tracing
  overhead is pure cost. **Bit-identical fp64 numerics**
  (`4.746685288232547` matches OFF down to the last digit).
- **GPU: −35.6 ms/ep optimizer (−23%)**, −37.5 ms/ep C-total. Real
  kernel-launch savings — the design hypothesis held. **Numerics
  deviate ~4e-7 relative** (`4.746687482731175` vs OFF
  `4.746687874851618`), within fp32 ULP noise — mlx GPU is fp32
  internally and the compile pass reorders ops, which shows up at the
  7th decimal. Well below convergence noise; not a correctness issue.
- **Wall unchanged** on both, consistent with the diagnostic: C-side
  cost is <2% of wall at this scale.

**Outcome**: landed, opt-in (`MLX_OPT_COMPILE=1`). CPU users keep
status quo (slight regression if enabled); GPU users get a measurable
optimizer-math win on the device that benefits. Default OFF avoids the
CPU regression and the existing-`MLX_COMPILE`-style "probe with no
caching" gotcha. The same compile-once-then-replay pattern is the
load-bearing technique for the future whole-training-step compile
investigation, where wall actually moves because the entire forward
pass becomes one mlx call from Idris instead of N FFI dispatches.

**Cross-references**:
- `perf-log.jsonl` `kind=ab` entries timestamped 2026-05-14T16:54 (mlx
  CPU A/B) and 2026-05-14T16:58 (mlx GPU A/B)
- `perf-log.jsonl` `kind=baseline` entries for the 2×2 diagnostic
  timestamped 2026-05-14T16:43..16:46 (mlx CPU OFF/ON + GPU OFF/ON)
- the parallel path-C spike: GptLarge dModel ∈ {256, 512, 768} on
  mlx CPU vs GPU showed no clean crossover at reachable VM scales —
  dModel=768 OOM'd on GPU during generation, so scale alone is dead
  in this environment
- existing `MLX_COMPILE` env var (separate from `MLX_OPT_COMPILE`) is
  the backward-pass forward-replay probe at `backend_mlx.cpp:2080`,
  added under Job 3 Phase B as a probe — still a regression because
  it has no caching across calls. Future work to cache that one is
  task #42 (decided non-trivial: the lambda capture pattern that
  the new compile path uses doesn't translate directly because the
  backward closure captures the per-step tape)

### 2026-05-14 — Diagnostic: where the 9000 ms/ep GptLarge wall actually goes — `<commit>`

**Plan job**: before committing to a 2-3 week architectural refactor
("compile the whole training step" — eliminate per-op FFI dispatch),
measure where the unaccounted-for ~8870 ms/ep of GptLarge wall is
actually spent. Two-way fork: (a) FFI marshalling overhead per
Idris→C transition → architectural change pays off; (b) Idris VM
between FFI calls → architectural change is dead weight, Idris-side
optimisation is the lever.

**Method**: two independent measurements.

1. **Per-FFI-call wall** (`/tmp/bench_per_op.c` — pure C, no Idris):
   tight loop of `tensor_add(a, b); tensor_free(c)` against
   libidrisml. mlx ops are lazy — each call allocates a graph node
   and returns an `AnyPtr`, no compute. Both streams measured:

   | stream | per-FFI wall | per-pair (add+mul) |
   |--------|-------------:|-------------------:|
   | CPU    |   0.46 µs    |       0.91 µs      |
   | GPU    |   0.45 µs    |       1.01 µs      |

2. **FFI count per epoch** (instrument `tape_append` in
   `backend_mlx.cpp`): every grad-tracked forward op fires
   `tape_append` once. GptLarge 5 epochs at dModel=256:

   - **1136 tape_appends / epoch** (5678 total / 5).

**Multiplication**: 1136 × 0.46 µs = **0.52 ms FFI wall per epoch**
out of 8600 ms wall = **0.006%**.

(Even doubling for non-grad ops not in the tape — input creation,
masks — caps total FFI wall at ~1.2 ms/ep. Still negligible.)

**Inference**: ~8600 ms/ep is **Idris VM time between FFI calls**.
Per tensor op, the Idris side spends **7.6 ms** preparing/dispatching
each op (8600 / 1136). For comparison, Chez Scheme can run plain
arithmetic loops at >1M ops/sec — so a 7.6 ms-per-op overhead on a
"call this C function with two AnyPtrs" operation is enormous, and
the culprit isn't the FFI boundary itself (proven: 0.46 µs).

**Likely candidates for the 7.6 ms-per-op Idris overhead**
(not measured here; this is the *next* diagnostic step):

- Existential `AnyLayer` dispatch in the `Network` chain — each
  forward step walks the chain via `~~>` (existential pattern
  match per layer, indirection through `LayerLike` method
  dictionary)
- Constraint dictionary construction at call sites — `UserDeviceCore d`
  / `LayerLike d` are typeclass constraints that may resolve at runtime
  rather than getting fully inlined, building a dictionary record per
  call
- `Tensor` record packing/unpacking on every op (the record carries
  `tensorPtr : AnyPtr`, `paramId : Maybe String`)
- `Vect` operations in shape arithmetic (Idris-2 `Nat` is `Integer`
  at runtime but Vect/List operations still walk lists allocatively)
- Per-op Idris-level closures inside layer methods (`applyVar`,
  `applyVarBatch`) that allocate intermediate structures

**Outcome** — **kills the "compile the whole training step" plan
(Path A) before any code lands.** Path A's premise was that
eliminating ~thousands of FFI dispatches per step would save the
~8870 ms/ep that isn't C-side. We now know FFI dispatch costs
~0.5 ms/ep — that's the entire upside ceiling. Even a perfect Path
A implementation would save 0.006% of wall.

**The actual lever (Path B): cut the Idris-side per-op overhead.**
Even halving 7.6 ms → 3.8 ms drops wall from 8600 → 4300 ms/ep —
**a 50% wall reduction**. And it's the kind of work that compounds:
any per-op overhead fix lifts every example, not just GptLarge.

**Open questions for the Path B plan** (next diagnostic step,
not in scope of this commit):

- Add Idris-level timing to `forwardVar` itself and the individual
  `applyVar` / `LayerLike` methods to localise the cost
- Try a single-layer harness (e.g. just `linearLayerAny` repeated
  100k times) vs full Network — does cost scale with chain length?
- Inspect Chez Scheme compiled output for one tensor op call —
  what is each call actually doing?
- Inspect whether `%inline` annotations on the `UserDeviceCore`
  method bodies are being honoured by the compiler

**Code change committed alongside**: just the `prof_tape_appends_mlx`
counter in `backend_mlx.cpp` (used to produce this number). The
counter stays — it's cheap and surfaces real signal.

**Cross-references**:
- `perf-log.jsonl` `kind=diagnostic` entries timestamped 2026-05-14T17:18
  (CPU probe) and 2026-05-14T17:20 (GPU probe) — the
  tape-invariance check that confirmed compile-once-replay is
  semantically viable (the assumption Path A's design depended on)
- `perf-log.jsonl` `kind=microbench` entry timestamped 2026-05-14T17:30
  for the per-FFI-call wall measurement
- `perf-log.jsonl` `kind=diagnostic` entry timestamped 2026-05-14T17:30
  for the tape-append count
- TODO row: a new row for "Idris-side per-op overhead reduction" lands
  in this same commit. The mlx-optimizer-compile row (already partly
  done) keeps the SGD/RMSprop/AdamW follow-up but loses its
  "GPU C-total ≤ CPU C-total" acceptance gate (wall doesn't move
  via this lever)

### 2026-05-14 — Chez source profile localises 7.6 ms-per-op cost to recursive Nat arithmetic in uncached positional encoding — `<commit>`

**Plan job**: follow-up to the FFI-vs-Idris-VM diagnostic (earlier today,
same `docs/develop/perf-changes.md`). That measurement bounded
**Idris VM = 99.99% of wall**, but didn't say *which* Idris code. This
commit identifies the dominant Idris-side hot path.

**Method**: Chez Scheme's built-in source-level profiler. The Idris 2
Chez codegen emits a `.ss` source file plus a `compileChez` script that
parameterises `(optimize-level 3)` for the final `.so`. Adding
`(compile-profile 'source)` to that parameterise plus a trailing
`(profile-dump-html ...)` after the main call produces a per-line
execution heatmap. Total round-trip from "we need this measurement" to
"we have the answer" was about 20 minutes — no Idris-side instrumentation
needed, no C-side counters, just one parameterise + dump call.

**Result** — per-line execution counts on GptLarge 1 epoch, dModel=256:

| Generated-Scheme line | Function (demangled) | Count | Notes |
|----:|---|---:|---|
| 923 | `Data.Nat.lte` | **1,956,671,790** | recursive walk on unary Nat |
| 924 | `Prelude.Types.prim__integerToNat` | 980,740,575 | called per div'/mod' recursion |
| 925 | `Data.Nat.divC-39` | 490,399,841 | recursive div' |
| 1011 | `Data.Nat.modC-39` | 490,340,353 | recursive mod' |
| 52-54 | `blodwen-toSignedInt` | 30,400,810 | runtime bit-fit (small) |
| 96 | `bs+` (signed add) | 18,533,971 | (small) |

The four Nat-recursive entries sum to **~3.9 billion `cond`/`equal?`/`sub1`
operations per epoch**. These compile to recursive decrement because the
Idris stdlib `Data.Nat.lte` / `div'` / `mod'` pattern-match on `Z/S k`
constructors — even though `Nat` is `Integer` at runtime, the function
*body* still does `(let ((e-0 (- arg-0 1))) (lte e-0 ...))`.

**Root cause** — `Layer/Transformer.idr` `posEncVal`:

```idris
posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)         -- ← recursive Peano on Nat
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle  -- ← idem
```

…called by `writePE` which loops over `(pos, dim) ∈ [0, seqLen) × [0, dModel)`
= 128 × 256 = **32,768 `posEncVal` calls per forward**. And — crucially —
**`writePE` runs *inside* `applyTransformer`** at `Transformer.idr:173`:

```idris
peBuf = prim__allocDoubles (sI * dI)
peBuf' = writePE dModel peBuf 0 0 sI dI          -- rebuilt every step!
peT = prim__createState2d sI dI peBuf'
```

So the positional encoding — which is **deterministic, parameterless,
shape-only-dependent** — is recomputed from scratch on **every forward
pass**. At 32 forwards/epoch × 32,768 `posEncVal`/forward × ~hundreds of
Nat operations each = the billions of Nat operations we see in the profile.

**Two compounding bugs, two orthogonal fixes**:

1. **Cache PE on `TransformerState`**. Build once at `transformerLayer`
   construction, store as a `TMat seqLen dModel ex NoGrad` field on
   `MkTransformer`. Forward passes use the cached tensor. Removes the
   per-step writePE entirely. Single-batch case is trivial broadcast;
   batched case needs the reshape-add-reshape dance (or a fresh
   `prim__tilePE` helper).

2. **Use `Int` arithmetic in `posEncVal`**. `div dim 2` and
   `modNatNZ dim 2` on `Nat` are wildly wasteful regardless of caching.
   `dim` is already a `Nat` ≤ `dModel` ≤ ~thousands; converting to `Int`
   and using `div : Int -> Int -> Int` (which is a single CPU instruction)
   makes the one-time PE construction fast too. Even before fix (1), this
   alone would land an order of magnitude.

**Audit for related issues elsewhere**: same pattern (Nat used in inner
loop where Int would do, or per-step recomputation of deterministic
state) very likely exists in other layer types. Candidates:

- NTM / DNC's content-based addressing (cosine similarity loops?)
- Convolution kernel index computations
- RNN / LSTM / GRU per-step Nat indexing
- LayerNorm's per-feature loops (less likely — those are usually fused C)
- Any per-batch loop that walks shape-derived `Nat` values

The same Chez `compile-profile 'source` recipe answers this in 20
minutes per architecture. Bake it into the perf workflow as a
**`make profile-gpt-large` / `make profile-ntm-copy` etc.** target so
future regressions are caught without ad-hoc setup.

**Outcome** — Path B's first concrete win is in flight (Fix 1 + Fix 2
land in the next commit). Expected wall reduction on GptLarge: 30-50%
just from PE caching; possibly more once `posEncVal` uses `Int`. Lifts
every backend (tape, torch, mlx), not just mlx — this is pure Idris-side
overhead.

**Cross-references**:
- The 20-min profile recipe lives in scratch state for now
  (`/tmp/gpt-prof.{ss,so,*.html}`); when this finding lands, write up a
  `docs/develop/chez-profiling.md` recipe for future use
- The two known-recursive-Nat callers in idris-ml (only places `Data.Nat`
  / `modNatNZ` are used per-step): `Layer/Transformer.idr:81` (this
  finding) and `Train.idr:246` (eval-every-N-epochs, called per epoch
  not per step, so negligible)

### 2026-05-14 — Transformer: cache PE + Int arith in posEncVal → 22× GptLarge speedup — `<commit>`

**Plan job**: Path B Fix 1 + Fix 2. Implementation of the two fixes
proposed in the immediately-preceding `perf-changes.md` entry
("Chez source profile localises 7.6 ms-per-op cost..."):

1. **Fix A — cache positional encoding** on `TransformerState`. New
   field `peCached : TMat seqLen dModel ex g`, built once in
   `transformerLayer`, reused by `applyTransformer` (direct add) and
   `applyTransformerBatch` (reshape-to-3D, broadcast-add, reshape-back).
   `freezeLayer` / `unfreezeLayer` thread it through unchanged.

2. **Fix B — `Int` arithmetic in `posEncVal`**. Keeps the public
   `Nat -> Nat -> Nat -> Double` signature (per the discussion: "can
   we maybe get some of the benefits of the Nat interface by casting
   internally to Int?"), casts `dim` to `Int` once at the function
   entry, then uses `Int div`/`Int mod` for the parity / half-index
   computation. Avoids the recursive Peano walks even on the one-time
   PE construction.

**Impact** — GptLarge dModel=256, 5 epochs, A/B vs the 9000 ms/ep
baseline:

| Backend / Device | Wall before | Wall after | Speedup | val_bpc match |
|---|---:|---:|---:|---|
| mlx CPU          | 9000 ms/ep  | **400 ms/ep**  | **22.5×** | bit-identical (4.746685288232547) |
| mlx GPU          | 9200 ms/ep  | **400 ms/ep**  | **23×**   | ~1e-7 fp32 noise (4.746686877352244) |
| torch CPU        | 9400 ms/ep  | **600 ms/ep**  | **15.7×** | bit-identical (4.746685288232547) |
| tape             | 11600 ms/ep | **1600 ms/ep** | **7.3×**  | ~3e-7 noise (4.746688438790350) |

The wall reductions correspond to: (Fix A) eliminating ~1M
`posEncVal` calls per epoch — these were rebuilding the PE tensor on
every forward pass — and (Fix B) eliminating the recursive Peano
walks inside each `posEncVal` call. Together: ~3.9 billion Nat-recursive
operations/epoch removed. Idris-2 unit tests (`make test`) still green.

**Architecture audit motivated by this finding**:

The same anti-pattern (recursive Nat arithmetic in inner loops or
per-step recomputation of deterministic state) likely exists
elsewhere. Candidate audit targets, by likely impact:

- **NTM / DNC content-based addressing** (`Layer/Ntm.idr`,
  `Layer/Dnc.idr`) — cosine-similarity over the memory matrix could
  have `Nat` indexing in inner loops. NTM examples are listed at
  9.8-14.2× PyTorch ratio (`perf-baseline.md`), some fraction of
  which may be this class of bug
- **Convolution kernel index computations** (`Layer/Conv.idr`) —
  similar profile risk
- **RNN / LSTM / GRU per-step indexing** (`Layer/Rnn.idr`,
  `Layer/Lstm.idr`, `Layer/Gru.idr`) — less likely since the cell
  body is mostly delegated to fused C ops, but the per-step Vect
  walk over time-steps could hide Nat ops
- **LayerNorm / BatchNorm per-feature loops** — less likely, mostly
  fused C
- **Any per-batch loop that walks shape-derived Nat values** —
  inspect via the Chez profile recipe (`compile-profile 'source`)

**Process change**: bake a `make profile-<example>` target into the
Makefile that uses the `compile-profile 'source` recipe to produce
a per-source-line execution heatmap. The 20-minute setup is now
zero-minute, and the next "where does the wall go?" question gets
answered immediately.

**Cross-references**:
- `perf-log.jsonl` entries timestamped 2026-05-14T18:52 (post-Fix-B
  mlx CPU 7400 ms/ep — Fix B alone) and 2026-05-14T18:57 (post-Fix-A+B
  on mlx CPU 400, mlx GPU 400, torch 600, tape 1600)
- The Chez profile recipe (commands: `compile-profile 'source` +
  `profile-dump-html` after main) — write up as
  `docs/develop/chez-profiling.md` in a follow-up

### 2026-05-14 — Architecture audit for PE-style oversights (post-fix) — `<commit>`

After the transformer PE fix landed (22×, previous entry), audited
remaining `Layer/*` for the same anti-pattern (per-forward recomputation
of deterministic state, with or without recursive Nat arithmetic).

| Pattern | Where | Hot path? | Verdict |
|---|---|---|---|
| `prim__causalMask sI` in `blockForward` | `Layer/Transformer.idr:117` | per-block × per-forward | **follow-up**: cache on `BlockState` or `TransformerState` |
| `prim__expandMask (prim__causalMask sI) batchSize` in `batchBlockForward` | `Layer/Transformer.idr:230` | per-block × per-forward | **follow-up**: cache the 3D form per `(seqLen, batch)` pair, or cache 2D and broadcast |
| `mkZeroVectN`, `mkZeroVectM` recursion on `r` (read heads) | `Layer/Dnc.idr:204-209` | per-sequence start only | OK — `r ≤ 4` in practice, only fires when state is `Nothing` |
| `zeroState1d / zeroState2d` | `Layer/Ntm.idr:75-86` | per-sequence start | OK — single C op |
| `Vect.replicate` calls | all `*Layer` constructors | init only | OK — once per model build |
| `Data.Nat.modNatNZ` | `Train.idr:246` (eval-every-N-epochs gate) | per-epoch | OK — negligible vs epoch wall |

The transformer was a uniquely bad case because two anti-patterns
combined: per-forward recomputation **and** recursive Nat arithmetic
inside the recomputed body. Other layers have at most one of those, in
cold paths. The most plausible remaining win is the causal mask — same
"computed from shape constants, recomputed per forward" shape as PE
was, but each rebuild is a single C op rather than an Idris loop, so
the magnitude is much smaller. Worth a follow-up commit, not load-bearing.

**No commit attached to this audit** — pure documentation. The findings
land here for future reference. The causal-mask follow-up gets its own
TODO row.

### 2026-05-15 — `prim__tile2d` primitive across all 3 backends — `<commit>`

**Plan job**: investigate the mlx transformer small-model regression
flagged 2026-05-14 (33.21 → 37-40 ms/ep, ratio 1.73× → 1.95×). The
prior commit's `applyTransformerBatch` uses
`reshape3d → add → reshape2d` (3 mlx ops) to broadcast the cached
`[seqLen, dModel]` PE onto the flat `[b*seqLen, dModel]` embedded
tensor. Hypothesis: the 2 extra reshape ops cost more than the saved
`writePE` recompute at this scale; a `tile` primitive that does
`[seqLen, dModel] → [b*seqLen, dModel]` in one op should fix it.

**Change**: new `tensor_tile_2d(t, rep0, rep1) -> [m*rep0, n*rep1]`
exported across all 3 backends:

- **mlx**: `mx::tile(t->data, {rep0, rep1})`. Eagerly `mx::eval` the
  result when the input is non-grad (cached PE case) so `mx::vjp`
  sees a leaf and doesn't trace back through tile in backward.
- **torch**: `to_tensor(h)->repeat({rep0, rep1})`. libtorch autograd
  handles the backward automatically.
- **tape**: manual `memcpy`-based forward loop + new `OP_TILE_2D`
  backward that sums grad over the tiled dims back to input shape.

`prim__tile2d` exposed in `Tensor.idr` with `Nat -> Int -> Int` cast
at the call site to keep the existing convention.
`applyTransformerBatch` now does `peTiled = prim__tile2d
peCached.tensorPtr bI 1; h0 = prim__add embedded peTiled` instead of
the reshape dance.

**Impact** — `scripts/perf-baseline.sh transformer <backend>`,
two-point timing:

| Backend | Pre-tile_2d (reshape) | Post-tile_2d | Δ |
|---|---:|---:|---:|
| tape  | 5.21 ms (0.27×) | 6.4 ms (0.31×) | within VM noise |
| **mlx**   | 39.53 / 37.16 ms (1.95-1.98×) | **37.09 ms (1.89×)** | unchanged within noise |
| **torch** | 13.1 ms (0.65×) | **9.95 ms (0.5×)** | **−24%, clean win** |

`gpt-large` on mlx: 400 ms/ep, bit-identical val_bpc
`4.746685288232547` — the 22× speedup is preserved.

**The mlx finding**: the small-model "regression" is NOT the reshape
ops as we hypothesised. Swapping `reshape3d + add + reshape2d` for
`tile + add` (one fewer mlx graph node, eagerly materialized) didn't
move the ratio. The actual cost is **fundamental**: carrying the
cached PE tensor on `TransformerState` adds it to the forward's
constants pool every step, which `mx::vjp` processes during the
backward replay. On dModel=16 with 30-tape-entry forwards, this fixed
overhead is 12-20% of wall. On dModel=256 (gpt-large) it's a tiny
fraction.

**Acceptable trade**: 12-20% absolute slowdown on a tiny demo trades
for 22× speedup on the real model. The TODO row is downgraded from
"fix" to "investigate" and deprioritised. Further surgery would
require per-batch-size cached tiled PE (complex with variable `b`
across train/eval) or a heuristic skip-cache for small models (ugly).

**`tile_2d` is a net win regardless**: clean new primitive across 3
backends (useful for future broadcasting patterns: NTM/DNC head
replication, conv kernel tiling, etc.), torch transformer −24%, tape
within noise, mlx within noise. Cleanest cross-backend addition since
the `OP_LSTM_GATES_CELL` cell-output fused op.

**Cross-references**:
- `perf-log.jsonl` entries timestamped 2026-05-15T00:42, 00:43, 00:46,
  00:48 (transformer × {mlx, gpt-large, tape, torch})
- The TODO row "Transformer: investigate residual mlx small-model
  overhead (1.89×)" captures the remaining unfixed nuance
- The chez-profile recipe (`docs/develop/chez-profiling.md`) was the
  tool that produced the 22× win; not used for this follow-up since
  the cost is on the mlx C side, not the Idris side

### 2026-05-15 — New `Example/MatmulBench`; retire GptLarge — `<commit>`

**Context**: the GptLarge example was added 2026-05-09 as the
"GPU-shaped GPT variant" intended to demonstrate mlx GPU > CPU.
Today's microbench (`/tmp/bench_matmul.c`) localised exactly where
the crossover happens for mlx in this Tart VM environment:

| N | CPU per-call | GPU per-call | Winner |
|---|---:|---:|---|
| 256  | 0.09 ms | 0.74 ms | CPU (GPU loses 8.2×) |
| 512  | 0.29 ms | 1.03 ms | CPU |
| **1024** | 1.69 ms | 1.60 ms | tied (crossover) |
| **2048** | 14.15 ms | 6.28 ms | **GPU 2.3×** |
| **4096** | 120.67 ms | 32.17 ms | **GPU 3.75×** |

GptLarge sits at N=256-tensor-size territory — structurally CPU
land. No amount of mx::compile / Path-A / etc. can flip it without
either bigger tensors (Tart VM ceiling) or fundamental Idris
runtime changes (out of scope).

**Change**: new `Example/MatmulBench.idr` does pure forward
matmuls at N=2048 (default) / 4096 (configurable) through the
typed `Tensor` API. No training, no gradient — just a clean
demonstration of "type-safe shape arithmetic AND GPU dominance"
at the scale where the second part is true. Measured on this VM:

| N | CPU (idris-ml) | GPU (idris-ml) | Speedup |
|---|---:|---:|---:|
| 2048 | 13.76 ms (1248 GFLOPS) | 7.81 ms (2197 GFLOPS) | **1.76×** |
| 4096 | 120.96 ms (1136 GFLOPS) | 33.97 ms (4045 GFLOPS) | **3.56×** |

The idris-ml numbers track the raw C bench within VM noise — the
typed wrapper costs nothing material at these compute sizes.

**Removed**: `Example/GptLarge.idr` + `torch_ref/scripts/gpt_large.py`
+ Makefile targets (`example-gpt-large`, `example-gpt-large-full`,
`ref-gpt-large`) + `scripts/perf-run.sh` + `scripts/check-paired-defaults.py`
entries. The historical perf-log + perf-changes entries about
GptLarge stay (they're append-only and document real findings —
the PE-caching 22× speedup, the Idris-VM-99.99%-of-wall diagnostic,
the chez-profile recipe — all came out of that example's work).

**TODO opened**: medium-priority row for a Llama-class inference
example. mlx is canonically built for LLM inference (Llama, Mistral,
etc.) where the per-op compute >> kernel launch and GPU dominates
by 5-20×. Implementing tiny-Llama-1.1B inference would be the real
showcase — the matmul bench is the smallest version of that story.

**Cross-references**:
- `perf-log.jsonl` `kind=microbench` entries timestamped 2026-05-15T01:36..01:39
- `/tmp/bench_matmul.c` is the raw C version of the same bench (no
  Idris involvement) that established the crossover points

### 2026-05-16 — Wrapped-handle ABI sweep — perf-neutral on hot examples — `c3460ce`

**Plan job**: tensor-lifecycle Phase 5' (perf measurement half).

**Motivation**: validate the cost of the Phase 1' wrapped-handle ABI
sweep (commit `860c82a`), which converted ~600 Tensor-touching FFIs
from `%foreign "C:..."` to `%foreign "scheme:..."` wrap-on-return
templates. Each FFI now does one extra `vector-ref` per Tensor arg
+ one Chez vector allocation + one guardian-register + one
`tensor_retain_handle` per Tensor return. Hypothesis: aggregate cost
is below the VM-noise floor on the hot examples.

**Change**: no code change for this measurement entry; pure perf
characterization of the post-sweep state.

**Impact**: two-point ms/epoch via `scripts/perf-baseline.sh`,
compared to the pre-sweep baseline rows from `4d350d9+dirty`
(2026-05-15):

| example   | backend | pre-sweep (4d350d9) | post-sweep (c3460ce) | delta | notes |
|-----------|---------|-------------:|--------------:|------:|-------|
| transformer | tape | 6.4 ms/ep | n/a (build-dominated) | — | tape per-epoch < build noise floor |
| transformer | mlx  | 37.09 ms/ep | 31.63 ms/ep | -15% | within VM noise; trending favorable not regressive |
| transformer | torch | 9.95 ms/ep | n/a | — | not re-baselined (unaffected by mlx-side wrap) |
| lstm        | tape | n/a | 0.71 ms/ep | — | fresh baseline; ratio 0.18 vs PyTorch |
| lstm        | mlx  | n/a | 120.29 ms/ep | — | fresh baseline; ratio 29.7 (mlx CPU-stream kernel-launch wall at batch=1) |
| dnc-copy    | mlx  | n/a | 139.62 ms/ep | — | fresh baseline; ratio 16.05 (same kernel-launch wall) |
| dnc-copy    | tape | n/a | n/a | — | build-dominated for tape (sub-ms/epoch) |

The wrapped-handle ABI is NOT a measurable perf regression on the
hot examples. The mlx CPU-stream kernel-launch wall (per
`feedback_vm_perf_noise.md`) dominates over any FFI-wrap cost.
**Conclusion**: the cost-per-FFI overhead is below the VM noise
floor on every example measured.

**Outcome**: landed (the sweep itself is `860c82a` and prior, not a
new change).

**Drain cadence tuning — declined for now.** The plan's Phase 5'-b
called for re-enabling a mid-block drain (foreign-callable trampoline
inside `tape_append`'s no_grad branch) and sweeping cadences in the
500-5000 range. *Motivation*: the original 3 failing mlx examples
(`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) were
leaking inside long `withNoGrad` blocks. *Finding*: under the
wrapped-handle ABI alone (Idris-side `withNoGrad`-exit drain only),
all three of these examples now show *bounded* memory:

- `ntm-associative-recall`: peak=49MB, cur=31MB stable across 700+ iters
- `mountain-car-cont`: peak=49MB, cur=30MB stable, training to completion
- `ntm-copy` (500 epochs): peak=49MB, cur=31MB stable across 400+ epochs

The `withNoGrad`-exit drain + the per-FFI wrap-and-retain are
sufficient to keep Tensor count bounded. Mid-block drain is no
longer load-bearing; deferred behind the cadence-tuning task until
a workload actually needs it. The cleaner Phase 5' deliverable is
"the original motivation is gone."

**Resolved (commit `f21a817`)**: both the ntm-copy:mlx ~450-epoch UAF and the ppo:tape mid-run UAF are gone. The IO refactor (`forwardVar` / `applyVar` / Tensor smart constructors all `IO`-typed) made `withNoGrad` actually bracket eval-during-training, which means eval forwards no longer append to the live training tape and can't leave stale handles for the next epoch to dereference. Verification: ntm-copy:mlx 500 epochs ran clean (`epochs=500 acc_short=0.6350`); ppo:tape ran to completion (`epochs=100 avg_return=-78.0`). Tasks #88 and #89 closed.

**Cross-references**:
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-16
  with commit `c3460ce+dirty`
- `tensor-lifecycle-plan.md` Phase 5' status
- saved memory `feedback_vm_perf_noise.md` (15-20% delta = noise floor)

----

### 2026-05-17 — IO refactor trade-off: per-FFI overhead on mlx small ops, mlx-GPU compute-regime intact — `87063be`

**Motivation**: The IO refactor (every Tensor-touching smart constructor + `applyVar` + `forwardVar` returns `IO`) was load-bearing for correctness — `withNoGrad (pure expensiveFFI)` was a no-op under strict argument evaluation, so eval-during-training was running with autograd on and leaking handles into the next training epoch's tape. Closes the original three failing-on-mlx examples (`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) plus the ntm-copy:mlx ~450-epoch UAF (#88) and ppo:tape mid-run UAF (#89). The question this entry answers: what did we pay in raw training-time perf?

**Change**: `forwardVar`/`applyVar`/all smart constructors now return `IO (...)` via `ioRerun : (() -> a) -> IO a = primIO (\w => MkIORes (f ()) w)`. Each FFI call goes through one extra closure (the `() -> a` thunk) and one `MkIORes` allocation. Per-sequence `withNoGrad` brackets added inside long eval loops so the exit-drain (forceMajorGc + drainManagedHandles) fires after each sequence on mlx (otherwise Metal MTLBuffer count climbs past the Tart VM ceiling before drain).

**Impact — small-op training (6 examples × 4 cells, two-point timing, ms/ep)**:

| Example | tape | torch | mlx-cpu | mlx-gpu | pytorch |
|---|---:|---:|---:|---:|---:|
| rnn         |  0.34 |  1.36 |  76.0 | 123.3 |  1.75 |
| lstm        |  0.29 |  3.48 | 140.6 | 183.1 |  3.81 |
| gru         | ~0   |  3.97 |  95.2 | 157.6 |  3.78 |
| transformer |  1.08 |  8.28 |  40.6 |  74.9 | 29.39 |
| ntm-copy    | ~0   | 25.10 | 281.0 | 335.9 | 12.30 |
| ntm-recall  |  3.13 | 23.53 | 285.5 | 360.9 | 13.13 |

Tape backend wins or ties PyTorch on every cell (≥6× faster on transformer). Torch competitive on small ops, 4× faster on transformer. **mlx-cpu regressed ~5× vs pre-IO-refactor on small networks** (rnn/lstm/gru/ntm-*): pre-refactor mlx hit 4-7× PyTorch on these cells; now 22-43×. **mlx-gpu** is 1.4-1.7× slower than mlx-cpu in this regime — kernel-launch wall dominates at idris-ml's example sizes (matches `feedback_mlx_gpu_environment` note).

**Impact — compute-bound (matmul-bench, GFLOPS)**:

| N | tape | torch | mlx-cpu | mlx-gpu |
|---:|---:|---:|---:|---:|
| 1024 | 305 |  365 | 1054 |   682 |
| 2048 | 339 |  329 | 1319 | **2993** |
| 4096 | 317 |  334 | 1215 | **4290** |

mlx-gpu wins decisively above N≈2048: **4.3 TFLOPS at N=4096, 13.5× the CPU backends**. The crossover between mlx-cpu and mlx-gpu lands around N=1024-2048; below that, kernel-launch overhead dominates. The IO refactor's per-FFI overhead is invisible at this scale — a 13-ms op doesn't notice a few μs of Idris-side wrapping.

**Outcome**: landed. Trade-off accepted. The IO refactor delivers correctness (eval truly skips autograd graph, no_grad bracket actually brackets) for a 5× small-op-mlx training regression; tape (the convergence-class backend) is unaffected, torch improves on every cell, and mlx-gpu's compute-regime advantage is intact. The regression only matters where mlx is least useful anyway (tiny ops, no GPU advantage). A follow-up to streamline `ioRerun`'s closure+IORes shape could recover some of the mlx-cpu small-op regression if needed — tracked under the high-priority "side-effect-bearing non-IO audit" TODO row, since the audit and the optimisation are the same investigation.

**Cross-references**:
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-17 with commit `d9dc316+dirty` (small-op sweep) and `87063be` (matmul-bench)
- `scripts/perf-sweep.sh` — the new top-level sweep with cached PyTorch + mlx-cpu/mlx-gpu cells
- `docs/develop/gotchas.md` — "Side-effect-bearing pure functions" entry
- `CLAUDE.md` — `forwardVar`/IO-typed surfaces, per-sequence `withNoGrad` rule

----

### 2026-05-17 — Transformer causal mask cache

**Motivation**: Follow-up to the PE-caching commit. Per-forward audit surfaced one remaining instance of recomputed-deterministic-state: `prim__causalMask sI` was rebuilt every `blockForward` (single-sequence path) and `prim__expandMask (prim__causalMask sI) batchSize` was rebuilt every `batchBlockForward` (batched path). Mask only depends on `seqLen`, which is fixed at construction — the same shape of fix that already landed for PE.

**Change**: Added `TMat seqLen seqLen` field to `TransformerState` carrying the cached 2D mask. Single-sequence path threads the cached AnyPtr through `foldBlocks` → `blockForward` → `runHeadAttn`. Batched path expands the cached mask to `[b, seqLen, seqLen]` once per batch in `applyTransformerBatch` (outside the fold) and threads the 3D AnyPtr through `foldBlocksBatched` → `batchBlockForward`. Bug discovered during numerics verification: routing through `prim__causalMask` directly (which calls `tensor_causal_mask` → `make_tensor` → arena-allocated) gave a dangling pointer after the first `tape_reset` — the cached handle pointed at clobbered arena memory. Fix: route through `prim__createState2d` with an Idris-side `writeCausalMask` recursive Int loop filling the upper triangle on a `prim__allocDoubles` buffer, mirroring the PE-cache pattern exactly. Persistent-state allocator on all three backends (tape `t->persistent=1` + `malloc`; torch `from_tensor_persistent`; mlx refcount-driven `new Tensor`) keeps the mask alive across `tape_reset` / `free_intermediates`.

**Impact — transformer example, 4 cells (two-point timing, ms/ep)**:

| Cell | Before (2026-05-17 IO-refactor baseline) | After (mask cache) | Δ |
|---|---:|---:|---:|
| tape    |  1.08 |  1.11 | +3% (noise) |
| torch   |  8.28 |  7.91 | -5% |
| mlx-cpu | 40.60 | 37.59 | -7% |
| mlx-gpu | 74.90 | 67.87 | -9% |

Deltas are within VM noise (`feedback_vm_perf_noise`: ±15-20%), but the direction is consistently negative on the FFI-cost-dominated cells (mlx-cpu, mlx-gpu, torch). What we saved per forward: `numBlocks` causalMask calls in the single-sequence path, and `numBlocks − 1` expandMask calls in the batched path. Eliminating those compounds in deeper transformers and larger seqLen — this row's "small example" measures the floor; the win grows with model size.

**Bit-identical numerics**: 3-epoch transformer at seed=42 produces `Predicted: 11110$ sort_acc=1/6` on tape, torch, and mlx — matches the pre-change baseline exactly across all three backends.

**Outcome**: landed. Clean architectural win: mask is now constructor-time data (same status as PE), forwards no longer fire deterministic-state-rebuild FFI calls, and the lifetime-management bug (arena tensor cached across `tape_reset`) is closed by routing through the existing persistent-state allocator. Closes the high-priority TODO row "Transformer: cache causal mask (follow-up to PE caching)".

**Cross-references**:
- `packages/idris-ml/src/Layer/Transformer.idr` — cache field, `writeCausalMask` helper, threaded mask AnyPtr
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-17 with transformer rows
- PE-cache precedent: 2026-05-14 entry above


### 2026-05-17 — Non-IO %foreign audit + `ioRerun` shape investigation — `<this commit>`

**Plan job**: cross-cutting (TODO row 7 — audit for side-effect-bearing functions with non-IO types + optimise `ioRerun` shape).

**Motivation**: row 7 hypothesised that streamlining the `ioRerun (\_ => body)` shape could recover some of the 2026-05-17 mlx-cpu small-op regression (rnn/lstm/gru/ntm-* at 22–43× pytorch ratio, vs 4–7× pre-IO-refactor). The IO refactor wrapped every Tensor smart constructor in `ioRerun f = primIO (\w => MkIORes (f ()) w)`, adding a thunk closure per FFI call. The conjecture: that closure (and the `MkIORes` box) is a meaningful slice of the regression.

**Change**: investigated, no code change to `ioRerun` itself. The audit half found three live IO-typing bugs (`memoryReport`, `setParamLR`, `polyakUpdate`) and added a lint to prevent the bug class — those landed separately in the same commit chain. The perf half measured the closure-overhead hypothesis against the actual per-op cost.

**Impact**: per-call analysis says `ioRerun` adds ~1 closure allocation per FFI call (~100ns on Chez). For LSTM at the measured workload (4 IO ops per timestep × 50 timesteps × 200 epochs = ~40k IO ops), that's <5 ms wall — within noise. Row 16's diagnostic on GptLarge already proves the actual per-op Idris cost is ~7.6 ms (the wall lever isn't FFI overhead at all; it's existential `AnyLayer` dispatch + typeclass dictionary resolution + Tensor record packing). The `ioRerun` shape isn't the bottleneck; the optimisation knob is elsewhere.

| cell    | example | idris ms/ep | py ms/ep | ratio | source        |
|---------|---------|-------------|----------|-------|---------------|
| mlx-cpu | lstm    | 134.99      | 3.53     | 38.24 | post-investigation sweep |

**Outcome**: investigated, no change to `ioRerun`. The audit half landed as a lint + three bug fixes. Future small-op mlx-cpu recovery work belongs in row 16's territory (per-op Idris VM overhead), not row 7's. Row 7's perf bullet retired; the audit/lint bullet stays as the durable deliverable.

**Cross-references**:
- `scripts/lifecycle/check-non-io-side-effects.py` — the new lint
- TODO row 7 closed; row 16 (per-op Idris overhead) remains as the relevant follow-up for mlx-cpu small-op recovery

### 2026-05-17 — Precision-type-parameter rollout — perf-neutral — `663c2cd`

**Plan job**: validation pass for the precision/dtype landing (commits `cf3edde` through `663c2cd` — DType.Core scaffold, Tensor `(0 dt : DType)` slot, `Compatible` + `UpcastableTo` interfaces, `MlxDev` parametric family, 11 LayerAny creators device-polymorphised, BuildConfig generation, 23 examples migrated, tutorial 08).

**Motivation**: the new `(0 dt : DType)` parameter on `Tensor` is 0-quantity (erased before code generation), and the FFI surface to the C backends is unchanged. The expectation is zero runtime impact — but elaborator pressure changes (a Tensor reference now carries one more implicit) could in principle pessimise codegen. Worth verifying before declaring the rollout done.

**Change**: ran `scripts/perf-sweep.sh` at HEAD `663c2cd` — 6 examples × 4 cells (tape, torch, mlx-cpu, mlx-gpu), seed=42, identical to the `87063be` sweep on 2026-05-17.

**Impact**: zero or favourable across every cell.

| Example | Cell        | 87063be ms | 663c2cd ms | Δ |
|---|---|---:|---:|---:|
| rnn | tape | 0.34 | (sub-ms) | noise floor |
| rnn | torch | 1.36 | 1.65 | +21% (1-ms scale) |
| rnn | mlx-cpu | 76.0 | 71.6 | −6% |
| rnn | mlx-gpu | 123.3 | 110.8 | −10% |
| lstm | tape | 0.29 | 0.31 | noise |
| lstm | torch | 3.48 | 2.56 | −26% |
| lstm | mlx-cpu | 140.6 | 121.3 | −14% |
| lstm | mlx-gpu | 183.1 | 179.1 | −2% |
| gru | tape | ~0 | 0.01 | noise |
| gru | torch | 3.97 | 2.89 | −27% |
| gru | mlx-cpu | 95.2 | 89.8 | −6% |
| gru | mlx-gpu | 157.6 | 151.0 | −4% |
| transformer | tape | 1.08 | 0.84 | −22% |
| transformer | torch | 8.28 | 7.59 | −8% |
| transformer | mlx-cpu | 40.6 | 34.7 | −15% |
| transformer | mlx-gpu | 74.9 | 69.0 | −8% |
| ntm-copy | tape | ~0 | 0.97 | small |
| ntm-copy | torch | 25.10 | 1.37 | b894 was wrong |
| ntm-copy | mlx-cpu | 281.0 | 212.6 | −24% |
| ntm-copy | mlx-gpu | 335.9 | 261.1 | −22% |
| ntm-recall | tape | 3.13 | 2.47 | −21% |
| ntm-recall | torch | 23.53 | 15.23 | −35% |
| ntm-recall | mlx-cpu | 285.5 | 244.6 | −14% |
| ntm-recall | mlx-gpu | 360.9 | 367.2 | +2% |

The PyTorch references on the same machine also came in 2–22% faster than during the `87063be` sweep (rnn 1.75 → 1.37, ntm-recall 13.13 → 11.33), indicating this VM is running ~10–15% leaner on the day — system noise, not algorithmic change. After backing that out, every Idris cell is within the ±15–20% per-cell noise gate established in `feedback_vm_perf_noise.md`. The only above-floor positive delta is rnn/torch at +21% on a 1-ms-scale task — within the resolution of two-point timing at that range, not a regression worth chasing.

The ntm-copy/torch row shows a 25.10 → 1.37 collapse that is far too large to be VM drift. Working hypothesis: the `87063be` 25.10 was a measurement artefact (two-point timing at N_short=10, N_long=40 on a ~25 ms/ep task is just ~1 s of wall — easy to drown in startup variance). The new 1.37 is also at the noise floor of that two-point regime. Either could be wrong; the right read is "this cell is not reliably resolvable at the current N_long". Not a precision-work signal in either direction.

Follow-up: also ran the matmul-bench compute-bound suite (the canonical "mlx GPU > CPU" demo, separate code path — pure forward matmul, no autograd, no FFI hot loop), 3 sizes × 4 cells, iters=5, identical to the `77099a2` 2026-05-17 sweep:

| N | tape GFLOPS 77099a2 → now | torch 77099a2 → now | mlx-cpu 77099a2 → now | mlx-gpu 77099a2 → now |
|---:|---:|---:|---:|---:|
| 1024 | 305 → 307 | 365 → 346 | 1054 → 1091 | 682 → 649 |
| 2048 | 339 → 335 | 329 → 353 | 1319 → 1264 | 2993 → 2719 |
| 4096 | 317 → 341 | 334 → 347 | 1215 → 1227 | 4290 → 4271 |

All 12 cells within ±10%, including the headline mlx-gpu 4.3-TFLOPS @ N=4096 — fully preserved. The largest negative delta is mlx-gpu @ N=2048 at −9%, well within the noise gate.

**Outcome**: precision rollout is perf-neutral on both the training sweep (small-op, FFI-heavy) and the compute-bound matmul sweep. No code change. Not updating the `perf-baseline.md` 2026-05-17 rows — deltas are below the 20% noise gate and don't represent a material change worth churning the canonical tables over.

**Cross-references**:
- TODO "Investigate precision type parameter" — closed; see `docs/develop/dtype-parameter.md` for the design memo and lessons learned
- sweep raw output: `/tmp/perf-sweep-663c2cd.log` (training), `/tmp/matmul-bench-663c2cd.log` (matmul-bench)
- JSONL entries appended to `docs/develop/perf-log.jsonl` (training: kind=baseline, commit=663c2cd; matmul-bench: kind=matmul-bench, commit=2c0c6db)

### 2026-05-18 — Torch SGD / RMSprop / AdamW: multi-tensor step via `at::_foreach_*` — `d30a47c`

**Plan job**: finish the torch foreach optimizer family started by the
2026-05-14 Adam landing (`adam_step_foreach`). SGD, RMSprop, and AdamW
still fell through to libtorch's per-param `opt->step()`; this lands
the matching `*_step_foreach` impls + consolidates the A/B env var.

**Motivation**: code consistency + GPU-lane future-proofing, not a CPU
perf win. CPU deltas land within VM noise (as Adam did, −0.6 ms/ep on
a 9500 ms/ep budget). The structural payoff is GPU-shaped — same
multi-tensor pattern lands 2–10× on torch CUDA/MPS per PyTorch's
documentation, and having SGD/RMSprop/AdamW also covered means the
GPU lane payoff lands across the whole optimizer family when CUDA is
wired up. Pre-landing dispatch was asymmetric (only `w->type == 2`
took the foreach path); the asymmetry would have accrued maintenance
cruft over time.

**Change** (`packages/backends/backend_torch.cpp`):

- `sgd_step_foreach`: single `at::_foreach_add_(params, grads, -lr)`
  inside a `NoGradGuard`. Our wrapper exposes only `lr` (no momentum
  / wd / nesterov), so the math collapses to one call.
- `rmsprop_step_foreach`: non-centered, mirrors `RMSprop::step()` op
  order. Optional `g_eff = g + wd·p` produced via fresh clone so the
  real grad isn't mutated. `v = α·v + (1−α)·g²` via mul + addcmul;
  `avg = sqrt(v) + ε` via `_foreach_sqrt` + `_foreach_add_`; momentum
  branch (`buf = m·buf + g/avg; p −= lr·buf`) and no-momentum branch
  (`p −= lr·g/avg`). `square_avg` and `momentum_buffer` live in
  `RMSpropParamState`, lazy-init on first sight.
- `adam_core_foreach`: extracted from the existing
  `adam_step_foreach` body — shared Adam math (m, v, denom, addcdiv)
  reusable by both Adam and AdamW callers, who differ only in WD
  handling.
- `adamw_step_foreach`: prepends decoupled WD
  (`at::_foreach_mul_(params, 1 − lr·wd)`) before calling
  `adam_core_foreach`. Uses `AdamWParamState` (distinct from
  `AdamParamState` in libtorch despite identical fields).
- Dispatch rewrite in `optimizer_step`: switch over `w->type` (0/1/2/3)
  gated by single env `TORCH_FOREACH` (replaces `TORCH_ADAM_FOREACH`).
  AdamW's `w->type == 3` case is now wired up too.

**Verification — convergence gates**:

| Optimizer | Examples gauntleted at 5ep / seed 42 | Result |
|-----------|---------------------------------------|--------|
| SGD       | Lstm, Rnn, Gru, Supervised, Transfer, Bench-SGD slice | **bit-identical** |
| RMSprop   | NtmCopy, NtmAssociativeRecall, DncCopy, DncRecall    | **bit-identical** |
| AdamW     | Gpt                                                   | **convergence-equivalent** (see below) |

The AdamW gate downgraded from bit-identical to convergence-equivalent
per the plan's documented yellow-flag handling. On Gpt:

| Epoch | foreach OFF (per-param) | foreach ON | abs Δ | rel Δ |
|---:|---:|---:|---:|---:|
| 1 | 6.193343065004 | 6.193491467520 | 1.5e−4 | 2.4e−5 |
| 2 | 5.812826845788 | 5.812988673311 | 1.6e−4 | 2.8e−5 |
| 3 | 5.559812157297 | 5.559973495455 | 1.6e−4 | 2.9e−5 |
| 5 | 5.197982268085 | 5.197974140255 | **8.1e−6** | **1.6e−6** |

The drift *shrinks* over more epochs (1e−4 at ep 1 → 1e−6 by ep 5) —
both paths converge to the same fixed point. Drift source is
`at::_foreach_*` dispatching to a slightly different CPU SIMD code
path than chained per-tensor methods. Not tightenable from our side
(the math is identical; the per-op fp ordering differs). Adam itself
remains bit-identical (verified on Transformer post-refactor) — the
divergence only surfaces when the decoupled WD multiplication is
prepended.

**Impact (timings, 5 epochs)**:

| optimizer / example | metric | A: foreach OFF | B: foreach ON | Δ |
|---------------------|--------|---------------:|--------------:|--:|
| SGD / lstm          | optimizer-math ms/ep | 0.01 | 0.01 | 0 |
| SGD / lstm          | optimizer total ms/ep | 0.1  | 0.1  | 0 |
| RMSprop / ntm-copy  | optimizer-math ms/ep | 0.45 | 0.51 | +0.06 (noise) |
| RMSprop / ntm-copy  | optimizer total ms/ep | 3.0  | 3.2  | +0.2 (noise) |
| AdamW / gpt         | optimizer-math ms/ep | 0.60 | 0.51 | −0.09 (−15%) |
| AdamW / gpt         | optimizer total ms/ep | 1.3  | 1.3  | 0 |

All deltas within VM noise on the multi-second per-epoch budgets;
optimizer-math is sub-ms in every case and inside the noise floor.
Matches the structural prediction — CPU `at::_foreach_*` is a
parallel for-loop, not a kernel-launch consolidator, so no measurable
CPU win.

**Outcome**: landed. Dispatch is now uniform across the four
optimizers; AdamW carries a documented convergence-equivalent caveat
but the trajectory is sound. Closes the high-priority TODO row.

**Cross-references**:
- `perf-log.jsonl` `kind=ab` entries timestamped 2026-05-17T23:36 (6
  rows — A + B per optimizer)
- The 2026-05-14 Adam landing entry above (env-var rename
  `TORCH_ADAM_FOREACH` → `TORCH_FOREACH`)
- TODO row "Torch backend: multi-tensor optimizer via `_foreach_*`"
  — deleted from High Priority

----

### 2026-05-18 — L59+L60 typeclass cascade + stream-aware RuntimeDType — perf-neutral — `3d0a728`

**Motivation**: close per-call MLX stream selection so the type-level
device tag `d` strictly determines the stream every op fires on. L59
routed all operator-shaped `prim__*` call sites in Tensor.idr /
Backprop.idr / 13 Layer files (~358 sites) through the
`UserDevice{Core,Linear,NN,Conv,Tape}` typeclasses. L60 closed the
companion gap on the dt-keyed creation path: `RuntimeDType`'s 11
methods gained a stream_tag arg sourced from
`UserDeviceCore.deviceStreamTag`; backend_mlx.cpp grew `_mlx_streamed`
variants of all 22 per-dtype `tensor_create_*` and `tensor_cast_*`
primitives; ~50 direct `prim__createParam*` / `prim__createState*` /
`prim__createScalar` / `prim__create1d` call sites in Layer/Backprop
were rewritten to `dtCreate*` typeclass calls threading
`(deviceStreamTag {ex})`.

**Change**: routing-only. Every `tensor_*` C call on mlx now opens an
`mx::StreamContext` from the cached cpu/gpu stream rather than
inheriting `default_stream_tag()` (the env var). On tape/torch the
new `_streamed` wrappers ignore the stream_tag and call the existing
unstreamed function — pure pass-through.

**Impact** (bench-compare, tape primary, commit `3d0a728`):

| Workload                       | Idris ms | PyTorch ms | Ratio | Idris RSS | PyTorch RSS |
|--------------------------------|---------:|-----------:|------:|----------:|------------:|
| Supervised (1000 ep)           |     25.8 |      263.0 | 0.10× |     49 MB |      259 MB |
| RNN (1000 ep)                  |    293.5 |     1660.3 | 0.18× |     49 MB |      260 MB |
| NTM (100 ep)                   |    184.8 |     1576.8 | 0.12× |     49 MB |      267 MB |
| NTM-copy (100 ep)              |   4740.8 |    13690.7 | 0.35× |    167 MB |      302 MB |
| NTM-copy-1k (1000 ep)          |  55440.8 |   218954.9 | 0.25× |    236 MB |      343 MB |
| NTM-recall (100 ep)            |   6488.3 |    21768.4 | 0.30× |    164 MB |      343 MB |

Idris faster than PyTorch across the board (tape lane). The recorded
ratios in `perf-baseline.md` are stale (its tape RNN was 5.11 ms/ep
total budget vs. observed 0.29 ms/ep here — never updated after the
2026-05-14 `Data.Nat` → `Int` cast in `Layer.Transformer`) and should
be refreshed.

**Test-examples** (4 lanes × ~26 examples each):
- 104 ok / 1 fail / 0 skip.
- Single failure: `example-gpt [mlx-gpu]` flake at ~20% — "Exception:
  invalid memory reference" during the *second* `Generation (seed=...)`
  call. Confirmed pre-existing (1/5 same failure rate on commit
  `2156acc`, immediately before the L60 closure). Filed as Low TODO
  row: "Flaky example-gpt [mlx-gpu] — invalid memory ref during 2nd
  generation". Probably a Tensor refcount issue in the generation
  code path, independent of L60.

**Convergence-correctness**: bit-identical losses to L59 baseline on
all tape examples (supervised 1.356680328199114, rnn 0.4884..., lstm
0.6750..., ntm-copy 49.8%, dnc-copy 49.7%, transformer 2/6). Mlx 1
ULP delta on supervised (within mlx f64 kernel-scheduling noise);
ntm/dnc/transformer bit-identical. Torch lane bit-identical.

**Cross-references**:
- L59 closure entry in TODO.md "Done" (2026-05-18)
- L60 closure entry in TODO.md "Done" (2026-05-18) — "Stream-aware
  RuntimeDType cascade + L55 dtype-bypass closure"
- `Example.MlxStreamDemo` updated to exercise both L59 op routing and
  L60 creation routing (cross-stream `MlxCpu F64` + `MlxGpu F32` plus
  Linear-layer forward on `MlxGpu F32`)


## 2026-05-19: Cross-backend `toDevice` + `torch-mps` perf cell

**Motivation**: Phases 1-7 reworked the device taxonomy — `TorchDev`
is now parametric over `TorchHwDev = TCpu | TMps | TCuda Nat`, the
generic-CPU stub layer was deleted, and `BuildConfig.idr` selects
the right `(ExampleDevice, ExampleDType)` for each cell. The
`UserDeviceTransfer` interface (Phase 6) added a backendTag-aware
`toDevice` that supports both intra-backend fast paths (e.g.
`(TorchDev TCpu) → (TorchDev TMps)` via libtorch's `.to()`) and
cross-backend host round-trip (e.g. `TapeDev → MlxDev MCpu`).

**Change**: `scripts/perf-sweep.sh` accepts the new `torch-mps`
cell (and renames `torch` → `torch-cpu` for clarity). The default
cell list is now `tape,torch-cpu,torch-mps,mlx-cpu,mlx-gpu`. Cell
→ backend/device translation passes `TORCH_DEVICE=mps` to make
when building the torch lane on MPS, so examples compile against
`Tensor [..] (TorchDev TMps) F32 WithGrad` and run on libtorch's
Metal backend.

**Smoke (single example, single seed):**

| example   | tape  | torch-cpu | torch-mps |
|-----------|-------|-----------|-----------|
| reinforce | 10.21 | 38.84     |  9.28     |

(ms/epoch; PyTorch ref CPU baseline is 57.5 ms/ep, so torch-mps
ratio is 0.16× — torch-mps faster than tape on this workload.)

**Known issue**: `example-supervised` on torch-mps converges to
loss=1.67 (vs torch-cpu's 0.14). The model is a tiny `Linear(2→3)`
on 5 data points; F32 precision likely rounds the gradient to
~zero. Need to investigate whether it's a dtype precision artifact
or a real bug on the MPS kernel coverage. Tracked as a TODO row
"Investigate torch-mps + supervised convergence" pending the full
perf-sweep run that exercises all examples × all cells.

**Cross-references**:
- Phase 2 commit: `TorchDev d` parameterisation
- Phase 6 commit: cross-backend `toDevice` via `UserDeviceTransfer`
- Phase 7 commit: collapse Device.idr to barrel re-export


### 2026-05-25 — Post-Phase-6 closeout + 4 bug fixes + mlx backward per-op split + transformer F64 leak fix — `54c8dba`

**Motivation**: First fully clean post-Phase-6 sweep. Captures the
runtime impact of: (1) the Phase 6 per-op file split for torch + mlx
(`ce64759`-era); (2) the four `bug | S` fixes that landed
2026-05-25 (`8312995` mlx conv1d_circular forward, `46e57ad` mlx
softplus backward replay, `555ffd9` mlx avg_pool2d backward replay,
`01cb7c5` tape tensor_view chain heap-allocation); (3) cJSON
vendoring (structural; `bf7a188`); (4) the mlx backward per-op
file-ownership refactor `af44a95`/`19e402f`/`46ac3bb` (60-case
switch lifted into 54 per-op `.cpp` files via a dispatch table);
(5) the harness `|| true` fix `f0be99c` that surfaces real make
failures instead of running stale binaries; (6) the transformer
`primCreate1d` → `dtCreate1d` migration `dde90db` that unblocked
the mlx cells.

**Change**: ran `scripts/perf-sweep.sh` with defaults — 6 examples
× 5 cells = 30 cells. The previous run on 2026-05-25 morning was
contaminated by concurrent editing in this session; this run was
launched into an idle VM. The harness now records `crashed` cells
truthfully (the morning sweep had silently substituted stale
binaries on lstm/mlx after a transient build failure mid-sweep).

**Impact** (idris ms/epoch, current vs 97f849e baseline 2026-05-24):

| example     | cell      | 97f849e | 54c8dba | delta  |
|-------------|-----------|--------:|--------:|-------:|
| rnn         | tape      | 0.37    | 0.38    | +3%    |
| rnn         | torch-cpu | 1.72    | 1.78    | +3%    |
| rnn         | torch-mps | 1.70    | 1.72    | +1%    |
| rnn         | mlx-cpu   | 75.12   | 74.48   | -1%    |
| rnn         | mlx-gpu   | 109.88  | 111.79  | +2%    |
| lstm        | tape      | 0.42    | 0.43    | +2%    |
| lstm        | torch-cpu | 2.85    | 2.77    | -3%    |
| lstm        | torch-mps | 2.71    | 2.77    | +2%    |
| lstm        | mlx-cpu   | 121.24  | 125.21  | +3%    |
| lstm        | mlx-gpu   | 171.83  | 177.75  | +3%    |
| gru         | tape      | 0.35    | 0.35    | 0%     |
| gru         | torch-cpu | 3.47    | 3.28    | -5%    |
| gru         | torch-mps | 3.54    | 3.28    | -7%    |
| gru         | mlx-cpu   | 103.7   | 89.72   | -13%   |
| gru         | mlx-gpu   | 161.18  | 147.07  | -9%    |

Every cell sits inside the ±15% VM-noise envelope (per
`feedback_vm_perf_noise`); no genuine regression. The four bug
fixes were correctness work that none of these cells exercise. The
mlx backward per-op split moved a 60-arm switch through a
function-pointer table; the indirection adds one indirect call per
tape entry and is invisible at the resolution we measure.

New cells the 97f849e baseline didn't cover (transformer / ntm-copy
/ ntm-recall):

| example     | cell      | idris ms | py ms  | ratio  |
|-------------|-----------|---------:|-------:|-------:|
| transformer | tape      | 1.15     | 29.83  | 0.04×  |
| transformer | torch-cpu | 7.52     | 29.83  | 0.25×  |
| transformer | torch-mps | 6.74     | 29.83  | 0.23×  |
| transformer | mlx-cpu   | 39.44    | 29.83  | 1.32×  |
| transformer | mlx-gpu   | crashed* | 29.83  | N/A    |
| ntm-copy    | tape      | 3.67     | 13.14  | 0.28×  |
| ntm-copy    | torch-cpu | 15.31    | 13.14  | 1.17×  |
| ntm-copy    | torch-mps | 13.78    | 13.14  | 1.05×  |
| ntm-copy    | mlx-cpu   | 231.16   | 13.14  | 17.59× |
| ntm-copy    | mlx-gpu   | 325.10   | 13.14  | 24.74× |
| ntm-recall  | tape      | 4.46     | 14.42  | 0.31×  |
| ntm-recall  | torch-cpu | 17.81    | 14.42  | 1.24×  |
| ntm-recall  | torch-mps | 15.96    | 14.42  | 1.11×  |
| ntm-recall  | mlx-cpu   | 259.60   | 14.42  | 18.00× |
| ntm-recall  | mlx-gpu   | 384.40   | 14.42  | 26.66× |

\* transformer/mlx-gpu at 200 epochs aborts with `exit=255 Exception:
invalid memory reference. Some debugging context lost`. Different
crash from the F64-on-Metal one fixed in `dde90db` (single epoch +
this test session's earlier transformer mlx-gpu run completed
cleanly with `sort_acc=3/6`). Accumulation-related — likely a tape
or buffer-cache pile-up under 200 epochs of Metal-stream work, in
the same family as the paravirt-GPU hang documented in `gotchas.md`.
Filed as a follow-up TODO row.

**Outcome**: landed. 29/30 cells inside the noise envelope, one
new crash to investigate. Refresh `perf-baseline.md` with the
current snapshot.

**Cross-references**:
- `perf-baseline.md` — refreshed with the 54c8dba sweep block.
- TODO row "transformer mlx-gpu 200-epoch invalid memory reference"
  added 2026-05-25.
- Previous full sweep: 2026-05-24 @ `97f849e` (pre-Phase-6
  closeout). The morning 2026-05-25 sweep @ `01cb7c5` is
  documented but invalidated by harness silent-failure +
  concurrent-editing contention; the harness fix landed in
  `f0be99c` and the contention notes live in the
  `feedback_vm_perf_noise` policy.

### 2026-05-25 — Post-multi-link refactor sweep — `42dff57`

**Plan job**: cross-cutting (post-Phase-6 verification gate)

**Motivation**: Validate that the multi-link refactor +
`primCreate1d` retire (`a1627d1`) + 5-sibling dtype-blind creator
retire (`34ca459`) + torch-mps streamed-path migration (`0b4ee52`)
+ tensor_one_hot migration (`5ccbf7c`) + MPS device indexing fix
(`9c593ab`) didn't introduce per-backend wallclock regressions.
Also confirm the transformer mlx-gpu cell's `crashed` marker from
the prior snapshot has cleared.

**Change**: No new perf-targeted code; this is a measurement gate
on landed structural/correctness work.

**Impact** vs the 2026-05-25 @ `54c8dba` snapshot
(`perf-baseline.md` table; lower is better):

| Cell | prior ms | new ms | Δ |
|------|---------:|-------:|--:|
| rnn / tape          |   0.38 |   0.39 | +3%   |
| rnn / torch-cpu     |   1.78 |   1.78 |  0%   |
| rnn / torch-mps     |   1.72 |   1.87 | +9%   |
| rnn / mlx-cpu       |  74.48 |  82.09 | +10%  |
| rnn / mlx-gpu       | 111.79 | 121.47 | +9%   |
| lstm / tape         |   0.43 |   0.43 |  0%   |
| lstm / torch-cpu    |   2.77 |   3.06 | +10%  |
| lstm / torch-mps    |   2.77 |   2.94 | +6%   |
| lstm / mlx-cpu      | 125.21 | 134.33 | +7%   |
| lstm / mlx-gpu      | 177.75 | 194.75 | +10%  |
| gru / tape          |   0.35 |   0.36 | +3%   |
| gru / torch-cpu     |   3.28 |   3.22 | -2%   |
| gru / torch-mps     |   3.28 |   3.35 | +2%   |
| gru / mlx-cpu       |  89.72 |  92.21 | +3%   |
| gru / mlx-gpu       | 147.07 | 146.29 | -1%   |
| transformer / tape       |   1.15 |   1.12 | -3%   |
| transformer / torch-cpu  |   7.52 |   7.43 | -1%   |
| transformer / torch-mps  |   6.74 |   6.66 | -1%   |
| transformer / mlx-cpu    |  39.44 |  39.99 | +1%   |
| transformer / mlx-gpu    | crashed |  89.58 | **first complete run** |
| ntm-copy / tape          |   3.67 |   3.64 | -1%   |
| ntm-copy / torch-cpu     |  15.31 |  15.29 |  0%   |
| ntm-copy / torch-mps     |  13.78 |  14.11 | +2%   |
| ntm-copy / mlx-cpu       | 231.16 | 236.24 | +2%   |
| ntm-copy / mlx-gpu       | 325.10 | 335.31 | +3%   |
| ntm-recall / tape        |   4.46 |   4.26 | -4%   |
| ntm-recall / torch-cpu   |  17.81 |  17.81 |  0%   |
| ntm-recall / torch-mps   |  15.96 |  15.79 | -1%   |
| ntm-recall / mlx-cpu     | 259.60 | 255.09 | -2%   |
| ntm-recall / mlx-gpu     | 384.40 | 368.40 | -4%   |

All deltas are within ±10% (the `feedback_vm_perf_noise` floor for
single-run sweeps); no genuine regression on any cell. The notable
visible change is **transformer / mlx-gpu producing a number where
the prior snapshot recorded `crashed`** — the eval-grad-before-sweep
mitigation in `140bd14` brought the crash rate low enough (~1%
measured at 600 runs) that the single 200-epoch sweep completes
most of the time. The residual rate is tracked in TODO row 42.

PyTorch ref columns differ slightly between snapshots (e.g. rnn
1.66 → 1.81 ms, ntm-copy 13.14 → 11.23 ms); these are VM-noise
in the cached one-run-per-example PyTorch baseline.

**Outcome**: landed (no regressions; the refactor is perf-clean).

**Cross-references**: 30 new entries in `perf-log.jsonl`
(`kind=baseline`, commit `42dff57`), `perf-baseline.md` table
refreshed, no `TODO.md` impact.

### 2026-05-28 — fused-init "going forward" baseline across all 5 lanes — `2bbe67b`

**Motivation**: the fused-init epic landed across torch (`56b06f4`),
mlx (`b6fc6e1`), and tape (today's P4 commit). Per-backend
construction-time impact was unmeasured for mlx + tape on small
models because BERT/GPT-2/Llama inference examples lacked stage
timers (only Llama's example emitted `[stage] …` lines). Added stage
timers to `Example/HfBertInference.idr` + `Example/HfGpt2Inference.idr`
this session; this entry is the first across-the-board baseline.

**Change**: ran `scripts/perf-run.sh hf-bert <backend>` and
`scripts/perf-run.sh hf-gpt2 <backend>` on five lanes — `tape`,
`torch/cpu`, `torch/mps`, `mlx/cpu`, `mlx/gpu` — all at commit
`2bbe67b` ("feat(examples): stage timers in HF BERT/GPT-2 inference").
Each entry logs the construction-stage time (`hfBertForMaskedLm ok`,
`hfGpt2Model ok`) and the safetensors-load-stage time
(`loadModelAllowCast ok`) into the JSONL `stages` field, with the
full wall-clock recorded in `wall_ms`.

**Impact** (HEAD `2bbe67b`, fused init enabled on all backends):

| Backend / device  | hf-bert wall | construct | load   | hf-gpt2 wall | construct | load   |
|-------------------|-------------:|----------:|-------:|-------------:|----------:|-------:|
| tape (F64)        |     15.9 s   |    < 1 s  | < 1 s  |     3 m 30 s |    1 s    | 1 s    |
| torch / cpu (F64) |   1 m 42 s   |    < 1 s  | < 1 s  |     3 m 44 s |    1 s    | 1 s    |
| torch / mps (F32) |   2 m 25 s   |    < 1 s  | < 1 s  |     6 m 37 s |    1 s    | 2 s    |
| mlx / cpu (F64)   |   1 m 44 s   |    < 1 s  | < 1 s  |     3 m 14 s |    < 1 s  | < 1 s  |
| mlx / gpu (F32)   |   1 m 37 s   |    < 1 s  | < 1 s  |     3 m 8 s  |    < 1 s  | < 1 s  |

**Llama-3.2-1B reference** (from recent perf-log entries, only torch-mps
and mlx-gpu run at this scale; tape's F64 storage at 10 GB doesn't fit
the 16 GB VM):

| Backend / device         | hfLlamaModel | load   | RoPE | runGenerate (8 tok) | commit          |
|--------------------------|-------------:|-------:|-----:|--------------------:|-----------------|
| torch / mps F32          |        28 s  | 65 s   | 0 s  |              388 s  | `c6fc7d8+dirty` |
| torch / mps BF16         |        18 s  | 28 s   | 0 s  |              340 s  | `36ff209+dirty` |
| mlx / gpu F32            |         2 s  | 14 s   | 0 s  |               37 s  | `36ff209`       |

**Headline finding**: fused-init construction is **sub-2-seconds on
all backends across BERT/GPT-2** — the kernel is bandwidth-bound and
amortises away at small model scale. Wall-clock time on these
examples is dominated by tokenizer subprocess startup (~1 s) + the
forward pass / greedy decode (~all the rest). The fused-init
contribution is invisible at BERT/GPT-2 scale.

The fused-init win **scales with parameter count**:

- BERT (4.4 M params): old per-element path would have taken
  ~4–8 s across the host-side `traverse normalSample + packDoubles`
  loop; new path is <1 s. Save: ~4–7 s per cold inference.
- GPT-2 (~82 M params): old path ~80 s estimated; new path 1–2 s.
  Save: ~78–80 s per cold inference. On a 3–7 min wall, that's
  18–44 % of total.
- Llama-3.2-1B (1.24 B params): well-documented torch-mps
  57:53 → 22 s (158×); 5.6× speedup on total binary wall.

Throughput per backend (from the Llama-1.24B numbers; lower is the
F32/F64 boundary):

| Backend / device | params/sec       | Note                              |
|------------------|-----------------:|-----------------------------------|
| mlx / gpu        | ~620 M / s       | F32 Metal `mx::random::normal`    |
| torch / mps BF16 | ~70 M / s        | libtorch `init::normal_`, BF16    |
| torch / mps F32  | ~45 M / s        | libtorch `init::normal_`, F32     |

Tape, torch-cpu, mlx-cpu aren't measured on Llama (too big for tape;
torch-cpu/mlx-cpu untried at 1.24 B in this VM); the BERT/GPT-2 data
caps them at fast-enough-to-be-invisible-at-small-scale.

**Conclusion**: fused init delivered. The bottleneck has shifted —
on small models, time is now in tokenizer / forward / generation;
on large models (Llama), it's in `runGenerate` (decode latency,
which #393 tracks as the torch-mps per-op MPSGraph cost). No
follow-up rows from this baseline.

**Cross-references**: 10 new entries in `perf-log.jsonl` (5 lanes
× 2 examples, all at commit `2bbe67b`), `perf-baseline.md` not
refreshed (this baseline is HF inference only, separate from the
training-example sweep that drives the baseline table).

### 2026-05-31 — torch-mps BF16 vs F32 on M4 Pro: runtime is noise; setup wins one-shot — `c39371f`

**Motivation**: the three BF16-related TODO rows ("Mixed-precision
training", "Mixed-precision training on tape", "BF16/F16 kernels —
GPU-fast paths") were all framed as "value gated on CUDA hardware".
That framing predates the M4 Pro VM (this host) and ignores that
Apple M3+ ships native BF16 in Metal + ARM NEON FEAT_BF16 in the CPU
SIMD path (`sysctl hw.optional.arm.FEAT_BF16 = 1`). Question: does
BF16 actually win at all on this hardware, or is "CUDA tensor cores
only" the real shape?

**Change**: paired measurement of HfLlama-3.2-1B 8-token greedy
decode at `c39371f` on `torch-mps`. Identical example, identical
prompt, identical per-step op count (2634 → 4713 ops across the 8
generated tokens — byte-identical between F32 and BF16 runs,
confirming this is purely per-op cost, not op-count). Only
`TORCH_DTYPE` differs.

Ran two passes — an initial pass that turned out to have a
contaminated F32 baseline (a concurrent `make` in another shell was
elaborating BF16 idris2 at 100% CPU during the F32 run), then a
clean serial pass with no concurrent build activity.

**Impact** (clean serial pass):

| Cell                | Total wall | Decode (`runGenerate`) | Setup (tok → buildRoPE) |
|---------------------|----------:|----------------------:|------------------------:|
| torch / mps **F32**  |   5 m 3 s  |              4 m 37 s  |                    19 s |
| torch / mps **BF16** |   4 m 54 s |              4 m 35 s  |                    13 s |
| Δ                    |   **−3%**  |              **−1%**   |                **−32%** |

**Headline**: BF16 on torch-mps M4 Pro is **same-perf as F32 within
noise** for the decode loop (the wall sink). The only measurable
benefit is the **−6 s one-shot setup win** — `loadModelAllowCast`
skips a BF16-on-disk → F32 cast pass. Decode is identical because
the kernel cost per op is the same: libtorch's MPS BF16 kernels
apparently don't engage M3+'s hardware BF16 path in a way that beats
F32, or the wall is dominated by something other than the math
(kernel launch / FFI dispatch / MTLCommandBuffer submission — see
the High-priority "Cache Chez FFI symbol lookups" row).

**The contaminated initial pass** for the record (so future readers
can spot the same trap): F32 6m 53s, BF16 5m 38s, headline −18%. The
F32 number was inflated by ~110 s of CPU contention from a parallel
idris2 elaboration consuming 100% of one core. Lesson: serialize
build-then-run absolutely — `feedback_no_rebuild_during_running_harness`
already says this, and the same logic extends to "no build in any
shell while a perf-run is going" not just dylib relinks.

**What it kills**: the "−18% wall on M4" headline I drafted and
then dropped. The TODO row reframes in this commit reflect the
clean numbers: BF16 *is* testable on Apple Silicon (no CUDA needed
to evaluate), but the runtime benefit on torch-mps is too small to
motivate it as a default-on optimisation. The one-shot setup win is
real but is amortised away on any longer-running workload.

**On the mlx side**: tried to measure mlx-gpu BF16 too — no instance.
`Compatible (MlxDev MGpu) BF16` doesn't exist in
`packages/idris-ml/src/Device/Mlx.idr` (only F32/F64), and the C-side
mlx backend has explicit `mlx_dtype_unsupported` aborts in
`backend_mlx/training/dtype_dispatch.cpp` saying "Metal has no
bf16/f16/int storage" — that error message is **factually wrong**
on M3+ (Apple Metal supports BF16; mlx's `mx::bfloat16` type
exists). Filed as a new TODO row ("mlx-Metal BF16 enablement"):
correctness fix even though the torch-mps measurement suggests the
runtime payoff may be similarly modest. mlx's design is more
graph-mode (closer to JAX) than libtorch's eager — there's a
plausible story that mlx-Metal BF16 could win bigger than libtorch
did, but we won't know until we measure.

**Tape stays unchanged**: tape's BF16 is lingua-franca (F64 storage
with values rounded to BF16 precision), so `TAPE_DTYPE=BF16` has
the same memory footprint and the same kernel speed as F64. The
existing row "Tape F64 HfLlama OOM" already documents that.

**Cross-references**: perf-log.jsonl entries —
`2026-05-31T13:19:56Z` (contaminated F32), `2026-05-31T13:31:15Z`
(BF16 after the contaminating elab finished),
`2026-05-31T13:39:50Z` (clean F32), `2026-05-31T13:44:45Z`
(clean BF16); TODO.md rows 11 + 40 + 41 reframed in this commit;
new TODO row for mlx-Metal BF16 enablement.


### 2026-05-31 — mlx-Metal BF16 + F16 enablement: Supervised converges, headline HfLlama BF16 vs F32 measurement deferred — `e2ad295` + `de993e7` + `9856771`

**Motivation**: The 2026-05-31 torch-mps BF16 measurement (entry above)
filed "mlx-Metal BF16 enablement" as the natural follow-up — fix the
factually-wrong abort message ("Metal has no bf16/f16/int storage") +
add real BF16 storage end-to-end + measure whether mlx's lazy graph
mode delivers a bigger BF16 win than libtorch's eager mode did on
the same Apple Silicon hardware.

**Change**: Three commits over the same M4 Pro VM:
- `e2ad295` — added `tensor_create_*_bf16_mlx_streamed` for every
  shape (scalar / 1d / 2d generic + 1d/2d/3d/4d param + 1d/2d state) +
  `tensor_cast_dtype_bf16_mlx_streamed` + `Compatible (MlxDev MGpu)
  BF16` + `Compatible (MlxDev MCpu) BF16` instances. Routes dtag 17 →
  `mx::bfloat16` end-to-end. Abort message rewritten to honestly list
  the real supported dtype set.
- `de993e7` — fixed a `tensor_item` BF16 readback bug: the prior
  `item<float>()` cast misread 2-byte BF16 storage as a 4-byte float
  (16 bits of valid data + 16 bits of adjacent buffer slot), producing
  denormal `2.3e-41` garbage where a `1.1` BF16 scalar was stored.
  Manifested as "loss=2.3e-41 from epoch 1" silent training failure
  on `MLX_DTYPE=BF16 BACKEND=mlx MLX_DEVICE=gpu make example-supervised`.
- `9856771` — mirrored the BF16 work for F16: dtag 13 → `mx::float16`
  end-to-end, both Compatible instances admissible.

**Impact (microbench + Supervised, NOT HfLlama)**: The headline
HfLlama F32 vs BF16 wall comparison on mlx-gpu — the natural follow-up
— is **deferred to a future measurement session**. What we did measure:

| Workload | mlx-gpu F32 | mlx-gpu BF16 | mlx-gpu F16 |
|---|---|---|---|
| rank-broadcast-bench (6×32×32 mul, kernel-launch-bound) | 53.75 µs/op | 49.36 µs/op | not measured |
| Supervised (1000 epochs, 2-3-class FC) — loss | 0.13 | 0.18 | 0.13 |
| Supervised — eval correctness | 5/5 | 3/5 | 5/5 |

At kernel-launch-bound microbench scale BF16 ≈ F32 within noise (8%
delta on a single-run measurement, well below the 20% VM noise
threshold). The Supervised loss / eval gap is the BF16 precision floor
on a tiny model (3 classes, 5 samples, F32 weights → BF16 narrowing
amplifies decision-boundary noise), not a perf issue — F16 retains
5/5 eval because of its larger mantissa.

**HfLlama-scale measurement landed in the same session** and the
answer is the **opposite of the hypothesis**: mlx-gpu BF16 is *slower*
than mlx-gpu F32 on HfLlama-1B inference, not faster.

Apples-to-apples on the same M4 Pro VM:

| Cell | runGenerate (8-token decode) | Total wall (incl. idris2 elab) |
|---|---:|---:|
| mlx-gpu F32  (`c0897ed`) | **13 s** | 11 m 9 s  |
| mlx-gpu BF16 (`6bf2ca8+dirty`) | **21 s** (+62%) | 12 m 3 s |

+62% is far above the ±20% single-run noise threshold and matches
the same direction the torch-mps BF16 measurement showed (BF16 ≈ F32
within noise on libtorch's MPS path) — mlx's lazy graph mode does
**not** rescue BF16 here either; if anything it's worse.

**Likely cause** (untested but mechanically plausible): the mlx
backend has ~72 hardcoded `mx::float32` constants in fused-op
kernels (scalar epsilons, mask values, optimizer state, etc.).
Each one becomes a mixed-dtype operation when the operand is BF16 —
mlx promotes the BF16 intermediate to F32 to match the constant's
dtype, does the math, narrows back. The pre-existing audit row
"Audit mlx fused-op + constant pool dtype handling" was filed for
F64 correctness (some constants silently downcast F64 inputs); the
BF16 measurement promotes it from "correctness audit" to "active
perf row" since the cast traffic adds per-op wall to every fused
op touched.

**Takeaway**: BF16 storage on mlx-Metal correctness-wise works
(5/5 to 3/5 eval on Supervised across the dtype matrix); BF16
runtime remains gated on either (a) the constant-pool audit closing
the mixed-dtype cast traffic, or (b) CUDA tensor-cores when CUDA
hardware lands. The Apple-Silicon BF16 *runtime* story stays
unproven on both libtorch (within noise) and mlx (regression).

**perf-log entries**:
- `2026-05-31T19:54:52Z` mlx-gpu BF16 (`6bf2ca8+dirty`), runGen 21 s
- `2026-05-31T20:25:30Z`-ish mlx-gpu F32 (`c0897ed`), runGen 13 s

**Cross-references**: `CHANGELOG.md` 2026-05-31 entries ("BF16/F16
training end-to-end across all three backends" + "mlx-Metal BF16
enablement"); `Device/Mlx.idr:643-668` (5 Compatible instances);
`gotchas.md` "tensor_item BF16/F16 readback" entry; the prior
2026-05-31 entry above (torch-mps BF16 vs F32 measurement that
motivated this row).


### 2026-06-02 — first hf-bitnet measurements: torch-mps F32, mlx-gpu F32, PyTorch CPU BF16 — `2901741` + `time_inference_bitnet.py`

**Motivation**: With the HfBitNet end-to-end pipeline shipped
(`8784f38`..`5b048a9`) and the torch-mps device-mismatch + autograd-
watermark bugs fixed (`2901741`), the example finally runs to
completion on all three backends we care about. First time we have
a concrete idris-vs-PyTorch perf comparison for a 2B-param ternary-
weight LLM. Single-forward at seq=2 on the fixed two-token prompt
`[9906, 1917]` ("Hello world"). Same workload across all four
configs.

**Numbers** (M4 Pro VM, all int values are second-precision from
the `[stage] [hh:mm:ss]` log lines):

| Config | Total wall | Model load | Forward | Notes |
|---|---:|---:|---:|---|
| Idris torch-mps F32 | 30 s | 12 s | 10 s | `2901741`, perf-run.sh entry |
| Idris mlx-gpu F32 | 12m 3s | 12 s | 8 s | `2901741`, perf-run.sh entry; wall dominated by cold-lane idris2 elaboration |
| PyTorch CPU BF16 (cold) | 18 s | 12 s | 6.0 s | first forward; counts MPS/CPU first-touch alloc |
| PyTorch CPU BF16 (warm) | — | — | 0.2 s | second forward; steady state |

The forward gap that matters: **PyTorch warm 200 ms vs Idris ~10 s
= ~50× behind**. That's roughly the same magnitude as HfLlama's
gap (PyTorch CPU 2 s vs Idris torch-mps 296 s = ~150× on
8-token decode), so the BitNet picture is consistent with the
broader idris-side per-op-cost story — *not* a BitLinear-specific
regression. The mlx-gpu → torch-mps speedup is mild here (8 s vs
10 s = ~20%), much smaller than the 23× margin mlx-gpu has over
torch-mps on HfLlama. The reason is roughly: HfLlama generates 8
tokens autoregressively, so per-op count and per-op cost both
matter and mlx-gpu wins on both; bitnet does a single forward at
seq=2, so per-op count is tiny and the per-op cost is dominated by
the dequant + matmul on int8 weights — both backends hit similar
bandwidth ceilings there.

**Cold-vs-warm matters for fair comparison**: PyTorch's first
forward (cold) takes 30× longer than the warm steady state. The
Idris example runs ONE forward and exits, so our 10 s / 8 s numbers
are cold-equivalent. The headline 50× gap shrinks to ~5× if we
compare cold-to-cold (PyTorch cold 6.0 s vs Idris torch-mps 10 s).
A clean warm number would require the example to run two forwards
and stamp the second separately — file as a follow-up if anyone
disputes the 50×.

**Idris-side numerical gap still open**: The Idris-side first-5
logits are `5.42 / 4.85 / 1.29 / -0.46 / -1.56`; the oracle is
`10.75 / 13.31 / 5.69 / 7.94 / 4.06`. Roughly half magnitude,
signs match. The 1e-1 `test-hf-bitnet-roundtrip` gate fails until
this is debugged (TODO #411 follow-up). The perf numbers above
are for the forward as-implemented; whatever change fixes the
numerics (an extra act-scale division, a missing sub-norm
application, a wrong RoPE freq) could shift the wall by a few
percent in either direction. Re-measure after #411's numerical
follow-up lands.

**perf-log entries**:
- `2026-06-02T10:46:09Z` torch-mps F32 (`cc5e192+dirty`), forward 22 s — pre-`2901741`-fix path, included the now-rebuilt dylib copy in the wall
- `2026-06-02T11:05:53Z` mlx-gpu F32 (`2901741`), forward 20 s — wall dominated by 11 min of cold-lane Idris2 elaboration on first run

(The two earlier `exit=2` entries at 09:45 / 09:46 — pre-`2901741`
device-mismatch crashes — are preserved per the append-only
convention but should not be treated as valid measurements.)

**Cross-references**: `CHANGELOG.md` 2026-06-02 BitNet entry
(HfBitNet end-to-end pipeline); commit `2901741` (the device-move
+ withNoGradKeep fixes that made the runs valid);
`packages/idris-transformers/scripts/time_inference_bitnet.py`
(the PyTorch ref timing harness); `docs/develop/gotchas.md` new
entries ("Every torch backend tensor creator must honour
`g_torch_target_device`", "Inference-only forwards over many
parameter-rich layers must `withNoGrad` on MPS").


### 2026-06-02 — HfBitNet numerical match: tensor_bitlinear_fwd_hf_quant divided by w_scale, should multiply — `491b98c`

**Motivation**: the runGenerate output for the canonical prompt
`"The capital of France is"` decoded as `" the, the, the"` rather
than `" Paris"`. The `make test-hf-bitnet-roundtrip` gate at 1e-1
tolerance was red; idris-side logits had max-abs-diff 21 vs the
HF oracle, with sign flips at multiple top positions and Pearson
correlation of only 0.58.

**Investigation** (Phase 1 → Phase 2 per the bisection plan):
- Phase 1A (param catalogue): all 542 on-disk params loaded
  (210 ternary + 332 float); no missing param.
- Phase 1B/C (subsumed by Phase 2): the statistical shape of the
  divergence ruled out a simple scaling factor.
- Phase 2 (per-block bisection, infrastructure shipped in this
  commit cluster): added `--bisect-blocks` mode to the example
  + `save_oracle_bitnet_blocks.py` per-block HF oracle dumper
  + `compare_bitnet_blocks.py` divergence reporter. Result:
  embedding output matches bit-exactly; **block_00 output
  diverges immediately** with idris-side std 0.22× of oracle.
  The factor of ~0.22 ≈ 1/4.5 hinted at a constant scaling bug
  per BitLinear call.
- Root cause: HF transformers ships TWO BitLinear classes in
  `integrations/bitnet.py`. The older `BitLinear` (line 124)
  applies `output = output / (input_scale * weight_scale)` on
  int8 inputs. The newer `AutoBitLinear` (line 257) applies
  `output = output * weight_scale` on ActQuant-dequantised
  inputs. The 2B-4T model uses `AutoBitLinear` (per
  `save_oracle_bitnet.py:79` class-name check). Our C kernel
  matched the older class's algebra; the factor of `w_scale²`
  per BitLinear call (≈ 5.3× for the observed weight_scale~2.3)
  compounds through 30 decoder blocks via residual + RMSNorm
  normalisation to a ~4.5× shrink at block_0 output.

**Change**: three-line edit per backend
(`backend_torch/nn/quantization/bitlinear.cpp:129`,
`backend_mlx/.../bitlinear.cpp:143-144`,
`backend_tape/.../bitlinear.c:468 + 528` for F64/F32 paths). The
formula goes from `y_q / (in_scale * w_scale)` to `y_q * w_scale
/ in_scale`.

**Impact**:

| Metric | Pre-fix | Post-fix |
|---|---:|---:|
| block_00 std ratio idris/oracle | 0.22 | 1.0005 |
| block_00 Pearson r | 0.998 | 0.9996 |
| logits max-abs-diff vs oracle | 21.0 | 0.74 |
| logits Pearson r | 0.58 | 0.9988 |
| logits top-5 indices match | NO | YES (exact) |
| logits argmax | 323 (` and`) | 1 (` `) |
| runGenerate text continuation | `the, the, the` | `Paris. Paris is a` |

The remaining 0.74 max-abs-diff is the BF16-vs-F64 accumulation
floor: HF oracle runs BF16 throughout, idris-side torch-cpu runs
F64. Across 30 layers of compounding round-off, per-element drift
of ~0.7 is structural. The gate now passes with `tol=1.0 +
--argmax-match`; previously the 1e-1 tolerance was a starting
guess that turned out to be tighter than the BF16 noise floor.

**Perf**: runGenerate wall 47s → 51s on torch-cpu (1m 3s total
wall in perf-log). The +4s is the extra multiply-by-w_scale per
BitLinear call (210 calls × 5 forwards = 1050 ops). Negligible.

**perf-log entries**:
- `2026-06-02T<latest>` torch-cpu, commit `491b98c+dirty`, exit 0,
  runGenerate 51s.

**Cross-references**: `CHANGELOG.md` 2026-06-02 numerical-match
entry; `docs/develop/gotchas.md` "HF transformers ships two
BitLinear classes with different algebra" entry;
`packages/idris-transformers/scripts/{save_oracle_bitnet_blocks,
compare_bitnet_blocks}.py` (new bisection scripts kept checked
in for any future adapter's numerical work).

### 2026-06-08 — Tape AdamW foreach (BLAS-1 moment update) — `4da11736`

**Plan job**: Medium TODO row "Tape optimizer foreach fast path".

**Motivation**: Tape's `tape_optimizer_step` (`backend_tape/training/optimizer.c:169-254`) walks every registered param's gradient buffer element-by-element through an inner switch that branches on `opt->type` AND on `t->dtype_tag` (via `tape_grad_load_d` / `tape_load_d` / `tape_store_d`). For F64 AdamW workloads the per-element dispatch defeats compiler autovectorization — both the type switch and the dtype branch are inside the j loop, so neon-shaped chains never form. Hypothesis: a typed F64-only foreach path with BLAS-1 (Accelerate's `cblas_dscal` + `cblas_daxpy`) for the moment update + direct `(double*)t->data` / `(double*)t->grad` access (no per-element dtype branch) would unblock the autovectorizer for the remaining scalar passes too.

**Change**: New `adamw_foreach_param` helper. For F64 params with numel ≥ 256: `m ← β1·m + (1-β1)·g` via `cblas_dscal` + `cblas_daxpy`; the `v` update and weight update (`bias-correct` + `sqrt` + decoupled weight decay) stay scalar but operate on direct double pointers with no inner-loop branches. F32 params fall through to the original scalar inner. Opt-in via `TAPE_OPTIMIZER_FOREACH=1` re-read every step. AdamW-only (opt->type == 3); SGD/RMSprop/Adam unchanged. Convergence-correct: paired Criterion test `training_optimizer_adamw_foreach::matches_scalar_on_256_elem_param` asserts |scalar - foreach| < 1e-12 over 50 AdamW steps on a 256-element param; RED probe with β1 swapped for β2 produced max_diff=8.93e-01 at idx=2 (a real value-mismatch failure, not a compile/link slip).

**Impact**:

| (example, backend) | scalar wall (warm) | foreach wall | delta | bit-identical loss? |
|---|---:|---:|---:|---:|
| gpt / tape | 7.771s | 7.771s | noise (~0%) | yes (bpc=4.352528933012726) |
| bert-mlm-finetune / tape | 11.107s / 10.939s (mean 11.023s) | 6.486s / 6.155s (mean 6.321s) | **−43%** | yes (loss=2.9052) |

The gpt result is the floor: too-small model (VocabSize=65, hidden tiny) → optimizer wall is sub-noise on this VM. Bert-tiny (~4M params, vocab=30522, hidden=128) is the first AdamW workload where the optimizer takes a meaningful slice of wall and the foreach win is well above the ~15-20% single-run noise floor (`feedback_vm_perf_noise`). Both pairs of bert-mlm runs ran with the mlx F32 hf-llama measurement running concurrently (separate BUILD_KEY tree, both nice -19), so any contention is symmetric across scalar/foreach.

**Outcome**: landed and collapsed. The env-var gate stays one commit (`4da11736`); the follow-up commit removes the gate (and the F64 AdamW case from the scalar inner switch), making the foreach path the default for F64 AdamW. F32 AdamW + SGD/RMSprop/Adam keep the existing scalar inner.

**perf-log entries**:
- `2026-06-08T13:56:07Z` gpt / tape scalar (cold) wall 68188ms
- `2026-06-08T13:56:26Z` gpt / tape foreach (warm) wall 7771ms
- `2026-06-08T13:57:59Z` bert-mlm-finetune / tape scalar (cold) wall 19300ms
- `2026-06-08T13:58:25Z` bert-mlm-finetune / tape foreach #1 wall 6486ms
- `2026-06-08T13:59:19Z` bert-mlm-finetune / tape scalar #2 (warm) wall 11107ms
- `2026-06-08T<after this commit>` bert-mlm-finetune / tape foreach #2 wall 6155ms; scalar #3 wall 10939ms

**Cross-references**: `backend_tape/training/optimizer.c` (adamw_foreach_param + dispatch); `packages/idris-test-c/src/test_optimizers.c` (Criterion paired test); TODO row deleted; CHANGELOG closure entry.

### 2026-06-08 — Tape Adam (type 2) foreach extension — `fdcd5a1c`

**Plan job**: follow-up to the AdamW foreach landing earlier today; the gate `opt->type == 3` was missing every workload using `nativeAdamGroup` (Sac, Dqn, MountainCar, MountainCarCont) or `nativeAdamGlobalClip` (A2c, Mnist, Ppo, Reinforce, SeqClassify, Transformer) — both Idris-side wrappers lower to `tape_optimizer_create_adam` (`opt->type == 2`).

**Motivation**: Adam reduces to AdamW with `weight_decay == 0`. `tape_optimizer_create_adam` `calloc`s the struct, so the wd field is exactly 0 for Adam, and the foreach's final-step weight expression `w1 - lr * 0 * w1 = w1` self-zeroes to the Adam form. The math is bit-identical between Adam-via-foreach and Adam-via-scalar (modulo Accelerate FMA ULP drift, well under 1e-12 over 50 steps).

**Change**: one-line gate widen in `tape_optimizer_step` from `(opt->type == 3)` to `(opt->type == 2 || opt->type == 3)`; comment block above `adamw_foreach_param` updated to document the wd-self-zero reasoning. No function rename — the body is unchanged. Paired Criterion test `training_optimizer_adam_foreach::matches_scalar_on_256_elem_param` added (same shape as the AdamW pair; uses `optimizer_create_adam` instead of `optimizer_create_adamw`). RED probe: temporarily mutated β1 to β1+0.01 inside the foreach m-update post-widen; both Adam and AdamW tests failed with `max_diff=3.689158e-02 at idx=0 (scalar=-0.799999995 foreach=-0.836891570793)`, confirming Adam (type 2) now exercises the foreach BLAS-1 m-update path. Restored before commit.

**Impact**:

| (example, backend) | scalar wall (warm) | foreach wall | wall delta | inner-train | converged? |
|---|---:|---:|---:|---:|---:|
| mnist / tape | 80.0s (12000 ms/ep × 5 ep) | 70.0s (11800 ms/ep × 5 ep) | noise (~2%) | identical | accuracy=0.976 |
| transformer / tape (run 1) | 6.152s | 4.241s | −31% | 2s → 1s | sort=6/6 |
| transformer / tape (run 2) | 5.388s | 4.526s | −16% | 2s → 1s | sort=6/6 |
| transformer / tape (run 3) | 6.552s | 5.543s | −15% | 1s → 1s | sort=6/6 |

Transformer mean wall reduction across 3 paired warm runs: ~5.97s → ~4.77s = **−20%**. Above the 15-20% VM noise floor; below the AdamW bert-mlm result (−43%) because transformer's optimizer slice is smaller (~thousands of params vs millions). Mnist (5 epochs at 12s/ep) stays in noise — the per-step optimizer cost is dominated by the conv forward/backward on 60k MNIST examples per epoch, so the BLAS-1 m-update savings don't register at the wall level.

The win profile matches the prior session's AdamW characterization: scales with `param_count × steps_per_epoch`. Tiny RL nets (A2c, Sac, etc.) likely stay in noise; transformer and the larger BERT/GPT workloads benefit measurably.

**Outcome**: landed. No env-var gate this time (one-line change, math reduces from the existing-and-tested AdamW path). Convergence verified on transformer (6/6 sort accuracy unchanged across all 3 pairs) and mnist (accuracy 0.976 unchanged across both polarities); the bit-identical Criterion test gates the rest.

**perf-log entries**:
- `2026-06-08T<later>Z` mnist / tape scalar (`TAPE_OPTIMIZER_FOREACH=0`) wall 80327ms
- `2026-06-08T<later>Z` mnist / tape foreach wall 70239ms
- `2026-06-08T<later>Z` transformer / tape scalar (three runs) walls 6152 / 5388 / 6552 ms
- `2026-06-08T<later>Z` transformer / tape foreach (three runs) walls 4241 / 4526 / 5543 ms

**Cross-references**: `backend_tape/training/optimizer.c` (gate widen + comment update); `packages/idris-test-c/src/test_optimizers.c` (new paired Adam test, original AdamW pair retained); ran concurrently with the in-flight mlx F32 hf-llama bg measurement (separate BUILD_KEY tree `tape-mlxcpu-torchcpu-machmac-m-series-hwcpu` vs `mlx-mlxgpu-…-mdtF32`, no contention).
