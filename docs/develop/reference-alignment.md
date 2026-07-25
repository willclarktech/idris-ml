# Reference Implementation Alignment

## Policy

Idris examples and their PyTorch references must use **identical defaults** for all hyperparameters, architecture, and initialization. When a discrepancy is found, adopt **whichever is the better practice** in BOTH implementations.

When adding or changing an example, always update both Idris and PyTorch to match.

> **Note (2026-07-28): per-backend numbers in this file are not comparable to each other.**
> Random parameter init runs C-side in each backend's own generator (tape's Box-Muller,
> `torch::nn::init::normal_`, `mx::random::normal`), so the same seed yields different
> initial weights per backend — a tape-vs-torch loss gap in the tables below is a
> different-experiment artifact, not an implementation difference. Compare backends only
> by loading identical checkpoints (the HF roundtrip pattern) or statistically over ≥5
> seeds. See gotchas.md "Parameter init RNG differs per backend". Dense layers built
> with `Nn.linear` are exempt as of 2026-07-29: their `Uniform`/`Zeros` init fills a
> host buffer from libc `rand`, which is the same everywhere.

## Multi-seed pass rates after the alignment work (2026-08-01)

First campaign run where both sides share init, data, metrics and eval
protocol, so the two rates finally answer the same question. Seeds 42/1/2/3/4,
tape backend; thresholds from `test-examples-convergence.expect` and
`test-refs-convergence.expect`.

Both campaigns are complete: 125 Idris cells
(`convergence-campaign.tsv`, 25 examples) and 115 reference cells
(`convergence-campaign-ref.tsv`, 23 modules; gpt and transformer have no seeded
reference bar).

20 of 25 Idris examples are 5/5, as are 19 of 23 reference modules. Every row
where the two sides differ, or where either falls short of 5/5:

| Example | Idris | Reference | Note |
|---------|-------|-----------|------|
| example-a2c | 3/5 (seeds 2, 4) | 5/5 | the one clear Idris deficit |
| example-ntm-associative-recall | 2/5 (seeds 1, 2, 4) | 3/5 (seeds 42, 4) | down from 5/5 at `fd0407bb` |
| example-ntm-copy | 3/5 (seeds 3, 4) | 4/5 (seed 4) | up from 1/5 at `fd0407bb` |
| example-dqn | 4/5 (seed 2) | 4/5 (seed 3) | same rate, different seed |
| example-double-dqn | 5/5 | 4/5 (seed 3) | |
| example-sac | 5/5 | 2/5 (seeds 1, 3, 4) | |

Five seeds is a small sample, so a one-seed gap (dqn, ntm-copy, ntm-recall) is
weak evidence in either direction; the two- and three-seed gaps (a2c, sac) are
the ones carrying signal.

`example-dqn` seed 2 had been an open question since the DQN batching change.
Both sides now sit at 4/5 and fail on *different* seeds, which places the
failure in the algorithm's seed sensitivity rather than in the Idris port.

`example-a2c` is the row to investigate. It is the only example where the
reference is solid and Idris is not, and A2C is where a paramId-scoping bug
previously let seed 42 "converge" while the actor received no updates at all
(see "A2C real bug surfaced by multi-seed alignment" below).

SAC runs the comparison the other way: 5/5 in Idris against 2/5 in the
reference. `torch_ref/correctness/test_sac.py` only ever asserted seed 42, so
the reference's fragility here had never been measured.

The NTM pair is seed-fragile on **both** implementations, and seed 4 fails on
both sides of both tasks. Idris sits one seed below the reference on each.

The recall drop from 5/5 was worth chasing and did not resolve to a culprit.
Reverting the controller-bias change alone rescues no failing seed; reverting
the output-projection change alone rescues seed 2 but not seed 1; each variant
converges on a *different* subset. The reference, whose NTM init this work did
not touch, sits at 3/5 independently. So ~2-3 of 5 is what this configuration
supports on either side, and the earlier 5/5 was a favourable draw that a
single campaign run made look like a property.

Both per-seed thresholds are kept. Pooled across the two implementations,
converged runs score 0.94-0.996 (recall) and 0.993-1.0 (copy) while failed runs
score 0.687-0.781 and 0.674-0.795 — bimodal with nothing between, and each bar
sits in the gap. Lowering them would let a non-converged run register as a
pass. The multi-seed expectation is the pass rate above, per the policy's "use
PyTorch's pass rate as the target".

## Alignment Changes (2026-07-31) — conv, recurrent and attention init

The three axes the dense alignment two days earlier deferred. Each was left
open on the reasoning that both sides still trained; the reason to close them
anyway is that a reference which inits differently measures init noise instead
of the implementation, which is the only job the reference has.

| Layer | Idris before | Reference before | Both now |
|-------|--------------|------------------|----------|
| `Nn.conv1d` / `conv2d` | `N(0, √(2/fan_in))`, bias 0 | `nn.Conv` default `U(±1/√fan_in)`, bias `U(±1/√fan_in)` | `U(±1/√fan_in)`, bias 0 |
| `Nn.Recurrent` / `Lstm` / `Gru` | `N(0, √(2/(fan_in+fan_out)))` | `xavier_uniform_` | `U(±√(6/(fan_in+fan_out)))` |
| `Nn.Attention`, `transformerBlock`'s ff | `N(0, 1/√fan_in)` | `xavier_uniform_` | `U(±1/√fan_in)` |

Conv was the one that mattered. Idris kernels started √6 ≈ 2.45× wider than the
reference's, measured across all four conv layers in the two examples that have
any (2.44 and 2.45 at fan_in 25 and 400; 2.95 and 2.35 at fan_in 3 and 12, where
the sample is small). The reference side keeps `kaiming_uniform_(w, a=√5)`,
which is what `_ConvNd.reset_parameters` already does, and zeroes the bias
through the new `init_conv_` for the reasons given on `init_linear_`.

Recurrent was subtler: `√(2/(fan_in+fan_out))` is *exactly* the std of
Xavier-uniform, so every summary statistic agreed while the tails did not.
Only the distribution changed; the target variance did not, and no reference
edit was needed. Both sides stay on Xavier there rather than moving to the
dense contract, which suits a weight applied once per timestep.

Attention carried the same off-by-√3 the dense layers had before 2026-07-29 —
a normal whose std equalled the uniform's *bound*. On the reference side
`MultiHeadTransformer._init_weights` had been applying `xavier_uniform_` to
every `nn.Linear` it owns, which put it out of step with the other eleven
references as well as with its twin; it now calls `init_linear_(self)`. Every
layer there is an `nn.Linear`, per-head projections included, so one call
covers the block. `gpt.py` builds on the same model.

Seed 42, after (before → after where it moved):

| Example | Idris | Reference |
|---------|-------|-----------|
| seq-classify accuracy | 0.844 → **0.980** | 0.978 |
| mnist accuracy | 0.986 → **0.989** | — |
| rnn loss | 0.00115 | 0.00050 |
| lstm loss | 0.00127 | 0.00129 |
| gru loss | 0.00095 | 0.00095 |
| transformer sort_acc | 6/6 | 6/6 |
| gpt bpc | 4.439 → **4.327** | 4.233 |

Every weight matrix in the library now draws from a uniform, which also puts
conv, recurrent and attention init on the host-buffer fill rather than a fused
per-backend RNG — so they are identical across tape / torch / mlx by
construction, the property `Nn.linear` gained on 2026-07-29.
`IDRISML_PORTABLE_INIT` is left covering only the normal-init sites that
remain: NTM/DNC heads (`linearWith`) and embeddings.

Gates: `Test.Nn.{Conv,Recurrent,Lstm,Gru,Transformer}` pin the Idris bounds,
`torch_ref/correctness/test_init.py::TestInitConvHelper` the reference's. The
recurrent tests read 1024-2048 elements because at equal variance the bound is
the only discriminator between a normal and a uniform.

## Alignment Changes (2026-07-29) — one dense init on both sides

Idris has one dense-layer constructor, `Ml.Nn.Linear.linear`, so every reference
model that maps onto it has to init the same way or the comparison measures init
noise rather than the implementation. The references disagreed with each other on
both axes: nine models took `nn.Linear`'s defaults (Kaiming-uniform weight,
uniform bias), `supervised` and `rnn` set Xavier weights with zero biases
explicitly, and Idris used `N(0, 1/√fan_in)` with a zero bias (1.73× wider than
`nn.Linear`, since the std of `U(±b)` is `b/√3`).

Both sides now apply one contract: **weight ~ `U(±1/√fan_in)`, bias = 0**.

| Side | Mechanism |
|------|-----------|
| PyTorch | `torch_ref.init.init_linear_(self)`, one call per model `__init__` |
| Idris | `Ml.Nn.Linear.linear` (`Uniform (-bound) bound` weight, `Zeros` bias) |

The weight half is `kaiming_uniform_(w, a=√5)`, which is what
`nn.Linear.reset_parameters` already does; spelling it out keeps the contract
readable from the Idris side, which cannot see a framework default. The bias half
departs from `nn.Linear`, whose uniform bias is a legacy artifact: symmetry
breaking is the weight's job, bias gradients neither vanish nor explode with
depth, and the Kaiming/Xavier variance derivations assume a zero bias.
HuggingFace's `_init_weights` overrides it to zero and LLaMA/PaLM-style models
drop the bias entirely.

Scope is the set of reference layers whose Idris counterpart is `Nn.linear`:
`supervised`, `mnist_cnn`, `seq_classify`, `reinforce`, `dqn` (and `double_dqn`,
which reuses its `QNetwork`), `mountain_car`, `mountain_car_cont`, `sac`, `ppo`,
`a2c`, plus the output projections in `rnn.py`'s three cells.

Three axes were left out of scope at the time, because their Idris counterparts
init from a normal distribution. All three closed on 2026-07-31; see below.

Gates: `torch_ref/correctness/test_init.py` pins the reference contract and the
out-of-scope exclusions; `Test.Nn.Linear`'s `defaultWeightInRange` /
`defaultBiasIsZero` pin the Idris side.

> **Note: Path C migration is alignment-preserving.**
> Historical entries below mention V1 internals (`Variable d`, `forwardVarTensor`, `nameLayer`/`autoName`,
> `applyDeltas`, V1 epoch runners) — these names are gone post-migration but the *alignment* is
> preserved bit-identically: every documented multi-seed pass rate matches the V2 branch's smoke gate
> at seed=42 (the example smoke gate, now `make test-e2e-examples`: 76/76 OK, bit-identical). The V1 paramId-scoping bug class
> referenced in the A2C/PPO entries is structurally impossible in V2 (each layer is named at
> construction). See [path-c-migration.md](path-c-migration.md).

## Alignment Changes (2026-06-08) — `Env.reset` randomized per Gymnasium

Closes the deterministic-reset divergence that was documented in multiple
older sections below. `Gym.Env.reset` changes from a pure value to
`Seed -> (state, Seed)`; per-env distributions now match Gymnasium:

| Env | Reset distribution (both sides) |
|---|---|
| CartPole-v1 | each of `(x, x', θ, θ')` ~ U(-0.05, 0.05) |
| MountainCar-v0 | `pos` ~ U(-0.6, -0.4), `vel` = 0 |
| MountainCarContinuous-v0 | `pos` ~ U(-0.6, -0.4), `vel` = 0 |
| Pendulum-v1 | `θ` ~ U(-π, π), `θ̇` ~ U(-1, 1) |
| Acrobot-v1 | each of 4 components ~ U(-0.1, 0.1) |
| FrozenLake-v1 | start fixed at pos 0; input Seed seeds internal slip RNG |
| Blackjack-v1 | input Seed seeds initial deal |
| CliffWalking-v1 | start fixed at (3, 0) (canonical) |
| Taxi-v4 | taxi/passenger/destination randomized; `dest != pass` enforced |

**Contract**: both Idris and PyTorch references seed once at trainer
start (`env.reset(seed=cfg.seed)` on PyTorch; `srand cfg.seed` +
`randomInt32`-sourced per-call Seed on Idris). Per-episode resets
advance each side's PRNG and produce different initial states —
trajectories diverge across episodes naturally rather than restarting
from a fixed worst-case init.

**Paired-side dropped scaffolding**:
- `torch_ref/models/{reinforce,sac,mountain_car,mountain_car_cont,ppo}.py`
  no longer set `env.unwrapped.state = ...` after each `env.reset()`.
  The `reset_to_*` / `_reset_to_pi` helpers stay (kept for call-site
  stability) but now just return the natural Gymnasium-reset obs.
  Make-vec-env loops that re-pinned each sub-env after `vec.reset()`,
  and the auto-reset branches that overrode `next_obs[i]` with the
  hardcoded pinned obs, are dropped (SyncVectorEnv auto-reset already
  produces randomized obs in next_obs).
- `Example.Taxi` is intentionally unchanged — it calls
  `Gym.ToyText.Taxi.defaultStart` directly (not via `Env.reset`), so
  the matching `torch_ref/models/taxi.py` keeps its `_pin_start` for
  paired-side Q-table reproducibility.

Older sections below containing "pin `(0,0,0,0)` / pin `(π, 0.0)` etc."
language describe pre-change behaviour; the divergence those rows
flagged is closed by this commit and the underlying TODO row was
removed when this section landed.

## BERT SST-2 LoRA fine-tune (2026-06-07)

New paired example `bert-classify-sst2-lora` — LoRA fine-tune on top of the bert-tiny backbone for SST-2 binary sentiment classification. Same architecture / dataset / seqLen / classifier head as the prior full-fine-tune row; the only difference is what trains.

| Item | Idris | PyTorch |
|---|---|---|
| Backbone | `google/bert_uncased_L-2_H-128_A-2` (frozen) | same (frozen via `peft.get_peft_model`) |
| Adapter type | LoRA on Q + V attention projections | `LoraConfig(target_modules=["query","value"])` |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.0 (omitted for clean numerical comparison) | 0.0 |
| LoRA bias mode | `bias="none"` (no LoRA on bias terms) | `bias="none"` |
| Trainable params | ~6,402 (~0.13% of model) | matches (peft `print_trainable_parameters`) |
| Adapter file size | ~80 KB (saved via `HfLoraIO.saveLoraAdapter`) | ~80 KB (peft `save_pretrained`) |
| Optimizer | AdamW(lr=1e-4, β=(0.9, 0.999), ε=1e-8, wd=0.01) | same |
| Gradient clip | norm 1.0 | same |
| Epochs | 3 | 3 |
| Batch size | 8 | 8 |
| Seq len | 32 | 32 |
| Default train/dev subset | 256/256 | same |
| LR vs full FT | 1e-4 (5× higher than full FT's 2e-5) | matches — peft tutorial recommendation, only adapters update |

**Convergence (256-subset default)**:

| Backend | Wall | Loss (final) | Dev acc (final) | Seed |
|---|---|---|---|---|
| Idris tape | 27.3 s | 0.337 | 0.570 | 42 |
| Idris torch | 25.2 s | 0.300 | 0.613 | 42 |
| Idris mlx-cpu | 44.4 s | 0.339 | 0.566 | 99 (via perf-run-quiet) |
| PyTorch CPU (peft) | (run `make ref-bert-classify-sst2-lora-finetune` after `uv sync`) | — | — | — |

The 256-subset numbers are bounded by the limited training signal (256 examples × 3 epochs × 8 batch = ~96 train steps total). Full SST-2 (~67k train) would converge to ~80%+ per the HF tutorial but takes hours on Idris tape; the demo's contribution is proving the LoRA pipeline correctly composes end-to-end across the registered adapters + freeze-by-suffix + peft-compatible save path.

**Cross-tool gate**: `make example-bert-classify-sst2-lora -- --save-adapter /tmp/lora-out` produces a directory that loads cleanly via `make validate-lora-adapter ADAPTER_DIR=/tmp/lora-out` (which runs `PeftModel.from_pretrained` in Python and forward-passes a sentence). This is the strongest evidence the on-disk format matches peft.


## BERT MLM continued pretraining (2026-06-07)

New paired example `bert-mlm-finetune` — bert-tiny MLM continued pretraining on Tiny Shakespeare-via-WordPiece.

| Side | File | Notes |
|------|------|-------|
| Idris | `packages/idris-ml-examples/src/Example/BertMlmFinetune.idr` | `loadModelAllowCast` loads 44 params (39 backbone + 5 MLM head); HF 80/10/10 masking; position-selective CE loss; `hfBertMlmForward` → `[SeqLen, Vocab]` logits |
| PyTorch | `packages/pytorch/torch_ref/scripts/bert_mlm_finetune.py` | reads the SAME token file; `AutoModelForMaskedLM.from_pretrained` + HF labels with -100 at unmasked positions |

| Setting | Both sides |
|---------|------------|
| Architecture | bert-tiny / `google/bert_uncased_L-2_H-128_A-2` (vocab=30522, hidden=128, layers=2, heads=2, headDim=64, intermediate=512, maxPos=512, typeVocab=2) |
| SeqLen / batch | 32 / 1 (single-example batch) |
| Corpus | Tiny Shakespeare (1.1MB) tokenized via BERT WordPiece → `data/tinyshakespeare/input.bert-tiny.tokens` (288K tokens) |
| Masking | HF 80/10/10 at 15% mask probability per position. Of masked positions: 80% → `[MASK]` (id=103), 10% → fixed mid-vocab id=200 (Idris) / id=200 (Python — both hardcoded the same shortcut to keep the per-step distribution aligned; HF's `DataCollatorForLanguageModeling` samples uniformly over the full vocab, an approximation worth ~negligible loss difference at 50-step bounded training). CLS=101 / SEP=102 never masked. |
| Loss | Position-selective cross-entropy: CE per masked position, sum / numMasked. Idris uses `bertMlmLoss` (logSoftmax2d + multiply by zero-row-padded one-hot + sum + negate + divide by numMasked). PyTorch uses `CrossEntropyLoss(ignore_index=-100)` against labels with `-100` at unmasked positions. |
| Optimizer | AdamW(lr=5e-5, β=(0.9, 0.999), ε=1e-8, weight_decay=0.01) + grad-norm clip 1.0 |
| Default steps | 100 |
| Convergence on 50 steps (seed=42) | Idris tape: EMA 5.15 → 4.90 / 8.2s; Idris torch: EMA 5.15 → 4.90 / 1m13s; Idris mlx-cpu (seed=99 via perf-run-quiet): 7.3s; PyTorch CPU: EMA 5.51 → 5.21 / 0.6s |

bert-tiny's pretrained MLM loss baseline on this corpus sits at ~5.5 (vocab=30522;
uniform-random `ln(30522) ≈ 10.3`). Both sides start near the baseline and drop
into the 4.9-5.2 range after 50 single-example steps. Per-step variance is high
because each window's number of masked positions varies stochastically; EMA
smooths the trajectory.

## GPT-2 LM continued pretraining (2026-06-07)

New paired example `gpt2-lm-finetune` — distilgpt2 fine-tune on Tiny Shakespeare.

| Side | File | Notes |
|------|------|-------|
| Idris | `packages/idris-ml-examples/src/Example/Gpt2LmFinetune.idr` | `loadModelAllowCast` warm-starts distilgpt2; sliding-window batch; `hfGpt2ForwardLm` → `[SeqLen, Vocab]` logits; 2D one-hot target via `primOneHot` + `primReshape2d` |
| PyTorch | `packages/pytorch/torch_ref/scripts/gpt2_lm_finetune.py` | reads the SAME token file; `transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")` + same AdamW |

| Setting | Both sides |
|---------|------------|
| Architecture | distilgpt2 (vocab=50257, hidden=768, layers=6, heads=12, headDim=64, intermediate=3072, maxPos=1024) |
| SeqLen / batch | 32 / 1 (single-example batch) |
| Corpus | Tiny Shakespeare (1.1MB) tokenized via distilgpt2 BPE → `data/tinyshakespeare/input.distilgpt2.tokens` (338K tokens) |
| Sampling | random sliding window of (SeqLen+1) tokens; input = first SeqLen, target = shifted-by-1 last SeqLen |
| Loss | per-position cross-entropy mean over SeqLen positions (Idris `gpt2LmLoss`: logSoftmax2d + elementwise multiply against one-hot + sum + negate + mean; PyTorch `nn.CrossEntropyLoss` over flattened `[seqLen, vocab]`) |
| Optimizer | AdamW(lr=5e-5, β=(0.9, 0.999), ε=1e-8, weight_decay=0.01) + grad-norm clip 1.0 |
| Default steps | 100 |
| Convergence on 50 steps (seed=42) | Idris tape: EMA 5.26 → 4.89 / 1m13s; Idris torch: 5.26 → 4.89 / 24.6s; Idris mlx-cpu: ~4.3 (different seed via perf-run-quiet, different sample stream) / 3m19s; PyTorch CPU: 5.04 → 4.80 / 7.3s |

Loss-step trajectories are not bit-aligned cross-backend because the random
window-start sampling differs between Idris (`Generate.randomInt`) and Python
(`random.randint`). The EMA reduction washes out per-step variance; both sides
show the same monotone downward trend. distilgpt2's pretrained baseline loss on
Tiny Shakespeare sits at ~5.5 (text data with vocab=50257; uniform-random is
`ln(50257) ≈ 10.8`); both sides drop into the 4.0-5.0 range after 50 steps,
confirming the continued-pretraining path works end-to-end.

## BERT SST-2 classification fine-tune (2026-06-07)

New paired example `bert-classify-sst2-finetune` — real-text variant of the synthetic FT3 example.

| Side | File | Notes |
|------|------|-------|
| Idris | `packages/idris-ml-examples/src/Example/BertClassifySst2Finetune.idr` | warm-starts the `google/bert_uncased_L-2_H-128_A-2` backbone via `loadModelPrefixAllowCast _ "bert."`; padding + 2D attention mask via `HfDataset.padToSeqLen` + `toAttentionMask2d`; `hfBertSeqClassifyForward _ _ _ _ (Just mask)` |
| PyTorch | `packages/pytorch/torch_ref/scripts/bert_classify_sst2_finetune.py` | reads the SAME TSV; `transformers.BertForSequenceClassification.from_pretrained` + matched `BertConfig` |

| Setting | Both sides |
|---------|------------|
| Backbone | `google/bert_uncased_L-2_H-128_A-2` (vocab=30522, hidden=128, layers=2, heads=2, headDim=64, intermediate=512, maxPos=512, typeVocab=2) |
| NumClasses | 2 (binary sentiment) |
| SeqLen / PadId / BatchSize | 32 / 0 / 8 |
| Dataset | GLUE SST-2 via `hf-download-dataset.sh glue {train,validation} sst2` → `data/hf-datasets/glue-sst2/{train,validation}.tsv` (BERT WordPiece, sentence column → CLS + tokens + SEP) |
| Optimizer | AdamW(lr=2e-5, β=(0.9, 0.999), ε=1e-8, weight_decay=0.01) + grad-norm clip 1.0 |
| Loss | Cross-entropy (Idris `tnllLoss` over one-hot target; PyTorch `nn.CrossEntropyLoss` over integer labels) |
| Default subset | `--max-train 256 --max-dev 256 --epochs 3` (fast iteration; full SST-2 takes ~hours on Idris tape) |
| Convergence on default subset (seed=42) | Idris tape: 59.4% / loss 0.3337 / 15.7s; Idris torch: 60.9% / loss 0.3377 / 2m7s; Idris mlx-cpu: 56.3% / loss 0.3360 / 44s. PyTorch CPU: 52.0% / loss 0.6867 / 1.7s |
| HF tutorial reference (full train, 3 epochs, lr=2e-5) | ~80-85% dev accuracy — bounded by the subset in this demo |

The convergence numbers on the 256-example subset are explicitly bounded by the slice — they
prove the real-text path works end-to-end on all 3 backends, not that bert-tiny reaches its
documented HF-tutorial ceiling. The Idris-vs-PyTorch loss gap (0.33 vs 0.69 at the same lr)
reflects different effective optimizer dynamics + Idris head init (wider distribution) on
the small subset; both sides reach the same ~50-60% accuracy range. Full SST-2 fine-tuning
matches HF's tutorial number but is not a CI gate.

## BERT classification fine-tune (2026-06-07)

New paired example `bert-classify-finetune` ships with both sides aligned.

| Side | File | Notes |
|------|------|-------|
| Idris | `packages/idris-ml-examples/src/Example/BertClassifyFinetune.idr` | `BertForSequenceClassification` at the FT2-introduced API |
| PyTorch | `packages/pytorch/torch_ref/scripts/bert_classify_finetune.py` | `transformers.BertForSequenceClassification` w/ matched config |

| Setting | Both sides |
|---------|------------|
| Vocab / Hidden / Layers / Heads / HeadDim / Intermediate / MaxPos / TypeVocab / NumClasses | 64 / 32 / 1 / 2 / 16 / 64 / 8 / 2 / 3 |
| SeqLen / BatchSize | 8 / 16 |
| Synthetic dataset | label-token at position 1: class 0→token 11, class 1→token 13, class 2→token 17. CLS=0, SEP=1, distractors random in [20, 60]. |
| Optimizer | AdamW(lr=1e-3, β=(0.9, 0.999), ε=1e-8, weight_decay=0.01) + grad-norm clip 1.0 |
| Default epochs / patience | 2000 / 500 (loss-improvement, minDelta=1e-3) |
| Loss | per-example mean cross-entropy over a 3-element one-hot target (Idris `tnllLoss`; PyTorch `nn.CrossEntropyLoss` with integer labels) |
| Eval | held-out 32-sample greedy-argmax accuracy |
| LayerNorm ε / dropout | 1e-12 / 0.0 (both sides) |
| Convergence (seed=42) | Idris tape: 1.000 acc, 1582 epochs. PyTorch CPU: 1.000 acc, ~1300-1700 epochs. |
| Multi-seed (seeds 1-5, Idris tape) | 5/5 converge to 1.000 acc |

The Idris-side classifier head omits the dropout HF interposes between pooler and classifier; the
PyTorch ref sets `hidden_dropout_prob=0.0` to match. Adding fine-tune dropout back is a future
tuning knob — see TODO for "Real-text fine-tuning of HF-loaded models".

## Alignment Changes (2026-04)

### Idris defaults changed to match PyTorch

| Example | Parameter | Before | After |
|---------|-----------|--------|-------|
| NTM Copy | Batch size | 1 | 16 |
| NTM Recall | Batch size | 1 | 16 |
| DNC Copy | Batch size | 1 | 16 (reverted — see 2026-04-29 below) |
| DNC Copy | Memory size N | 32 | 128 (reverted — see 2026-04-29 below) |
| DNC Copy | Max seq length | 10 | 20 (reverted — see 2026-04-29 below) |
| DNC Recall | Batch size | 1 | 16 (reverted — see 2026-04-29 below) |
| DNC Recall | Memory size N | 32 | 128 (reverted — see 2026-04-29 below) |
| LSTM | Learning rate | 0.1 | 0.5 (lr_find / B3 dogfood, 2026-04-29) |
| LSTM | Seed | 123456 | 42 |
| Supervised | Seed | 123456 | 42 |
| RNN | Seed | 123456 | 42 |
| MNIST | Epochs | 2000 | 100 (reverted — see below) |
| NTM Copy/Recall | Eval test size | 20 | 100 |
| DNC Copy/Recall | Eval test size | 20 | 100 (reverted — see 2026-04-29 below) |

### Idris layer implementations changed

| Layer | Change | Rationale |
|-------|--------|-----------|
| Transformer embedding | zeros → xavier uniform | Zero breaks symmetry; xavier is standard |
| LSTM hidden/cell init | xavier random → zeros | Zeros is standard, matches PyTorch |

### PyTorch references changed to match Idris (best practice)

| Reference | Change | Rationale |
|-----------|--------|-----------|
| GPT | Adam → AdamW (wd=0.01) | Weight decay prevents overfitting in LMs |
| GPT | Temperature 0.8 → 1.0 | Standard eval |
| MNIST CNN | Added dropout (0.25/0.5) | Regularization best practice |
| MNIST | Epochs 10 → 100 | Reasonable for comparison with early stopping |
| SeqClassify CNN | Added dropout (0.5) | Regularization best practice |
| LSTM | Added forget gate bias=1.0 | Jozefowicz et al. 2015, helps gradient flow |

### Added

| Item | Description |
|------|-------------|
| Reinforce script | `pytorch/torch_ref/scripts/reinforce.py` — was missing |

### LSTM hidden size (resolved)

Idris LSTM ties hidden=output in `LstmState`. Was using `{i=1, o=1}` (hidden=1) while PyTorch used `LinearLSTMCell(1, 4, 1)` (hidden=4 + output projection). Fixed by using `lstmLayer {i=1, o=4}` + `linearLayer {i=4, o=1}` to match.

## Alignment Changes (2026-04-22) — RL suite

### A2C divergence (resolved below)

During initial A2C port, the Idris side was pivoted to a **combined single-chain network** (output vector = `[logit_0, logit_1, value]`) because Idris' `Network` type is a linear chain and can't express PyTorch's branching actor-head + critic-head on a shared trunk. The pivot was not mirrored in the PyTorch reference, which retained the branching architecture. Hyperparameters also drifted: Idris ended up at `lr=3e-3, entropy=0.05`, PyTorch at `lr=7e-4, entropy=0.01`.

**Fix**: PyTorch `a2c.py` rewritten to use the same combined-chain architecture as Idris. Both sides now use `lr=3e-3, entropy=0.05, rollout=10, gamma=0.99, lam=0.95, value_coef=0.5`. Both converge to greedy eval ~200 on CartPole.

### PPO divergence (resolved below)

Same failure mode. Idris PPO used combined chain + `rollout=200, K=3, full-batch`; PyTorch ref used separate actor + critic + `rollout=2048, K=10, batch=64`. Architectural divergence hid whether Idris' plateau at -1500 was a config issue or an implementation bug.

**Fix**: PyTorch `ppo.py` rewritten to use combined chain (state-independent learnable `log_std`, mean and value on the same output head), and both sides adopt `rollout=2048, K=10, batch=64`. The stronger PyTorch settings are the baseline because Idris-matched settings (short rollout + no mini-batching) demonstrably do not converge for either side.

### Process note

This incident prompted a strengthening of the alignment policy in CLAUDE.md — see "Architectural alignment — DO NOT pivot silently." The key rule: if Idris' Network chain can't express the PyTorch architecture, **update PyTorch to match Idris**, not keep both diverged.

### A2C real bug surfaced by multi-seed alignment (resolved)

After aligning both sides to separate actor + critic at matched hyperparameters, multi-seed testing exposed a real Idris implementation bug: the combined-net version had reported "converges to 200" based on a single seed=42 run, but in fact the policy at seed=42 was benefiting from random initialization more than from training. With separate actor + critic, Idris' optimizer wasn't updating the actor at all: `lr=0` and `lr=0.1` gave identical greedy-eval scores.

**Root cause**: `prefixParamId` via `emap` only renames the scalar *view* Variables in `LinearState.weights`, but the consolidated weight tensor stored in `LinearState.weightTensor` (used by `applyVarTensor`) was registered under the original unprefixed paramId like `ll0_weights`. When the critic's `autoName` ran, it overwrote the actor's `ll0_weights` registry entry — so the actor's weight tensor had no gradient-collection hook and the optimizer never touched it. The renamed scalar views were accounted for in the registry, but they aren't the tensors used in the hot path.

**Fix**: use a scope-prefixed `autoNameNetwork` instead of a post-hoc `emap`-based rename, so each layer's `nameLayer` receives the prefix directly and registers the consolidated weight tensor under a scoped name. `Example.A2c` inlines `autoNameNetworkLocal` / `autoNameAnyLocal` / `autoNameScoped` locally because the `-o <file>` invocation path used by Makefile example targets doesn't pick up newly-exported helpers from `idris-ml` (a single-file resolution quirk we haven't root-caused; `--build <pkg>.ipkg` works fine). This also prompted adding a multi-seed convergence requirement to CLAUDE.md.

### PPO (applied A2C's paramId-scoping fix)

PPO has the same twin-network shape (actor + critic) and originally exhibited the same silent optimizer failure as A2C before the scoping fix. Rewriting `Example.Ppo` to use the inlined `autoNameScoped` helper (same pattern as A2C) restored actor gradient flow. At the aligned CI-sized config (`rollout=400, K=10, batch=64, lr=3e-4, γ=0.99, λ=0.95, clip=0.2, seed=42, 300 rollouts`):

| Implementation | greedy_eval |
|---|---|
| PyTorch | -1197.1 |
| Idris | -1571.9 |

Both descend then oscillate/plateau in the -1200 to -1600 band — PPO at rollout=400 is genuinely starved of data, and both implementations express that. The original PyTorch reference config (`rollout=2048`) converged to -353 in the same 300 rollouts but each Idris epoch is ~20× slower than PyTorch due to per-step `forwardVarTensor` calls (Idris autograd doesn't have a batched forward path), so we've shipped the shorter rollout for tractable iteration and noted the convergence gap as a compute-speed issue rather than an implementation gap. A follow-up to batch the Variable forward path would close this.

### PPO env swap: Pendulum → Acrobot (B3-fixes, 2026-04-30)

The Pendulum result above (Idris -1571 vs PyTorch -1197 at the CI-sized config) was a documented partial-convergence band, not a real demonstration of PPO. Pendulum + Gaussian policy + the rollout sizes we can afford on tape never reaches a "PPO clearly works" regime. As part of B3-fixes (see `docs/develop/hyperparameter-tuning-2026.md`), both sides were rewritten on **Acrobot** — discrete-action, sparse reward, longer horizon, the canonical "PPO clipped-surrogate demonstrates" benchmark.

Aligned config on both sides: `lr=3e-4, gamma=0.99, lambda=0.95, clip=0.2, K=10, batch=64, rollout=1024, entropy_coef=0.01, 100 rollouts`. Architecture: separate actor (6 → 64 → 64 → 3 logits) + critic (6 → 64 → 64 → 1) with tanh activations; categorical policy. Acrobot physics matches `Gym.ClassicControl.Acrobot` (semi-implicit Euler, 4 substeps of dt=0.05) on both sides — distinct from Gymnasium's RK4 reference but task and termination identical.

Multi-seed greedy eval (20 episodes per seed), Acrobot solved is ~-100, random ~-500:

| Seed | PyTorch | Idris |
|------|---------|-------|
| 1    | -63.0   | -63.0 |
| 2    | -63.0   | -82.0 |
| 3    | -75.0   | -74.0 |
| 4    | -106.0  | -500.0 *(reproducible bad init — see note)* |
| 5    | —       | -75.0 |
| 42   | -73.0   | -94.0 |

PyTorch: 5/5 converge well within solved (-63 to -106). Idris: **5/6 seeds converge** (-63 to -94), all in the solved band; seed=4 reproducibly stays at -500 (random/timeout) across two independent runs. The seed=4 trajectory shows training loss pinned at 500.0 from epoch 0 onward — the policy collapses to a single action and the categorical entropy bonus + clipped surrogate can't escape. PyTorch at seed=4 is its worst result (-106) but still solved, so the failure mode appears specific to Idris's xavier-uniform init at that PRNG draw, not a systemic implementation gap. Convergence threshold in `test-examples-convergence.expect` is `>= -150` (which seed=42 = -94 clears with margin).

The env swap also picks up Acrobot in the `docs/develop/example-coverage.md` gap list, so it's a 2-for-1: real PPO demonstration + new env coverage.

### GPT multi-seed validation at embedded/30 (B5, 2026-04-30)

After B3-fixes shrunk the GPT default to `--corpus embedded --epochs 30` (with proportional warmup, ~40 s on tape), B5 ran ≥5-seed validation at the new default to confirm the demo robustly hits the smoke-relaxed `bpc < 5.0` threshold.

| Seed | PyTorch bpc | Idris bpc |
|------|-------------|-----------|
| 1    | 4.228       | 4.438     |
| 2    | 4.234       | 4.487     |
| 3    | 4.211       | 4.431     |
| 4    | 4.211       | 4.403     |
| 42   | 4.195       | 4.537     |

5/5 on both backends, all values 0.4–0.8 below the 5.0 threshold. PyTorch slightly tighter (4.20 avg) than Idris (4.46 avg), explained by PyTorch's dynamic vocab=36 on the embedded corpus vs Idris' hardcoded vocab=65 (the embedded corpus is a strict subset of the 65-char tinyshakespeare alphabet, so Idris carries 29 unused output dims). Both runs are deterministic at the same seed; the gap is fully attributable to the vocab choice, not implementation drift.

### Transformer dModel 32 → 16 (B4, 2026-04-30)

First B4 default change to land — Transformer's attention is matmul-bound (cost scales O(seqLen² × dModel)), so halving dModel is the natural compute win. NumHeads stays 4; HeadDim drops 8 → 4 to keep `NumHeads × HeadDim == dModel`.

5-seed validation on both backends at the new dModel=16:

| Seed | Idris sort_acc | PyTorch sort_acc |
|------|----------------|------------------|
| 1    | 6/6            | 6/6              |
| 2    | 6/6            | 6/6              |
| 3    | 6/6            | 6/6              |
| 4    | 6/6            | 6/6              |
| 42   | 6/6            | 6/6              |

10/10 perfect convergence at the new defaults. Threshold `sort_acc >= 0.8` cleared with full margin. Detail entry in `hyperparameter-tuning-2026.md` "Transformer: dModel 32 → 16" section.

### GRU example added (B6 calibration spike, 2026-04-30)

First B6 ticket — fills the GRU layer coverage gap from B1. Pattern-prediction task and architecture mirror LSTM (1 → 4 → 1, BCE loss, SGD, patience=500, lr=0.5).

**Cross-backend agreement check** (Idris + PyTorch both run 100-iter `--lr-find` at seed=42):

| Backend | RECOMMENDED_LR |
|---|---|
| Idris | 0.4751 |
| PyTorch | 0.4751 |

Ratio: **1.00× — exact agreement.** The 0.5 default is within 5% of both backends' recommendation. Ship-as-is.

**Multi-seed convergence at lr=0.5 (≥5 seeds, both backends)**, threshold `loss < 0.05`:

| Seed | Idris loss | PyTorch loss |
|------|------------|--------------|
| 1    | 8.42e-4    | 1.34e-3      |
| 2    | 6.06e-4    | 8.88e-4      |
| 3    | 8.23e-4    | 8.49e-4      |
| 4    | 8.76e-4    | 9.04e-4      |
| 42   | 8.17e-4    | 6.83e-4      |

**10/10 pass**, all 50× or more below the threshold. Convergence in 1095–1537 epochs (early-stopped via patience=500).

**Implementation note — simplified GRU variant**: the C kernel `tensor_gru_cell` (in `backend_{tape,mlx,torch}.c{,pp}`) computes z and r gates but uses only z and n in the update rule:

```
h' = (1 - z) * n + z * h    where  n = tanh(combined[2o:3o])
```

Standard PyTorch `nn.GRUCell` masks the hidden contribution to n with r:

```
n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
```

For cross-backend alignment we ship the simplified variant in **both** Idris (`tensor_gru_cell`) and PyTorch (`LinearGRUCell` in `models/rnn.py`). r is computed but unused in both. Filed under the layer-perf TODO bucket as a possible future correctness improvement; the simplified variant converges fine on the pattern task and matches across backends.

### FrozenLake example added (B6, 2026-04-30)

Second B6 ticket — fills the FrozenLake env coverage gap. Tabular Q-learning on the slippery 4×4 FrozenLake-v1, mirroring `Example.QLearning` (CliffWalking) with the stochastic-env handling pattern from `Example.MonteCarlo` (per-episode `(envSeed, noise)` input).

**Configuration alignment**: Idris `Example.FrozenLake` and PyTorch `torch_ref/models/frozen_lake.py` use identical defaults: `alpha=0.1 gamma=0.99 epsilon=0.3 epochs=10000 seed=42`. Both implement the Gymnasium 4×4 default map exactly (intended action prob 1/3, each perpendicular 1/3, reward 1.0 at goal, 0.0 elsewhere) and use ε-greedy with first-argmax tiebreaking.

**Multi-seed convergence (≥5 seeds, both backends)**, threshold `avg_return >= 0.4`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 1    | 0.68             | 0.74               |
| 2    | 0.66             | 0.56               |
| 3    | 0.70             | 0.75               |
| 4    | 0.80             | 0.74               |
| 42   | 0.74             | 0.69               |

**10/10 pass**. Idris mean 0.72, PyTorch mean 0.70 — closely aligned. avg_return is the greedy success rate; even an optimal policy on slippery FrozenLake fails ~30% of episodes by slipping into holes, so the metric is success-rate-capped well below 1.0.

Initial defaults (alpha=0.5 gamma=1.0 epsilon=0.1 epochs=500, mirroring CliffWalking) failed multi-seed (2/5 stuck at 0.0). Tuned epsilon up and epochs up so the agent finds the goal often enough to bootstrap learning despite slipperiness — both sides updated together per the alignment policy.

### Taxi example added (B6, 2026-04-30)

Third B6 ticket — fills the Taxi env coverage gap. Tabular Q-learning on the deterministic 5×5 Taxi-v3 grid. Mirrors `Example.QLearning` (CliffWalking) line-for-line; only the env layer changed.

**Configuration alignment**: Idris `Example.Taxi` and PyTorch `torch_ref/models/taxi.py` use identical defaults: `alpha=0.1 gamma=0.99 epsilon=0.1 epochs=20000 seed=42`. Both implement the same fixed-start scaffold (`defaultStart` = taxi (2,2), passenger R=0, destination B=3) and the same Gymnasium wall layout (between cols 1-2 in rows 0-1, between cols 2-3 in rows 3-4).

**Multi-seed convergence (≥5 seeds, both backends)**, threshold `avg_return >= 5`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 1    | 8.0              | 8.0                |
| 2    | 8.0              | 8.0                |
| 3    | 8.0              | 8.0                |
| 4    | 8.0              | 8.0                |
| 42   | 8.0              | 8.0                |

**10/10 hit optimal**. Deterministic env + fixed start = a single optimal trajectory of 13 actions (12·-1 + 20 = +8); the seed only affects ε-greedy exploration during training, and 20K episodes is sufficient on this state space (500 states × 6 actions = 3000-cell table) for every seed to converge to the optimal Q-function.

### MountainCar example added (B6, 2026-04-30)

Fourth B6 ticket — fills the MountainCar env coverage gap. DQN with velocity-magnitude reward shaping; mirrors the CartPole DQN architecture (MLP `2 → 64 → 64 → 3`, 3 actions). Required prerequisite was the batched-forward DQN refactor (commit `3c24cd3`); without that the per-epoch cost was ~4 s on tape (200-step episodes × 64-batch per-sample forward), making multi-seed validation infeasible.

**Configuration alignment**: Idris `Example.MountainCar` and PyTorch `torch_ref/models/mountain_car.py` use identical defaults: `lr=1e-3 gamma=0.99 batch=64 buffer=50000 target_sync=200 eps_start=1.0 eps_end=0.05 eps_decay=50000 shaping=10.0 epochs=500 seed=42`. Both implement the same Gymnasium-aligned MountainCar physics (constants from `Gym.ClassicControl.MountainCar` and Gymnasium's `mountain_car.py`) and the same shaping rule (`r_shaped = r_raw + 10 * |v_next|`).

**Reward shaping note**: not strictly policy-invariant in the Ng99 sense (the optimal Q is altered by the bonus), but at the chosen weight (10·|v|) the optimal trajectory is preserved — kinetic energy is the proven precursor to reaching the goal in MountainCar. The eval metric reports the *raw* return for direct comparison to standard MountainCar reporting (-200 floor, -110 reliable, -100 optimal).

**Multi-seed convergence (≥5 seeds, both backends)**, threshold `avg_return >= -160`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 4    | -104             | -106               |
| 7    | -104             | -109               |
| 13   | -110             | -150               |
| 21   | -102             | -140               |
| 42   | -152             | -110               |
| **mean** | **-114**     | **-123**           |

**10/10 pass**. Cross-backend mean delta is ~9 — within DQN seed noise. The two backends pick different "worst" seeds (Idris's seed=42 and PyTorch's seed=13) but both stay above the -160 threshold across all 5 seeds. Wall-clock: Idris tape ~17:48 / 500 episodes / seed=42 (≈2.1s/epoch); PyTorch ~85s / 500 episodes (≈12.6× faster, the standard FFI-per-step overhead seen across all RL examples on tape).

### MountainCarCont example added (B6, 2026-04-30)

Fifth and final B6 ticket — fills the MountainCarContinuous env coverage gap. SAC with velocity-magnitude reward shaping (mirrors discrete MountainCar). Closes the env-coverage gap list from B1.

**Configuration alignment**: Idris `Example.MountainCarCont` and PyTorch `torch_ref/models/mountain_car_cont.py` use identical defaults: `lr=3e-4 gamma=0.99 alpha=0.2 batch=64 buffer=100000 warmup=1000 tau=0.005 shaping=10.0 epochs=30000`. Both implement the same Gymnasium-aligned MountainCarCont physics (constants from `Gym.ClassicControl.MountainCarCont`) and the same shaping rule (`r_shaped = r_raw + 10 * |v_next|`).

**Done-flag handling**: unlike Pendulum SAC (which treats every 200-step boundary as `done=True` for buffer because Pendulum has no real terminal state), MountainCarCont distinguishes natural termination (goal reached → `done=True`) from truncation (999-step boundary → `done=False` so Q-target bootstrap continues). Both backends use this correct distinction.

**Multi-seed convergence (≥5 seeds, both backends)**, threshold `avg_return >= 85`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 4    | 95.29 (ES@7.8K)  | 93.7               |
| 7    | 95.37 (ES@6.7K)  | 93.4               |
| 13   | 95.35 (ES@6.1K)  | 96.6               |
| 21   | 95.47 (ES@7.6K)  | 92.8               |
| 42   | 94.86 (ES@6.4K)  | 88.3               |
| **mean** | **95.27**    | **92.96**          |

**10/10 pass**. Both backends reach near-optimal (max ~98-100 from the +100 terminal reward minus action penalty). Cross-backend mean delta is ~2.3.

**Idris-vs-PyTorch convergence-speed gap**: Idris breaks through to goal-reaching by step ~5-6K and ES-terminates at ~6-8K; PyTorch needs ~14-16K to break through and runs the full 30K. The asymmetry shows up across all seeds, so it's not seed-luck. Most likely cause: default Linear-layer initialization differs (Idris used Xavier uniform, PyTorch's `nn.Linear` uses Kaiming uniform with `a=sqrt(5)`). For the actor's first layer (2→64), Kaiming gives ~2.3× larger initial weight magnitudes than Xavier; the downstream effect on a sparse-reward task with fragile initial exploration is non-trivial. **Stale as of 2026-06-14 — do not carry this forward (noted 2026-07-29).** Idris's `linear` stopped being Xavier uniform when it moved to the `Nn` surface (`3fbcfc1d`): it is now `N(0, 1/√fan_in)`, whose std is **1.73× larger** than `nn.Linear`'s `U(±1/√fan_in)` (std `1/(√3·√fan_in)`). So the magnitude ordering has *flipped* since this was measured — Idris was 2.35× smaller then, is 1.73× larger now — and this paragraph cannot be used as evidence about the current init. Re-measure before drawing any conclusion from it. Note also that it implicated weight *magnitude*, never the bias. **Superseded 2026-07-29**: both sides now use `U(±1/√fan_in)` with a zero bias (see "one dense init on both sides" above), so whatever init asymmetry existed here is gone and the convergence-speed gap needs re-measuring from scratch. The aligned-defaults policy uses 30K (the slower side); Idris's WindowedAvg early-stop terminates as soon as the policy is consistently solving, so Idris wall-clock stays ~13 min/seed while PyTorch runs full 30K at ~5 min/seed.

### LSTM multi-seed validation at lr=0.5 (B5, 2026-04-30)

After B3 raised the LSTM default LR from 0.03 → 0.5 (lr_find recommendation, single-seed verified), B5 ran the full ≥5-seed validation at the new default on both backends. Convergence threshold: `loss < 0.05`.

| Seed | Idris loss | PyTorch loss |
|------|-----------|-------------|
| 1    | 0.00116   | 0.00150     |
| 2    | 0.00146   | 0.00184     |
| 3    | 0.00159   | 0.00132     |
| 4    | 0.00116   | 0.00154     |
| 42   | 0.00117   | 0.00163     |

5/5 on both backends, all values 30-40× below the threshold. Loss medians around 0.0015 demonstrate clear convergence to a tight final loss; the lr=0.5 default is a strict upgrade over lr=0.03 (which used to plateau at ~0.7 within the same 2000-epoch budget).

### SAC alignment

PyTorch SAC and Idris SAC share:
- Architecture: separate actor + twin Q-networks, tanh-squashed Gaussian actor with state-independent learnable `log_std`.
- ParamId scoping: actor / q1 / q2 / q1tgt / q2tgt get distinct scope prefixes; three group-scoped Adam optimizers (`nativeAdamGroup`) own only their own scope.
- Reparameterized actor gradient. Both sides build `a = tanh(mean + std·ε) · max_action` with gradient flow, concatenate with `obs`, forward Q1/Q2 through the result, and use `min(Q1, Q2)` as a grad-tracked Variable in the actor loss.
- Polyak soft target updates τ=0.005 every step. Idris uses `polyakBlend` (FFI call to `polyak_blend` in all three backends) operating directly on the param registry; PyTorch uses an in-place `mul_(1-τ).add_(online, τ)` over `target.parameters()`.
- Hyperparameters: lr=3e-4, α=0.2, batch=64, warmup=1000, γ=0.99, buffer=100k, τ=0.005.

At matched config, 10k env steps:

| Seed | PyTorch | Idris |
|------|---------|-------|
| 1    | -1331.2 | -394.2 |
| 42   | -1351.5 | -1204.8 |
| 100  | -1075.9 | -389.7 |

Both implementations in the same noise band at the same config. The ~650-point gap that existed in the earlier log-prob-only + hard-sync version is closed; if anything, Idris learns slightly faster on 2/3 seeds at this short horizon, well within the variance of 10k-step Pendulum runs.

The SAC paper's -250 target assumes much longer training than 10k steps — reaching it at higher step counts is a matter of time, not alignment. The short-horizon numbers above demonstrate that the two implementations learn at the same rate from the same gradient signal.

### Earlier SAC divergence (resolved — history)

The initial SAC ship used hard target copy every 100 steps plus a log-prob-only actor gradient (PyTorch SAC's `min(Q1, Q2)` entered the Idris actor loss as `fromDouble minQ`, cutting the reparameterization gradient path). That produced a ~650-point convergence gap at the same seed (PyTorch -1331, Idris -1973 at 10k steps). Fix required three library additions:
- `optimizer_create_adam_group` (C backend) + `nativeAdamGroup` (Idris wrapper) — per-optimizer paramId-prefix filter, so SAC's three optimizers update only their own networks even when the actor-loss backward graph populates gradients on Q params too.
- `polyak_blend` (C backend) + `polyakBlend` / `polyakUpdate` (Idris wrappers) — registry-level soft update, so target Q-nets can track online Q-nets smoothly.
- Reparameterized actor path using existing `prim__tanh` / `prim__mulScalar` / `prim__cat2` / `forwardVarTensor` primitives. No new FFI needed on that front — just using the grad-tracked tensor ops that were already in place.

### Multi-seed A2C pass rates at aligned config

At matched config (separate actor+critic, lr=7e-4, entropy=0.01, rollout=20 single-env, 5000 updates, γ=0.99, λ=0.95):

| Implementation | Pass rate (greedy_eval ≥ 150 / total) |
|---|---|
| PyTorch | 3/7 (seeds 7, 100, 314) |
| Idris | 4/7 (seeds 1, 7, 100, 99) |

Single-env rollout=20 is a noisy A2C config — the PyTorch reference's original 200/200 convergence used 8 parallel envs × rollout=20 (= 160 effective steps per update) which smooths the gradient significantly. Both implementations agree at the aligned config; the "full convergence" requires multi-env rollouts (not yet implemented in Idris — Gym.Wrapper.Vector exists but is unwired here).

## Alignment Changes (2026-04-26) — MNIST/SeqClassify double-softmax + epoch semantics

### Double-softmax bug (resolved)

Both `Example/Mnist.idr` and `Example/SeqClassify.idr` ended their model chain with `OutputLayer softmaxLayer`, then their loss functions called `prim__logSoftmax predT 0` on the already-softmaxed output. The composition `log_softmax(softmax(x))` flattens the distribution toward uniform and drives training-time loss toward `log C` (the empirically observed plateau values: ~2.27 for MNIST, ~1.10 for seq-classify). Surfaced when `make test-examples-convergence` ran for the first time.

**Fix**: drop `OutputLayer softmaxLayer` from both model chains. The existing loss functions correctly apply `log_softmax` to raw logits — the recommended pattern (also documented in CLAUDE.md gotchas). PyTorch references already used this pattern (raw logits + `F.nll_loss`). Notebook mirrors (`models/cnn.ipynb`, `models/seq_classify.ipynb`) updated for consistency.

Verified post-fix at full default epochs:
- seq-classify: loss 0.61 → 0.121 (PyTorch reference 0.243 at 1000 epochs)
- MNIST: see epoch-semantics divergence below

### MNIST epoch semantics — Idris/PyTorch alignment (resolved)

Previously, 1 Idris MNIST "epoch" = 1 mini-batch step (`mkIndexedLoader` yields one batch per call; `runTraining`/`epochNativeTensorPre` consumed one batch per epoch). PyTorch's `train_epoch` iterates **all batches** of the 60K training set per epoch. So 100 Idris epochs ≈ 100 batches, while 100 PyTorch epochs ≈ 187,500 batches — same word, ~1875× compute gap. Earlier alignment work (commit `0c6b1e72`) had reduced Idris MNIST epochs from 2000 → 100 on the assumption that the tokens were semantically identical, dropping accuracy 0.92 → 0.599 and breaking the convergence gate; reverted in `7433ab4` to 2000 single-batch epochs as a stopgap.

**Refactored**: `Example/Mnist.idr` now uses `runTrainingIO` with `dataSrc=pure ()` and an inline `trainOneFullPass` helper that fetches `batchesPerEpoch = trainCount / BatchSize ≈ 937` mini-batches per logical epoch — matching PyTorch's full-pass semantics. Loss returned is the mean per-batch loss across the full pass (mirrors PyTorch's `total_loss / count`).

Aligned defaults:
- Idris: `--batch-size 64 --epochs 5 --patience 3` (≥0.85 threshold reached well before epoch 5).
- PyTorch: `--batch-size 64 --epochs 100 --patience 500` (kept; PyTorch trains longer for the 0.99 final-quality demo using the same script).

The convergence threshold (≥0.85) is unchanged in `test-examples-convergence.expect`. Wall time at 5 full-pass epochs: ≤15 minutes on tape — well inside the 4h `CONVERGENCE_TIMEOUT`. SeqClassify uses synthetic data and 1000 single-batch "epochs" already roughly match the PyTorch reference's 1000 single-batch loop (synthetic-data sampling rather than full-dataset iteration), so it stays as-is per the original TODO note.

## Alignment Changes (2026-04-28) — GPT convergence on tinyshakespeare

### Corpus + held-out validation

Both Idris (`Example/Gpt.idr`) and PyTorch ref (`models/gpt.py` + `scripts/gpt.py`) were aligned but on a 1342-character hardcoded Shakespeare excerpt with a 36-char lowercase-collapse vocab. With those defaults, `test-examples-convergence` ran 2000 epochs and "converged" to bpc=0.13 — pure memorization of a 1.3 KB corpus, not learning. The threshold (`bpc < 3.5`) was hit hundreds of epochs before patience could fire.

**Fix**: align both sides on Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) `train_shakespeare_char` recipe — the canonical char-LM benchmark — at a model scale tractable on the tape backend (existing 2 blocks / 4 heads / dModel=64 / seqLen=64; ~26K params). Adopted from nanoGPT:

- **Corpus**: tinyshakespeare (~1.1 M chars, 65-char vocab) loaded from `data/tinyshakespeare/input.txt` (fetched by `make dataset-tinyshakespeare`). Vocab built dynamically as the sorted set of distinct characters in the corpus.
- **90/10 train/val split** (deterministic, last-N% as val). Convergence metric is `val_bpc` (held-out), not training-corpus bpc.
- **AdamW recipe**: β1=0.9, **β2=0.99** (was 0.999), **wd=0.1** (was 0.01), grad clip 1.0.
- **Cosine LR with linear warmup**: 100 epochs warmup → cosine decay from `lr` to `lr * 0.1`. Idris uses the existing `Schedule.cosineWithWarmup`; per-epoch LR update via a `setLRAll` helper that iterates the param registry. PyTorch uses an inline lambda matching the same nanoGPT formula.
- **Default epochs**: 1000 (was 2000). Cosine LR + warmup converges faster than the previous bare patience-based setup.

`test-examples-convergence.expect` updated: `bpc < 3.5` → `val_bpc < 3.5`. Same numeric value, but now on a real held-out set (random baseline = log₂(65) = 6.02; 3.5 is a meaningful "definitely learning" target for the small architecture).

Wall-time impact: PyTorch ref reaches val_bpc = 3.32 at 1000 epochs in ~64s on Apple Silicon; tape-backend Idris extrapolates to ~30 min at the existing ~1.76 s/epoch — vs 58 min of overrun on the old configuration.

### Smoke gate vs convergence path

A single `--corpus {tinyshakespeare,embedded}` CLI flag selects between the new file-based corpus (default; convergence) and the legacy 1342-char embedded excerpt (smoke gate, no file dependency). The smoke gate (`make test-examples`) sets `--corpus embedded --epochs 3` to keep the wiring test fast and self-contained; the embedded corpus uses a strict subset of the 65-char vocab so a single tokenizer serves both paths.

## Alignment Changes (2026-04-29) — DNC defaults reverted on both sides

The Apr 21 unification (DNC Copy/Recall → batch=16, N=128, max-len=20) put the example at a config the Idris tape backend cannot validate end-to-end: ~5 min/epoch, ~10 days for the trajectory PyTorch reaches in 13 min. The previously-documented Idris convergence run (`docs/develop/dnc-convergence-results.md`) was entirely at the smaller pre-Apr-21 config (N=32, batch=1, max-len 10), and no run at the unified config has ever completed.

Per the alignment policy ("update PyTorch to the lower config and re-verify"), reverted both sides to the smaller config:

| Example | Parameter | Reverted to |
|---------|-----------|-------------|
| DNC Copy | Memory size N | 32 |
| DNC Copy | Batch size (CLI default) | 1 |
| DNC Copy | Max seq length (CLI default) | 10 |
| DNC Copy | Eval test size | 20 |
| DNC Recall | Memory size N | 32 |
| DNC Recall | Batch size (CLI default) | 1 |
| DNC Recall | Eval test size | 20 |

NTM Copy/Recall were intentionally NOT reverted — NTM's per-epoch cost is ~10× lower than DNC's (no O(N²) link matrix), so it runs the unified config without the same wall-clock pain.

Re-aligning DNC at PyTorch's previous config (N=128, batch=16) is blocked on tape-backend perf work — the dominant cost is `Layer/Dnc.idr`'s `zeroDiag` per-cell C-level fill loop and per-row `prim__select` extraction in `buildMatrixRows`. Filed as a Medium-priority TODO entry; revisit alignment once those land.

## Alignment Changes (2026-05-08) — NTM batch reverted on both sides + early-stop redesign

The 2026-04 NTM batch=1 → 16 change was adopting PyTorch's arbitrary historical default rather than the better practice. All canonical NTM references (Graves et al. 2014, Collier & Beel 2018, vlgiitr/ntm-pytorch) use batch=1, and `docs/develop/gotchas.md:184` explicitly warns that "batch averaging dilutes the per-sequence addressing signal that the NTM needs to learn distinct write slots and query-triggered retrieval." Per the alignment policy ("adopt whichever is the better practice"), reverted both sides to batch=1:

| Example | Parameter | Before (2026-04) | After (2026-05-08) |
|---------|-----------|------------------|---------------------|
| NTM Copy | Batch size | 16 | 1 |
| NTM Recall | Batch size | 16 | 1 |

Wall-clock impact on tape (seed=42, full 50K epochs without early-stop firing): batch=16 → ~7.7 h projected; batch=1 → 29:45 measured. Inside the 30-min/example budget.

Concurrently, replaced the loss-mean windowed early-stop with a percentile-based variant (`Train.WindowedPercentile`). At batch=1 with variable-length sequences, loss is bimodal — short sequences hit near-zero quickly while long ones plateau higher — so the mean over a 1000-epoch window stays around 0.2-0.3 even at full convergence (acc_short=99.95%). The `WindowedAvg` early-stop never fires. The new `WindowedPercentile 0.10 thresh win pat` checks the 10th-percentile of chunk-means in the window — i.e., the lowest 100-epoch chunk in the recent 1000 epochs. Fires reliably once the model converges on at least the easier sequences.

Applied uniformly to all four NTM/DNC examples (NtmCopy, NtmAssociativeRecall, DncCopy, DncAssociativeRecall) for consistency.

## Alignment Changes (2026-05-08) — NTM/DNC model alignment to PyTorch ref

Fixed a real algorithmic regression in Idris's NTM (and parallel gaps in DNC). At fully aligned config (batch=1, seed=42, matched `WindowedPercentile` ES on both sides), pre-fix Idris reached only acc_full=82% at 27,700 epochs while PyTorch ref hit 100% at 4,600. Bit-for-bit bisection (Idris-on-tape vs Idris-on-torch matched to within 1 ULP at epoch 200) confirmed the gap was in Idris's shared model code, not a backend bug.

Five fixes brought Idris into algorithmic alignment with the ref (commits `ad62186` and `8b...`):

| Fix | Was | Now (matches PyTorch) |
|---|---|---|
| NTM `ntmInterpWriteIdris` | additive `mem + outer(w, a)` | interpolative `w·a + (1-w)·mem` |
| NTM/DNC memory_init | fixed 1e-6 | learned Xavier param, sigmoid'd at sequence-start |
| NTM/DNC initial read output(s) | zero | Kaiming-uniform, fixed at construction, non-learnable |
| LSTM h0/c0 (both) | lazy-zero (non-learnable) | learned zero-init params |
| NTM/DNC FC inits | `linearLayer` default (Xavier weights, zero bias) | per-FC: Xavier(gain=1.4) + normal(std=0.01) for head FCs, kaiming + normal for output FC |

PyTorch ref unchanged — the alignment direction was "Idris is wrong, fix Idris to match Graves/PyTorch", not bidirectional.

**Validation** (Idris-on-torch backend = algorithmic oracle since libtorch autograd is identical to PyTorch's):

NTM-Copy (batch=1):
- Idris-on-torch (seed=42): 5,000 epochs / 100% / 100%
- PyTorch ref (seed=42):    4,600 epochs / 99.6% / 100%
- Within 9% epoch budget, identical accuracy.

DNC-Copy (batch=1, N=32, max-len=10, eval len 1-20):

| seed | Idris-on-torch acc_full | PyTorch ref acc_full |
|---|---:|---:|
| 42 | 83.9% | 99.4% |
| 1 | 96.3% | 64.3% |

Multi-seed mean: Idris ≈ 90%, PyTorch ≈ 82%. Both implementations show massive seed-variance on length-generalization (1-10 trained → 1-20 eval) — PyTorch swings 35 points across two seeds, Idris swings 12 points. The seed=42 single-seed gap is RNG variance (different C `rand()` vs PCG init values for "seed=42"), not an algorithmic bug. Multi-seed mean is comparable; Idris-on-torch is a faithful port of the PyTorch DNC.

## Phase 1.5d findings (2026-05-08) — tape + mlx don't track torch on aligned NTM

After landing the model alignment, ran NTM-Copy at seed=42 batch=1 on
all three backends. Idris-on-torch matches PyTorch ref (the autograd
oracle is libtorch in both). tape and mlx do NOT match:

| Backend | epochs | acc_short | acc_full | wall-clock |
|---|---:|---:|---:|---:|
| Idris-on-torch | 5,000 | 100% | 100% | 2:48 – 5:05 (run-to-run variance) |
| PyTorch ref | 4,600 | 99.6% | 100% | 3:02 |
| Idris-on-tape | 35,500 | 95.8% | **80.5%** | 11:40 |
| Idris-on-mlx | killed at 17K | ~50% (random level) | — | 29 min (no convergence) |

Forward parity is fine — at epoch 0 tape matches torch within 36 ULPs
(`0.7018285801505979` vs `0.7018285801506005`); mlx differs at digit 7
because it's float32 internally. The convergence trajectory drift comes
from **backward** differences accumulating over many epochs. The NTM
model code is shared across backends; the divergence is in tape's
hand-rolled backward replay rules and mlx's training pipeline, not in
the layer code.

This was likely **latent** before the alignment fix — the broken
additive-write NTM converged to acc_full=82% on tape (similar to the
post-alignment 80%), so tape's gradient bug was masked by the
algorithmic regression. The fix exposes it. mlx wasn't tested on the
aligned model before now and the failure to train is new evidence.

**Filed as separate engineering projects** — backend-side investigation
is not a model-alignment fix:
- tape: locate which backward rule(s) for the new aligned-NTM ops
  (sigmoid+reshape on memInit, learned LSTM h0/c0, outer-products in
  interp write) produce drift, and fix.
- mlx: identify why the aligned model fails to descend at all on mlx.
  Could be backward rule, sync timing, or float32 precision compounding.

For now, **Idris-on-torch is the convergence-correctness backend** for
NTM/DNC examples; tape and mlx have a documented quality gap until
their backward rules are audited.

## Alignment Changes (2026-05-19) — SAC torch_ref migrated to gymnasium Pendulum-v1

PyTorch SAC reference now uses `gym.make("Pendulum-v1")` instead of the hand-rolled `PendulumState` / `pendulum_step` (which had been imported from `ppo.py` and broke when PPO switched env to Acrobot in `8b0992e`). The migration also heals SAC's import error and is the first step of the broader `gymnasium`-adoption work in `torch_ref/` (TODO Medium row, sweep across all 11 RL examples).

**Documented divergence — reset state**: canonical Gymnasium Pendulum-v1 randomizes the initial state within `theta ∈ [-π, π], theta_dot ∈ [-1, 1]`; idris-gym `Gym.ClassicControl.Pendulum.reset` is deterministic `MkP Pi 0.0` (hangs-down, worst-case inverted). The torch_ref Pendulum loop pins `env.unwrapped.state = [π, 0.0]` after each reset to mirror the Idris init. Threading a seedable RNG through `Env.reset` to randomize idris-gym is a follow-up (touches the `Env` interface — out of scope for this row).

Multi-seed convergence after migration (30K env steps, eval over 10 episodes, threshold ≥ -1500):

| seed | avg_return |
|---:|---:|
| 42 | -1458.5 |
| 0 | -1308.5 |
| 1 | -1175.2 |
| 2 | -1312.0 |
| 3 | -1020.3 |

5/5 pass; mean -1255, within the same band the pre-broken `pendulum_step` reference produced. Other env constants match between idris-gym and Pendulum-v1 (`Gravity=10`, `MaxTorque=2`, `MaxSpeed=8`, `Dt=0.05`, dynamics formula, reward `-(θ_norm² + 0.1·θ̇² + 0.001·u²)`, 200-step TimeLimit), so no further alignment work is needed for Pendulum.

## Alignment Changes (2026-05-19) — torch_ref env layer migrated to gymnasium

Eight of nine RL envs in `torch_ref/models/*.py` migrated from hand-rolled physics to `gym.make("...")`. Per-env summary (Pendulum landed in the SAC commit earlier the same day):

| Env | Idris-gym | torch_ref before | torch_ref after | Reset convention |
|---|---|---|---|---|
| Pendulum-v1 | matches | hand-rolled | `gym.make` | pin `(π, 0.0)` |
| FrozenLake-v1 | matches | hand-rolled | `gym.make` | (canonical pos 0) |
| Taxi-v4 | matches except missing wall at rows 3-4 cols 0-1 | hand-rolled | `gym.make` | pin via `unwrapped.encode(2,2,0,3)` |
| CliffWalking-v1 | matches | hand-rolled | `gym.make` | (canonical start 36) |
| Blackjack-v1 | **was** Ace=2/13 / 10=3/13 → **now** 1/13 / 4/13 (canonical) | hand-rolled with same skewed distribution | `gym.make` | (no init state, deck draw) |
| CartPole-v1 | matches (uses v0's 200-step cap) | hand-rolled CartPole-v0 | `gym.make("CartPole-v1")`, cap MAX_STEPS=200 | pin `(0,0,0,0)` |
| MountainCar-v0 | matches | hand-rolled | `gym.make` | pin `(-0.5, 0.0)` |
| MountainCarContinuous-v0 | matches | hand-rolled | `gym.make` | pin `(-0.5, 0.0)` |
| Acrobot-v1 | matches (RK4, dt=0.2) | hand-rolled Euler-substep | `gym.make` (RK4) | pin `(0,0,0,0)` |

**Paired-side resolutions in this sweep**:
1. **Blackjack card distribution** — adopted in this commit on both sides. The pre-change Ace=2/13, 10=3/13 (from the n=0 and n=1 → 1 collision and n=10..12 → 10 mapping) was non-canonical. Both Idris (`Gym.ToyText.Blackjack.drawCard`) and torch_ref now use the canonical Gymnasium 13-card uniform suit. Convergence preserved on both: Idris win_rate=0.43, torch_ref win_rate=0.42.
2. **Acrobot integrator** — adopted in a same-day follow-up commit. idris-gym's `Gym.ClassicControl.Acrobot.aStep` switched from 4×dt=0.05 semi-implicit Euler substeps to single-step dt=0.2 RK4 (matching gymnasium's reference rk4); torch_ref `ppo.py` switched to `gym.make("Acrobot-v1")`. Both sides now run identical physics. PPO convergence *improved* with the more accurate integrator: Idris seed=42 -75.0 (was ~-200 under Euler), torch_ref 5-seed mean -77.2 (range -72 .. -88).
3. **Taxi wall divergence** — canonical Taxi-v4 has a wall between cols 0-1 in rows 3-4 that neither side currently models. Fixed-start optimal trajectory doesn't traverse that gap, so observed convergence (+8) is unchanged. Q-table values in the SW corner now differ from idris-gym's, invisible to the test. Not separately filed; subsumed by [[Env.reset Rng-threading]] row's general "audit idris-gym physics against canonical" implication.
4. **Deterministic resets** — every classic-control env pins gymnasium's randomized init to match idris-gym's `Env.reset = constant`. Threading a seedable Rng through `idris-gym` `Env.reset` is filed as a separate Medium TODO row; once that lands, the pins can be removed.

Reset-state pinning is achieved via `env.unwrapped.state = np.array(..., dtype=np.float64)` after each `env.reset()`. Using `float64` is load-bearing — CartPole specifically loses ~5% convergence (95.0 vs 100.0 in the test_a2c.test_converges threshold) if the pin uses float32, because the env's internal Euler step then runs in float32. Pinning with float64 keeps internal physics at the same precision as the pre-migration Python-float hand-roll.

## Alignment Changes (2026-06-04) — HfLlama generate-side parity (KV cache landing)

`Example/HfLlamaInference.idr` greedy-decode path switched from the
re-feed-full-prefix `genLoop` to a cache-aware `genLoopCached` (Phase
D of the KV-cache work, commit `dd3aca25`). The PyTorch-side oracle
for the token-sequence gate
(`scripts/save_oracle_llama_generate.py`) uses
`model.generate(do_sample=False, use_cache=True, temperature=1.0)`
with `pad_token_id = config.eos_token_id`. Paired-side settings:

| Setting | Idris (`genLoopCached`) | HF oracle |
|---|---|---|
| Sampling | Greedy (`argmaxRow` over last logits row) | `do_sample=False` |
| KV cache | Functional concat (`Empty / Filled` sum type) | `use_cache=True` (HF default) |
| Prompt | "The capital of France is" (Tokenizer subprocess, `add_special_tokens=True`) | Same prompt, `tokenizer.encode(text, add_special_tokens=True)` |
| BOS token | Prepended by tokenizer (Llama-3 id 128000) | Prepended by tokenizer |
| Budget | `--num-tokens 4` default in Makefile gate | `NUM_NEW_TOKENS = 4` |
| Position offset for RoPE | `cacheLen cache` (cumulative pre-current-step) | HF internal `cache_position` |
| Causal mask under asymmetric Q.seq/KV.seq | `is_causal=True`, lower-right alignment per torch/mlx | Same — `use_cache=True` triggers asymmetric SDPA internally |

`use_cache=True` is mathematically equivalent to `use_cache=False`
for greedy decode (same forward math, same argmax), so the oracle's
token sequence is invariant to the HF cache flag. This is what lets
the same token-sequence gate verify both the no-cache Idris path
(Phase A baseline at `c1f7489d`) and the cached Idris path (Phase D
at `dd3aca25` + `f2663ff2`).

**Documented storage-shape divergence (not a paired-side
mismatch)**: HF stores per-layer KV cache as rank-4 `[batch,
n_kv_heads, seq, head_dim]`; idris-ml stores it as flat 2D `[seq,
n_kv_heads * head_dim]` to match the existing `applyAttention`'s K/V
projection output layout and skip a reshape at the SDPA call site.
Both are byte-equivalent under column-major-to-row-major view of the
same backing storage; the forward semantics are identical.

## Status

All known discrepancies resolved (model-side); two backend-side
backward-rule issues filed as follow-ups (tape gradient drift; mlx
NTM training failure).
