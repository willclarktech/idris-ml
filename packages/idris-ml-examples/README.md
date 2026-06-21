# idris-ml-examples

Runnable example programs demonstrating [idris-ml](../idris-ml/) across supervised learning,
sequence modelling, transformers, and reinforcement learning — plus the end-to-end benchmark
harness. Depends on `idris-ml`, `idris-gym`, and `idris-transformers`.

Each example is a self-contained `Example/*.idr` module run via `make example-<name>`. Targets
type-check, link, and execute the program against the active backend. Every example accepts
`--epochs`, `--lr`, `--seed` and task-specific flags. Examples don't hardcode device or dtype —
they reference the build's `(ExampleDevice, ExampleDType)` cell, so the same source runs on tape,
torch, or mlx by choosing the backend at `make install` time.

## Supervised & vision

| Example | Description | Command |
| --- | --- | --- |
| Supervised | 3-class classification with softmax | `make example-supervised` |
| MNIST | CNN digit classification (Conv2D + MaxPool2D) | `make example-mnist` |
| SeqClassify | 1D waveform classification (Conv1D + MaxPool1D) | `make example-seq-classify` |

## Recurrent & memory-augmented

| Example | Description | Command |
| --- | --- | --- |
| RNN / LSTM / GRU | Sequence prediction on a repeating pattern | `make example-rnn` · `example-lstm` · `example-gru` |
| NTM Copy / Recall | Neural Turing Machine copy + associative recall | `make example-ntm-copy` · `example-ntm-associative-recall` |
| DNC Copy / Recall | Differentiable Neural Computer copy + recall | `make example-dnc-copy` · `example-dnc-recall` |

## Transformers & language models

| Example | Description | Command |
| --- | --- | --- |
| Transformer | Autoregressive next-token prediction (causal self-attention) | `make example-transformer` |
| GPT | Character-level LM on Shakespeare | `make example-gpt` |
| GPT-2 / Llama / BitNet inference | Load real HF checkpoints, run forward | `make example-gpt2-inference` · `example-llama-inference` · `example-bitnet-inference` |
| BERT inference / fine-tune / LoRA | HF BERT forward + classification fine-tuning + LoRA | `make example-bert-inference` · `example-bert-classify-finetune` · `example-bert-classify-sst2-lora` |

## Reinforcement learning (on [idris-gym](../idris-gym/))

| Example | Description | Command |
| --- | --- | --- |
| REINFORCE / A2C / PPO | Policy-gradient & actor-critic on CartPole | `make example-reinforce` · `example-a2c` · `example-ppo` |
| DQN / Double DQN / SAC | Value-based & off-policy control | `make example-dqn` · `example-double-dqn` · `example-sac` |
| Q-Learning / SARSA / Monte Carlo | Tabular methods | `make example-q-learning` · `example-sarsa` · `example-monte-carlo` |
| FrozenLake / Taxi / MountainCar | Classic environments | `make example-frozen-lake` · `example-taxi` · `example-mountain-car` |

## Demos & benchmarks

Smaller programs exercise specific features: `example-transfer` (multi-backend tensor transfer),
`example-bring-your-own` (declare a custom backend), `example-dtype-pitch` / `example-precision-demo`
(lossless-cast typing), `example-checkpoint` (save/resume), `example-tcast-demo` (explicit casts).

`Bench.idr` (`make example-bench`) is the end-to-end training microbenchmark timing
forward/backward/step across model families; `make bench-compare` runs it side by side against the
PyTorch reference. See the [performance regime in CLAUDE.md](../../CLAUDE.md) and
[docs/benchmarks.md](../../docs/benchmarks.md).
