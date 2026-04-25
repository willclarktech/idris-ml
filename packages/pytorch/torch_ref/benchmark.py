"""Timing benchmark.

- Supervised: 100 warmup + 1000 timed epochs, SGD lr=0.03
- RNN: 100 warmup + 1000 timed epochs, SGD lr=0.03
- NTM: 10 warmup + 100 timed epochs, RMSprop lr=1e-4 alpha=0.95 value clip 10.0
- NTM-copy: 10 warmup + 100 timed epochs, same optimizer, production scale
- NTM-copy-1k: 10 warmup + 1000 timed epochs, fresh data each epoch, momentum=0.9
- NTM-recall: 10 warmup + 100 timed epochs, recall task, batch=16, momentum=0.9
"""

import platform
import random
import resource
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch_ref.data.copy_task import generate_copy_batch
from torch_ref.data.recall_task import generate_recall_batch
from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    generate_reversal_data,
    train_reversal_epoch,
)
from torch_ref.models.ntm import NtmConfig, NtmModel
from torch_ref.models.rnn import LinearRNNCell, generate_rnn_dataset, train_rnn_epoch
from torch_ref.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch


def _peak_rss_mb() -> float:
    """Return peak RSS in MB (ru_maxrss from getrusage)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _run_benchmark(
    run_epoch: Callable[[], float],
    warmup: int,
    epochs: int,
) -> tuple[float, float, float]:
    """Generic benchmark runner: warmup + timed loop + RSS.

    Args:
        run_epoch: Callable that runs one epoch and returns loss.
        warmup: Number of warmup epochs.
        epochs: Number of timed epochs.

    Returns:
        (elapsed_ms, final_loss, peak_rss_mb)
    """
    for _ in range(warmup):
        run_epoch()

    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(epochs):
        loss_val = run_epoch()
    t1 = time.monotonic()

    return (t1 - t0) * 1000, loss_val, _peak_rss_mb()


def _train_ntm_epoch(
    model: NtmModel,
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    clip_value: float,
) -> float:
    """Train one NTM epoch: forward all sequences in batch, single backward.

    Matches Idris epochTwoPhaseDense: accumulate loss over batch, one backward pass.
    Two-phase: encoding (feed inputs, discard outputs) then output (feed zeros, compute loss).
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)

    for input_seq, target_seq in batch:
        model.reset_state()
        seq_len = target_seq.shape[0]
        input_width = input_seq.shape[1]

        # Encoding phase: feed input sequence, discard outputs
        for t in range(input_seq.shape[0]):
            model(input_seq[t])

        # Output phase: feed zeros, compute loss on targets
        zero_input = torch.zeros(input_width)
        outputs = []
        for _ in range(seq_len):
            out = model(zero_input)
            outputs.append(out)

        pred = torch.stack(outputs)  # (seq_len, output_width) — raw logits
        loss = F.binary_cross_entropy_with_logits(pred, target_seq)
        total_loss = total_loss + loss

    avg_loss = total_loss / len(batch)
    avg_loss.backward()
    clip_grad_value_(model.parameters(), clip_value)
    optimizer.step()
    return avg_loss.item()


def bench_supervised() -> tuple[float, float, float]:
    """Benchmark supervised model. Returns (elapsed_ms, final_loss, peak_rss_mb)."""
    torch.manual_seed(123456)
    model = SupervisedModel()
    data = SUPERVISED_DATA
    lr = 0.03

    def run_epoch() -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return train_supervised_epoch(model, data, optimizer)

    return _run_benchmark(run_epoch, warmup=100, epochs=1000)


def bench_rnn() -> tuple[float, float, float]:
    """Benchmark RNN model. Returns (elapsed_ms, final_loss, peak_rss_mb)."""
    torch.manual_seed(123456)
    model = LinearRNNCell(1, 1)
    data = generate_rnn_dataset(8)
    lr = 0.03

    def run_epoch() -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return train_rnn_epoch(model, data, optimizer)

    return _run_benchmark(run_epoch, warmup=100, epochs=1000)


def bench_ntm() -> tuple[float, float, float]:
    """Benchmark small NTM. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Small NTM (w=3, n=10, m=5, h=20, batch=5) matching Idris benchNtm.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 3, 10, 5, 20
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1, output_width=w, n=n, m=m, controller_size=h, lr=lr, clip_value=clip_value
    )
    model = NtmModel(cfg)
    batch = generate_copy_batch(5, seq_min=2, seq_max=4, seq_width=w)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    def run_epoch() -> float:
        return _train_ntm_epoch(model, batch, optimizer, clip_value)

    return _run_benchmark(run_epoch, warmup=10, epochs=100)


def bench_ntm_copy() -> tuple[float, float, float]:
    """Benchmark production-scale NTM copy. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Production NTM (w=8, n=128, m=20, h=100, batch=16) matching Idris NtmCopy.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 8, 128, 20, 100
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1, output_width=w, n=n, m=m, controller_size=h, lr=lr, clip_value=clip_value
    )
    model = NtmModel(cfg)
    batch = generate_copy_batch(16, seq_min=1, seq_max=20, seq_width=w)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    def run_epoch() -> float:
        return _train_ntm_epoch(model, batch, optimizer, clip_value)

    return _run_benchmark(run_epoch, warmup=10, epochs=100)


def bench_ntm_copy_1k() -> tuple[float, float, float]:
    """Benchmark 1000-epoch NTM copy with fresh data. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Realistic benchmark: fresh batch each epoch, momentum=0.9, matching Idris benchNtmCopy1k.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 8, 128, 20, 100
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1, output_width=w, n=n, m=m, controller_size=h, lr=lr, clip_value=clip_value
    )
    model = NtmModel(cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    def run_epoch() -> float:
        batch = generate_copy_batch(16, seq_min=1, seq_max=20, seq_width=w)
        return _train_ntm_epoch(model, batch, optimizer, clip_value)

    return _run_benchmark(run_epoch, warmup=10, epochs=1000)


def bench_ntm_recall() -> tuple[float, float, float]:
    """Benchmark production-scale NTM recall. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Production NTM recall (w=6, n=128, m=20, h=100, batch=16) matching Idris NtmAssociativeRecall.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 6, 128, 20, 100
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 2, output_width=w, n=n, m=m, controller_size=h, lr=lr, clip_value=clip_value
    )
    model = NtmModel(cfg)
    batch = generate_recall_batch(16, 2, 6, 3, w)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    def run_epoch() -> float:
        return _train_ntm_epoch(model, batch, optimizer, clip_value)

    return _run_benchmark(run_epoch, warmup=10, epochs=100)


def bench_multi_head_transformer() -> tuple[float, float, float]:
    """Benchmark multi-head transformer on reversal task.

    Fresh data each epoch, loss only on reversal portion (matching Idris).
    Returns (elapsed_ms, final_loss, peak_rss_mb).
    """
    torch.manual_seed(123456)
    vocab_size, input_len, seq_len = 10, 5, 11
    sep_token, eos_token = 8, 9
    d_model, num_heads = 32, 4

    model = MultiHeadTransformer(vocab_size, seq_len, d_model, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def run_epoch() -> float:
        data = generate_reversal_data(16, input_len, vocab_size, sep_token, eos_token)
        return train_reversal_epoch(model, data, optimizer, reversal_start=input_len)

    return _run_benchmark(run_epoch, warmup=100, epochs=500)


BENCHMARKS: dict[str, tuple[str, Callable[[], tuple[float, float, float]]]] = {
    "supervised": ("Supervised (1000 epochs)", bench_supervised),
    "rnn": ("RNN (1000 epochs)", bench_rnn),
    "ntm": ("NTM (100 epochs)", bench_ntm),
    "ntm-copy": ("NTM-copy (100 epochs)", bench_ntm_copy),
    "ntm-copy-1k": ("NTM-copy-1k (1000 epochs)", bench_ntm_copy_1k),
    "ntm-recall": ("NTM-recall (100 epochs)", bench_ntm_recall),
    "transformer": ("Transformer (1000 epochs)", bench_multi_head_transformer),
}


def main() -> None:
    import sys

    requested = sys.argv[1:] or list(BENCHMARKS.keys())
    unknown = [r for r in requested if r not in BENCHMARKS]
    if unknown:
        print(f"Unknown benchmarks: {', '.join(unknown)}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    print("PyTorch Benchmark")
    print("=" * 50)

    for name in requested:
        label, fn = BENCHMARKS[name]
        elapsed, loss, rss = fn()
        print(f"{label + ':':<30s} {elapsed:.1f} ms")
        print(f"  Final loss: {loss:.6f}")
        print(f"  Peak RSS: {rss:.0f} MB")


if __name__ == "__main__":
    main()
