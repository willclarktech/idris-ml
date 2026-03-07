"""Timing benchmark.

- Supervised: 100 warmup + 1000 timed epochs, SGD lr=0.03
- RNN: 100 warmup + 1000 timed epochs, SGD lr=0.03
- NTM: 10 warmup + 100 timed epochs, RMSprop lr=1e-4 alpha=0.95 value clip 10.0
- NTM-copy: 10 warmup + 100 timed epochs, same optimizer, production scale
- NTM-copy-1k: 10 warmup + 1000 timed epochs, fresh data each epoch, momentum=0.9
"""

import platform
import random
import resource
import time

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from torch_ref.data.copy_task import generate_copy_batch
from torch_ref.models.ntm import NtmConfig, NtmModel
from torch_ref.models.rnn import LinearRNNCell, generate_rnn_dataset, train_rnn_epoch
from torch_ref.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch


def _peak_rss_mb() -> float:
    """Return peak RSS in MB (ru_maxrss from getrusage)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def bench_supervised() -> tuple[float, float, float]:
    """Benchmark supervised model. Returns (elapsed_ms, final_loss, peak_rss_mb)."""
    torch.manual_seed(123456)
    model = SupervisedModel()
    data = SUPERVISED_DATA
    lr = 0.03

    # Warmup: 100 epochs
    for _ in range(100):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_supervised_epoch(model, data, optimizer)

    # Benchmark: 1000 epochs
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(1000):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_val = train_supervised_epoch(model, data, optimizer)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val, _peak_rss_mb()


def bench_rnn() -> tuple[float, float, float]:
    """Benchmark RNN model. Returns (elapsed_ms, final_loss, peak_rss_mb)."""
    torch.manual_seed(123456)
    model = LinearRNNCell(1, 1)
    data = generate_rnn_dataset(8)
    lr = 0.03

    # Warmup: 100 epochs
    for _ in range(100):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_rnn_epoch(model, data, optimizer)

    # Benchmark: 1000 epochs
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(1000):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_val = train_rnn_epoch(model, data, optimizer)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val, _peak_rss_mb()


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


def bench_ntm() -> tuple[float, float, float]:
    """Benchmark small NTM. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Small NTM (w=3, n=10, m=5, h=20, batch=5) matching Idris benchNtm.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 3, 10, 5, 20
    batch_size = 5
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1,
        output_width=w,
        n=n,
        m=m,
        controller_size=h,
        lr=lr,
        clip_value=clip_value,
    )
    model = NtmModel(cfg)

    # Generate fixed batch
    batch = generate_copy_batch(batch_size, seq_min=2, seq_max=4, seq_width=w)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    # Warmup: 10 epochs
    for _ in range(10):
        _train_ntm_epoch(model, batch, optimizer, clip_value)

    # Benchmark: 100 epochs
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(100):
        loss_val = _train_ntm_epoch(model, batch, optimizer, clip_value)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val, _peak_rss_mb()


def bench_ntm_copy() -> tuple[float, float, float]:
    """Benchmark production-scale NTM copy. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Production NTM (w=8, n=128, m=20, h=100, batch=16) matching Idris NtmCopy.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 8, 128, 20, 100
    batch_size = 16
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1,
        output_width=w,
        n=n,
        m=m,
        controller_size=h,
        lr=lr,
        clip_value=clip_value,
    )
    model = NtmModel(cfg)

    # Generate fixed batch
    batch = generate_copy_batch(batch_size, seq_min=1, seq_max=20, seq_width=w)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    # Warmup: 10 epochs
    for _ in range(10):
        _train_ntm_epoch(model, batch, optimizer, clip_value)

    # Benchmark: 100 epochs
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(100):
        loss_val = _train_ntm_epoch(model, batch, optimizer, clip_value)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val, _peak_rss_mb()


def bench_ntm_copy_1k() -> tuple[float, float, float]:
    """Benchmark 1000-epoch NTM copy with fresh data. Returns (elapsed_ms, final_loss, peak_rss_mb).

    Realistic benchmark: fresh batch each epoch, momentum=0.9, matching Idris benchNtmCopy1k.
    """
    random.seed(123456)
    torch.manual_seed(123456)

    w, n, m, h = 8, 128, 20, 100
    batch_size = 16
    lr, alpha, clip_value = 0.0001, 0.95, 10.0

    cfg = NtmConfig(
        input_width=w + 1,
        output_width=w,
        n=n,
        m=m,
        controller_size=h,
        lr=lr,
        clip_value=clip_value,
    )
    model = NtmModel(cfg)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9)

    # Warmup: 10 epochs (fresh data)
    for _ in range(10):
        batch = generate_copy_batch(batch_size, seq_min=1, seq_max=20, seq_width=w)
        _train_ntm_epoch(model, batch, optimizer, clip_value)

    # Benchmark: 1000 epochs (fresh data)
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(1000):
        batch = generate_copy_batch(batch_size, seq_min=1, seq_max=20, seq_width=w)
        loss_val = _train_ntm_epoch(model, batch, optimizer, clip_value)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val, _peak_rss_mb()


def main() -> None:
    print("PyTorch Benchmark")
    print("=" * 50)

    elapsed, loss, rss = bench_supervised()
    print(f"Supervised (1000 epochs): {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")
    print(f"  Peak RSS: {rss:.0f} MB")

    elapsed, loss, rss = bench_rnn()
    print(f"RNN (1000 epochs):        {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")
    print(f"  Peak RSS: {rss:.0f} MB")

    elapsed, loss, rss = bench_ntm()
    print(f"NTM (100 epochs):         {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")
    print(f"  Peak RSS: {rss:.0f} MB")

    elapsed, loss, rss = bench_ntm_copy()
    print(f"NTM-copy (100 epochs):    {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")
    print(f"  Peak RSS: {rss:.0f} MB")

    elapsed, loss, rss = bench_ntm_copy_1k()
    print(f"NTM-copy-1k (1000 epochs): {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")
    print(f"  Peak RSS: {rss:.0f} MB")


if __name__ == "__main__":
    main()
