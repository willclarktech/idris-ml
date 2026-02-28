"""Timing benchmark matching idris-ml's Example/Bench.idr.

- Supervised: 100 warmup + 1000 timed epochs, SGD lr=0.03
- RNN: 100 warmup + 1000 timed epochs, SGD lr=0.03
- NTM: 10 warmup + 100 timed epochs, Adam lr=0.001 maxNorm=5.0

NOTE: The NTM benchmark uses sigmoid activation (not tanh), matching Bench.idr
which differs from NtmCopy.idr. Bench.idr also uses maxNorm=5.0, not 50.0.
"""

import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from bench.data.copy_task import copy_task_point
from bench.models.rnn import LinearRNNCell, generate_rnn_dataset, train_rnn_epoch
from bench.models.supervised import SupervisedModel, SUPERVISED_DATA, train_supervised_epoch
from bench.ntm.ntm_layer import NTMLayer, ntm_input_width, ntm_output_width
from bench.training.losses import cross_entropy, nll_loss


def bench_supervised() -> tuple[float, float]:
    """Benchmark supervised model. Returns (elapsed_ms, final_loss)."""
    torch.manual_seed(123456)
    model = SupervisedModel()
    data = SUPERVISED_DATA
    lr = 0.03

    # Warmup: 100 epochs
    for _ in range(100):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_supervised_epoch(model, data, optimizer)

    # Benchmark: 1000 epochs
    t0 = time.monotonic()
    for _ in range(1000):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_val = train_supervised_epoch(model, data, optimizer)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val


def bench_rnn() -> tuple[float, float]:
    """Benchmark RNN model. Returns (elapsed_ms, final_loss)."""
    torch.manual_seed(123456)
    model = LinearRNNCell(1, 1)
    data = generate_rnn_dataset(8)
    lr = 0.03

    # Warmup: 100 epochs
    for _ in range(100):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_rnn_epoch(model, data, optimizer)

    # Benchmark: 1000 epochs
    t0 = time.monotonic()
    for _ in range(1000):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_val = train_rnn_epoch(model, data, optimizer)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val


# Same 5 NTM sequences as Bench.idr
_NTM_SEQUENCES: list[list[int]] = [
    [1, 2, 1, 2],
    [1, 1, 2, 2, 1],
    [2, 1, 1, 2, 2, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [1, 2, 1, 1, 2, 2, 1, 2],
]


def bench_ntm() -> tuple[float, float]:
    """Benchmark NTM model. Returns (elapsed_ms, final_loss).

    NOTE: Uses sigmoid activation (not tanh) and maxNorm=5.0, matching Bench.idr.
    """
    torch.manual_seed(123456)

    w, n, h = 3, 10, 20
    input_w = ntm_input_width(w)
    output_w = ntm_output_width(n, w)

    # Controller: Linear → Sigmoid → Linear (Bench.idr uses sigmoidLayer)
    controller = nn.Sequential(
        nn.Linear(input_w, h),
        nn.Sigmoid(),
        nn.Linear(h, output_w),
    )
    for m in controller.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    ntm = NTMLayer(controller, n, w)

    # Prepare data
    data = [copy_task_point(seq, w) for seq in _NTM_SEQUENCES]

    max_norm = 5.0

    def train_one_epoch() -> float:
        optimizer = torch.optim.Adam(ntm.parameters(), lr=0.001)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for xs, ys in data:
            ntm.reset_state()
            seq_loss = torch.tensor(0.0)
            for x, y in zip(xs, ys):
                raw = ntm(x)
                pred = torch.log_softmax(raw, dim=-1)
                seq_loss = seq_loss + nll_loss(pred, y)
            total_loss = total_loss + seq_loss / len(xs)
        loss = total_loss / len(data)
        loss.backward()
        clip_grad_norm_(ntm.parameters(), max_norm)
        optimizer.step()
        ntm.project_addressing()
        return loss.item()

    # Warmup: 10 epochs
    for _ in range(10):
        train_one_epoch()

    # Benchmark: 100 epochs
    t0 = time.monotonic()
    for _ in range(100):
        loss_val = train_one_epoch()
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val


def main() -> None:
    print("PyTorch Benchmark")
    print("=" * 50)

    elapsed, loss = bench_supervised()
    print(f"Supervised (1000 epochs): {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")

    elapsed, loss = bench_rnn()
    print(f"RNN (1000 epochs):        {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")

    elapsed, loss = bench_ntm()
    print(f"NTM (100 epochs):         {elapsed:.1f} ms")
    print(f"  Final loss: {loss:.6f}")


if __name__ == "__main__":
    main()
