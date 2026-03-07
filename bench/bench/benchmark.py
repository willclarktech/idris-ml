"""Timing benchmark matching idris-ml's Example/Bench.idr.

- Supervised: 100 warmup + 1000 timed epochs, SGD lr=0.03
- RNN: 100 warmup + 1000 timed epochs, SGD lr=0.03
- NTM: 10 warmup + 100 timed epochs, Adam lr=0.001 maxNorm=5.0

NOTE: The NTM benchmark uses a simple copy-like task for timing comparison
with idris-ml's Bench.idr. It is NOT the reference copy task architecture.
"""

import time

import torch
from torch.nn.utils import clip_grad_norm_

from bench.models.rnn import LinearRNNCell, generate_rnn_dataset, train_rnn_epoch
from bench.models.supervised import SUPERVISED_DATA, SupervisedModel, train_supervised_epoch
from bench.ntm.controller import LSTMController
from bench.ntm.layer import NTMLayer


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
    loss_val = 0.0
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
    loss_val = 0.0
    t0 = time.monotonic()
    for _ in range(1000):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_val = train_rnn_epoch(model, data, optimizer)
    t1 = time.monotonic()

    elapsed = (t1 - t0) * 1000
    return elapsed, loss_val


# Simple copy-like sequences for NTM timing benchmark
_NTM_SEQUENCES: list[list[int]] = [
    [1, 2, 1, 2],
    [1, 1, 2, 2, 1],
    [2, 1, 1, 2, 2, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [1, 2, 1, 1, 2, 2, 1, 2],
]


def _make_bench_data(
    sequences: list[list[int]], w: int
) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Convert symbol sequences to one-hot input/target pairs for timing."""
    data = []
    for symbols in sequences:
        seq_len = len(symbols)
        blanks = [0] * seq_len

        def one_hot(idx: int) -> torch.Tensor:
            v = torch.zeros(w)
            v[idx] = 1.0
            return v

        inputs = [one_hot(s) for s in symbols + blanks]
        targets = [one_hot(s) for s in blanks + symbols]
        data.append((inputs, targets))
    return data


def bench_ntm() -> tuple[float, float]:
    """Benchmark NTM model. Returns (elapsed_ms, final_loss).

    Uses a small NTM (w=3, n=10) with LSTM controller for timing comparison
    with idris-ml's Bench.idr.
    """
    torch.manual_seed(123456)

    w, n, h = 3, 10, 20
    m = w  # memory width = input width for this simple benchmark

    controller = LSTMController(w + m, h)  # input + prev read vector

    ntm = NTMLayer(
        controller=controller,
        n=n,
        m=m,
        num_inputs=w,
        num_outputs=w,
        controller_hidden_size=h,
    )

    # Prepare data
    data = _make_bench_data(_NTM_SEQUENCES, w)

    max_norm = 5.0

    def nll_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -(targets * log_probs).mean()

    def train_one_epoch() -> float:
        optimizer = torch.optim.Adam(ntm.parameters(), lr=0.001)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for xs, ys in data:
            ntm.reset_state()
            controller.reset_state()
            seq_loss = torch.tensor(0.0)
            for x, y in zip(xs, ys, strict=True):
                raw = ntm(x)
                pred = torch.log_softmax(raw, dim=-1)
                seq_loss = seq_loss + nll_loss(pred, y)
            total_loss = total_loss + seq_loss / len(xs)
        loss = total_loss / len(data)
        loss.backward()
        clip_grad_norm_(ntm.parameters(), max_norm)
        optimizer.step()
        return loss.item()

    # Warmup: 10 epochs
    for _ in range(10):
        train_one_epoch()

    # Benchmark: 100 epochs
    loss_val = 0.0
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
