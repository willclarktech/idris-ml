"""Single-layer forward+backward+step benchmarks (Axis B).

PyTorch reference for `packages/idris-ml-examples/src/Example/LayersBench.idr`.
Same shapes, same iteration counts, same output format the Idris-side bench
emits — so `scripts/perf-fast.sh` reads both with one regex and emits paired
JSONL entries (`runtime: "tape"` vs `runtime: "pytorch"`) for
`scripts/render-benchmarks.py`.

Output format (one line per workload, matches bench_ops.py):
    <label>:\t<wall_ms> ms  (<iters> iters)
"""

import time

import torch
import torch.nn as nn


def wall_ms() -> float:
    return time.monotonic() * 1000.0


# --- Linear (batch=32, in=512, out=512) ---


def bench_linear(in_dim: int, out_dim: int, batch: int, iters: int, warmup: int) -> None:
    """Dense matmul + bias + autograd graph fwd+bwd+step.

    Mirrors Idris-side `Example.LayersBench.benchLinear`: 100 fwd+bwd+step
    cycles at batch=32, in=512, out=512 on F64 CPU; sum-MSE loss; SGD lr=0.01.
    """
    torch.manual_seed(42)
    model = nn.Linear(in_dim, out_dim).double()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inp = torch.full((batch, in_dim), 0.1, dtype=torch.float64)
    tgt = torch.full((batch, out_dim), 0.1, dtype=torch.float64)

    for _ in range(warmup):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(f"linear bs={batch} i={in_dim} o={out_dim}:\t{elapsed:.3f} ms\t({iters} iters)")


# --- LstmCell (hidden=256, unbatched) ---


def bench_lstm_cell(hidden: int, iters: int, warmup: int) -> None:
    """Single-sample LSTM cell fwd+bwd+step.

    Mirrors Idris-side `benchLstmCell`: same hidden dim, same iters,
    same MSE-against-constant-target target. The Idris LstmState carries
    h/c across iters; here we explicitly thread them through.
    """
    torch.manual_seed(42)
    cell = nn.LSTMCell(input_size=hidden, hidden_size=hidden).double()
    opt = torch.optim.SGD(cell.parameters(), lr=0.01)
    inp = torch.full((1, hidden), 0.1, dtype=torch.float64)
    tgt = torch.full((1, hidden), 0.1, dtype=torch.float64)
    h = torch.zeros(1, hidden, dtype=torch.float64)
    c = torch.zeros(1, hidden, dtype=torch.float64)

    for _ in range(warmup):
        opt.zero_grad()
        h, c = cell(inp, (h.detach(), c.detach()))
        diff = h - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        opt.zero_grad()
        h, c = cell(inp, (h.detach(), c.detach()))
        diff = h - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(f"lstm_cell hidden={hidden}:\t{elapsed:.3f} ms\t({iters} iters)")


# --- Conv2dBlock (batch=8, c_in=3, h=w=16, c_out=16, k=3) ---


def bench_conv2d_block(
    in_c: int,
    out_c: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    batch: int,
    iters: int,
    warmup: int,
) -> None:
    """Single Conv2D layer fwd+bwd+step.

    Mirrors Idris-side `benchConv2dBlock`: input shape is rank-4
    `[batch, c_in, h, w]` (the Idris-side flattens to rank-2
    `[batch, c_in*h*w]` and the layer reshapes internally; here we
    pass 4D since PyTorch's nn.Conv2d takes that natively).
    """
    torch.manual_seed(42)
    conv = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), padding=0).double()
    opt = torch.optim.SGD(conv.parameters(), lr=0.01)
    inp = torch.full((batch, in_c, h, w), 0.1, dtype=torch.float64)
    out_h = h - kh + 1
    out_w = w - kw + 1
    tgt = torch.full((batch, out_c, out_h, out_w), 0.1, dtype=torch.float64)

    for _ in range(warmup):
        opt.zero_grad()
        pred = conv(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        opt.zero_grad()
        pred = conv(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(
        f"conv2d_block bs={batch} {in_c}x{h}x{w}->{out_c} k={kh}x{kw}:"
        f"\t{elapsed:.3f} ms\t({iters} iters)"
    )


# --- TransformerBlock (1-block transformer at small dims) ---


def bench_transformer_block(
    seq_len: int,
    d_model: int,
    n_heads: int,
    vocab_size: int,
    batch: int,
    iters: int,
    warmup: int,
) -> None:
    """1-block transformer encoder fwd+bwd+step.

    Mirrors Idris-side `benchTransformerBlock`: embedding + one encoder
    layer + final norm + vocab projection at small dims so the
    measurement is dominated by the attention + FFN + LayerNorm pattern,
    not the embedding/projection wrappers.
    """
    torch.manual_seed(42)

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True,
            )
            self.final_norm = nn.LayerNorm(d_model)
            self.vocab_proj = nn.Linear(d_model, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embed(x)
            h = self.block(h)
            h = self.final_norm(h)
            return self.vocab_proj(h)

    model = TinyTransformer().double()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inp = torch.zeros((batch, seq_len), dtype=torch.long)
    tgt = torch.full((batch, seq_len, vocab_size), 0.1, dtype=torch.float64)

    for _ in range(warmup):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()

    t0 = wall_ms()
    for _ in range(iters):
        opt.zero_grad()
        pred = model(inp)
        diff = pred - tgt
        loss = (diff * diff).sum()
        loss.backward()
        opt.step()
    elapsed = wall_ms() - t0

    print(
        f"transformer_block bs={batch} seq={seq_len} d={d_model} heads={n_heads}:"
        f"\t{elapsed:.3f} ms\t({iters} iters)"
    )


def main() -> None:
    print("--- Linear ---")
    bench_linear(in_dim=512, out_dim=512, batch=32, iters=100, warmup=10)
    print()
    print("--- LstmCell ---")
    bench_lstm_cell(hidden=256, iters=100, warmup=10)
    print()
    print("--- Conv2dBlock ---")
    bench_conv2d_block(in_c=3, out_c=16, h=16, w=16, kh=3, kw=3, batch=8, iters=100, warmup=10)
    print()
    # NTM has no canonical PyTorch reference (the Idris-side bench
    # mirrors the model architecture in `packages/pytorch/torch_ref/ntm/`
    # but that's not designed for op-level perf comparison). Axis B's
    # NTM entry intentionally shows "—" for the pytorch column.
    print("--- TransformerBlock ---")
    bench_transformer_block(
        seq_len=16, d_model=64, n_heads=4, vocab_size=32, batch=2, iters=50, warmup=5
    )
    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
