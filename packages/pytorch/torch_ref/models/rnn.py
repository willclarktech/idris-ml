"""RNN model — `nn.RNNCell`-equivalent (tanh activation, two biases).

Mirrors Idris's `Layer.Rnn`:
  h' = activation( W_ih @ x + b_ih + W_hh @ h + b_hh )
with `activation = tanh` (default), zero-initialised hidden state.

The previous `LinearRNNCell` (no activation, single bias, learned
initial hidden state) was a non-standard variant chosen arbitrarily;
we've moved both Idris and the reference to the standard PyTorch
`nn.RNNCell` shape so the example demonstrates the canonical RNN
recipe.
"""

import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.training.losses import bce_with_logits
from torch_ref.training.runner import get_device


class LinearRNNCell(nn.Module):
    """RNN cell + linear output projection (mirrors Idris example shape).

    Cell equation matches PyTorch's `nn.RNNCell` (tanh activation, two
    biases):
      h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)

    Plus an output projection W_out @ h + b_out, so the full forward
    is `out = W_out @ tanh(...) + b_out`. The Idris example has the
    same structure: `RnnLayer(1, 4) ~~> Linear(4, 1)`.

    Class name kept (`LinearRNNCell`) for stable script imports; the
    "Linear" prefix here means "with a linear output projection on
    top", not "linear-recurrence" (which was the pre-2026-05-09
    non-standard variant).
    """

    def __init__(self, input_size: int, hidden_size: int = 4, output_size: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size))

        self.weight_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias_out = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_out)

    def reset_state(self) -> None:
        self._h = torch.zeros(self.hidden_size, device=self.weight_ih.device)

    def forward(self, x: Tensor) -> Tensor:
        self._h = torch.tanh(
            self.weight_ih @ x + self.bias_ih + self.weight_hh @ self._h + self.bias_hh
        )
        return self.weight_out @ self._h + self.bias_out


class LinearLSTMCell(nn.Module):
    """LSTM cell with learned initial state + linear output projection.

    Wraps nn.LSTMCell (two biases as standard) plus learned h0/c0
    parameters and an output projection. Matches Idris LstmState
    which carries learned h0/c0 (added in Phase 1.5b for NTM/DNC
    alignment; reused here for the standalone LSTM example so all
    callers see the same shape).

    Pre-2026-05-09 there was no forget-gate-bias=1.0 init in this
    cell because the Idris side didn't have it either; we drop the
    Jozefowicz default to keep the example aligned with what
    Layer.Lstm produces.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        # Learned initial hidden + cell state (matches Idris LstmState).
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.xavier_uniform_(self.lstm.weight_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def reset_state(self) -> None:
        self._h = self.h0.clone()
        self._c = self.c0.clone()

    def forward(self, x: Tensor) -> Tensor:
        # LSTMCell expects (batch, input_size), add batch dim
        h_in = self._h.unsqueeze(0)
        c_in = self._c.unsqueeze(0)
        x_in = x.unsqueeze(0)
        h_out, c_out = self.lstm(x_in, (h_in, c_in))
        self._h = h_out.squeeze(0)
        self._c = c_out.squeeze(0)
        return self.output_proj(self._h)


class LinearGRUCell(nn.Module):
    """GRU cell + linear output projection (mirrors Idris example shape).

    Cell equation matches PyTorch's `nn.GRUCell` (the standard GRU):
        ih = W_ih @ x + b_ih           (size 3*o)
        hh = W_hh @ h + b_hh           (size 3*o)
        z = sigmoid(ih_z + hh_z)
        r = sigmoid(ih_r + hh_r)
        n = tanh(ih_n + r * hh_n)
        h' = (1 - z) * n + z * h
    Plus an output projection W_out @ h + b_out.

    Pre-2026-05-09 this was a "simplified GRU" where r was computed
    but not used — non-standard, kept for parity with a C kernel that
    also ignored r. Both sides now use the standard `nn.GRU` equation.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def reset_state(self) -> None:
        self._h = torch.zeros(self.hidden_size, device=self.weight_ih.device)

    def forward(self, x: Tensor) -> Tensor:
        ih = self.weight_ih @ x + self.bias_ih
        hh = self.weight_hh @ self._h + self.bias_hh
        o = self.hidden_size
        z = torch.sigmoid(ih[0:o] + hh[0:o])
        r = torch.sigmoid(ih[o:2 * o] + hh[o:2 * o])
        n = torch.tanh(ih[2 * o : 3 * o] + r * hh[2 * o : 3 * o])
        self._h = (1.0 - z) * n + z * self._h
        return self.output_proj(self._h)


def generate_rnn_data(n: int) -> tuple[list[float], list[float]]:
    """Generate cyclic [0,1,0] pattern of length n."""
    pattern = [0.0, 1.0, 0.0]
    inputs = [pattern[i % 3] for i in range(n)]
    outputs = [pattern[(i + 1) % 3] for i in range(n)]
    return inputs, outputs


def generate_rnn_dataset(n: int) -> list[tuple[list[Tensor], list[Tensor]]]:
    """Generate n sequences with lengths 3, 4, ..., n+2."""
    dataset = []
    device = get_device()
    for i in range(n):
        length = i + 3
        inputs, outputs = generate_rnn_data(length)
        xs = [torch.tensor([v], device=device) for v in inputs]
        ys = [torch.tensor([v], device=device) for v in outputs]
        dataset.append((xs, ys))
    return dataset


# Module-level dataset constant — built on whatever device is active
# at import time (typically "cpu"). Scripts that switch to MPS/CUDA
# should rebuild via `generate_rnn_dataset(...)` after `set_device`.
RNN_DATA = generate_rnn_dataset(8)


def train_lstm_epoch(
    model: LinearLSTMCell,
    data: list[tuple[list[Tensor], list[Tensor]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one LSTM epoch, same structure as train_rnn_epoch."""
    optimizer.zero_grad()
    device = get_device()
    total_loss = torch.tensor(0.0, device=device)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0, device=device)
        for x, y in zip(xs, ys, strict=True):
            pred = model(x)
            seq_loss = seq_loss + bce_with_logits(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_gru_epoch(
    model: LinearGRUCell,
    data: list[tuple[list[Tensor], list[Tensor]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one GRU epoch, same structure as train_lstm_epoch."""
    optimizer.zero_grad()
    device = get_device()
    total_loss = torch.tensor(0.0, device=device)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0, device=device)
        for x, y in zip(xs, ys, strict=True):
            pred = model(x)
            seq_loss = seq_loss + bce_with_logits(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_rnn_epoch(
    model: LinearRNNCell,
    data: list[tuple[list[Tensor], list[Tensor]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one recurrent epoch, return loss value.

    For each sequence: reset state, forward all timesteps, accumulate loss.
    Average across sequences.
    """
    optimizer.zero_grad()
    device = get_device()
    total_loss = torch.tensor(0.0, device=device)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0, device=device)
        for x, y in zip(xs, ys, strict=True):
            pred = model(x)
            seq_loss = seq_loss + bce_with_logits(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()
