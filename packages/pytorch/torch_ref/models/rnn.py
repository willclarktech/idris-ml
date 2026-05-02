"""RNN model with linear recurrence (no activation).

Uses a custom LinearRNNCell instead of nn.RNN because the Idris-side
RnnLayer has no activation — it's a raw linear recurrence:
  h' = W_ih @ x + W_hh @ h + b
PyTorch's nn.RNN always applies tanh.
"""

import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.training.losses import bce_with_logits


class LinearRNNCell(nn.Module):
    """Linear RNN cell with no activation."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.hidden = nn.Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def reset_state(self) -> None:
        self._h = self.hidden.clone()

    def forward(self, x: Tensor) -> Tensor:
        self._h = self.weight_ih @ x + self.weight_hh @ self._h + self.bias
        return self._h


class LinearLSTMCell(nn.Module):
    """LSTM cell with linear output projection.

    Matches Idris LSTM example: LSTM(1, hidden) -> Linear(hidden, 1).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.xavier_uniform_(self.lstm.weight_hh)
        # Forget gate bias = 1.0 (Jozefowicz et al. 2015, helps gradient flow)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        with torch.no_grad():
            self.lstm.bias_ih[hidden_size : 2 * hidden_size].fill_(1.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def reset_state(self) -> None:
        self._h = torch.zeros(self.hidden_size)
        self._c = torch.zeros(self.hidden_size)

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
    """GRU cell with linear output projection.

    Matches the simplified-GRU variant implemented in
    `packages/backends/backend_{tape,mlx,torch}.c{,pp}` (`tensor_gru_cell`):
    z and n gates are used; r is computed but NOT used to mask n.
    This deviates from PyTorch's standard `nn.GRUCell` (where r masks
    the hidden contribution to n) — kept for cross-backend alignment
    with the Idris-side C kernel.

    Equations:
        combined = W_ih @ x + b_ih + W_hh @ h + b_hh   (size 3*o)
        z = sigmoid(combined[0:o])
        r = sigmoid(combined[o:2o])           # computed, not used
        n = tanh(combined[2o:3o])
        h' = (1 - z) * n + z * h
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
        self._h = torch.zeros(self.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        combined = self.weight_ih @ x + self.bias_ih + self.weight_hh @ self._h + self.bias_hh
        o = self.hidden_size
        z = torch.sigmoid(combined[0:o])
        # r = torch.sigmoid(combined[o:2*o])  # simplified GRU: r unused
        n = torch.tanh(combined[2 * o : 3 * o])
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
    for i in range(n):
        length = i + 3
        inputs, outputs = generate_rnn_data(length)
        xs = [torch.tensor([v]) for v in inputs]
        ys = [torch.tensor([v]) for v in outputs]
        dataset.append((xs, ys))
    return dataset


RNN_DATA = generate_rnn_dataset(8)


def train_lstm_epoch(
    model: LinearLSTMCell,
    data: list[tuple[list[Tensor], list[Tensor]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one LSTM epoch, same structure as train_rnn_epoch."""
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
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
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
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
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
        for x, y in zip(xs, ys, strict=True):
            pred = model(x)
            seq_loss = seq_loss + bce_with_logits(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()
