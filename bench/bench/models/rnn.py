"""RNN model matching idris-ml's Example/Rnn.idr.

NOTE: We use a custom LinearRNNCell instead of nn.RNN because idris-ml's
RnnLayer has NO activation function — it's a raw linear recurrence:
  h' = W_ih @ x + W_hh @ h + b
PyTorch's nn.RNN always applies tanh, which would diverge from idris-ml.
"""

import torch
import torch.nn as nn
from torch import Tensor

from bench.training.losses import bce_with_logits


class LinearRNNCell(nn.Module):
    """Linear RNN cell with no activation, matching idris-ml RnnLayer."""

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


def generate_rnn_data(n: int) -> tuple[list[float], list[float]]:
    """Generate cyclic [0,1,0] pattern of length n.

    Matches Rnn.idr generateData.
    """
    pattern = [0.0, 1.0, 0.0]
    inputs = [pattern[i % 3] for i in range(n)]
    outputs = [pattern[(i + 1) % 3] for i in range(n)]
    return inputs, outputs


def generate_rnn_dataset(n: int) -> list[tuple[list[Tensor], list[Tensor]]]:
    """Generate n sequences with lengths 3, 4, ..., n+2.

    Matches Rnn.idr generateDataSet (range mapped with +3).
    """
    dataset = []
    for i in range(n):
        length = i + 3
        inputs, outputs = generate_rnn_data(length)
        xs = [torch.tensor([v]) for v in inputs]
        ys = [torch.tensor([v]) for v in outputs]
        dataset.append((xs, ys))
    return dataset


# Same 8 sequences as Rnn.idr
RNN_DATA = generate_rnn_dataset(8)


def train_rnn_epoch(
    model: LinearRNNCell,
    data: list[tuple[list[Tensor], list[Tensor]]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one recurrent epoch, return loss value.

    Matches Backprop.idr epochRecurrent: for each sequence, reset state,
    forward all timesteps, accumulate loss. Average across sequences.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
        for x, y in zip(xs, ys):
            pred = model(x)
            seq_loss = seq_loss + bce_with_logits(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()
