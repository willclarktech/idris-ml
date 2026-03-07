"""LSTM controller for NTM.

Learned initial state via direct nn.Parameter (non-zero initial h and c).
"""

import torch
import torch.nn as nn
from torch import Tensor


class LSTMController(nn.Module):
    """LSTM controller with learned initial state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # Learned initial state: direct parameters
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

    def reset_state(self) -> None:
        self._h = self.h0.clone()
        self._c = self.c0.clone()

    @property
    def last_hidden(self) -> Tensor:
        return self._h

    @property
    def last_cell(self) -> Tensor:
        return self._c

    def forward(self, x: Tensor) -> Tensor:
        h, c = self.lstm(x.unsqueeze(0), (self._h.unsqueeze(0), self._c.unsqueeze(0)))
        self._h = h.squeeze(0)
        self._c = c.squeeze(0)
        return self._h
