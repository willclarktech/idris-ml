"""LSTM controller for NTM.

Learned initial state via FC from dummy input (non-zero initial h and c).
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

        # Learned initial state: FC from dummy → non-zero h0, c0
        self.h_bias_fc = nn.Linear(1, hidden_size)
        self.c_bias_fc = nn.Linear(1, hidden_size)

    def reset_state(self) -> None:
        dummy = torch.tensor([[0.0]])
        self._h = self.h_bias_fc(dummy).squeeze(0)
        self._c = self.c_bias_fc(dummy).squeeze(0)

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
