"""NTM copy task model matching loudinthecloud/pytorch-ntm reference.

LSTM controller (hidden=100) → separate head FCs → NTMLayer → sigmoid output.
Loss: BCELoss. Optimizer: RMSprop lr=1e-4. Value clip [-10,10].
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from bench.ntm.ntm_layer import NTMLayer


class LSTMController(nn.Module):
    """LSTM controller for NTM."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

    def reset_state(self) -> None:
        self._h = self.h0.clone()
        self._c = self.c0.clone()

    @property
    def last_hidden(self) -> Tensor:
        return self._h

    def forward(self, x: Tensor) -> Tensor:
        h, c = self.lstm(x.unsqueeze(0), (self._h.unsqueeze(0), self._c.unsqueeze(0)))
        self._h = h.squeeze(0)
        self._c = c.squeeze(0)
        return self._h


class RNNController(nn.Module):
    """Simple RNN controller for NTM (vanilla RNNCell + tanh)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
        self.h0 = nn.Parameter(torch.zeros(hidden_size))

    def reset_state(self) -> None:
        self._h = self.h0.clone()

    @property
    def last_hidden(self) -> Tensor:
        return self._h

    def forward(self, x: Tensor) -> Tensor:
        self._h = self.rnn(x.unsqueeze(0), self._h.unsqueeze(0)).squeeze(0)
        return self._h


@dataclass
class NtmCopyConfig:
    seq_width: int = 8  # bits per vector
    seq_min: int = 1  # min sequence length
    seq_max: int = 20  # max sequence length
    n: int = 128  # memory slots
    m: int = 20  # memory width
    controller_size: int = 100  # LSTM hidden size
    lr: float = 1e-4
    iterations: int = 50000
    clip_value: float = 10.0


class NtmCopyModel(nn.Module):
    """NTM model for copy task matching loudinthecloud reference."""

    def __init__(self, cfg: NtmCopyConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = NtmCopyConfig()
        self.cfg = cfg

        # Input width: seq_width + 1 (delimiter channel) + m (prev read)
        num_inputs = cfg.seq_width + 1
        num_outputs = cfg.seq_width  # output is seq_width bits
        controller_input_size = num_inputs + cfg.m  # input + prev read vector

        controller = LSTMController(controller_input_size, cfg.controller_size)

        self.ntm = NTMLayer(
            controller=controller,
            n=cfg.n,
            m=cfg.m,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            controller_hidden_size=cfg.controller_size,
        )

    def reset_state(self) -> None:
        self.ntm.reset_state()
        self.ntm.controller.reset_state()  # type: ignore[operator]

    def forward(self, x: Tensor) -> Tensor:
        """Forward one timestep, returns sigmoid output."""
        raw = self.ntm(x)
        return torch.sigmoid(raw)


def train_ntm_copy_step(
    model: NtmCopyModel,
    input_seq: Tensor,
    target_seq: Tensor,
    optimizer: torch.optim.Optimizer,
    clip_mode: Literal["value", "norm"] = "value",
) -> float:
    """Train one sequence (batch_size=1).

    input_seq: (seq_len+1, seq_width+1)
    target_seq: (seq_len, seq_width)
    """
    optimizer.zero_grad()
    model.reset_state()

    seq_len = target_seq.shape[0]

    # Input phase: feed all input rows (data + delimiter)
    for t in range(input_seq.shape[0]):
        model(input_seq[t])

    # Output phase: feed zeros, collect outputs
    num_inputs = input_seq.shape[1]
    outputs = []
    for _ in range(seq_len):
        out = model(torch.zeros(num_inputs))
        outputs.append(out)

    # Stack outputs and compute BCE loss
    pred = torch.stack(outputs)  # (seq_len, seq_width)
    loss = nn.functional.binary_cross_entropy(pred, target_seq)

    loss.backward()
    if clip_mode == "norm":
        clip_grad_norm_(model.parameters(), model.cfg.clip_value)
    else:
        clip_grad_value_(model.parameters(), model.cfg.clip_value)
    optimizer.step()

    return loss.item()
