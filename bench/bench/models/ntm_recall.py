"""NTM associative recall model matching idris-ml's NtmAssociativeRecall.idr.

Supports two controller types:
  LSTM (default, matches Graves et al. 2014):
    LSTMCell(NtmInputWidth→H) → Linear(H→NtmOutputWidth) → [NTM] → LogSoftmax
  RNN (vanilla baseline):
    LinearRNNCell(NtmInputWidth→H) → Tanh → Linear(H→NtmOutputWidth) → [NTM] → LogSoftmax
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from bench.models.rnn import LinearRNNCell
from bench.ntm.ntm_layer import NTMLayer, ntm_input_width, ntm_output_width
from bench.training.losses import weighted_nll_loss


@dataclass
class NtmRecallConfig:
    w: int = 8
    n: int = 128
    h: int = 100
    controller: str = "lstm"  # "lstm" (Graves et al. 2014) or "rnn"
    batch_size: int = 1
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_norm: float = 50.0
    clip_mode: str = "value"  # "norm" (global L2) or "value" (per-param clamp)
    clip_value: float = 10.0  # used when clip_mode="value"
    optimizer: str = "rmsprop"  # "adam" or "rmsprop"
    div_final: float = 10.0
    epochs: int = 100000
    patience: int = 2000
    chunk_size: int = 25
    recall_weight: float = 3.0


class RNNController(nn.Module):
    """RNN controller: LinearRNNCell → Tanh → Linear."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = LinearRNNCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_state(self) -> None:
        self.rnn.reset_state()

    def forward(self, x: Tensor) -> Tensor:
        h = self.rnn(x)
        h = torch.tanh(h)
        return self.output(h)


class LSTMController(nn.Module):
    """LSTM controller matching Graves et al. 2014.

    LSTMCell already has tanh in its output gate, so no extra activation
    is needed (unlike RNNController which adds explicit tanh).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Learnable initial states (matching RNNController's h0 pattern)
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.c0 = nn.Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_state(self) -> None:
        self._h = self.h0.clone()
        self._c = self.c0.clone()

    def forward(self, x: Tensor) -> Tensor:
        # LSTMCell expects (batch, features) — unsqueeze/squeeze for unbatched
        h, c = self.lstm(x.unsqueeze(0), (self._h.unsqueeze(0), self._c.unsqueeze(0)))
        self._h = h.squeeze(0)
        self._c = c.squeeze(0)
        return self.output(self._h)


class NtmRecallModel(nn.Module):
    """NTM model for associative recall."""

    def __init__(self, cfg: NtmRecallConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = NtmRecallConfig()
        self.cfg = cfg

        input_w = ntm_input_width(cfg.w)
        output_w = ntm_output_width(cfg.n, cfg.w)

        if cfg.controller == "lstm":
            controller: nn.Module = LSTMController(input_w, cfg.h, output_w)
        else:
            controller = RNNController(input_w, cfg.h, output_w)
        self.ntm = NTMLayer(controller, cfg.n, cfg.w)

    def reset_state(self) -> None:
        self.ntm.reset_state()
        # pyright doesn't see reset_state() through nn.Module-typed controller field
        self.ntm.controller.reset_state()  # type: ignore[operator]

    def forward(self, x: Tensor) -> Tensor:
        """Forward one timestep, returns log-softmax output."""
        raw = self.ntm(x)
        return torch.log_softmax(raw, dim=-1)

    def project_addressing(self) -> None:
        self.ntm.project_addressing()


def train_ntm_recall_step(
    model: NtmRecallModel,
    data: list[tuple[list[Tensor], list[Tensor]]],
    loss_fn: object,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch on NTM recall data."""
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
        for x, y in zip(xs, ys, strict=True):
            pred = model(x)
            seq_loss = seq_loss + weighted_nll_loss(pred, y, weight=model.cfg.recall_weight)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    if model.cfg.clip_mode == "value":
        clip_grad_value_(model.parameters(), model.cfg.clip_value)
    else:
        clip_grad_norm_(model.parameters(), model.cfg.max_norm)
    optimizer.step()
    model.project_addressing()
    return loss.item()
