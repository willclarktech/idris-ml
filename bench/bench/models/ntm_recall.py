"""NTM associative recall model matching idris-ml's NtmAssociativeRecall.idr.

RNN controller + tanh activation:
  LinearRNNCell(NtmInputWidth→H) → Tanh → Linear(H→NtmOutputWidth) → [NTM] → LogSoftmax
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from bench.models.rnn import LinearRNNCell
from bench.ntm.ntm_layer import NTMLayer, ntm_input_width, ntm_output_width
from bench.training.losses import weighted_nll_loss


@dataclass
class NtmRecallConfig:
    w: int = 8
    n: int = 128
    h: int = 100
    batch_size: int = 48
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_norm: float = 50.0
    div_final: float = 10.0
    epochs: int = 30000
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


class NtmRecallModel(nn.Module):
    """NTM model for associative recall with RNN controller."""

    def __init__(self, cfg: NtmRecallConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = NtmRecallConfig()
        self.cfg = cfg

        input_w = ntm_input_width(cfg.w)
        output_w = ntm_output_width(cfg.n, cfg.w)

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
    clip_grad_norm_(model.parameters(), model.cfg.max_norm)
    optimizer.step()
    model.project_addressing()
    return loss.item()
