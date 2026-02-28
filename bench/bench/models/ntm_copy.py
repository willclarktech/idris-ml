"""NTM copy task model matching idris-ml's Example/NtmCopy.idr.

Linear controller + tanh activation:
  Linear(NtmInputWidth→H) → Tanh → Linear(H→NtmOutputWidth) → [NTM] → LogSoftmax
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from bench.ntm.ntm_layer import NTMLayer, ntm_input_width, ntm_output_width
from bench.training.losses import nll_loss


@dataclass
class NtmCopyConfig:
    w: int = 3
    n: int = 10
    h: int = 20
    batch_size: int = 16
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_norm: float = 50.0
    div_final: float = 10.0
    epochs: int = 6000
    patience: int = 200
    chunk_size: int = 100


class NtmCopyModel(nn.Module):
    """NTM model for copy task with linear controller + tanh."""

    def __init__(self, cfg: NtmCopyConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = NtmCopyConfig()
        self.cfg = cfg

        input_w = ntm_input_width(cfg.w)
        output_w = ntm_output_width(cfg.n, cfg.w)

        # Controller: Linear → Tanh → Linear
        controller = nn.Sequential(
            nn.Linear(input_w, cfg.h),
            nn.Tanh(),
            nn.Linear(cfg.h, output_w),
        )
        # Xavier uniform init, zero bias
        for m in controller.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ntm = NTMLayer(controller, cfg.n, cfg.w)

    def reset_state(self) -> None:
        self.ntm.reset_state()

    def forward(self, x: Tensor) -> Tensor:
        """Forward one timestep, returns log-softmax output."""
        raw = self.ntm(x)
        return torch.log_softmax(raw, dim=-1)

    def project_addressing(self) -> None:
        self.ntm.project_addressing()


def train_ntm_copy_step(
    model: NtmCopyModel,
    data: list[tuple[list[Tensor], list[Tensor]]],
    loss_fn: object,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch on NTM copy data.

    Matches Backprop.idr epochRecurrent.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    for xs, ys in data:
        model.reset_state()
        seq_loss = torch.tensor(0.0)
        for x, y in zip(xs, ys):
            pred = model(x)
            seq_loss = seq_loss + nll_loss(pred, y)
        total_loss = total_loss + seq_loss / len(xs)
    loss = total_loss / len(data)
    loss.backward()
    clip_grad_norm_(model.parameters(), model.cfg.max_norm)
    optimizer.step()
    model.project_addressing()
    return loss.item()
