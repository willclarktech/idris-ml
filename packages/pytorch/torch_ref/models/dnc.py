"""DNC model for sequence-to-sequence tasks.

LSTM controller (hidden=100) -> separate head FCs -> DNCLayer -> logit output.
Loss: BCEWithLogitsLoss. Optimizer: RMSprop lr=1e-4. Value clip [-10,10].
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from torch_ref.dnc.layer import DNCLayer


@dataclass
class DncConfig:
    input_width: int  # seq_width+1 for copy, seq_width+2 for recall
    output_width: int  # seq_width for both
    n: int = 128  # memory slots
    m: int = 20  # memory width
    num_reads: int = 1  # number of read heads
    controller_size: int = 100  # LSTM hidden size
    lr: float = 1e-4
    clip_value: float = 10.0


class DncModel(nn.Module):
    """DNC model for sequence-to-sequence tasks."""

    def __init__(self, cfg: DncConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dnc = DNCLayer(
            n=cfg.n,
            m=cfg.m,
            num_reads=cfg.num_reads,
            num_inputs=cfg.input_width,
            num_outputs=cfg.output_width,
            controller_hidden_size=cfg.controller_size,
        )

    def reset_state(self) -> None:
        self.dnc.reset_state()

    def forward(self, x: Tensor) -> Tensor:
        """Forward one timestep, returns raw logits."""
        return self.dnc(x)


def train_dnc_step(
    model: DncModel,
    input_seq: Tensor,
    target_seq: Tensor,
    optimizer: torch.optim.Optimizer,
    clip_mode: Literal["value", "norm"] = "value",
) -> tuple[float, float]:
    """Train one sequence (two-phase: encode then decode).

    input_seq: (total_timesteps, input_width)
    target_seq: (output_len, output_width)
    Returns: (loss, bit_error_count)
    """
    optimizer.zero_grad()
    model.reset_state()

    seq_len = target_seq.shape[0]
    num_inputs = input_seq.shape[1]

    # Encoding phase: feed entire input sequence, discard outputs
    for t in range(input_seq.shape[0]):
        model(input_seq[t])

    # Output phase: feed zeros, collect seq_len outputs
    zero_input = torch.zeros(num_inputs)
    outputs = []
    for _ in range(seq_len):
        out = model(zero_input)
        outputs.append(out)

    pred = torch.stack(outputs)  # (seq_len, output_width) -- raw logits
    loss = F.binary_cross_entropy_with_logits(pred, target_seq)

    loss.backward()
    if clip_mode == "norm":
        clip_grad_norm_(model.parameters(), model.cfg.clip_value)
    else:
        clip_grad_value_(model.parameters(), model.cfg.clip_value)
    optimizer.step()

    # Bit error
    with torch.no_grad():
        pred_bits = (torch.sigmoid(pred) >= 0.5).float()
        bit_error = torch.sum(torch.abs(pred_bits - target_seq)).item()

    return loss.item(), bit_error
