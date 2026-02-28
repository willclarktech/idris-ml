"""NTM associative recall model matching Graves 2014 / vlgiitr reference.

Same NTM architecture as copy task (LSTM controller, N=128, M=20),
just different input/output widths and data format.
Loss: BCELoss. Optimizer: RMSprop lr=1e-4, alpha=0.95, momentum=0.9.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_value_

from bench.models.ntm_copy import LSTMController
from bench.ntm.ntm_layer import NTMLayer


@dataclass
class NtmRecallConfig:
    seq_width: int = 6  # bits per vector within item
    seq_len: int = 3  # vectors per item
    min_items: int = 2
    max_items: int = 6
    n: int = 128  # memory slots
    m: int = 20  # memory width
    controller_size: int = 100  # LSTM hidden size
    lr: float = 1e-4
    iterations: int = 100000
    clip_value: float = 10.0


class NtmRecallModel(nn.Module):
    """NTM model for associative recall."""

    def __init__(self, cfg: NtmRecallConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = NtmRecallConfig()
        self.cfg = cfg

        # Input width: seq_width + 2 (item_delim + query_delim)
        num_inputs = cfg.seq_width + 2
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

    def project_addressing(self) -> None:
        self.ntm.project_addressing()


def train_ntm_recall_step(
    model: NtmRecallModel,
    input_seq: Tensor,
    target_seq: Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one sequence (batch_size=1).

    input_seq: (total_timesteps, input_width)
    target_seq: (seq_len, seq_width)
    """
    optimizer.zero_grad()
    model.reset_state()

    seq_len = target_seq.shape[0]
    total_timesteps = input_seq.shape[0]
    num_inputs = input_seq.shape[1]

    # Encoding phase: all timesteps except last seq_len
    encode_len = total_timesteps - seq_len
    for t in range(encode_len):
        model(input_seq[t])

    # Output phase: last seq_len timesteps, collect outputs
    outputs = []
    for t in range(encode_len, total_timesteps):
        out = model(input_seq[t])
        outputs.append(out)

    # If not enough output timesteps (only query_delim left), feed zeros
    while len(outputs) < seq_len:
        out = model(torch.zeros(num_inputs))
        outputs.append(out)

    pred = torch.stack(outputs[:seq_len])  # (seq_len, seq_width)
    loss = nn.functional.binary_cross_entropy(pred, target_seq)

    loss.backward()
    clip_grad_value_(model.parameters(), model.cfg.clip_value)
    optimizer.step()
    model.project_addressing()

    return loss.item()
