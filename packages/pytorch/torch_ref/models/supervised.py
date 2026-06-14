"""Supervised model: Linear(2->3) outputting raw logits, multiclass NLL loss.

Xavier uniform init, zero bias. Matches Idris Example/Bench.idr (and
Example/Supervised.idr) — raw logits in the model, loss applies log_softmax
then negative-log-likelihood. The task is argmax over 3 mutually-exclusive
classes, so multiclass NLL (softmax-coupled) is the correct loss, matching
Idris's `tnllLoss`; the earlier `bce_with_logits` modelled the classes as
independent binaries (the wrong tool — corrected on both sides 2026-06-14).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.losses import nll_loss
from torch_ref.training.runner import get_device, get_dtype


class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def _make_supervised_data() -> list[tuple[Tensor, Tensor]]:
    """Build the (5, 2) supervised dataset on the active device/dtype."""
    device, dtype = get_device(), get_dtype()
    raw = [
        ([1.5, -2.7], [0.0, 1.0, 0.0]),
        ([-3.2, 4.1], [0.0, 1.0, 0.0]),
        ([5.7, 0.0], [0.0, 0.0, 1.0]),
        ([-1.3, 8.8], [0.0, 1.0, 0.0]),
        ([2.9, -1.4], [1.0, 0.0, 0.0]),
    ]
    return [
        (torch.tensor(x, device=device, dtype=dtype), torch.tensor(y, device=device, dtype=dtype))
        for x, y in raw
    ]


# Module-level constant — built at import time under the default
# device (cpu). Scripts that switch to MPS should call
# `_make_supervised_data()` after `set_device`.
SUPERVISED_DATA = _make_supervised_data()


def train_supervised_epoch(
    model: SupervisedModel,
    data: list[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch, return loss value.

    Forward all samples, compute mean loss, backward, step.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=get_device())
    for x, y in data:
        logits = model(x)
        total_loss = total_loss + nll_loss(F.log_softmax(logits, dim=-1), y)
    loss = total_loss / len(data)
    # torch's Tensor.backward stub leaves its params unannotated.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()
    return loss.item()
