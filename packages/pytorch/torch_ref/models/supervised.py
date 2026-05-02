"""Supervised model: Linear(2->3) outputting raw logits, BCE-with-logits loss.

Xavier uniform init, zero bias. Matches Idris Example/Bench.idr (and
Example/Supervised.idr) — raw logits in the model, loss applies the
log_softmax-or-sigmoid transform with numerical stability.
"""

import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.training.losses import bce_with_logits


class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


SUPERVISED_DATA = [
    (torch.tensor([1.5, -2.7]), torch.tensor([0.0, 1.0, 0.0])),
    (torch.tensor([-3.2, 4.1]), torch.tensor([0.0, 1.0, 0.0])),
    (torch.tensor([5.7, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
    (torch.tensor([-1.3, 8.8]), torch.tensor([0.0, 1.0, 0.0])),
    (torch.tensor([2.9, -1.4]), torch.tensor([1.0, 0.0, 0.0])),
]


def train_supervised_epoch(
    model: SupervisedModel,
    data: list[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch, return loss value.

    Forward all samples, compute mean loss, backward, step.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    for x, y in data:
        logits = model(x)
        total_loss = total_loss + bce_with_logits(logits, y)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()
