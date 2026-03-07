"""Supervised model: Linear(2->3) + softmax with cross-entropy loss.

Xavier uniform init, zero bias.
"""

import torch
import torch.nn as nn
from torch import Tensor

from bench.training.losses import cross_entropy


class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(self.linear(x), dim=-1)


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
        pred = model(x)
        total_loss = total_loss + cross_entropy(pred, y)
    loss = total_loss / len(data)
    loss.backward()
    optimizer.step()
    return loss.item()
