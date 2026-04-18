"""MNIST CNN (LeNet-style).

Simple convolutional neural network for MNIST digit classification.
Architecture matches the Idris implementation:
  Conv2d(1->16, k=5) -> ReLU -> MaxPool(2) ->
  Conv2d(16->32, k=5) -> ReLU -> MaxPool(2) ->
  Linear(512->10) -> LogSoftmax

Uses torchvision for data loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MnistCNN(nn.Module):
    """LeNet-style CNN matching the Idris Layer.Conv architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, bias=True)
        self.fc = nn.Linear(512, 10)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, 28, 28]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # [B, 16, 12, 12]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # [B, 32, 4, 4]
        x = x.view(x.size(0), -1)  # [B, 512]
        return F.log_softmax(self.fc(x), dim=1)  # [B, 10]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "data/mnist",
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST from torchvision, returns (train_loader, test_loader)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: MnistCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        count += data.size(0)
    return total_loss / count


def evaluate(model: MnistCNN, loader: DataLoader) -> tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            count += data.size(0)
    return total_loss / count, correct / count
