"""MNIST CNN (LeNet-style).

Simple convolutional neural network for MNIST digit classification.
Architecture matches the Idris implementation
(packages/idris-ml-examples/src/Example/Mnist.idr):
  Conv2d(1->16, k=5) -> ReLU -> MaxPool(2) ->
  Conv2d(16->32, k=5) -> ReLU -> MaxPool(2) -> Dropout(0.5) ->
  Linear(512->10) -> LogSoftmax

Uses torchvision for data loading.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Runtime imports (not TYPE_CHECKING): the unquoted Dataset cast below
# evaluates at runtime so vulture can see the usage; torch.utils.data
# is loaded by the DataLoader construction anyway.
from torch.utils.data import DataLoader, Dataset  # noqa: TC002

from torch_ref.init import init_conv_, init_linear_
from torch_ref.training.losses import nll_loss
from torch_ref.models.masked_dropout import MaskedDropout
from torch_ref.training.runner import get_device, get_dtype

# Batch type yielded by the MNIST loaders: (images [B,1,28,28], labels [B]).
MnistBatch = tuple[Tensor, Tensor]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MnistCNN(nn.Module):
    """LeNet-style CNN matching the Idris Layer.Conv architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, bias=True)
        # Explicit-mask twin of nn.Dropout: the step oracle records its
        # keep-bits for the Idris side's replay mask channel.
        self.drop = MaskedDropout(0.5)
        self.fc = nn.Linear(512, 10)
        init_linear_(self)
        init_conv_(self)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, 28, 28]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # [B, 16, 12, 12]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # [B, 32, 4, 4]
        x = self.drop(x.view(x.size(0), -1))  # [B, 512]
        return F.log_softmax(self.fc(x), dim=1)  # [B, 10]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "data/mnist",
    train_count: int = 0,
) -> tuple[DataLoader[MnistBatch], DataLoader[MnistBatch]]:
    """Load MNIST from torchvision, returns (train_loader, test_loader).

    `train_count`: if > 0, cap training set to first N images (used by smoke tests).
    """
    # torchvision ships no py.typed marker, so its datasets are untyped.
    from torchvision import datasets, transforms  # pyright: ignore[reportMissingTypeStubs]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    # MNIST's item type is unknown (no stubs); after ToTensor + default
    # collate the loader yields (image batch, label batch) Tensor pairs.
    train = cast(Dataset[MnistBatch], mnist_train)  # noqa: TC006 - unquoted so vulture sees the import used
    test = cast("Dataset[MnistBatch]", datasets.MNIST(data_dir, train=False, transform=transform))
    if train_count > 0 and train_count < len(mnist_train):
        train = torch.utils.data.Subset(train, range(train_count))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: MnistCNN,
    loader: DataLoader[MnistBatch],
    optimizer: torch.optim.Optimizer,
    clip_norm: float = 1.0,
) -> float:
    """Train one epoch, return average loss.

    Matches the Idris-side `nativeAdamGlobalClip` step: global-norm
    gradient clip at 1.0 applied between backward and optimizer step.
    """
    model.train()
    total_loss = 0.0
    count = 0
    device = get_device()
    for data32, target32 in loader:
        # torchvision yields float32; the reference trains float64 like the
        # Idris side (the 2026-08-01 reference-precision alignment missed
        # this example until its step oracle refused the F32 fixture).
        data, target = data32.to(device, dtype=get_dtype()), target32.to(device)
        optimizer.zero_grad()
        output = model(data)
        # The repo's reference loss convention (torch_ref.training.losses):
        # -(target * logprob).mean() over b*n, matching Idris tnllLossMean.
        # F.nll_loss means over b only — 10x the gradient scale; the step
        # oracle caught this side training a different experiment.
        loss = nll_loss(output, F.one_hot(target, 10).to(output.dtype))
        # torch's Tensor.backward stub leaves its params unannotated.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        count += data.size(0)
    return total_loss / count


def evaluate(model: MnistCNN, loader: DataLoader[MnistBatch]) -> tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    device = get_device()
    with torch.no_grad():
        for data32, target32 in loader:
            data, target = data32.to(device, dtype=get_dtype()), target32.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            count += data.size(0)
    return total_loss / count, correct / count
