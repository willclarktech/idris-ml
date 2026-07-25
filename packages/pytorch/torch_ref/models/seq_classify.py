"""1D sequence classification with Conv1D.

Classify synthetic waveforms (sine, square, triangle) using a small CNN.
Architecture matches the Idris implementation with small dims to avoid
Idris 2 Peano Nat type-checker limitations.

Input: [1, 32] (single channel, 32 timesteps)
Conv1d(1->4, k=3) -> ReLU -> MaxPool1d(2) ->
Conv1d(4->8, k=3) -> ReLU -> MaxPool1d(2) ->
Dropout(0.5) -> Linear(48->3) -> LogSoftmax
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.init import init_linear_
from torch_ref.training.runner import get_device, get_dtype

SEQ_LEN = 32
NUM_CLASSES = 3


class SeqClassifyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(48, NUM_CLASSES)
        init_linear_(self)

    def forward(self, x: Tensor) -> Tensor:
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)  # [B,4,15]
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)  # [B,8,6]
        x = x.view(x.size(0), -1)  # [B,48]
        x = self.dropout(x)
        return F.log_softmax(self.fc(x), dim=1)


def generate_waveform(label: int, length: int = SEQ_LEN) -> list[float]:
    """Generate a waveform: 0=sine, 1=square, 2=triangle."""
    freq = random.uniform(1.0, 3.0)
    phase = random.uniform(0, 2 * math.pi)
    noise = 0.1
    result: list[float] = []
    for i in range(length):
        t = i / length * 2 * math.pi * freq + phase
        if label == 0:  # sine
            val = math.sin(t)
        elif label == 1:  # square
            val = 1.0 if math.sin(t) > 0 else -1.0
        else:  # triangle
            val = 2.0 * abs(2.0 * (t / (2 * math.pi) - math.floor(t / (2 * math.pi) + 0.5))) - 1.0
        result.append(val + random.gauss(0, noise))
    return result


def generate_batch(
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Generate a batch of (input [B,1,SEQ_LEN], labels [B])."""
    inputs: list[list[float]] = []
    labels: list[int] = []
    for _ in range(batch_size):
        label = random.randint(0, NUM_CLASSES - 1)
        wave = generate_waveform(label)
        inputs.append(wave)
        labels.append(label)
    device = get_device()
    return (
        torch.tensor(inputs, dtype=get_dtype(), device=device).unsqueeze(1),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def train_epoch(
    model: SeqClassifyCNN,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
) -> float:
    model.train()
    data, target = generate_batch(batch_size)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    # torch's Tensor.backward stub leaves its params unannotated.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()
    return loss.item()


def evaluate(model: SeqClassifyCNN, n_samples: int = 200) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        data, target = generate_batch(n_samples)
        output = model(data)
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
    return correct / n_samples
