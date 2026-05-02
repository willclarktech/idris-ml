"""Copy task data generation matching loudinthecloud/pytorch-ntm reference.

Input:  [random_bits x seq_len] [delimiter]   — shape (seq_len+1, seq_width+1)
Target: [random_bits x seq_len]               — shape (seq_len, seq_width)

Binary vectors are seq_width bits wide. The delimiter is a 1 in an extra
channel (index seq_width), with zeros in the data channels.
"""

import random

import torch
from torch import Tensor

from torch_ref.training.runner import get_device


def generate_copy_sequence(
    seq_len: int,
    seq_width: int = 8,
) -> tuple[Tensor, Tensor]:
    """Generate a single copy task sequence.

    Returns:
        input_seq: (seq_len+1, seq_width+1) — data rows + delimiter row
        target_seq: (seq_len, seq_width) — the binary vectors to reproduce
    """
    device = get_device()
    # Random binary vectors
    data = torch.bernoulli(torch.full((seq_len, seq_width), 0.5, device=device))

    # Input: data with extra delimiter channel (0 during data, 1 at delimiter)
    input_data = torch.zeros(seq_len, seq_width + 1, device=device)
    input_data[:, :seq_width] = data

    delimiter = torch.zeros(1, seq_width + 1, device=device)
    delimiter[0, seq_width] = 1.0  # delimiter channel

    input_seq = torch.cat([input_data, delimiter], dim=0)  # (seq_len+1, seq_width+1)
    target_seq = data  # (seq_len, seq_width)

    return input_seq, target_seq


def generate_copy_batch(
    batch_size: int,
    seq_min: int,
    seq_max: int,
    seq_width: int = 8,
) -> list[tuple[Tensor, Tensor]]:
    """Generate a batch of copy task sequences with random lengths.

    Each element is (input_seq, target_seq) with varying seq_len.
    """
    batch = []
    for _ in range(batch_size):
        seq_len = random.randint(seq_min, seq_max)
        batch.append(generate_copy_sequence(seq_len, seq_width))
    return batch
