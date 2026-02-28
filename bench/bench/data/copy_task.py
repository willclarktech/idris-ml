"""Copy task data generation matching idris-ml's Generate.idr copyTaskPoint.

Input:  symbols ++ blanks  (write phase)
Output: blanks ++ symbols  (read phase)
Symbol 0 = blank, symbols 1..w-1 are data.
"""

import random

import torch
from torch import Tensor


def copy_task_point(symbols: list[int], w: int) -> tuple[list[Tensor], list[Tensor]]:
    """Generate a single copy task data point.

    Matches Generate.idr copyTaskPoint.
    symbols: list of symbol indices (1..w-1)
    w: alphabet size (including blank=0)
    Returns: (inputs, targets) as lists of one-hot tensors
    """
    seq_len = len(symbols)
    blanks = [0] * seq_len

    inp_indices = symbols + blanks
    out_indices = blanks + symbols

    def one_hot(idx: int) -> Tensor:
        v = torch.zeros(w)
        v[idx] = 1.0
        return v

    inputs = [one_hot(i) for i in inp_indices]
    targets = [one_hot(i) for i in out_indices]
    return inputs, targets


def generate_copy_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    w: int,
) -> list[tuple[list[Tensor], list[Tensor]]]:
    """Generate a batch of random copy task data points.

    Matches Generate.randomBatchVect with copyTask.
    """
    batch = []
    for _ in range(batch_size):
        seq_len = random.randint(min_len, max_len)
        symbols = [random.randint(1, w - 1) for _ in range(seq_len)]
        batch.append(copy_task_point(symbols, w))
    return batch
