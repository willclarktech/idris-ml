"""Unit tests for the shared memory-task accuracy metric.

The headline property: per-sequence accuracy is strictly more sensitive than
per-bit accuracy — a single wrong bit per sequence leaves per-bit near 1.0 but
drives per-sequence to 0.0.
"""

import torch

from torch_ref.metrics import bit_and_sequence_accuracy


def test_sequence_accuracy_distinguishes_from_bit_accuracy() -> None:
    # 10 sequences of width 10; flip exactly one bit in each prediction.
    # per-bit: 9/10 correct per sequence -> 0.9. per-sequence: every sequence
    # has a mismatch -> 0.0.
    targets = [torch.ones(1, 10) for _ in range(10)]
    preds: list[torch.Tensor] = []
    for _ in range(10):
        p = torch.ones(1, 10)
        p[0, 0] = 0.0
        preds.append(p)
    bit_acc, seq_acc = bit_and_sequence_accuracy(preds, targets)
    assert bit_acc == 0.9
    assert seq_acc == 0.0


def test_all_correct_is_one_on_both() -> None:
    targets = [torch.ones(2, 8) for _ in range(5)]
    preds = [t.clone() for t in targets]
    bit_acc, seq_acc = bit_and_sequence_accuracy(preds, targets)
    assert bit_acc == 1.0
    assert seq_acc == 1.0


def test_some_sequences_fully_correct() -> None:
    # 4 sequences; 2 fully correct, 2 with one wrong bit -> seq 0.5.
    targets = [torch.ones(1, 4) for _ in range(4)]
    good = torch.ones(1, 4)
    bad = torch.ones(1, 4)
    bad[0, 0] = 0.0
    preds = [good.clone(), bad.clone(), good.clone(), bad.clone()]
    _, seq_acc = bit_and_sequence_accuracy(preds, targets)
    assert seq_acc == 0.5
