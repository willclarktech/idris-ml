"""Per-bit and per-sequence accuracy for the memory tasks (copy / recall).

Pure metric over thresholded predictions vs targets — no model coupling, so it
is unit-testable directly. A sequence counts as correct only when *every* bit
matches: per-bit accuracy can sit near 1.0 while many whole sequences are still
wrong, so per-sequence accuracy is the stricter, more sensitive signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def bit_and_sequence_accuracy(
    preds: list[torch.Tensor], targets: list[torch.Tensor]
) -> tuple[float, float]:
    """Return (per-bit accuracy, per-sequence accuracy).

    Each ``preds[i]`` / ``targets[i]`` is a ``[seq_len, width]`` tensor of 0/1
    values (predictions already thresholded). Per-bit aggregates over every bit
    of every sequence; per-sequence counts a sequence iff *all* its bits match.
    """
    bit_correct = 0
    bit_total = 0
    seq_correct = 0
    seq_total = 0
    for pred_bits, target in zip(preds, targets, strict=True):
        bit_correct += int((pred_bits == target).sum().item())
        bit_total += int(target.numel())
        seq_correct += int(bool((pred_bits == target).all().item()))
        seq_total += 1
    bit_acc = bit_correct / bit_total if bit_total > 0 else 0.0
    seq_acc = seq_correct / seq_total if seq_total > 0 else 0.0
    return bit_acc, seq_acc
