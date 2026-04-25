"""Loss functions.

NOTE: We do NOT use nn.CrossEntropyLoss or nn.NLLLoss because our reduction
semantics differ: we compute mean(pointwise(pred, target)) per sample, then
average across samples. PyTorch's built-in losses use different reduction.

NOTE: nllLoss takes soft target vectors (not class indices), computing
-(target * logprob).mean(). PyTorch's nn.NLLLoss takes class indices.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy with explicit softmax output, clamped log.

    predictions: softmax output probabilities
    targets: one-hot or soft targets
    """
    eps = 1e-6
    pp = predictions.clamp(eps, 1 - eps)
    pointwise = -(targets * torch.log(pp)) + -(1 - targets) * torch.log(1 - pp)
    return pointwise.mean()


def nll_loss(log_probs: Tensor, targets: Tensor) -> Tensor:
    """Negative log-likelihood for logSoftmax outputs.

    log_probs: logSoftmax output
    targets: one-hot or soft targets
    """
    return -(targets * log_probs).mean()


def weighted_nll_loss(log_probs: Tensor, targets: Tensor, weight: float = 3.0) -> Tensor:
    """NLL loss weighting non-blank timesteps more heavily.

    Blank = target[0] >= 0.5 (symbol 0 is blank).
    """
    base = -(targets * log_probs).mean(dim=-1)
    is_blank = targets[..., 0] >= 0.5
    weights = torch.where(is_blank, torch.ones_like(base), torch.full_like(base, weight))
    return (base * weights).mean()


def bce_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy with logits.

    Uses PyTorch's fused kernel: numerically stable, no clamp needed.
    """
    return F.binary_cross_entropy_with_logits(logits, targets)
