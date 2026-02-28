"""Loss functions matching idris-ml's Math.idr.

NOTE: We do NOT use nn.CrossEntropyLoss or nn.NLLLoss because idris-ml's
reduction semantics differ: idris-ml uses reduceLoss which computes
mean(pointwise(pred, target)) per sample, then the training loop averages
across samples. PyTorch's built-in losses use different reduction patterns.

NOTE: idris-ml's nllLoss takes soft target vectors (not class indices),
computing -(target * logprob).mean(). PyTorch's nn.NLLLoss takes class
indices, so we implement our own.
"""

import torch
from torch import Tensor


def cross_entropy(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy with explicit softmax output, clamped log.

    Matches Math.idr crossEntropy: clampedLoss with eps=1e-6.
    predictions: softmax output probabilities
    targets: one-hot or soft targets
    """
    eps = 1e-6
    pp = predictions.clamp(eps, 1 - eps)
    pointwise = -(targets * torch.log(pp)) + -(1 - targets) * torch.log(1 - pp)
    return pointwise.mean()


def nll_loss(log_probs: Tensor, targets: Tensor) -> Tensor:
    """Negative log-likelihood for logSoftmax outputs.

    Matches Math.idr nllLoss: -(target * logprob), mean reduction.
    log_probs: logSoftmax output
    targets: one-hot or soft targets
    """
    return -(targets * log_probs).mean()


def weighted_nll_loss(log_probs: Tensor, targets: Tensor, weight: float = 3.0) -> Tensor:
    """NLL loss weighting non-blank timesteps more heavily.

    Matches NtmAssociativeRecall.idr weightedNllLoss.
    Blank = target[0] >= 0.5 (symbol 0 is blank).
    """
    base = -(targets * log_probs).mean(dim=-1)
    is_blank = targets[..., 0] >= 0.5
    weights = torch.where(is_blank, torch.ones_like(base), torch.full_like(base, weight))
    return (base * weights).mean()


def bce_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy with logits.

    Matches Math.idr binaryCrossEntropyWithLogits: applies sigmoid then BCE.
    Clamps sigmoid output to avoid log(0).
    """
    eps = 1e-6
    sigp = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pointwise = -(targets * torch.log(sigp) + (1 - targets) * torch.log(1 - sigp))
    return pointwise.mean()
