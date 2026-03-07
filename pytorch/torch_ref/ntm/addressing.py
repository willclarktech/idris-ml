"""NTM addressing operations: content addressing, interpolation, shift, focus."""

import torch.nn.functional as F
from torch import Tensor

SHIFT_KERNEL_SIZE = 3  # circular convolution kernel: [left, stay, right]
NUM_ADDRESSING_SCALARS = 3  # beta (key strength), g (interpolation gate), gamma (sharpening)


def addressing_params_width(key_size: int) -> int:
    """Width of addressing parameter vector: key + shift kernel + scalars."""
    return key_size + SHIFT_KERNEL_SIZE + NUM_ADDRESSING_SCALARS


def cosine_similarity(key: Tensor, row: Tensor, eps: float = 1e-6) -> Tensor:
    """Cosine similarity between key and row vectors."""
    dot = (key * row).sum(dim=-1)
    norm_key = key.norm(dim=-1).clamp(min=eps)
    norm_row = row.norm(dim=-1).clamp(min=eps)
    return dot / (norm_key * norm_row)


def content_address(beta: Tensor, memory: Tensor, key: Tensor) -> Tensor:
    """Content-based addressing: beta * cosine_sim then softmax.

    memory: (n, w)
    key: (w,)
    beta: scalar
    Returns: (n,) addressing weights
    """
    sims = cosine_similarity(key.unsqueeze(0), memory)  # (n,)
    return F.softmax(beta * sims, dim=-1)


def interpolate(g: Tensor, content_w: Tensor, location_w: Tensor) -> Tensor:
    """Interpolate between content and location addressing: c*g + l*(1-g)."""
    return content_w * g + location_w * (1 - g)


def shift(weights: Tensor, kernel: Tensor) -> Tensor:
    """Circular convolution with shift kernel.

    kernel[0] (sl): roll(w, -1) — shift left
    kernel[1] (ss): w — stay
    kernel[2] (sr): roll(w, +1) — shift right
    """
    n = weights.shape[-1]
    if n <= 1:
        return weights
    sk = F.softmax(kernel, dim=-1)
    sl, ss, sr = sk[0], sk[1], sk[2]
    return sl * weights.roll(-1) + ss * weights + sr * weights.roll(1)


def focus(gamma: Tensor, weights: Tensor) -> Tensor:
    """Sharpen addressing weights: w^gamma / sum(w^gamma)."""
    raised = weights.pow(gamma)
    return raised / raised.sum(dim=-1, keepdim=True)
