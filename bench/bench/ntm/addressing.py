"""NTM addressing operations matching idris-ml's Memory.idr.

NOTE: Add vectors use 2*sigmoid(2*x)-1 instead of plain tanh, matching
idris-ml's Math.tanh definition: tanh x = 2 * sigmoid(2*x) - 1.
"""

import torch.nn.functional as F
from torch import Tensor


def cosine_similarity(key: Tensor, row: Tensor, eps: float = 1e-6) -> Tensor:
    """Cosine similarity between key and row vectors.

    Matches Math.idr cosineSimilarity with l2Norm eps=1e-6.
    """
    dot = (key * row).sum(dim=-1)
    norm_key = key.norm(dim=-1).clamp(min=eps)
    norm_row = row.norm(dim=-1).clamp(min=eps)
    return dot / (norm_key * norm_row)


def content_address(beta: Tensor, memory: Tensor, key: Tensor) -> Tensor:
    """Content-based addressing.

    Matches Memory.idr getContentAddress: beta * cosine_sim then softmax.
    memory: (n, w)
    key: (w,)
    beta: scalar
    Returns: (n,) addressing weights
    """
    sims = cosine_similarity(key.unsqueeze(0), memory)  # (n,)
    return F.softmax(beta * sims, dim=-1)


def interpolate(g: Tensor, content_w: Tensor, location_w: Tensor) -> Tensor:
    """Interpolate between content and location addressing.

    Matches Memory.idr interpolate: c*g + l*(1-g).
    """
    return content_w * g + location_w * (1 - g)


def shift(weights: Tensor, kernel: Tensor) -> Tensor:
    """Circular convolution with 3-element shift kernel.

    Matches Memory.idr shift: kernel[0]*roll(w,+1) + kernel[1]*w + kernel[2]*roll(w,-1).

    Memory.idr's cycleForward semantics:
    - kernel[0] (sl) multiplies cycleForward(1, ws) which shifts indices forward by 1
      (element at position i comes from position i+1), equivalent to roll(w, -1)
    - kernel[1] (ss) multiplies ws (stay)
    - kernel[2] (sr) multiplies cycleForward(last, ws) which shifts indices forward by n-1
      (element at position i comes from position i-1), equivalent to roll(w, +1)
    """
    n = weights.shape[-1]
    if n <= 1:
        return weights
    sk = F.softmax(kernel, dim=-1)
    sl, ss, sr = sk[0], sk[1], sk[2]
    return sl * weights.roll(-1) + ss * weights + sr * weights.roll(1)


def focus(gamma: Tensor, weights: Tensor) -> Tensor:
    """Sharpen addressing weights.

    Matches Memory.idr focus: w^gamma / sum(w^gamma).
    """
    raised = weights.pow(gamma)
    return raised / raised.sum(dim=-1, keepdim=True)
