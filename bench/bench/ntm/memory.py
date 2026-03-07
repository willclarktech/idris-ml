"""NTM read/write head operations."""

import torch
from torch import Tensor

from bench.ntm.addressing import (
    SHIFT_KERNEL_SIZE,
    content_address,
    focus,
    interpolate,
    shift,
)


def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))


def read_op(weights: Tensor, memory: Tensor) -> Tensor:
    """Weighted sum of memory rows.

    weights: (n,)
    memory: (n, w)
    Returns: (w,)
    """
    return (weights.unsqueeze(-1) * memory).sum(dim=0)


def forward_read_head(
    memory: Tensor,
    addressing_weights: Tensor,
    head_input: Tensor,
    w: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Forward pass for read head.

    head_input: (w + 3 + 3) = key(w) + shift(3) + params(3: beta, g, gamma)
    Returns: (new_addressing_weights, read_output, new_addressing_weights)
    """
    main_input = head_input[: w + SHIFT_KERNEL_SIZE]
    params = head_input[w + SHIFT_KERNEL_SIZE :]
    key_vector = main_input[:w]
    shift_vector = main_input[w : w + SHIFT_KERNEL_SIZE]

    beta = softplus(params[0])
    g = torch.sigmoid(params[1])
    gamma = 1 + softplus(params[2])

    content_weights = content_address(beta, memory, key_vector)
    interpolated = interpolate(g, content_weights, addressing_weights)
    shifted = shift(interpolated, shift_vector)
    focused = focus(gamma, shifted)

    output = read_op(focused, memory)
    return focused, output, focused


def write_memory(memory: Tensor, weights: Tensor, add_vector: Tensor) -> Tensor:
    """Interpolation write: w*data + (1-w)*mem.

    No separate erase vector — the write weight itself controls how much of
    the old memory to keep vs replace with the new data.
    """
    return weights.unsqueeze(-1) * add_vector.unsqueeze(0) + (1 - weights.unsqueeze(-1)) * memory


def forward_write_head(
    memory: Tensor,
    addressing_weights: Tensor,
    head_input: Tensor,
    w: int,
) -> tuple[Tensor, Tensor]:
    """Forward pass for write head (interpolation mechanism).

    head_input: (w + 3 + 3 + w) = addressing_params + add(w). No erase vector.
    Returns: (new_addressing_weights, new_memory)
    """
    read_head_size = w + SHIFT_KERNEL_SIZE + 3
    read_head_input = head_input[:read_head_size]
    raw_add = head_input[read_head_size : read_head_size + w]
    add_vector = raw_add

    # Compute new addressing weights (reuse read head logic)
    new_weights, _, _ = forward_read_head(memory, addressing_weights, read_head_input, w)

    # Write to memory via interpolation
    new_memory = write_memory(memory, new_weights, add_vector)

    return new_weights, new_memory
