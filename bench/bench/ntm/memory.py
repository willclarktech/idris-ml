"""NTM read/write head operations matching idris-ml's Memory.idr and Layer.idr."""

import torch
from torch import Tensor

from bench.ntm.addressing import content_address, focus, interpolate, shift


def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))


def sig(x: Tensor) -> Tensor:
    """Sigmoid matching Memory.idr sig."""
    return torch.sigmoid(x)


def read_op(weights: Tensor, memory: Tensor) -> Tensor:
    """Weighted sum of memory rows.

    Matches Memory.idr readOp: sum(w[i] * memory[i]).
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

    Matches Memory.idr forwardReadHead.
    head_input: (w + 3 + 3) = key(w) + shift(3) + params(3: beta, g, gamma)
    Returns: (new_addressing_weights, read_output, new_addressing_weights)
    """
    shift_kernel_size = 3
    main_input = head_input[: w + shift_kernel_size]
    params = head_input[w + shift_kernel_size :]
    key_vector = main_input[:w]
    shift_vector = main_input[w : w + shift_kernel_size]

    beta = softplus(params[0])
    g = sig(params[1])
    gamma = 1 + softplus(params[2])

    content_weights = content_address(beta, memory, key_vector)
    interpolated = interpolate(g, content_weights, addressing_weights)
    shifted = shift(interpolated, shift_vector)
    focused = focus(gamma, shifted)

    output = read_op(focused, memory)
    return focused, output, focused


def erase_memory(memory: Tensor, weights: Tensor, erase_vector: Tensor) -> Tensor:
    """Erase from memory.

    Matches Memory.idr eraseMemory: memory * (1 - w[i] * erase[j]).
    """
    erase_matrix = weights.unsqueeze(-1) * erase_vector.unsqueeze(0)
    return memory * (1 - erase_matrix)


def add_memory(memory: Tensor, weights: Tensor, add_vector: Tensor) -> Tensor:
    """Add to memory.

    Matches Memory.idr addMemory: memory + w[i] * add[j].
    """
    return memory + weights.unsqueeze(-1) * add_vector.unsqueeze(0)


def forward_write_head(
    memory: Tensor,
    addressing_weights: Tensor,
    head_input: Tensor,
    w: int,
) -> tuple[Tensor, Tensor]:
    """Forward pass for write head.

    Matches Memory.idr forwardWriteHead.
    head_input: (w + 3 + 3 + w + w) = read_head_input + erase(w) + add(w)
    Returns: (new_addressing_weights, new_memory)
    """
    shift_kernel_size = 3
    read_head_size = w + shift_kernel_size + 3
    read_head_input = head_input[:read_head_size]
    raw_erase = head_input[read_head_size : read_head_size + w]
    raw_add = head_input[read_head_size + w : read_head_size + 2 * w]

    erase_vector = sig(raw_erase)
    # NOTE: Reference impls (loudinthecloud, vlgiitr) use raw linear add vectors.
    # idris-ml uses 2*sigmoid(2*x)-1 for add vectors.
    add_vector = raw_add

    # Compute new addressing weights (reuse read head logic)
    new_weights, _, _ = forward_read_head(memory, addressing_weights, read_head_input, w)

    # Write to memory
    erased = erase_memory(memory, new_weights, erase_vector)
    new_memory = add_memory(erased, new_weights, add_vector)

    return new_weights, new_memory


def tanh_bound(memory: Tensor) -> Tensor:
    """Bound memory values to [-1, 1] via tanh.

    Matches Layer.idr tanhBound applied after every write.
    """
    return torch.tanh(memory)
