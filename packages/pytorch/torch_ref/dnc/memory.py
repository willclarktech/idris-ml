"""DNC memory read/write operations.

Erase+add write (unlike NTM's interpolation write):
  M' = M * (1 - outer(w, e)) + outer(w, a)

Read is the same as NTM: weighted sum of memory rows.
"""

from torch import Tensor


def read_op(weights: Tensor, memory: Tensor) -> Tensor:
    """Weighted sum of memory rows.

    weights: [n]
    memory: [n, m]
    Returns: [m]
    """
    return (weights.unsqueeze(-1) * memory).sum(dim=0)


def erase_add_write(
    memory: Tensor,
    write_weights: Tensor,
    erase_vector: Tensor,
    add_vector: Tensor,
) -> Tensor:
    """Erase+add memory write.

    M' = M * (1 - outer(w, e)) + outer(w, a)

    memory: [n, m]
    write_weights: [n]
    erase_vector: [m] (sigmoid-activated, values in [0,1])
    add_vector: [m]
    Returns: [n, m]
    """
    # outer(w, e) -> [n, m]
    erase_gate = write_weights.unsqueeze(-1) * erase_vector.unsqueeze(0)
    # outer(w, a) -> [n, m]
    add_gate = write_weights.unsqueeze(-1) * add_vector.unsqueeze(0)

    return memory * (1 - erase_gate) + add_gate
