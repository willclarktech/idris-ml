"""DNC addressing operations.

Content-based addressing (shared with NTM), plus DNC-specific:
- Usage-based allocation weighting
- Temporal link matrix update
- Multi-mode read weighting (backward + content + forward)

Reference: Graves et al. 2016, "Hybrid computing using a neural network
with dynamic external memory", Nature 538.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Content-based addressing (same as NTM)
# ---------------------------------------------------------------------------


def cosine_similarity(key: Tensor, memory: Tensor, eps: float = 1e-6) -> Tensor:
    """Cosine similarity between key [m] and each memory row [n, m] -> [n]."""
    dot = (key.unsqueeze(0) * memory).sum(dim=-1)
    norm_key = key.norm().clamp(min=eps)
    norm_rows = memory.norm(dim=-1).clamp(min=eps)
    return dot / (norm_key * norm_rows)


def content_address(beta: Tensor, memory: Tensor, key: Tensor) -> Tensor:
    """Content-based addressing: softmax(beta * cosine_sim(key, M)).

    beta: scalar (key strength)
    memory: [n, m]
    key: [m]
    Returns: [n] addressing weights
    """
    sims = cosine_similarity(key, memory)
    return F.softmax(beta * sims, dim=-1)


# ---------------------------------------------------------------------------
# Usage and allocation
# ---------------------------------------------------------------------------


def update_usage(
    prev_usage: Tensor,
    prev_write_weights: Tensor,
    free_gates: Tensor,
    prev_read_weights: list[Tensor],
) -> Tensor:
    """Update memory usage vector.

    u_t = (u_{t-1} + w^w_{t-1} - u_{t-1} * w^w_{t-1}) * retention
    retention = prod_j(1 - f^j_t * w^{r,j}_{t-1})

    prev_usage: [n]
    prev_write_weights: [n]
    free_gates: [R] (sigmoid-activated)
    prev_read_weights: list of R tensors, each [n]
    Returns: [n]
    """
    # Write usage: u + w - u*w = 1 - (1-u)*(1-w)
    write_usage = prev_usage + prev_write_weights - prev_usage * prev_write_weights

    # Retention: product over read heads of (1 - f_j * w^r_j)
    retention = torch.ones_like(prev_usage)
    for j, rw in enumerate(prev_read_weights):
        retention = retention * (1 - free_gates[j] * rw)
    # Clamp retention to prevent usage zeroing
    retention = retention.clamp(min=1e-10)

    return write_usage * retention


def allocation_weighting(usage: Tensor) -> Tensor:
    """Compute allocation weights from usage vector.

    Sort usage ascending, compute:
      a[phi[j]] = (1 - u[phi[j]]) * prod_{i=1}^{j-1} u[phi[i]]

    usage: [n]
    Returns: [n] allocation weights (sum <= 1)
    """
    n = usage.shape[0]

    # Sort usage ascending, clamp to prevent cumprod underflow
    sorted_usage, sorted_indices = torch.sort(usage, dim=0)
    sorted_usage = sorted_usage.clamp(min=1e-6)

    # Cumulative product of sorted usage (shifted by 1: first element is 1.0)
    # cumprod_term[j] = prod_{i=0}^{j-1} sorted_usage[i]
    cumprod_vals = torch.cumprod(sorted_usage, dim=0)
    # Shift right: [1, u[0], u[0]*u[1], ...]
    shifted_cumprod = torch.cat([torch.ones(1), cumprod_vals[:-1]])

    # Allocation in sorted order
    sorted_alloc = (1 - sorted_usage) * shifted_cumprod

    # Unsort: scatter back to original positions
    alloc = torch.zeros(n)
    alloc.scatter_(0, sorted_indices, sorted_alloc)

    return alloc


def write_weighting(
    content_w: Tensor,
    alloc_w: Tensor,
    write_gate: Tensor,
    alloc_gate: Tensor,
) -> Tensor:
    """Compute write weighting.

    w^w_t = g^w * [g^a * a_t + (1-g^a) * c^w_t]

    content_w: [n] content-based write weights
    alloc_w: [n] allocation weights
    write_gate: scalar (sigmoid)
    alloc_gate: scalar (sigmoid)
    Returns: [n]
    """
    return write_gate * (alloc_gate * alloc_w + (1 - alloc_gate) * content_w)


# ---------------------------------------------------------------------------
# Temporal link matrix
# ---------------------------------------------------------------------------


def update_link_matrix(
    prev_link: Tensor,
    prev_precedence: Tensor,
    write_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Update temporal link matrix and precedence weights.

    L'[i,j] = (1 - w^w[i] - w^w[j]) * L[i,j] + w^w[i] * p[j]
    L'[i,i] = 0
    p' = (1 - sum(w^w)) * p + w^w

    prev_link: [n, n]
    prev_precedence: [n]
    write_weights: [n]
    Returns: (new_link [n,n], new_precedence [n])
    """
    # Precedence update
    new_precedence = (1 - write_weights.sum()) * prev_precedence + write_weights

    # Link matrix update
    # (1 - w_i - w_j) * L[i,j] + w_i * p_j
    w_i = write_weights.unsqueeze(1)  # [n, 1]
    w_j = write_weights.unsqueeze(0)  # [1, n]
    p_j = prev_precedence.unsqueeze(0)  # [1, n]

    # Clamp decay to [0, inf) — prevents negative decay when w_i + w_j > 1
    decay = (1 - w_i - w_j).clamp(min=0.0)
    new_link = decay * prev_link + w_i * p_j

    # Zero diagonal and clamp entries non-negative
    new_link = new_link * (1 - torch.eye(new_link.shape[0]))
    new_link = new_link.clamp(min=0.0)

    return new_link, new_precedence


# ---------------------------------------------------------------------------
# Read addressing
# ---------------------------------------------------------------------------


def read_weighting(
    link_matrix: Tensor,
    prev_read_weights: Tensor,
    content_w: Tensor,
    mode_params: Tensor,
) -> Tensor:
    """Compute read weighting for one read head.

    pi = softmax(mode_params)  -- [3]: backward, content, forward
    forward_w = L @ prev_read_weights
    backward_w = L^T @ prev_read_weights
    w^r = pi[0]*backward + pi[1]*content + pi[2]*forward

    link_matrix: [n, n]
    prev_read_weights: [n]
    content_w: [n]
    mode_params: [3]
    Returns: [n]
    """
    pi = F.softmax(mode_params, dim=0)

    forward_w = link_matrix @ prev_read_weights
    backward_w = link_matrix.t() @ prev_read_weights

    result = pi[0] * backward_w + pi[1] * content_w + pi[2] * forward_w
    # Clamp and normalize read weights
    result = result.clamp(min=1e-10)
    return result / (result.sum() + 1e-10)
