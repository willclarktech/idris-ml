"""Associative recall data generation matching Graves 2014 / vlgiitr reference.

Each item is a sequence of `seq_len` binary vectors of `seq_width` bits.
Items are separated by item delimiters. The query phase presents a random
non-last item bracketed by query delimiters. The target is the item that
follows the queried item in the original list.

Input width: seq_width + 2 (data channels + item_delim + query_delim)

Sequence structure for num_items items, each of seq_len vectors:
  [item_delim item₁] [item_delim item₂] ... [item_delim itemₙ]
  [query_delim query_item query_delim]

Target: the item following the query item (seq_len vectors of seq_width bits).
Output is produced during the query_item timesteps (seq_len outputs).
"""

import random

import torch
from torch import Tensor


def generate_recall_sequence(
    num_items: int,
    seq_len: int = 3,
    seq_width: int = 6,
) -> tuple[Tensor, Tensor]:
    """Generate a single associative recall sequence.

    Args:
        num_items: Number of items to present (must be >= 2).
        seq_len: Vectors per item.
        seq_width: Bits per vector.

    Returns:
        input_seq: (total_timesteps, seq_width+2)
        target_seq: (seq_len, seq_width) — the item following the query
    """
    input_width = seq_width + 2
    item_delim_ch = seq_width  # channel index for item delimiter
    query_delim_ch = seq_width + 1  # channel index for query delimiter

    # Generate random binary items
    items = [torch.bernoulli(torch.full((seq_len, seq_width), 0.5)) for _ in range(num_items)]

    # Build input sequence: [item_delim item₁] [item_delim item₂] ...
    input_rows: list[Tensor] = []
    for item in items:
        # Item delimiter row
        delim = torch.zeros(1, input_width)
        delim[0, item_delim_ch] = 1.0
        input_rows.append(delim)

        # Item data rows (data in first seq_width channels)
        item_rows = torch.zeros(seq_len, input_width)
        item_rows[:, :seq_width] = item
        input_rows.append(item_rows)

    # Choose query: random item that is NOT the last one
    query_idx = random.randint(0, num_items - 2)
    query_item = items[query_idx]
    target_item = items[query_idx + 1]  # the item after the query

    # Query phase: [query_delim] [query_item] [query_delim]
    qd1 = torch.zeros(1, input_width)
    qd1[0, query_delim_ch] = 1.0
    input_rows.append(qd1)

    query_rows = torch.zeros(seq_len, input_width)
    query_rows[:, :seq_width] = query_item
    input_rows.append(query_rows)

    qd2 = torch.zeros(1, input_width)
    qd2[0, query_delim_ch] = 1.0
    input_rows.append(qd2)

    input_seq = torch.cat(input_rows, dim=0)
    target_seq = target_item  # (seq_len, seq_width)

    return input_seq, target_seq


def generate_recall_batch(
    batch_size: int,
    min_items: int,
    max_items: int,
    seq_len: int = 3,
    seq_width: int = 6,
) -> list[tuple[Tensor, Tensor]]:
    """Generate a batch of associative recall sequences."""
    batch = []
    for _ in range(batch_size):
        num_items = random.randint(min_items, max_items)
        batch.append(generate_recall_sequence(num_items, seq_len, seq_width))
    return batch
