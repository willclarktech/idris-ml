"""Associative recall data generation matching idris-ml's Generate.idr.

Sequence structure for K pairs (4K+1 timesteps):
  Store:  k1 v1 k2 v2 ... kK vK   (2K steps)
  Delim:  blank                     (1 step)
  Query:  q1 blank q2 blank ... qK blank  (2K steps)

Output is blank everywhere except on blank-input timesteps during
the query phase, where the correct value appears.
"""

import random

import torch
from torch import Tensor


def associative_recall_point(
    pairs: list[tuple[int, int]],
    query_order: list[int],
    w: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Generate a single associative recall data point.

    Matches Generate.idr associativeRecallPoint.
    pairs: list of (key, value) symbol indices
    query_order: keys in shuffled query order
    w: alphabet size (including blank=0)
    """
    blank = 0
    lookup = dict(pairs)

    def one_hot(idx: int) -> Tensor:
        v = torch.zeros(w)
        v[idx] = 1.0
        return v

    # Store phase: k1 v1 k2 v2 ...
    store_in = []
    store_out = []
    for key, val in pairs:
        store_in.extend([key, val])
        store_out.extend([blank, blank])

    # Delimiter
    delim_in = [blank]
    delim_out = [blank]

    # Query phase: q1 blank q2 blank ... qK blank
    query_in = []
    query_out = []
    for q in query_order:
        query_in.extend([q, blank])
        query_out.extend([blank, lookup.get(q, blank)])

    inp_indices = store_in + delim_in + query_in
    out_indices = store_out + delim_out + query_out

    inputs = [one_hot(i) for i in inp_indices]
    targets = [one_hot(i) for i in out_indices]
    return inputs, targets


def generate_recall_batch(
    batch_size: int,
    min_k: int,
    max_k: int,
    w: int,
) -> list[tuple[list[Tensor], list[Tensor]]]:
    """Generate a batch of random associative recall data points.

    Matches Generate.randomBatchVect with associativeRecallTask.
    """
    non_blank_symbols = list(range(1, w))
    batch = []
    for _ in range(batch_size):
        k = random.randint(min_k, max_k)
        k = min(k, len(non_blank_symbols))
        shuffled = non_blank_symbols.copy()
        random.shuffle(shuffled)
        keys = shuffled[:k]
        values = [random.randint(1, w - 1) for _ in range(k)]
        pairs = list(zip(keys, values))
        query_keys = keys.copy()
        random.shuffle(query_keys)
        batch.append(associative_recall_point(pairs, query_keys, w))
    return batch
