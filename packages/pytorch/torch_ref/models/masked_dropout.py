"""Inverted dropout with the mask materialized as data.

The ONNX `Dropout` / torch `native_dropout` shape: the Bernoulli keep-mask
is ordinary data rather than kernel-internal state, so a recorder can log
it and a replay (the Idris side's `--replay` mask channel) can reproduce
the exact drop pattern. The distribution matches `nn.Dropout` — drop with
probability p, scale survivors by 1/(1-p), identity in eval mode — but the
RNG stream differs (`rand_like` vs `bernoulli_`), so swapping this in
changes which masks a seeded run draws.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class MaskedDropout(nn.Module):
    """Drop-in `nn.Dropout` replacement (no parameters, no state-dict
    impact). When `recorder` is set, each training-mode forward appends its
    keep-bits as a '0'/'1' string in element order (1 = kept) — exactly the
    replay file's `mask` line format (`torch_ref.replay.write_replay`)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        self.recorder: list[str] | None = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = torch.rand_like(x) >= self.p
        if self.recorder is not None:
            # torch stub: Tensor.tolist() returns list[Unknown].
            bits = cast("list[int]", keep.flatten().to(torch.int8).tolist())  # pyright: ignore[reportUnknownMemberType]
            self.recorder.append("".join("1" if b else "0" for b in bits))
        return x * keep * (1.0 / (1.0 - self.p))
