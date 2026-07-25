"""Shared parameter init for the reference models.

Idris has one dense-layer constructor, `Ml.Nn.Linear.linear`, so every
reference model that maps onto it has to init the same way or the
implementation-vs-reference comparison measures init noise instead of the
implementation. Before 2026-07-29 the references disagreed among themselves:
nine models took `nn.Linear`'s defaults (Kaiming-uniform weight, uniform
bias), `supervised` and `rnn` set Xavier weights and zero biases explicitly,
and `multi_head_transformer` set Xavier weights over a default bias.

The agreed contract, applied by `init_linear_` below and matched by Idris
`linear`:

  * weight ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))
  * bias   = 0

The weight half is `kaiming_uniform_(w, a=sqrt(5))`, which is what
`nn.Linear.reset_parameters` already does: gain = sqrt(2/(1+a**2)) =
sqrt(1/3), and bound = sqrt(3) * gain / sqrt(fan_in) = 1/sqrt(fan_in).
Spelling it out keeps the contract visible at the call site instead of
resting on a framework default that the Idris side cannot read.

The bias half departs from `nn.Linear`, whose uniform bias is a legacy
artifact: symmetry breaking is the weight's job, bias gradients neither
vanish nor explode with depth, and the Kaiming/Xavier variance derivations
assume a zero bias. HuggingFace's `_init_weights` overrides it to zero and
LLaMA/PaLM-style models drop the bias entirely.

`init_conv_` applies the same contract to `nn.Conv1d` / `nn.Conv2d`, whose
Idris counterparts (`Nn.conv1d` / `Nn.conv2d`) match it as of 2026-07-31.

Recurrent weight matrices are the one deliberate exception, and they are
aligned on their own terms: `LinearRNNCell` / `LSTMCell` / `GRUCell` here and
`Nn.Recurrent` / `Lstm` / `Gru` in Idris both use Xavier-uniform,
`U(+-sqrt(6/(fan_in+fan_out)))`, which suits a weight applied once per
timestep. Everything else dense — including `multi_head_transformer`'s
per-head projections and feed-forward layers, which are all `nn.Linear` —
goes through this helper.
"""

import math

import torch.nn as nn
from torch import Tensor

# `kaiming_uniform_`'s `a` for the 1/sqrt(fan_in) bound. Not a slope that
# any activation here uses — it is the value `nn.Linear.reset_parameters`
# passes, preserved so the two sides agree bit-for-bit in intent.
KAIMING_A = math.sqrt(5)


def init_linear_weight_(weight: Tensor) -> None:
    """Kaiming-uniform a dense weight in place, bound 1/sqrt(fan_in).

    The raw-`nn.Parameter` entry point, for cells that hold their output
    projection as a bare tensor rather than an `nn.Linear` (`rnn.py`'s
    `LinearRNNCell.weight_out`).
    """
    nn.init.kaiming_uniform_(weight, a=KAIMING_A)


def init_linear_(module: nn.Module) -> None:
    """Apply the shared dense init to every `nn.Linear` under `module`.

    Walks submodules, so this is one call per model rather than one per
    layer — the actor/critic models nest six dense layers each.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            init_linear_weight_(submodule.weight)
            # torch's stub types `bias` as `Parameter`, but `nn.Linear(...,
            # bias=False)` really does leave it `None`, so the guard stays.
            if submodule.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(submodule.bias)


def init_conv_(module: nn.Module) -> None:
    """Apply the shared init to every `nn.Conv1d` / `nn.Conv2d` under `module`.

    Same contract as `init_linear_`: weight ~ U(-1/sqrt(fan_in),
    +1/sqrt(fan_in)) with fan_in = in_channels * prod(kernel_size), bias = 0.
    The weight half is already what `_ConvNd.reset_parameters` does; spelling
    it out keeps the contract visible from the Idris side, which cannot read a
    framework default. The bias half is the departure, for the reason given on
    `init_linear_`.

    Idris `Nn.conv1d` / `Nn.conv2d` were He-normal, `N(0, sqrt(2/fan_in))`,
    until 2026-07-31 — sqrt(6) ~ 2.45x wider than this — so mnist_cnn and
    seq_classify diverged from their Idris twins at step zero.
    """
    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(submodule.weight, a=KAIMING_A)
            if submodule.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(submodule.bias)
