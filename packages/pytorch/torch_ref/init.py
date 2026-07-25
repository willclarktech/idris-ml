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

Not everything dense is in scope. Convolution kernels, recurrent weight
matrices and attention projections have their own Idris counterparts
(`Nn.Conv`, `Nn.Recurrent`/`Lstm`/`Gru`, `Nn.Attention`), all of which init
from a normal distribution; aligning those is a separate axis and this
helper deliberately leaves them untouched.
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
