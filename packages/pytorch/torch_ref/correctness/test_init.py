"""Correctness tests for the shared linear-layer init (torch_ref.init).

Every reference model whose Idris counterpart builds its dense layers with
`Nn.linear` must agree with it on init: Kaiming-uniform weight (bound
1/sqrt(fan_in)) and an exactly-zero bias. Before 2026-07-29 the references
disagreed with each other — nine models took `nn.Linear`'s defaults (uniform
bias), supervised/rnn set Xavier weights explicitly — so "aligned with the
reference" had no single meaning.

Out of scope here, deliberately: `multi_head_transformer`'s nine `nn.Linear`
layers (their Idris counterpart is `Nn.Attention`, which inits from a normal
distribution, not `Nn.linear`) and the recurrent weight matrices in `rnn.py`
(counterpart `Nn.Recurrent` / `Nn.Lstm` / `Nn.Gru`, also normal).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from torch_ref.init import init_conv_, init_linear_, init_linear_weight_
from torch_ref.models.a2c import Actor as A2cActor
from torch_ref.models.a2c import Critic as A2cCritic
from torch_ref.models.dqn import QNetwork as DqnQNetwork
from torch_ref.models.mnist_cnn import MnistCNN
from torch_ref.models.mountain_car import QNetwork as MountainCarQNetwork
from torch_ref.models.mountain_car_cont import Actor as MccActor
from torch_ref.models.mountain_car_cont import QNet as MccQNet
from torch_ref.models.ppo import Actor as PpoActor
from torch_ref.models.ppo import Critic as PpoCritic
from torch_ref.models.reinforce import PolicyNetwork
from torch_ref.models.rnn import LinearGRUCell, LinearLSTMCell, LinearRNNCell
from torch_ref.models.sac import Actor as SacActor
from torch_ref.models.sac import QNet as SacQNet
from torch_ref.models.seq_classify import SeqClassifyCNN
from torch_ref.models.supervised import SupervisedModel

# Every model whose dense layers map onto Idris `Nn.linear`. Built with
# default constructor args — the init contract is shape-independent.
ALIGNED_MODELS: list[type[nn.Module]] = [
    SupervisedModel,
    MnistCNN,
    SeqClassifyCNN,
    PolicyNetwork,
    DqnQNetwork,
    MountainCarQNetwork,
    MccActor,
    MccQNet,
    SacActor,
    SacQNet,
    PpoActor,
    PpoCritic,
    A2cActor,
    A2cCritic,
]


def kaiming_bound(fan_in: int) -> float:
    """`kaiming_uniform_(w, a=sqrt(5))`'s bound, which reduces to 1/sqrt(fan_in).

    gain = sqrt(2 / (1 + a**2)) = sqrt(1/3), and the uniform bound is
    sqrt(3) * gain / sqrt(fan_in). This is also `nn.Linear`'s default and
    the literal `1.0 / sqrt i` in Idris `Ml.Nn.Linear.linear`.
    """
    return 1.0 / math.sqrt(fan_in)


def assert_kaiming_uniform(weight: Tensor, fan_in: int) -> None:
    bound = kaiming_bound(fan_in)
    weight = weight.detach()
    assert float(weight.abs().max()) <= bound, "weight outside the Kaiming bound"
    # A degenerate (all-zero / constant) weight would pass the bound check,
    # so pin the spread too. U(+-b) has std b/sqrt(3); allow a wide band
    # because the smallest layer here has only a handful of elements.
    assert 0.3 * bound < float(weight.std()) < 1.2 * bound, "weight spread implausible"


class TestInitLinearHelper:
    def test_weight_is_kaiming_uniform(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        layer = nn.Linear(64, 32)
        init_linear_(layer)
        assert_kaiming_uniform(layer.weight, fan_in=64)

    def test_bias_is_exactly_zero(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        layer = nn.Linear(64, 32)
        init_linear_(layer)
        assert layer.bias is not None
        assert float(layer.bias.abs().max()) == 0.0

    def test_walks_nested_submodules(self) -> None:
        """One call per model, not one per layer — the models nest 6 deep."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Sequential(nn.Linear(16, 4)))
        init_linear_(net)
        for module in net.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is not None
                assert float(module.bias.abs().max()) == 0.0

    def test_leaves_non_linear_modules_alone(self) -> None:
        """Convs go through `init_conv_`; recurrent init is still a separate axis."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        conv = nn.Conv2d(1, 4, 3)
        before = conv.weight.detach().clone()
        init_linear_(nn.Sequential(conv))
        assert torch.equal(conv.weight, before)

    def test_weight_only_helper_matches(self) -> None:
        """`init_linear_weight_` is the raw-Parameter entry point (rnn.py)."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        weight = nn.Parameter(torch.empty(4, 16))
        init_linear_weight_(weight)
        assert_kaiming_uniform(weight, fan_in=16)


class TestInitConvHelper:
    """The conv half of the contract, paired with Idris `Nn.conv1d`/`conv2d`.

    fan_in = in_channels * prod(kernel_size), so a Conv2d(1, 16, 5) has
    fan_in 25 and a Conv1d(4, 8, 3) has fan_in 12 — the mnist_cnn and
    seq_classify shapes.
    """

    def test_conv2d_weight_is_kaiming_uniform(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        conv = nn.Conv2d(1, 16, 5)
        init_conv_(conv)
        assert_kaiming_uniform(conv.weight, fan_in=1 * 5 * 5)

    def test_conv1d_weight_is_kaiming_uniform(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        conv = nn.Conv1d(4, 8, 3)
        init_conv_(conv)
        assert_kaiming_uniform(conv.weight, fan_in=4 * 3)

    def test_conv_bias_is_exactly_zero(self) -> None:
        """The departure from `_ConvNd.reset_parameters`, whose bias is uniform."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        conv = nn.Conv1d(4, 8, 3)
        init_conv_(conv)
        assert conv.bias is not None
        assert float(conv.bias.abs().max()) == 0.0

    def test_leaves_linear_alone(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        layer = nn.Linear(8, 4)
        before = layer.weight.detach().clone()
        init_conv_(nn.Sequential(layer))
        assert torch.equal(layer.weight, before)


class TestAlignedModels:
    def test_every_linear_bias_is_zero(self) -> None:
        for cls in ALIGNED_MODELS:
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
            model = cls()
            for module in model.modules():
                # `bias` is stubbed as `Parameter`, but `bias=False` yields None.
                if isinstance(module, nn.Linear) and module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                    assert float(module.bias.detach().abs().max()) == 0.0, (
                        f"{cls.__name__} has a nonzero bias on a Linear({module.in_features}, "
                        f"{module.out_features})"
                    )

    def test_every_linear_weight_is_kaiming_uniform(self) -> None:
        for cls in ALIGNED_MODELS:
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
            model = cls()
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    assert_kaiming_uniform(module.weight, fan_in=module.in_features)


class TestRnnOutputProjections:
    """`rnn.py`'s output heads map to Idris `Nn.linear`; its cells do not."""

    def test_rnn_cell_output_weights(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        cell = LinearRNNCell(1, 4, 1)
        assert_kaiming_uniform(cell.weight_out, fan_in=4)
        assert float(cell.bias_out.abs().max()) == 0.0

    def test_lstm_and_gru_output_projections(self) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        for cell in (LinearLSTMCell(1, 4, 1), LinearGRUCell(1, 4, 1)):
            proj = cell.output_proj
            assert_kaiming_uniform(proj.weight, fan_in=4)
            assert proj.bias is not None
            assert float(proj.bias.abs().max()) == 0.0
