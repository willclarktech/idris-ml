#!/usr/bin/env python3
"""The paired Idris/PyTorch example table — the single source of truth for
which reference belongs to which example.

Three gates read it, so a pair declared here is checked from three angles:

  * `check-example-pairing.py`  — every campaign example HAS an entry, and the
    files it names exist. Stops a new example shipping without a reference.
  * `check-paired-defaults.py`  — the two sides' CLI flag defaults agree.
  * `check-paired-metrics.py`   — the two sides' RESULT lines carry the same
    metric keys.

Kept in one module because a second copy of this list is exactly the drift the
gates exist to catch. `example-<name>` is the make target; the Idris source
basename does not always match it (`example-dnc-recall` builds
`DncAssociativeRecall.idr`), which is why the paths are spelled out rather than
derived.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: the gates run under whatever `python3` is on PATH (3.9 on
    # macOS CommandLineTools), where NotRequired doesn't exist at runtime.
    # `from __future__ import annotations` keeps all uses stringified.
    from typing import NotRequired, TypedDict

    class ExampleSpec(TypedDict):
        """One Idris/Python paired-example mapping row (see EXAMPLES below)."""

        name: str
        idris: str
        python: str
        idris_only: NotRequired[list[str]]
        python_only: NotRequired[list[str]]
        metrics_only_idris: NotRequired[list[str]]
        metrics_only_python: NotRequired[list[str]]
        init_manifest: NotRequired[bool]
        target: NotRequired[str]


REPO_ROOT = Path(__file__).resolve().parent.parent


# Mapping table: short name -> (idris file, python file, optional overrides).
# `idris_only` / `python_only` declare flags that legitimately exist on only
# one side. Anything not declared falls through as drift.
#
# Common pattern for python-only `--lr-find`: that example doesn't have the
# lr_find machinery wired on the Idris side. Mostly the supervised/RNN-family
# ones. Adding it is a separate small task per example, tracked elsewhere.
EXAMPLES: list[ExampleSpec] = [
    {
        "name": "supervised",
        "idris": "packages/idris-ml-examples/src/Example/Supervised.idr",
        "python": "packages/pytorch/torch_ref/scripts/supervised.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "rnn",
        "idris": "packages/idris-ml-examples/src/Example/Rnn.idr",
        "python": "packages/pytorch/torch_ref/scripts/rnn.py",
        "idris_only": ["--patience"],  # idris-side windowed-avg ES; py runs fixed-epoch
        "python_only": ["--lr-find"],
    },
    {
        "name": "lstm",
        "idris": "packages/idris-ml-examples/src/Example/Lstm.idr",
        "python": "packages/pytorch/torch_ref/scripts/lstm.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "gru",
        "idris": "packages/idris-ml-examples/src/Example/Gru.idr",
        "python": "packages/pytorch/torch_ref/scripts/gru.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "mnist",
        "idris": "packages/idris-ml-examples/src/Example/Mnist.idr",
        "python": "packages/pytorch/torch_ref/scripts/mnist.py",
        "idris_only": ["--data"],  # idris loads from local path; py uses torchvision
        "python_only": ["--batch-size"],
    },  # py exposes batch knob; idris bakes it
    {
        "name": "seq-classify",
        "idris": "packages/idris-ml-examples/src/Example/SeqClassify.idr",
        "python": "packages/pytorch/torch_ref/scripts/seq_classify.py",
        "idris_only": ["--patience"],
    },
    {
        "name": "transformer",
        "idris": "packages/idris-ml-examples/src/Example/Transformer.idr",
        "python": "packages/pytorch/torch_ref/scripts/transformer.py",
        "python_only": ["--blocks"],
    },  # py parameterises blocks; idris bakes it
    {
        "name": "gpt",
        "idris": "packages/idris-ml-examples/src/Example/Gpt.idr",
        "python": "packages/pytorch/torch_ref/scripts/gpt.py",
    },
    # NTM/DNC family: alpha/eps/momentum are RMSprop tuning. Idris exposes
    # them as CLI flags; Python bakes them into `torch.optim.RMSprop(...)`.
    # Same values used on both sides (verified at call site).
    {
        "name": "ntm-copy",
        "idris": "packages/idris-ml-examples/src/Example/NtmCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "ntm-recall",
        # The make target does not follow the short name here.
        "target": "example-ntm-associative-recall",
        "idris": "packages/idris-ml-examples/src/Example/NtmAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-copy",
        "idris": "packages/idris-ml-examples/src/Example/DncCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-recall",
        "idris": "packages/idris-ml-examples/src/Example/DncAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "reinforce",
        "idris": "packages/idris-ml-examples/src/Example/Reinforce.idr",
        "python": "packages/pytorch/torch_ref/scripts/reinforce.py",
        "idris_only": ["--batched"],
    },  # Job 4 Phase B; py doesn't have it
    {
        "name": "a2c",
        "idris": "packages/idris-ml-examples/src/Example/A2c.idr",
        "python": "packages/pytorch/torch_ref/scripts/a2c.py",
        "python_only": ["--rollout"],
    },  # rollout len exposed py-side; baked idris-side
    {
        "name": "ppo",
        "idris": "packages/idris-ml-examples/src/Example/Ppo.idr",
        "python": "packages/pytorch/torch_ref/scripts/ppo.py",
        "idris_only": ["--value-coef"],
        "python_only": ["--batch-size", "--max-ep-len", "--rollout"],
    },
    {
        "name": "dqn",
        "idris": "packages/idris-ml-examples/src/Example/Dqn.idr",
        "python": "packages/pytorch/torch_ref/scripts/dqn.py",
        "idris_only": ["--eps-start", "--eps-end", "--eps-decay"],
    },
    {
        "name": "mountain-car",
        "idris": "packages/idris-ml-examples/src/Example/MountainCar.idr",
        "python": "packages/pytorch/torch_ref/scripts/mountain_car.py",
    },
    {
        "name": "mountain-car-cont",
        "idris": "packages/idris-ml-examples/src/Example/MountainCarCont.idr",
        "python": "packages/pytorch/torch_ref/scripts/mountain_car_cont.py",
        "idris_only": ["--clip", "--es-threshold", "--es-window", "--es-patience"],
    },
    {
        "name": "q-learning",
        # Tabular: a Q-table, no registered parameters on either side, so there
        # is no init to compare.
        "init_manifest": False,
        "idris": "packages/idris-ml-examples/src/Example/QLearning.idr",
        "python": "packages/pytorch/torch_ref/scripts/q_learning.py",
    },
    {
        "name": "sarsa",
        # Tabular: a Q-table, no registered parameters on either side, so there
        # is no init to compare.
        "init_manifest": False,
        "idris": "packages/idris-ml-examples/src/Example/Sarsa.idr",
        "python": "packages/pytorch/torch_ref/scripts/sarsa.py",
    },
    {
        "name": "frozen-lake",
        # Tabular: a Q-table, no registered parameters on either side, so there
        # is no init to compare.
        "init_manifest": False,
        "idris": "packages/idris-ml-examples/src/Example/FrozenLake.idr",
        "python": "packages/pytorch/torch_ref/scripts/frozen_lake.py",
    },
    {
        "name": "taxi",
        # Tabular: a Q-table, no registered parameters on either side, so there
        # is no init to compare.
        "init_manifest": False,
        "idris": "packages/idris-ml-examples/src/Example/Taxi.idr",
        "python": "packages/pytorch/torch_ref/scripts/taxi.py",
    },
    {
        "name": "monte-carlo",
        # Tabular: a Q-table, no registered parameters on either side, so there
        # is no init to compare.
        "init_manifest": False,
        "idris": "packages/idris-ml-examples/src/Example/MonteCarlo.idr",
        "python": "packages/pytorch/torch_ref/scripts/monte_carlo.py",
    },
    {
        "name": "double-dqn",
        "idris": "packages/idris-ml-examples/src/Example/DoubleDqn.idr",
        "python": "packages/pytorch/torch_ref/scripts/double_dqn.py",
    },
    {
        "name": "sac",
        "idris": "packages/idris-ml-examples/src/Example/Sac.idr",
        "python": "packages/pytorch/torch_ref/scripts/sac.py",
    },
]
