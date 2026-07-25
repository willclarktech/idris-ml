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
        data_manifest: NotRequired[bool]
        params: NotRequired[dict[str, str]]
        step_oracle: NotRequired[bool]
        tolerance: NotRequired[float]
        # The reference's oracle run writes <fixture>.replay (recorded
        # draws); the Idris run receives it via its --replay flag.
        replay: NotRequired[bool]
        # Extra CLI args for the step-oracle run, applied to BOTH sides —
        # for examples whose default config would leave the first epoch
        # without an optimizer step (sac's warmup).
        oracle_args: NotRequired[list[str]]


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
        # Step-oracle bound, measured with `--tolerance 0`:
        # worst 2.2e-19 (linear_0.bias); the weight is bit-identical
        "tolerance": 1e-15,
        # One optimizer step per epoch and a two-parameter model, so a wrong
        # post-step weight is unambiguous. See check-step-oracle.py.
        "step_oracle": True,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "linear_0.bias": "0.linear.bias",
            "linear_0.weight": "0.linear.weight",
        },
        # Fixed 5-sample dataset, generated identically on both sides.
        "data_manifest": True,
        "idris": "packages/idris-ml-examples/src/Example/Supervised.idr",
        "python": "packages/pytorch/torch_ref/scripts/supervised.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "rnn",
        # Step-oracle bound, measured with `--tolerance 0`:
        # worst 5.2e-18 (both recurrent biases)
        "tolerance": 1e-14,
        # Deterministic pattern sequences on both sides, so the oracle only
        # transfers weights (maybeDumpOracleWeights / maybe_load_oracle_weights).
        "step_oracle": True,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "linear_0.bias": "0.bias_out",
            "linear_0.weight": "0.weight_out",
            "rnn_0.bias_hh": "0.bias_hh",
            "rnn_0.bias_ih": "0.bias_ih",
            "rnn_0.weight_hh": "0.weight_hh",
            "rnn_0.weight_ih": "0.weight_ih",
        },
        "idris": "packages/idris-ml-examples/src/Example/Rnn.idr",
        "python": "packages/pytorch/torch_ref/scripts/rnn.py",
        "idris_only": ["--patience"],  # idris-side windowed-avg ES; py runs fixed-epoch
        "python_only": ["--lr-find"],
    },
    {
        "name": "lstm",
        # Step-oracle bound, measured with `--tolerance 0`:
        # worst 8.7e-19 (both gate biases)
        "tolerance": 1e-15,
        # Deterministic pattern sequences on both sides, so the oracle only
        # transfers weights (maybeDumpOracleWeights / maybe_load_oracle_weights).
        "step_oracle": True,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "linear_0.bias": "0.output_proj.bias",
            "linear_0.weight": "0.output_proj.weight",
            "lstm_0.bias_hh": "0.lstm.bias_hh",
            "lstm_0.bias_ih": "0.lstm.bias_ih",
            "lstm_0.c0": "0.c0",
            "lstm_0.h0": "0.h0",
            "lstm_0.weight_hh": "0.lstm.weight_hh",
            "lstm_0.weight_ih": "0.lstm.weight_ih",
        },
        "idris": "packages/idris-ml-examples/src/Example/Lstm.idr",
        "python": "packages/pytorch/torch_ref/scripts/lstm.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "gru",
        # Step-oracle bound, measured with `--tolerance 0`:
        # worst 1.7e-18 (bias_hh)
        "tolerance": 1e-15,
        # Deterministic pattern sequences on both sides, so the oracle only
        # transfers weights (maybeDumpOracleWeights / maybe_load_oracle_weights).
        "step_oracle": True,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "gru_0.bias_hh": "0.bias_hh",
            "gru_0.bias_ih": "0.bias_ih",
            "gru_0.weight_hh": "0.weight_hh",
            "gru_0.weight_ih": "0.weight_ih",
            "linear_0.bias": "0.output_proj.bias",
            "linear_0.weight": "0.output_proj.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Gru.idr",
        "python": "packages/pytorch/torch_ref/scripts/gru.py",
        "python_only": ["--lr-find"],
    },
    {
        "name": "mnist",
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "conv2d_0.bias": "0.conv1.bias",
            "conv2d_0.weight": "0.conv1.weight",
            "conv2d_1.bias": "0.conv2.bias",
            "conv2d_1.weight": "0.conv2.weight",
            "linear_0.bias": "0.fc.bias",
            "linear_0.weight": "0.fc.weight",
        },
        # Same idx files on both sides; the gate pins the loader/normalisation.
        "data_manifest": True,
        "idris": "packages/idris-ml-examples/src/Example/Mnist.idr",
        "python": "packages/pytorch/torch_ref/scripts/mnist.py",
        "idris_only": ["--data"],  # idris loads from local path; py uses torchvision
        "python_only": ["--batch-size"],
    },  # py exposes batch knob; idris bakes it
    {
        "name": "seq-classify",
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "conv1d_0.bias": "0.conv1.bias",
            "conv1d_0.weight": "0.conv1.weight",
            "conv1d_1.bias": "0.conv2.bias",
            "conv1d_1.weight": "0.conv2.weight",
            "linear_0.bias": "0.fc.bias",
            "linear_0.weight": "0.fc.weight",
        },
        # Synthetic waveform generator on both sides — the case that motivated
        # the data gate.
        "data_manifest": True,
        "idris": "packages/idris-ml-examples/src/Example/SeqClassify.idr",
        "python": "packages/pytorch/torch_ref/scripts/seq_classify.py",
        "idris_only": ["--patience"],
    },
    {
        "name": "transformer",
        # The batch is RNG-driven (16 samples x 5 token draws), so the
        # reference records the raw tokens and Idris rebuilds the identical
        # batch by replaying them — sample construction included.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 6.4e-11
        # (block_0.ff2_0.weight) — Adam + global-norm clip, as on a2c.
        "tolerance": 1e-9,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "block_0.attn_0.key_0.weight": "0.blocks.0.key_ws.0.weight",
            "block_0.attn_0.key_1.weight": "0.blocks.0.key_ws.1.weight",
            "block_0.attn_0.key_2.weight": "0.blocks.0.key_ws.2.weight",
            "block_0.attn_0.key_3.weight": "0.blocks.0.key_ws.3.weight",
            "block_0.attn_0.out_proj_0.weight": "0.blocks.0.out_proj_ws.0.weight",
            "block_0.attn_0.out_proj_1.weight": "0.blocks.0.out_proj_ws.1.weight",
            "block_0.attn_0.out_proj_2.weight": "0.blocks.0.out_proj_ws.2.weight",
            "block_0.attn_0.out_proj_3.weight": "0.blocks.0.out_proj_ws.3.weight",
            "block_0.attn_0.query_0.weight": "0.blocks.0.query_ws.0.weight",
            "block_0.attn_0.query_1.weight": "0.blocks.0.query_ws.1.weight",
            "block_0.attn_0.query_2.weight": "0.blocks.0.query_ws.2.weight",
            "block_0.attn_0.query_3.weight": "0.blocks.0.query_ws.3.weight",
            "block_0.attn_0.value_0.weight": "0.blocks.0.value_ws.0.weight",
            "block_0.attn_0.value_1.weight": "0.blocks.0.value_ws.1.weight",
            "block_0.attn_0.value_2.weight": "0.blocks.0.value_ws.2.weight",
            "block_0.attn_0.value_3.weight": "0.blocks.0.value_ws.3.weight",
            "block_0.ff1_0.weight": "0.blocks.0.ff1.weight",
            "block_0.ff2_0.weight": "0.blocks.0.ff2.weight",
            "block_0.norm1.bias": "0.blocks.0.norm1.bias",
            "block_0.norm1.weight": "0.blocks.0.norm1.weight",
            "block_0.norm2.bias": "0.blocks.0.norm2.bias",
            "block_0.norm2.weight": "0.blocks.0.norm2.weight",
            "block_1.attn_0.key_0.weight": "0.blocks.1.key_ws.0.weight",
            "block_1.attn_0.key_1.weight": "0.blocks.1.key_ws.1.weight",
            "block_1.attn_0.key_2.weight": "0.blocks.1.key_ws.2.weight",
            "block_1.attn_0.key_3.weight": "0.blocks.1.key_ws.3.weight",
            "block_1.attn_0.out_proj_0.weight": "0.blocks.1.out_proj_ws.0.weight",
            "block_1.attn_0.out_proj_1.weight": "0.blocks.1.out_proj_ws.1.weight",
            "block_1.attn_0.out_proj_2.weight": "0.blocks.1.out_proj_ws.2.weight",
            "block_1.attn_0.out_proj_3.weight": "0.blocks.1.out_proj_ws.3.weight",
            "block_1.attn_0.query_0.weight": "0.blocks.1.query_ws.0.weight",
            "block_1.attn_0.query_1.weight": "0.blocks.1.query_ws.1.weight",
            "block_1.attn_0.query_2.weight": "0.blocks.1.query_ws.2.weight",
            "block_1.attn_0.query_3.weight": "0.blocks.1.query_ws.3.weight",
            "block_1.attn_0.value_0.weight": "0.blocks.1.value_ws.0.weight",
            "block_1.attn_0.value_1.weight": "0.blocks.1.value_ws.1.weight",
            "block_1.attn_0.value_2.weight": "0.blocks.1.value_ws.2.weight",
            "block_1.attn_0.value_3.weight": "0.blocks.1.value_ws.3.weight",
            "block_1.ff1_0.weight": "0.blocks.1.ff1.weight",
            "block_1.ff2_0.weight": "0.blocks.1.ff2.weight",
            "block_1.norm1.bias": "0.blocks.1.norm1.bias",
            "block_1.norm1.weight": "0.blocks.1.norm1.weight",
            "block_1.norm2.bias": "0.blocks.1.norm2.bias",
            "block_1.norm2.weight": "0.blocks.1.norm2.weight",
            "embed.embedding_0.weight": "0.token_embed.weight",
            "head_0.weight": "0.vocab_proj.weight",
            "layer_norm_0.bias": "0.norm_final.bias",
            "layer_norm_0.weight": "0.norm_final.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Transformer.idr",
        "python": "packages/pytorch/torch_ref/scripts/transformer.py",
        "python_only": ["--blocks"],
    },  # py parameterises blocks; idris bakes it
    {
        "name": "gpt",
        # The batch is RNG-driven (32 random window offsets into the shared
        # embedded corpus), so the reference records the offsets and Idris
        # rebuilds the identical batch by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.4e-11
        # (block_1.ff2_0.weight) — AdamW + global-norm clip.
        "tolerance": 1e-9,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "block_0.attn_0.key_0.weight": "0.blocks.0.key_ws.0.weight",
            "block_0.attn_0.key_1.weight": "0.blocks.0.key_ws.1.weight",
            "block_0.attn_0.key_2.weight": "0.blocks.0.key_ws.2.weight",
            "block_0.attn_0.key_3.weight": "0.blocks.0.key_ws.3.weight",
            "block_0.attn_0.out_proj_0.weight": "0.blocks.0.out_proj_ws.0.weight",
            "block_0.attn_0.out_proj_1.weight": "0.blocks.0.out_proj_ws.1.weight",
            "block_0.attn_0.out_proj_2.weight": "0.blocks.0.out_proj_ws.2.weight",
            "block_0.attn_0.out_proj_3.weight": "0.blocks.0.out_proj_ws.3.weight",
            "block_0.attn_0.query_0.weight": "0.blocks.0.query_ws.0.weight",
            "block_0.attn_0.query_1.weight": "0.blocks.0.query_ws.1.weight",
            "block_0.attn_0.query_2.weight": "0.blocks.0.query_ws.2.weight",
            "block_0.attn_0.query_3.weight": "0.blocks.0.query_ws.3.weight",
            "block_0.attn_0.value_0.weight": "0.blocks.0.value_ws.0.weight",
            "block_0.attn_0.value_1.weight": "0.blocks.0.value_ws.1.weight",
            "block_0.attn_0.value_2.weight": "0.blocks.0.value_ws.2.weight",
            "block_0.attn_0.value_3.weight": "0.blocks.0.value_ws.3.weight",
            "block_0.ff1_0.weight": "0.blocks.0.ff1.weight",
            "block_0.ff2_0.weight": "0.blocks.0.ff2.weight",
            "block_0.norm1.bias": "0.blocks.0.norm1.bias",
            "block_0.norm1.weight": "0.blocks.0.norm1.weight",
            "block_0.norm2.bias": "0.blocks.0.norm2.bias",
            "block_0.norm2.weight": "0.blocks.0.norm2.weight",
            "block_1.attn_0.key_0.weight": "0.blocks.1.key_ws.0.weight",
            "block_1.attn_0.key_1.weight": "0.blocks.1.key_ws.1.weight",
            "block_1.attn_0.key_2.weight": "0.blocks.1.key_ws.2.weight",
            "block_1.attn_0.key_3.weight": "0.blocks.1.key_ws.3.weight",
            "block_1.attn_0.out_proj_0.weight": "0.blocks.1.out_proj_ws.0.weight",
            "block_1.attn_0.out_proj_1.weight": "0.blocks.1.out_proj_ws.1.weight",
            "block_1.attn_0.out_proj_2.weight": "0.blocks.1.out_proj_ws.2.weight",
            "block_1.attn_0.out_proj_3.weight": "0.blocks.1.out_proj_ws.3.weight",
            "block_1.attn_0.query_0.weight": "0.blocks.1.query_ws.0.weight",
            "block_1.attn_0.query_1.weight": "0.blocks.1.query_ws.1.weight",
            "block_1.attn_0.query_2.weight": "0.blocks.1.query_ws.2.weight",
            "block_1.attn_0.query_3.weight": "0.blocks.1.query_ws.3.weight",
            "block_1.attn_0.value_0.weight": "0.blocks.1.value_ws.0.weight",
            "block_1.attn_0.value_1.weight": "0.blocks.1.value_ws.1.weight",
            "block_1.attn_0.value_2.weight": "0.blocks.1.value_ws.2.weight",
            "block_1.attn_0.value_3.weight": "0.blocks.1.value_ws.3.weight",
            "block_1.ff1_0.weight": "0.blocks.1.ff1.weight",
            "block_1.ff2_0.weight": "0.blocks.1.ff2.weight",
            "block_1.norm1.bias": "0.blocks.1.norm1.bias",
            "block_1.norm1.weight": "0.blocks.1.norm1.weight",
            "block_1.norm2.bias": "0.blocks.1.norm2.bias",
            "block_1.norm2.weight": "0.blocks.1.norm2.weight",
            "embed.embedding_0.weight": "0.token_embed.weight",
            "head_0.weight": "0.vocab_proj.weight",
            "layer_norm_0.bias": "0.norm_final.bias",
            "layer_norm_0.weight": "0.norm_final.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Gpt.idr",
        "python": "packages/pytorch/torch_ref/scripts/gpt.py",
    },
    # NTM/DNC family: alpha/eps/momentum are RMSprop tuning. Idris exposes
    # them as CLI flags; Python bakes them into `torch.optim.RMSprop(...)`.
    # Same values used on both sides (verified at call site).
    {
        "name": "ntm-copy",
        # The batch is RNG-driven (sequence length + random bits), so the
        # reference records each sequence's draws and Idris rebuilds the
        # identical batch by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.2e-7
        # (read_fc.weight). Looser than the dense examples because RMSprop's
        # first step is ill-conditioned where a gradient sits near its
        # epsilon: read_fc's median |g| is 2.4e-7, and those entries are
        # sums of cancelling terms, so cross-side rounding lands at the
        # term scale rather than the sum scale.
        "tolerance": 1e-5,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "ntm_0.controller.bias_hh": "0.ntm.controller.lstm.bias_hh",
            "ntm_0.controller.bias_ih": "0.ntm.controller.lstm.bias_ih",
            "ntm_0.controller.c0": "0.ntm.controller.c0",
            "ntm_0.controller.h0": "0.ntm.controller.h0",
            "ntm_0.controller.weight_hh": "0.ntm.controller.lstm.weight_hh",
            "ntm_0.controller.weight_ih": "0.ntm.controller.lstm.weight_ih",
            "ntm_0.memory_init_0": "0.ntm.memory_init",
            "ntm_0.output_fc.bias": "0.ntm.output_fc.bias",
            "ntm_0.read_init_0": "0.ntm.read_init",
            "ntm_0.output_fc.weight": "0.ntm.output_fc.weight",
            "ntm_0.read_fc.bias": "0.ntm.read_fc.bias",
            "ntm_0.read_fc.weight": "0.ntm.read_fc.weight",
            "ntm_0.write_fc.bias": "0.ntm.write_fc.bias",
            "ntm_0.write_fc.weight": "0.ntm.write_fc.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/NtmCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "ntm-recall",
        # The batch is RNG-driven (item count, item bits, query index), so
        # the reference records each sample's draws and Idris rebuilds the
        # identical batch by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 8.1e-8
        # (read_fc.weight) — the same RMSprop eps-regime floor as ntm-copy.
        "tolerance": 1e-5,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "ntm_0.controller.bias_hh": "0.ntm.controller.lstm.bias_hh",
            "ntm_0.controller.bias_ih": "0.ntm.controller.lstm.bias_ih",
            "ntm_0.controller.c0": "0.ntm.controller.c0",
            "ntm_0.controller.h0": "0.ntm.controller.h0",
            "ntm_0.controller.weight_hh": "0.ntm.controller.lstm.weight_hh",
            "ntm_0.controller.weight_ih": "0.ntm.controller.lstm.weight_ih",
            "ntm_0.memory_init_0": "0.ntm.memory_init",
            "ntm_0.output_fc.bias": "0.ntm.output_fc.bias",
            "ntm_0.read_init_0": "0.ntm.read_init",
            "ntm_0.output_fc.weight": "0.ntm.output_fc.weight",
            "ntm_0.read_fc.bias": "0.ntm.read_fc.bias",
            "ntm_0.read_fc.weight": "0.ntm.read_fc.weight",
            "ntm_0.write_fc.bias": "0.ntm.write_fc.bias",
            "ntm_0.write_fc.weight": "0.ntm.write_fc.weight",
        },
        # The make target does not follow the short name here.
        "target": "example-ntm-associative-recall",
        "idris": "packages/idris-ml-examples/src/Example/NtmAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/ntm_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-copy",
        # The batch is RNG-driven (sequence length + random bits), so the
        # reference records each sequence's draws and Idris rebuilds the
        # identical batch by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 3.4e-10
        # (memory_init).
        "tolerance": 1e-8,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "dnc_0.add.bias": "0.dnc.add_fc.bias",
            "dnc_0.add.weight": "0.dnc.add_fc.weight",
            "dnc_0.alloc_gate.bias": "0.dnc.alloc_gate_fc.bias",
            "dnc_0.alloc_gate.weight": "0.dnc.alloc_gate_fc.weight",
            "dnc_0.controller.bias_hh": "0.dnc.controller.lstm.bias_hh",
            "dnc_0.controller.bias_ih": "0.dnc.controller.lstm.bias_ih",
            "dnc_0.controller.c0": "0.dnc.controller.c0",
            "dnc_0.controller.h0": "0.dnc.controller.h0",
            "dnc_0.controller.weight_hh": "0.dnc.controller.lstm.weight_hh",
            "dnc_0.controller.weight_ih": "0.dnc.controller.lstm.weight_ih",
            "dnc_0.erase.bias": "0.dnc.erase_fc.bias",
            "dnc_0.erase.weight": "0.dnc.erase_fc.weight",
            "dnc_0.free_gates.bias": "0.dnc.free_gates_fc.bias",
            "dnc_0.free_gates.weight": "0.dnc.free_gates_fc.weight",
            "dnc_0.memory_init_0": "0.dnc.memory_init",
            "dnc_0.output.bias": "0.dnc.output_fc.bias",
            "dnc_0.read_init_0": "0.dnc.read_init",
            "dnc_0.output.weight": "0.dnc.output_fc.weight",
            "dnc_0.read_betas.bias": "0.dnc.read_betas_fc.bias",
            "dnc_0.read_betas.weight": "0.dnc.read_betas_fc.weight",
            "dnc_0.read_keys.bias": "0.dnc.read_keys_fc.bias",
            "dnc_0.read_keys.weight": "0.dnc.read_keys_fc.weight",
            "dnc_0.read_modes.bias": "0.dnc.read_modes_fc.bias",
            "dnc_0.read_modes.weight": "0.dnc.read_modes_fc.weight",
            "dnc_0.write_beta.bias": "0.dnc.write_beta_fc.bias",
            "dnc_0.write_beta.weight": "0.dnc.write_beta_fc.weight",
            "dnc_0.write_gate.bias": "0.dnc.write_gate_fc.bias",
            "dnc_0.write_gate.weight": "0.dnc.write_gate_fc.weight",
            "dnc_0.write_key.bias": "0.dnc.write_key_fc.bias",
            "dnc_0.write_key.weight": "0.dnc.write_key_fc.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/DncCopy.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_copy.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "dnc-recall",
        # The batch is RNG-driven (item count, item bits, query index), so
        # the reference records each sample's draws and Idris rebuilds the
        # identical batch by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 9.2e-11
        # (alloc_gate.weight).
        "tolerance": 1e-8,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "dnc_0.add.bias": "0.dnc.add_fc.bias",
            "dnc_0.add.weight": "0.dnc.add_fc.weight",
            "dnc_0.alloc_gate.bias": "0.dnc.alloc_gate_fc.bias",
            "dnc_0.alloc_gate.weight": "0.dnc.alloc_gate_fc.weight",
            "dnc_0.controller.bias_hh": "0.dnc.controller.lstm.bias_hh",
            "dnc_0.controller.bias_ih": "0.dnc.controller.lstm.bias_ih",
            "dnc_0.controller.c0": "0.dnc.controller.c0",
            "dnc_0.controller.h0": "0.dnc.controller.h0",
            "dnc_0.controller.weight_hh": "0.dnc.controller.lstm.weight_hh",
            "dnc_0.controller.weight_ih": "0.dnc.controller.lstm.weight_ih",
            "dnc_0.erase.bias": "0.dnc.erase_fc.bias",
            "dnc_0.erase.weight": "0.dnc.erase_fc.weight",
            "dnc_0.free_gates.bias": "0.dnc.free_gates_fc.bias",
            "dnc_0.free_gates.weight": "0.dnc.free_gates_fc.weight",
            "dnc_0.memory_init_0": "0.dnc.memory_init",
            "dnc_0.output.bias": "0.dnc.output_fc.bias",
            "dnc_0.read_init_0": "0.dnc.read_init",
            "dnc_0.output.weight": "0.dnc.output_fc.weight",
            "dnc_0.read_betas.bias": "0.dnc.read_betas_fc.bias",
            "dnc_0.read_betas.weight": "0.dnc.read_betas_fc.weight",
            "dnc_0.read_keys.bias": "0.dnc.read_keys_fc.bias",
            "dnc_0.read_keys.weight": "0.dnc.read_keys_fc.weight",
            "dnc_0.read_modes.bias": "0.dnc.read_modes_fc.bias",
            "dnc_0.read_modes.weight": "0.dnc.read_modes_fc.weight",
            "dnc_0.write_beta.bias": "0.dnc.write_beta_fc.bias",
            "dnc_0.write_beta.weight": "0.dnc.write_beta_fc.weight",
            "dnc_0.write_gate.bias": "0.dnc.write_gate_fc.bias",
            "dnc_0.write_gate.weight": "0.dnc.write_gate_fc.weight",
            "dnc_0.write_key.bias": "0.dnc.write_key_fc.bias",
            "dnc_0.write_key.weight": "0.dnc.write_key_fc.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/DncAssociativeRecall.idr",
        "python": "packages/pytorch/torch_ref/scripts/dnc_recall.py",
        "idris_only": ["--alpha", "--eps", "--momentum"],
    },
    {
        "name": "reinforce",
        # The rollout is RNG-driven, so the reference records its draws
        # (action decisions; per-episode reset states as the uniforms that
        # produced them) and Idris regenerates the identical episodes by
        # replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 6.8e-12
        # (linear_0.bias) — Adam + global-norm clip.
        "tolerance": 1e-9,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "linear_0.bias": "0.fc1.bias",
            "linear_0.weight": "0.fc1.weight",
            "linear_1.bias": "0.fc2.bias",
            "linear_1.weight": "0.fc2.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Reinforce.idr",
        "python": "packages/pytorch/torch_ref/scripts/reinforce.py",
        "idris_only": ["--batched"],
    },  # Job 4 Phase B; py doesn't have it
    {
        "name": "a2c",
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.4e-11
        # with the rollout regenerated from replayed draws over exact-state
        # obs. Global-norm clipping spreads one scalar's rounding across
        # every parameter, and Adam's first step divides by sqrt(v)+1e-8.
        "tolerance": 1e-9,
        # The rollout is RNG-driven, so the reference records its draws
        # (action choices; reset states as the uniforms that produced them)
        # and Idris regenerates the identical rollout by replaying them.
        "step_oracle": True,
        "replay": True,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "actor.linear_0.bias": "0.fc1.bias",
            "actor.linear_0.weight": "0.fc1.weight",
            "actor.linear_1.bias": "0.fc2.bias",
            "actor.linear_1.weight": "0.fc2.weight",
            "actor.linear_2.bias": "0.head.bias",
            "actor.linear_2.weight": "0.head.weight",
            "critic.linear_0.bias": "1.fc1.bias",
            "critic.linear_0.weight": "1.fc1.weight",
            "critic.linear_1.bias": "1.fc2.bias",
            "critic.linear_1.weight": "1.fc2.weight",
            "critic.linear_2.bias": "1.head.bias",
            "critic.linear_2.weight": "1.head.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/A2c.idr",
        "python": "packages/pytorch/torch_ref/scripts/a2c.py",
        "python_only": ["--rollout"],
    },  # rollout len exposed py-side; baked idris-side
    {
        "name": "ppo",
        # The rollout and the K-epoch minibatch permutations are RNG-driven,
        # so the reference records its draws (action decisions; reset states
        # as the uniforms that produced them; each shuffle's permutation as
        # rank/total tags) and Idris regenerates the identical epoch by
        # replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 6.0e-10
        # (critic.linear_1.weight) across one full update — 10 k-epochs x 16
        # minibatches = 160 Adam steps, each behind a global-norm clip, with
        # the 256-step Acrobot rollout regenerated from replayed draws. The
        # planted entropy x1.01 probe lands 1.3e-05..2.4e-04, actor only.
        "tolerance": 1e-7,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "actor.linear_0.bias": "0.fc1.bias",
            "actor.linear_0.weight": "0.fc1.weight",
            "actor.linear_1.bias": "0.fc2.bias",
            "actor.linear_1.weight": "0.fc2.weight",
            "actor.linear_2.bias": "0.head.bias",
            "actor.linear_2.weight": "0.head.weight",
            "critic.linear_0.bias": "1.fc1.bias",
            "critic.linear_0.weight": "1.fc1.weight",
            "critic.linear_1.bias": "1.fc2.bias",
            "critic.linear_1.weight": "1.fc2.weight",
            "critic.linear_2.bias": "1.head.bias",
            "critic.linear_2.weight": "1.head.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Ppo.idr",
        "python": "packages/pytorch/torch_ref/scripts/ppo.py",
        "python_only": ["--batch-size", "--max-ep-len", "--rollout"],
    },
    {
        "name": "dqn",
        # The episode is RNG-driven, so the reference records its draws
        # (explore gates as uniforms; explored actions and minibatch indices
        # as decisions; reset states as the uniforms that produced them) and
        # Idris regenerates the identical episode by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.1e-17
        # (online.linear_1.weight) across one episode's replay updates; the
        # target net comes out bit-identical. Planted gamma x1.01 probe lands
        # 1.2e-08..1.3e-05, online only.
        "tolerance": 1e-14,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "online.linear_0.bias": "0.fc1.bias",
            "online.linear_0.weight": "0.fc1.weight",
            "online.linear_1.bias": "0.fc2.bias",
            "online.linear_1.weight": "0.fc2.weight",
            "online.linear_2.bias": "0.fc3.bias",
            "online.linear_2.weight": "0.fc3.weight",
            "target.linear_0.bias": "1.fc1.bias",
            "target.linear_0.weight": "1.fc1.weight",
            "target.linear_1.bias": "1.fc2.bias",
            "target.linear_1.weight": "1.fc2.weight",
            "target.linear_2.bias": "1.fc3.bias",
            "target.linear_2.weight": "1.fc3.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Dqn.idr",
        "python": "packages/pytorch/torch_ref/scripts/dqn.py",
        "idris_only": ["--eps-start", "--eps-end", "--eps-decay"],
    },
    {
        "name": "mountain-car",
        # The episode is RNG-driven, so the reference records its draws
        # (explore gates as uniforms; explored actions and minibatch indices
        # as decisions; reset states as the uniforms that produced them) and
        # Idris regenerates the identical episode by replaying them. The
        # episode always ends by 200-step truncation, so the TimeLimit
        # done-flag semantics are pinned here in every run.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 6.7e-16
        # (both fc1.weights — the step-200 sync copies online into target)
        # across a full 200-step truncated episode with 185 replay updates.
        # Planted shaping x1.01 probe lands 4.9e-06..1.4e-04 on both nets.
        "tolerance": 1e-12,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "online.linear_0.bias": "0.fc1.bias",
            "online.linear_0.weight": "0.fc1.weight",
            "online.linear_1.bias": "0.fc2.bias",
            "online.linear_1.weight": "0.fc2.weight",
            "online.linear_2.bias": "0.fc3.bias",
            "online.linear_2.weight": "0.fc3.weight",
            "target.linear_0.bias": "1.fc1.bias",
            "target.linear_0.weight": "1.fc1.weight",
            "target.linear_1.bias": "1.fc2.bias",
            "target.linear_1.weight": "1.fc2.weight",
            "target.linear_2.bias": "1.fc3.bias",
            "target.linear_2.weight": "1.fc3.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/MountainCar.idr",
        "python": "packages/pytorch/torch_ref/scripts/mountain_car.py",
    },
    {
        "name": "mountain-car-cont",
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "actor_.linear_0.bias": "0.fc1.bias",
            "actor_.linear_0.weight": "0.fc1.weight",
            "actor_.linear_1.bias": "0.fc2.bias",
            "actor_.linear_1.weight": "0.fc2.weight",
            "actor_.linear_2.bias": "0.mean_head.bias",
            "actor_.linear_2.weight": "0.mean_head.weight",
            "actor_log_std": "0.log_std",
            "q1_.linear_0.bias": "1.fc1.bias",
            "q1_.linear_0.weight": "1.fc1.weight",
            "q1_.linear_1.bias": "1.fc2.bias",
            "q1_.linear_1.weight": "1.fc2.weight",
            "q1_.linear_2.bias": "1.head.bias",
            "q1_.linear_2.weight": "1.head.weight",
            "q1tgt_.linear_0.bias": "2.fc1.bias",
            "q1tgt_.linear_0.weight": "2.fc1.weight",
            "q1tgt_.linear_1.bias": "2.fc2.bias",
            "q1tgt_.linear_1.weight": "2.fc2.weight",
            "q1tgt_.linear_2.bias": "2.head.bias",
            "q1tgt_.linear_2.weight": "2.head.weight",
            "q2_.linear_0.bias": "3.fc1.bias",
            "q2_.linear_0.weight": "3.fc1.weight",
            "q2_.linear_1.bias": "3.fc2.bias",
            "q2_.linear_1.weight": "3.fc2.weight",
            "q2_.linear_2.bias": "3.head.bias",
            "q2_.linear_2.weight": "3.head.weight",
            "q2tgt_.linear_0.bias": "4.fc1.bias",
            "q2tgt_.linear_0.weight": "4.fc1.weight",
            "q2tgt_.linear_1.bias": "4.fc2.bias",
            "q2tgt_.linear_1.weight": "4.fc2.weight",
            "q2tgt_.linear_2.bias": "4.head.bias",
            "q2tgt_.linear_2.weight": "4.head.weight",
        },
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
        # The episode is RNG-driven, so the reference records its draws
        # (explore gates as uniforms; explored actions and minibatch indices
        # as decisions; reset states as the uniforms that produced them) and
        # Idris regenerates the identical episode by replaying them.
        "step_oracle": True,
        "replay": True,
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.8e-17
        # (online.linear_0.weight); the target net comes out bit-identical.
        # Planted probe: running the reference with the VANILLA dqn update
        # lands 8.8e-09..3.0e-07 online-only — the oracle tells the two
        # algorithms apart.
        "tolerance": 1e-14,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "online.linear_0.bias": "0.fc1.bias",
            "online.linear_0.weight": "0.fc1.weight",
            "online.linear_1.bias": "0.fc2.bias",
            "online.linear_1.weight": "0.fc2.weight",
            "online.linear_2.bias": "0.fc3.bias",
            "online.linear_2.weight": "0.fc3.weight",
            "target.linear_0.bias": "1.fc1.bias",
            "target.linear_0.weight": "1.fc1.weight",
            "target.linear_1.bias": "1.fc2.bias",
            "target.linear_1.weight": "1.fc2.weight",
            "target.linear_2.bias": "1.fc3.bias",
            "target.linear_2.weight": "1.fc3.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/DoubleDqn.idr",
        "python": "packages/pytorch/torch_ref/scripts/double_dqn.py",
    },
    {
        "name": "sac",
        # The step is RNG-driven, so the reference records its draws (action,
        # target and reparameterization noise on the normal channel; minibatch
        # indices as decisions; reset states as the uniforms that produced
        # them) and Idris regenerates the identical step by replaying them.
        # The default config leaves the first epoch inside warmup with an
        # empty buffer, so the oracle runs both sides warmup-free with a
        # batch one lockstep step can fill.
        "step_oracle": True,
        "replay": True,
        "oracle_args": ["--warmup", "0", "--batch", "4"],
        # Step-oracle bound, measured with `--tolerance 0`: worst 2.8e-10
        # (actor_log_std, which sums every squash-correction term); all
        # network weights land at or below 5.6e-17. Planted alpha x1.01
        # probe lands 6.9e-07..1.1e-05 across actor and Q nets.
        "tolerance": 1e-7,
        # Idris registry name -> reference parameter (prefixed by model index,
        # so an actor/critic pair stays distinguishable). Verified as a
        # shape-consistent bijection by check-init-manifest.py.
        "params": {
            "actor_.linear_0.bias": "0.fc1.bias",
            "actor_.linear_0.weight": "0.fc1.weight",
            "actor_.linear_1.bias": "0.fc2.bias",
            "actor_.linear_1.weight": "0.fc2.weight",
            "actor_.linear_2.bias": "0.mean_head.bias",
            "actor_.linear_2.weight": "0.mean_head.weight",
            "actor_log_std": "0.log_std",
            "q1_.linear_0.bias": "1.fc1.bias",
            "q1_.linear_0.weight": "1.fc1.weight",
            "q1_.linear_1.bias": "1.fc2.bias",
            "q1_.linear_1.weight": "1.fc2.weight",
            "q1_.linear_2.bias": "1.head.bias",
            "q1_.linear_2.weight": "1.head.weight",
            "q1tgt_.linear_0.bias": "2.fc1.bias",
            "q1tgt_.linear_0.weight": "2.fc1.weight",
            "q1tgt_.linear_1.bias": "2.fc2.bias",
            "q1tgt_.linear_1.weight": "2.fc2.weight",
            "q1tgt_.linear_2.bias": "2.head.bias",
            "q1tgt_.linear_2.weight": "2.head.weight",
            "q2_.linear_0.bias": "3.fc1.bias",
            "q2_.linear_0.weight": "3.fc1.weight",
            "q2_.linear_1.bias": "3.fc2.bias",
            "q2_.linear_1.weight": "3.fc2.weight",
            "q2_.linear_2.bias": "3.head.bias",
            "q2_.linear_2.weight": "3.head.weight",
            "q2tgt_.linear_0.bias": "4.fc1.bias",
            "q2tgt_.linear_0.weight": "4.fc1.weight",
            "q2tgt_.linear_1.bias": "4.fc2.bias",
            "q2tgt_.linear_1.weight": "4.fc2.weight",
            "q2tgt_.linear_2.bias": "4.head.bias",
            "q2tgt_.linear_2.weight": "4.head.weight",
        },
        "idris": "packages/idris-ml-examples/src/Example/Sac.idr",
        "python": "packages/pytorch/torch_ref/scripts/sac.py",
    },
]
