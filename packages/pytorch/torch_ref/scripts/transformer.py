"""Transformer sequence sorting training script.

Output format matches the Idris Example.Transformer exactly:
banner, per-epoch progress, timing summary, evaluation, RESULT line.

Usage:
    python -m torch_ref.scripts.transformer [--lr 0.001] [--epochs 1000] [--blocks 2] [--seed 42]
"""

import argparse
import os
import sys
from typing import cast

import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    eval_reversal_accuracy,
    generate_sorting_data,
    train_reversal_epoch,
)
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import TrainConfig, format_result, run_training, set_device

# Task config matching Idris
VOCAB_SIZE = 8  # digits 0-5 + SEP + EOS
INPUT_LEN = 5
SEQ_LEN = 2 * INPUT_LEN + 1  # 11
SEP_TOKEN = 6
EOS_TOKEN = 7
D_MODEL = 16
NUM_HEADS = 4
HEAD_DIM = 4
BATCH_SIZE = 16

# Idris registry name -> this script's parameter name, model-index prefixed.
# Mirrors the entry in scripts/paired_examples.py, which check-step-oracle.py
# cross-checks.
PAIRED_PARAMS = {
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
}


def _int_list(t: torch.Tensor) -> list[int]:
    """Typed view of an integer tensor — tolist() returns list[Unknown] in torch's stubs."""
    return cast("list[int]", t.tolist())  # pyright: ignore[reportUnknownMemberType]


def token_name(n: int) -> str:
    if n < 6:
        return str(n)
    if n == SEP_TOKEN:
        return "|"
    if n == EOS_TOKEN:
        return "$"
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="Run lr_find (LR-range test) instead of training, then exit.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    set_device(args.device)
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]

    print("=== Transformer: Sequence Sorting ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} patience={args.patience} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} headDim={HEAD_DIM}"
        f" blocks={args.blocks} vocab={VOCAB_SIZE}"
    )

    model = MultiHeadTransformer(
        VOCAB_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_HEADS,
        num_blocks=args.blocks,
    ).to(args.device)
    maybe_dump_init(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"Model: Transformer<{SEQ_LEN}x{D_MODEL} h={NUM_HEADS} blocks={args.blocks} v={VOCAB_SIZE}>"
    )
    print()

    # Oracle run: publish the parameters and the batch's raw token draws (a
    # sample's tokens are the first INPUT_LEN ids of its input; recorded
    # sample-major, the order the Idris side draws them), take exactly one
    # update and publish the result. Idris replays the tokens through
    # --replay and rebuilds the same batch — sample construction, forward,
    # masked CE, clip and Adam are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        data = generate_sorting_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        tokens = [t for inp, _tgt in data for t in _int_list(inp.argmax(dim=-1))[:INPUT_LEN]]
        maybe_dump_oracle((model,), PAIRED_PARAMS)
        write_replay(os.environ["IDRISML_ORACLE_DUMP"] + ".replay", choices=tokens)
        train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)
        maybe_dump_after_step((model,), PAIRED_PARAMS)

    def epoch_fn() -> float:
        data = generate_sorting_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        return train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

    def metrics_fn() -> list[tuple[str, str]]:
        eval_data = generate_sorting_data(BATCH_SIZE, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        _, acc = eval_reversal_accuracy(model, eval_data, INPUT_LEN)
        correct = int(acc * BATCH_SIZE * (SEQ_LEN - INPUT_LEN))
        total = BATCH_SIZE * (SEQ_LEN - INPUT_LEN)
        return [("sort_acc", f"{correct}/{total}")]

    if args.lr_find:
        lr_find(LrFindConfig(num_iters=100), epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
        device=args.device,
    )

    epochs_done, _final_loss = run_training(epoch_fn, config, metrics_fn)

    # Evaluation
    print()
    print("Evaluation:")
    eval_data = generate_sorting_data(1, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
    inp_onehot, target_indices = eval_data[0]
    with torch.no_grad():
        logits = model(inp_onehot)
        preds = logits.argmax(dim=-1)

    input_decoded = _int_list(inp_onehot.argmax(dim=-1))
    target_decoded = _int_list(target_indices)
    predicted = _int_list(preds)

    input_tokens = input_decoded[:INPUT_LEN]
    sort_target = target_decoded[INPUT_LEN:]
    sort_predicted = predicted[INPUT_LEN:]
    print(f"  Input:      {''.join(token_name(t) for t in input_tokens)}")
    print(f"  Target:     {''.join(token_name(t) for t in sort_target)}")
    print(f"  Predicted:  {''.join(token_name(t) for t in sort_predicted)}")

    sort_correct = sum(
        1 for p, t in zip(predicted[INPUT_LEN:], target_decoded[INPUT_LEN:], strict=True) if p == t
    )
    sort_total = SEQ_LEN - INPUT_LEN

    print(f"  Sort acc:   {sort_correct}/{sort_total}")

    print()
    print(
        format_result(
            [
                ("epochs", str(epochs_done)),
                ("sort_acc", f"{sort_correct}/{sort_total}"),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()
