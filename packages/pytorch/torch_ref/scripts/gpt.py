"""GPT character-level language model training script.

Output format matches the Idris Example.Gpt exactly:
banner, per-epoch progress, timing summary, generation sample, RESULT line.

Two corpus paths:
- `--corpus tinyshakespeare` (default): 1.1 M chars, 65-char dynamic vocab
  loaded from data/tinyshakespeare/input.txt. Train/val split 90/10. The
  RESULT line emits `val_bpc` (held-out). This is the canonical char-LM
  benchmark setup used by nanoGPT.
- `--corpus embedded`: 1342-char hardcoded excerpt on the same 65-char vocab.
  Fast wiring test; emits `bpc` on the (training) corpus. Used by the
  smoke gate.

Optimizer recipe is aligned with nanoGPT/train_shakespeare_char.py for the
tinyshakespeare path: AdamW β2=0.99 wd=0.1 + cosine LR with warmup.

Usage:
    python -m torch_ref.scripts.gpt [--corpus tinyshakespeare|embedded]
                                    [--lr 1e-3] [--epochs 1000] [--seed 42]
"""

import argparse
import math
import os
import random
import sys

import torch

from torch_ref.init_manifest import (
    maybe_dump_after_step,
    maybe_dump_init,
    maybe_dump_oracle,
)
from torch_ref.models.gpt import (
    CORPUS_INDICES,
    VOCAB_SIZE,
    evaluate_bpc,
    generate_gpt_data,
    generate_text,
    load_tinyshakespeare,
    train_gpt_epoch,
    train_val_split,
)
from torch_ref.models.multi_head_transformer import MultiHeadTransformer
from torch_ref.replay import write_replay
from torch_ref.training.lr_finder import LrFindConfig, lr_find
from torch_ref.training.runner import TrainConfig, format_result, run_training, set_device

# Architecture config matching Idris (kept small for tape-backend tractability;
# nanoGPT's full GPU defaults are 6 layers / 6 heads / 384 dim).
SEQ_LEN = 64
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
BATCH_SIZE = 32

# Idris registry name -> this script's parameter name, model-index prefixed.
# Mirrors the entry in scripts/paired_examples.py, which check-step-oracle.py
# cross-checks. Same map as transformer.py — both drive MultiHeadTransformer.
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

# nanoGPT optimizer/schedule defaults (train_shakespeare_char.py).
BETA1 = 0.9
BETA2 = 0.99  # default torch is 0.999; nanoGPT uses 0.99 for char-LM
WEIGHT_DECAY = 0.1  # default 0.01; nanoGPT uses 0.1
GRAD_CLIP = 1.0
MIN_LR_FACTOR = 0.1  # min_lr = base_lr * MIN_LR_FACTOR


def cosine_lr(epoch: int, base_lr: float, max_epochs: int) -> float:
    """Cosine LR schedule with linear warmup. Verbatim nanoGPT formula at
    max_epochs >= 1000; for shorter smoke runs (embedded/30) the warmup
    is capped at max_epochs/10 so the LR actually ramps."""
    warmup_epochs = min(100, max_epochs // 10)
    min_lr = base_lr * MIN_LR_FACTOR
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / (warmup_epochs + 1)
    if epoch >= max_epochs:
        return min_lr
    decay_ratio = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["tinyshakespeare", "embedded"], default="embedded")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="0 disables patience; rely on cosine LR for annealing",
    )
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
    random.seed(args.seed)

    # --- Corpus + vocab -----------------------------------------------------
    if args.corpus == "tinyshakespeare":
        text, vocab, all_indices = load_tinyshakespeare()
        train_indices, val_indices = train_val_split(all_indices, val_frac=0.1)
        vocab_size = vocab.size
        corpus_label = (
            f"tinyshakespeare ({len(text)} chars, vocab={vocab_size}, "
            f"train={len(train_indices)}, val={len(val_indices)})"
        )
    else:
        vocab = None
        train_indices = CORPUS_INDICES
        val_indices = CORPUS_INDICES  # smoke path: same set
        vocab_size = VOCAB_SIZE
        corpus_label = f"embedded ({len(CORPUS_INDICES)} chars, vocab={vocab_size})"

    print("=== GPT: Character-Level Language Model ===")
    print(f"Config: corpus={args.corpus} lr={args.lr} epochs={args.epochs} seed={args.seed}")
    print(
        f"Architecture: seqLen={SEQ_LEN} dModel={D_MODEL}"
        f" heads={NUM_HEADS} blocks={NUM_BLOCKS} vocab={vocab_size}"
    )
    print(f"Corpus: {corpus_label}")

    model = MultiHeadTransformer(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
    ).to(args.device)
    maybe_dump_init(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")
    print()

    # Oracle run: publish the parameters and the batch's window offsets
    # (recorded in draw order), take exactly one update — at the schedule's
    # epoch-0 LR, as epoch_fn would — and publish the result. Idris replays
    # the offsets through --replay and rebuilds the same batch; window
    # slicing, one-hot construction, forward, LM loss, clip, cosine warmup
    # and AdamW are all under test.
    if os.environ.get("IDRISML_ORACLE_DUMP"):
        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(0, args.lr, args.epochs)
        starts: list[int] = []
        data = generate_gpt_data(train_indices, BATCH_SIZE, SEQ_LEN, vocab_size, starts_log=starts)
        maybe_dump_oracle((model,), PAIRED_PARAMS)
        write_replay(os.environ["IDRISML_ORACLE_DUMP"] + ".replay", choices=starts)
        train_gpt_epoch(model, data, optimizer)
        maybe_dump_after_step((model,), PAIRED_PARAMS)

    if args.lr_find:
        # One mini-batch update per iter, no LR schedule (lrFind controls LR).
        def lr_find_epoch_fn() -> float:
            data = generate_gpt_data(train_indices, BATCH_SIZE, SEQ_LEN, vocab_size)
            return train_gpt_epoch(model, data, optimizer)

        lr_find(LrFindConfig(num_iters=100), lr_find_epoch_fn, optimizer)
        print()
        print("Done — re-run without --lr-find at the recommended LR.")
        sys.exit(0)

    epoch_counter = {"i": 0}

    def epoch_fn() -> float:
        # Cosine LR with warmup
        ep = epoch_counter["i"]
        lr = cosine_lr(ep, args.lr, args.epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr
        epoch_counter["i"] = ep + 1

        data = generate_gpt_data(train_indices, BATCH_SIZE, SEQ_LEN, vocab_size)
        loss = train_gpt_epoch(model, data, optimizer)
        # Apply gradient clipping is done inside train_gpt_epoch (clip 1.0).
        return loss

    def metrics_fn() -> list[tuple[str, str]]:
        # Periodic val_bpc for progress logging
        val_bpc = evaluate_bpc(
            model,
            val_indices,
            SEQ_LEN,
            n_samples=20,
            vocab_size=vocab_size,
        )
        cur_lr = cosine_lr(epoch_counter["i"], args.lr, args.epochs)
        return [("val_bpc", f"{val_bpc:.3f}"), ("lr", f"{cur_lr:.5f}")]

    config = TrainConfig(
        total_epochs=args.epochs,
        log_every=100,
        patience=args.patience,
        min_delta=0.001,
        device=args.device,
    )

    epochs_done, _final_loss = run_training(epoch_fn, config, metrics_fn)

    # Final eval — held-out val_bpc on a larger sample
    print()
    val_bpc = evaluate_bpc(
        model,
        val_indices,
        SEQ_LEN,
        n_samples=50,
        vocab_size=vocab_size,
    )
    train_bpc = evaluate_bpc(
        model,
        train_indices,
        SEQ_LEN,
        n_samples=50,
        vocab_size=vocab_size,
    )
    print(f"Final val_bpc: {val_bpc:.3f}  (train_bpc: {train_bpc:.3f})")
    print()

    # Generation samples — use a corpus-appropriate seed
    seed1 = "to be or " if args.corpus == "tinyshakespeare" else "to be or "
    seed2 = "the " if args.corpus == "tinyshakespeare" else "the "
    print(f"Generation (seed={seed1!r}):")
    sample = generate_text(model, seed1, length=200, temperature=1.0, vocab=vocab)
    print(f"  {sample!r}")
    print()
    print(f"Generation (seed={seed2!r}):")
    sample2 = generate_text(model, seed2, length=200, temperature=1.0, vocab=vocab)
    print(f"  {sample2!r}")

    print()
    # RESULT line: emit val_bpc (held-out) for tinyshakespeare,
    # bpc (training-corpus) for embedded — keeps backward compat with
    # the smoke-gate threshold but switches the convergence path to a
    # real held-out metric.
    metric_key = "val_bpc" if args.corpus == "tinyshakespeare" else "bpc"
    metric_value = val_bpc
    print(
        format_result(
            [
                (metric_key, f"{metric_value:.3f}"),
                ("epochs", str(epochs_done)),
                ("seed", str(args.seed)),
            ]
        )
    )


if __name__ == "__main__":
    main()
