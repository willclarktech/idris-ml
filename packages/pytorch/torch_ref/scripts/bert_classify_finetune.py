"""BERT classification fine-tune on a synthetic 3-class task.

Paired reference for the Idris-side `Example/BertClassifyFinetune.idr`.
Same architecture, same hyperparameters, same synthetic dataset
generator. Convergence on tape (Idris) and CPU (PyTorch) at the same
seed should produce similar accuracy at the same epoch count.

Arch: a tiny BERT (vocab=64, hidden=32, layers=1, heads=2,
intermediate=64, maxPos=8, typeVocab=2) + BertForSequenceClassification
classifier head (numClasses=3). Built from a `BertConfig` so we get the
exact same module layout HF's `transformers` package ships.

Task: 8-token sequences with the class-encoding token at position 1.
Class 0 -> token 11, class 1 -> token 13, class 2 -> token 17. Positions
2-6 are random distractor tokens in [20, 63]; positions 0 and 7 are
CLS/SEP equivalents (100, 101).

Usage:
    python -m torch_ref.scripts.bert_classify_finetune [--lr 1e-3] [--epochs 2000] [--seed 42]
"""

import argparse
import random
import time

import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

# Config (matches Idris-side BertClassifyFinetune.idr)
VOCAB = 64
HIDDEN = 32
NUM_LAYERS = 1
NUM_HEADS = 2
INTERMEDIATE = 64
MAX_POS = 8
TYPE_VOCAB = 2
NUM_CLASSES = 3
SEQ_LEN = 8
BATCH_SIZE = 16

LABEL_TOKENS = [11, 13, 17]
DISTRACTOR_LO = 20
DISTRACTOR_HI = 60
CLS_TOKEN = 0
SEP_TOKEN = 1


def build_example(label: int) -> list:
    label_tok = LABEL_TOKENS[label]
    distractors = [random.randint(DISTRACTOR_LO, DISTRACTOR_HI) for _ in range(5)]
    return [CLS_TOKEN, label_tok, *distractors, SEP_TOKEN]


def gen_batch(n: int, device: torch.device):
    inputs, labels = [], []
    for _ in range(n):
        c = random.randint(0, NUM_CLASSES - 1)
        inputs.append(build_example(c))
        labels.append(c)
    input_ids = torch.tensor(inputs, dtype=torch.long, device=device)
    return input_ids, torch.tensor(labels, dtype=torch.long, device=device)


@torch.no_grad()
def held_out_accuracy(model, device) -> float:
    was_training = model.training
    model.train(False)
    ids, lbls = gen_batch(32, device)
    out = model(input_ids=ids)
    preds = out.logits.argmax(dim=-1)
    hits = (preds == lbls).sum().item()
    model.train(was_training)
    return hits / 32.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the BERT encoder + pooler; only the classifier head trains.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for tensor ops (default: cpu)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    print("=== BertClassifyFinetune (PyTorch ref): synthetic 3-class fine-tune ===")
    print(
        f"Config: lr={args.lr} epochs={args.epochs}"
        f" patience={args.patience} seed={args.seed}"
        f" freeze-backbone={args.freeze_backbone}"
    )
    print(
        f"Arch: vocab={VOCAB} hidden={HIDDEN}"
        f" layers={NUM_LAYERS} heads={NUM_HEADS} classes={NUM_CLASSES}"
    )

    cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POS,
        type_vocab_size=TYPE_VOCAB,
        num_labels=NUM_CLASSES,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = BertForSequenceClassification(cfg).to(device, dtype=torch.float64)

    if args.freeze_backbone:
        print("Freezing backbone (`bert.*`); only classifier head trains.")
        for name, p in model.named_parameters():
            if name.startswith("bert."):
                p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    loss_fn = nn.CrossEntropyLoss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    print("Training...")
    t0 = time.time()
    best_loss = float("inf")
    stale = 0
    epochs_done = 0
    last_loss_val = float("nan")
    for epoch in range(args.epochs):
        ids, lbls = gen_batch(BATCH_SIZE, device)
        logits = model(input_ids=ids).logits
        loss = loss_fn(logits, lbls)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epochs_done = epoch + 1
        last_loss_val = loss.item()
        if epoch == 0 or (epoch + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:07.2f}s] {epoch + 1}\tloss={last_loss_val:.6f}")

        if last_loss_val < best_loss - 1e-3:
            best_loss = last_loss_val
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  Early stop at epoch {epoch + 1} (patience={args.patience})")
                break

    train_elapsed = time.time() - t0
    ms_per_ep = train_elapsed * 1000.0 / epochs_done if epochs_done > 0 else 0.0
    print(f"PERF_MS_PER_EP={ms_per_ep:.6f}")

    accuracy = held_out_accuracy(model, device)
    print(
        f"RESULT\tloss={last_loss_val:.4f}\taccuracy={accuracy:.3f}"
        f"\tepochs={epochs_done}\tseed={args.seed}"
    )


if __name__ == "__main__":
    main()
