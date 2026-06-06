"""BERT SST-2 LoRA fine-tune.

Paired reference for `Example/BertClassifySst2Lora.idr`. Mirrors the
full-fine-tune reference but wraps the model with HuggingFace `peft`:

    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                        target_modules=["query", "value"],
                        lora_dropout=0.0, bias="none")
    peft_model = get_peft_model(base_model, config)

Same architecture / dataset / seqLen / optimizer (AdamW lr=1e-4,
weight_decay=0.01, gradient clip 1.0) as the Idris side.

Usage:
    python torch_ref/scripts/bert_classify_sst2_lora_finetune.py \\
        --max-train 256 --max-dev 256 --epochs 3 --lora-rank 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BertConfig, BertForSequenceClassification


# Architecture (matches the on-disk google/bert_uncased_L-2_H-128_A-2
# tiny checkpoint + the Idris-side BertClassifySst2Lora config).
VOCAB = 30522
HIDDEN = 128
NUM_LAYERS = 2
NUM_HEADS = 2
INTERMEDIATE = 512
MAX_POS = 512
TYPE_VOCAB = 2
NUM_CLASSES = 2
SEQ_LEN = 32
PAD_ID = 0

REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_TSV = REPO_ROOT / "data" / "hf-datasets" / "glue-sst2" / "train.tsv"
DEV_TSV = REPO_ROOT / "data" / "hf-datasets" / "glue-sst2" / "validation.tsv"
BACKBONE_DIR = REPO_ROOT / "models" / "google" / "bert_uncased_L-2_H-128_A-2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # peft tutorial reports lr=1e-4 as a sweet spot for BERT-tiny LoRA;
    # higher than full FT's 2e-5 because only the adapters update.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=256,
                   help="Cap on train examples (0 = use all).")
    p.add_argument("--max-dev", type=int, default=256,
                   help="Cap on dev examples for held-out eval (0 = all).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA rank `r` (peft default = 8).")
    p.add_argument("--lora-alpha", type=float, default=16.0,
                   help="LoRA alpha scaling factor (peft default = 16).")
    p.add_argument("--save-adapter", type=str, default="",
                   help="If set, save the trained adapter to this directory via peft.save_pretrained.")
    return p.parse_args()


def load_tsv(path: Path, cap: int) -> list[tuple[list[int], int]]:
    if not path.exists():
        print(f"ERROR: {path} not found - run `make data-sst2`.", file=sys.stderr)
        sys.exit(1)
    out: list[tuple[list[int], int]] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label_str, ids_str = line.split("\t", 1)
            label = int(label_str)
            ids = [int(x) for x in ids_str.split(",") if x.strip()]
            out.append((ids, label))
    if cap > 0:
        out = out[:cap]
    return out


def pad_or_truncate(ids: list[int], seq_len: int, pad_id: int) -> tuple[list[int], list[int]]:
    if len(ids) >= seq_len:
        return ids[:seq_len], [1] * seq_len
    pad_n = seq_len - len(ids)
    return ids + [pad_id] * pad_n, [1] * len(ids) + [0] * pad_n


def evaluate_model(model, examples: list[tuple[list[int], int]],
                   device: torch.device) -> float:
    was_training = model.training
    model.train(False)
    hits = 0
    with torch.no_grad():
        for ids, label in examples:
            padded_ids, mask = pad_or_truncate(ids, SEQ_LEN, PAD_ID)
            input_ids = torch.tensor([padded_ids], device=device)
            attention_mask = torch.tensor([mask], device=device)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            pred = int(logits.argmax(dim=-1).item())
            if pred == label:
                hits += 1
    model.train(was_training)
    return hits / max(1, len(examples))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=== BertClassifySst2Lora (PyTorch ref) ===")
    print(f"Config: lr={args.lr} epochs={args.epochs} seed={args.seed}"
          f" lora-rank={args.lora_rank} lora-alpha={args.lora_alpha}")
    print(f"Subset: max-train={args.max_train} max-dev={args.max_dev}"
          f" batch={args.batch_size}")

    train_items = load_tsv(TRAIN_TSV, args.max_train)
    dev_items = load_tsv(DEV_TSV, args.max_dev)
    print(f"Loaded: train={len(train_items)} dev={len(dev_items)}")

    cfg = BertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POS,
        type_vocab_size=TYPE_VOCAB,
        num_labels=NUM_CLASSES,
    )
    base = BertForSequenceClassification.from_pretrained(
        str(BACKBONE_DIR), config=cfg, ignore_mismatched_sizes=True)
    print("Backbone warm-started; head at fresh init.")

    # LoRA wrap: target_modules=["query","value"] is peft's canonical
    # default for BERT — matches the Idris-side loraInjectBert call.
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.print_trainable_parameters()
    device = torch.device("cpu")
    peft_model.to(device)

    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    last_loss = 0.0
    for epoch in range(args.epochs):
        peft_model.train(True)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_items), args.batch_size):
            batch = train_items[i:i + args.batch_size]
            ids_batch = []
            mask_batch = []
            label_batch = []
            for ids, label in batch:
                padded_ids, mask = pad_or_truncate(ids, SEQ_LEN, PAD_ID)
                ids_batch.append(padded_ids)
                mask_batch.append(mask)
                label_batch.append(label)
            input_ids = torch.tensor(ids_batch, device=device)
            attention_mask = torch.tensor(mask_batch, device=device)
            labels = torch.tensor(label_batch, device=device)

            opt.zero_grad()
            logits = peft_model(input_ids=input_ids,
                                attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
        last_loss = epoch_loss / max(1, n_batches)
        acc = evaluate_model(peft_model, dev_items, device)
        print(f"Epoch {epoch + 1}: loss={last_loss:.4f}  dev-acc={acc:.3f}")

    final_acc = evaluate_model(peft_model, dev_items, device)
    wall = time.time() - start_time

    if args.save_adapter:
        peft_model.save_pretrained(args.save_adapter)
        print(f"Saved LoRA adapter to {args.save_adapter}")

    print()
    print(f"RESULT\tloss={last_loss:.4f}"
          f"\taccuracy={final_acc:.3f}"
          f"\tepochs={args.epochs}"
          f"\tseed={args.seed}"
          f"\twall_s={wall:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
