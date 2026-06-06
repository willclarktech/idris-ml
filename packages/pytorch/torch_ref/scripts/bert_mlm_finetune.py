"""BERT MLM continued pretraining on Tiny Shakespeare.

Paired reference for the Idris-side `Example/BertMlmFinetune.idr`. Loads
the bert-tiny backbone (`google/bert_uncased_L-2_H-128_A-2`) WITH the
MLM head, reads the same pre-tokenized Tiny Shakespeare corpus,
samples sliding windows + applies HF's 80/10/10 masking, runs the
position-selective CE loss with the same hyperparameters.

Pre-requisites (run once):
    make data-hf-bert-tiny                # fetches the backbone
    make data-tinyshakespeare-bert-tiny   # tokenizes the corpus

Usage:
    python torch_ref/scripts/bert_mlm_finetune.py --steps 100
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM

SEQ_LEN = 32

# BERT WordPiece special tokens.
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
PAD_ID = 0

# HF's standard mask probability for MLM.
MASK_PROB = 0.15

REPO_ROOT = Path(__file__).resolve().parents[4]
TOKEN_PATH = REPO_ROOT / "data" / "tinyshakespeare" / "input.bert-tiny.tokens"
BACKBONE_DIR = REPO_ROOT / "models" / "google" / "bert_uncased_L-2_H-128_A-2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-start",
        type=int,
        default=0,
        help="Cap on the random start position (0 = full corpus).",
    )
    return p.parse_args()


def load_tokens(path: Path) -> list[int]:
    if not path.exists():
        print(
            f"ERROR: {path} not found - run `make data-tinyshakespeare-bert-tiny`.", file=sys.stderr
        )
        sys.exit(1)
    with path.open() as f:
        text = f.read().strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def apply_hf_masking(window: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Returns (masked_input_ids, target_ids, mask_flags). Mirrors
    the Idris-side `applyHfMasking`. Mask probability is 0.15;
    of those, 80% become [MASK], 10% become a fixed mid-vocab id
    (id=200, matching the Idris shortcut), 10% keep the original.
    CLS / SEP are never masked."""
    masked_input = []
    target = []
    flags = []
    for tok in window:
        if tok in (CLS_ID, SEP_ID) or random.random() > MASK_PROB:
            masked_input.append(tok)
            target.append(tok)
            flags.append(0)
        else:
            r = random.random()
            if r < 0.8:
                new_id = MASK_ID
            elif r < 0.9:
                new_id = 200
            else:
                new_id = tok
            masked_input.append(new_id)
            target.append(tok)
            flags.append(1)
    return masked_input, target, flags


def sample_window(tokens: list[int], cap_max_start: int) -> tuple[list[int], list[int], list[int]]:
    abs_max = max(0, len(tokens) - SEQ_LEN)
    cap = abs_max if cap_max_start == 0 else min(abs_max, cap_max_start)
    start = random.randint(0, cap)
    window = tokens[start : start + SEQ_LEN]
    return apply_hf_masking(window)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=== BertMlmFinetune (PyTorch ref) ===")
    print(f"Config: lr={args.lr} steps={args.steps} seed={args.seed} max-start={args.max_start}")

    print(f"Loading tokens from {TOKEN_PATH}...")
    tokens = load_tokens(TOKEN_PATH)
    print(f"Loaded {len(tokens)} tokens.")

    if len(tokens) < SEQ_LEN:
        print(f"ERROR: corpus has only {len(tokens)} tokens (need >= {SEQ_LEN}).", file=sys.stderr)
        return 1

    model = AutoModelForMaskedLM.from_pretrained(str(BACKBONE_DIR))
    device = torch.device("cpu")
    model.to(device)
    model.train(True)
    print("bert-tiny backbone + MLM head warm-started.")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    start_time = time.time()
    acc_loss = 0.0
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        masked_input, target, flags = sample_window(tokens, args.max_start)
        input_ids = torch.tensor([masked_input], device=device)
        # HF expects -100 at non-masked positions to disable loss there.
        # Match the Idris-side semantics: only masked positions
        # contribute to the loss.
        labels = torch.tensor(
            [[tgt if flag else -100 for tgt, flag in zip(target, flags, strict=True)]],
            device=device,
        )
        opt.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
        acc_loss += last_loss
        if step % 10 == 0:
            ema = acc_loss / step
            print(f"  step {step}/{args.steps} loss={last_loss:.4f}  ema={ema:.4f}")
    wall = time.time() - start_time

    print()
    print(f"RESULT\tloss={last_loss:.4f}\tsteps={args.steps}\tseed={args.seed}\twall_s={wall:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
