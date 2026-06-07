"""GPT-2 LM continued pretraining on Tiny Shakespeare.

Paired reference for the Idris-side `Example/Gpt2LmFinetune.idr`. Loads
the distilgpt2 backbone via `transformers.AutoModelForCausalLM`, reads
the same pre-tokenized Tiny Shakespeare corpus, runs the same sliding-
window next-token CE loss with the same hyperparameters.

Pre-requisites (run once):
    make data-hf-distilgpt2                  # fetches the backbone
    make data-tinyshakespeare-distilgpt2     # tokenizes the corpus

Usage:
    python torch_ref/scripts/gpt2_lm_finetune.py --steps 100
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

SEQ_LEN = 32

REPO_ROOT = Path(__file__).resolve().parents[4]
TOKEN_PATH = REPO_ROOT / "data" / "tinyshakespeare" / "input.distilgpt2.tokens"
BACKBONE_DIR = REPO_ROOT / "models" / "distilgpt2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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
            f"ERROR: {path} not found - run `make data-tinyshakespeare-distilgpt2`.",
            file=sys.stderr,
        )
        sys.exit(1)
    with path.open() as f:
        text = f.read().strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def sample_window(tokens: list[int], cap_max_start: int) -> tuple[list[int], list[int]]:
    abs_max = max(0, len(tokens) - SEQ_LEN - 1)
    cap = abs_max if cap_max_start == 0 else min(abs_max, cap_max_start)
    start = random.randint(0, cap)
    window = tokens[start : start + SEQ_LEN + 1]
    input_tok = window[:SEQ_LEN]
    target_tok = window[1 : SEQ_LEN + 1]
    return input_tok, target_tok


def main() -> int:
    args = parse_args()
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]
    random.seed(args.seed)

    print("=== Gpt2LmFinetune (PyTorch ref) ===")
    print(f"Config: lr={args.lr} steps={args.steps} seed={args.seed} max-start={args.max_start}")

    print(f"Loading tokens from {TOKEN_PATH}...")
    tokens = load_tokens(TOKEN_PATH)
    print(f"Loaded {len(tokens)} tokens.")

    if len(tokens) < SEQ_LEN + 1:
        print(
            f"ERROR: corpus has only {len(tokens)} tokens (need >= {SEQ_LEN + 1}).", file=sys.stderr
        )
        return 1

    # transformers 5.x lazy attrs make Auto* from_pretrained return a
    # loose union pyright can't narrow; cast to the concrete class.
    model = cast(
        GPT2LMHeadModel,  # noqa: TC006 - unquoted so vulture sees the import used
        AutoModelForCausalLM.from_pretrained(str(BACKBONE_DIR)),  # pyright: ignore[reportUnknownMemberType]
    )
    device = torch.device("cpu")
    # transformers 5.x wraps Module.to in a decorator whose _Wrapped
    # type pyright can't bind as a method; the call is fine at runtime.
    model.to(device)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
    model.train(True)
    print("distilgpt2 backbone warm-started.")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    acc_loss = 0.0
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        input_tok, target_tok = sample_window(tokens, args.max_start)
        input_ids = torch.tensor([input_tok], device=device)
        targets = torch.tensor(target_tok, device=device)
        opt.zero_grad()
        logits = model(input_ids=input_ids).logits.squeeze(0)  # [seqLen, vocab]
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # torch's Optimizer.step stub leaves `closure` unannotated.
        opt.step()  # pyright: ignore[reportUnknownMemberType]
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
