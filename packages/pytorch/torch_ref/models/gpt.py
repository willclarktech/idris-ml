"""Character-level GPT on Shakespeare text.

Minimal char-level language model following Karpathy's char-rnn/minGPT tradition.
Uses the same MultiHeadTransformer architecture as the sorting example but trained
on next-character prediction over an embedded Shakespeare corpus.

Architecture matches the Idris implementation:
- Pre-LN with per-head Q/K/V weights, sum-not-concat output
- Learned token embeddings + sinusoidal positional encoding
- Cross-entropy loss on all positions (no masking)
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.multi_head_transformer import MultiHeadTransformer

# ---------------------------------------------------------------------------
# Corpus & vocabulary
# ---------------------------------------------------------------------------

# Shakespeare — "All the world's a stage" + Hamlet soliloquy, lowercased
CORPUS = (
    "all the world's a stage, and all the men and women merely players; "
    "they have their exits and their entrances, and one man in his time "
    "plays many parts, his acts being seven ages. at first, the infant, "
    "mewling and puking in the nurse's arms. then the whining schoolboy, "
    "with his satchel and shining morning face, creeping like snail "
    "unwillingly to school. and then the lover, sighing like a furnace, "
    "with a woeful ballad made to his mistress' eyebrow. then a soldier, "
    "full of strange oaths and bearded like the pard, jealous in honour, "
    "sudden and quick in quarrel, seeking the bubble reputation even in "
    "the cannon's mouth. and then the justice, in fair round belly with "
    "good capon lined, with eyes severe and beard of formal cut, full of "
    "wise saws and modern instances; and so he plays his part. "
    "to be or not to be, that is the question; whether 'tis nobler in "
    "the mind to suffer the slings and arrows of outrageous fortune, or "
    "to take arms against a sea of troubles, and by opposing end them. "
    "to die, to sleep; no more; and by a sleep to say we end the "
    "heartache and the thousand natural shocks that flesh is heir to; "
    "'tis a consummation devoutly to be wished. to die, to sleep; to "
    "sleep, perchance to dream. ay, there's the rub, for in that sleep "
    "of death what dreams may come, when we have shuffled off this mortal "
    "coil, must give us pause."
)

# 36-char vocab: a-z (0-25), space (26), newline (27), . , ' ; : ! ? - (28-35)
VOCAB = "abcdefghijklmnopqrstuvwxyz \n.,';:!?-"
VOCAB_SIZE = len(VOCAB)  # 36

_CHAR_TO_IDX: dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}


def char_to_idx(ch: str) -> int:
    """Map character to token index. Unknown chars map to space (26)."""
    return _CHAR_TO_IDX.get(ch, 26)


def idx_to_char(idx: int) -> str:
    """Map token index back to character."""
    if 0 <= idx < VOCAB_SIZE:
        return VOCAB[idx]
    return " "


def encode_corpus(text: str) -> list[int]:
    """Convert text to list of token indices."""
    return [char_to_idx(ch) for ch in text]


CORPUS_INDICES = encode_corpus(CORPUS)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_gpt_data(
    corpus: list[int],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Generate sliding-window training data from corpus.

    Each sample: random offset into corpus, extract seq_len+1 tokens.
    Input = tokens[0:seq_len] one-hot, Target = tokens[1:seq_len+1] indices.
    """
    max_start = len(corpus) - seq_len - 1
    data = []
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        window = corpus[start : start + seq_len + 1]
        inp = torch.tensor(window[:seq_len], dtype=torch.long)
        tgt = torch.tensor(window[1 : seq_len + 1], dtype=torch.long)
        inp_onehot = F.one_hot(inp, vocab_size).float()
        data.append((inp_onehot, tgt))
    return data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gpt_epoch(
    model: MultiHeadTransformer,
    data: list[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one epoch: CE loss on all positions."""
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0)

    for inp_onehot, target_indices in data:
        logits = model(inp_onehot)  # [seq_len, vocab_size]
        loss = F.cross_entropy(logits, target_indices)
        total_loss = total_loss + loss

    avg_loss = total_loss / len(data)
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return avg_loss.item()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_text(
    model: MultiHeadTransformer,
    seed_text: str,
    length: int,
    temperature: float = 1.0,
) -> str:
    """Autoregressive text generation.

    Seeds with seed_text, generates `length` additional characters.
    """
    seq_len = model.seq_len
    vocab_size = model.vocab_size

    # Encode seed, pad/truncate to seq_len
    tokens = encode_corpus(seed_text)[-seq_len:]
    if len(tokens) < seq_len:
        tokens = [26] * (seq_len - len(tokens)) + tokens  # pad with spaces

    result = list(seed_text)

    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor(tokens, dtype=torch.long)
            inp_onehot = F.one_hot(inp, vocab_size).float()
            logits = model(inp_onehot)  # [seq_len, vocab_size]

            # Take logits at last position
            last_logits = logits[-1] / temperature
            probs = F.softmax(last_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            result.append(idx_to_char(int(next_token)))
            tokens = tokens[1:] + [int(next_token)]

    return "".join(result)


# ---------------------------------------------------------------------------
# Convenience training function
# ---------------------------------------------------------------------------


def train_gpt(
    epochs: int = 2000,
    seed: int = 42,
    lr: float = 0.001,
    seq_len: int = 64,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    batch_size: int = 32,
) -> tuple[MultiHeadTransformer, list[float]]:
    """Train a char-level GPT and return (model, loss_history)."""
    torch.manual_seed(seed)
    random.seed(seed)

    model = MultiHeadTransformer(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    for _ in range(epochs):
        data = generate_gpt_data(CORPUS_INDICES, batch_size, seq_len, VOCAB_SIZE)
        loss = train_gpt_epoch(model, data, optimizer)
        history.append(loss)

    return model, history


def evaluate_bpc(
    model: MultiHeadTransformer,
    corpus: list[int],
    seq_len: int,
    n_samples: int = 50,
) -> float:
    """Evaluate bits per character on random windows from corpus."""
    total_loss = 0.0
    max_start = len(corpus) - seq_len - 1

    with torch.no_grad():
        for _ in range(n_samples):
            start = random.randint(0, max_start)
            window = corpus[start : start + seq_len + 1]
            inp = torch.tensor(window[:seq_len], dtype=torch.long)
            tgt = torch.tensor(window[1 : seq_len + 1], dtype=torch.long)
            inp_onehot = F.one_hot(inp, VOCAB_SIZE).float()
            logits = model(inp_onehot)
            loss = F.cross_entropy(logits, tgt)
            total_loss += loss.item()

    avg_loss = total_loss / n_samples
    return avg_loss / math.log(2)  # nats -> bits
