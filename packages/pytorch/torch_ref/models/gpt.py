"""Character-level GPT on Shakespeare text.

Minimal char-level language model following Karpathy's char-rnn/minGPT tradition.
Uses the same MultiHeadTransformer architecture as the sorting example but trained
on next-character prediction over a Shakespeare corpus.

Two corpus paths are supported:
- Embedded: a 1342-char hardcoded excerpt on the shared 65-char vocab
  (legacy; used by the smoke gate where a fast wiring test is enough).
- tinyshakespeare: 1.1 M chars, 65-char vocab (every distinct char in the file)
  loaded from `data/tinyshakespeare/input.txt`. The canonical char-LM benchmark
  used by nanoGPT; required for any "actually learned the task" claim.

Architecture matches the Idris implementation:
- Pre-LN with per-head Q/K/V weights, sum-not-concat output
- Learned token embeddings + sinusoidal positional encoding
- Cross-entropy loss on all positions (no masking)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_ref.models.multi_head_transformer import MultiHeadTransformer
from torch_ref.training.runner import get_device, get_dtype, multinomial_safe

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

# The tinyshakespeare 65-char set, byte-identical to Idris `Example.Gpt`'s
# `vocabChars` (same string, same order, so the two sides encode the corpus to
# the same token ids).
#
# This used to be a 36-char set covering only what the embedded corpus
# contains. Idris cannot follow: its `VocabSize` is a type-level `Nat`, so the
# vocab is fixed at compile time and cannot be derived per corpus. The two
# sides were therefore training different-sized models and reporting `bpc`
# against different denominators — the numbers were never comparable.
VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
VOCAB_SIZE = len(VOCAB)  # 65

_CHAR_TO_IDX: dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
_SPACE_IDX = _CHAR_TO_IDX[" "]


def char_to_idx(ch: str) -> int:
    """Map character to token index. Unknown chars map to space."""
    return _CHAR_TO_IDX.get(ch, _SPACE_IDX)


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
# Dynamic vocab + tinyshakespeare corpus loader (for convergence runs).
#
# nanoGPT builds a fresh vocab from every distinct character in the corpus.
# tinyshakespeare's 65-char vocab includes uppercase, numerals, and assorted
# punctuation that the embedded mapping above collapses or drops.
# ---------------------------------------------------------------------------


@dataclass
class Vocabulary:
    chars: list[str]  # ordered: chars[i] is the char at id i
    char_to_idx: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # Unknown chars map to space if present, else id 0.
        unk = self.char_to_idx.get(" ", 0)
        return [self.char_to_idx.get(ch, unk) for ch in text]

    def decode_idx(self, idx: int) -> str:
        if 0 <= idx < self.size:
            return self.chars[idx]
        return " "

    def decode(self, indices: list[int]) -> str:
        return "".join(self.decode_idx(i) for i in indices)

    @classmethod
    def from_text(cls, text: str) -> Vocabulary:
        chars = sorted(set(text))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        return cls(chars=chars, char_to_idx=char_to_idx)


def _default_tinyshakespeare_path() -> Path:
    # repo_root/data/tinyshakespeare/input.txt
    # __file__ = packages/pytorch/torch_ref/models/gpt.py
    # parents:    0=models 1=torch_ref 2=pytorch 3=packages 4=repo_root
    return Path(__file__).resolve().parents[4] / "data" / "tinyshakespeare" / "input.txt"


def load_tinyshakespeare(
    path: str | Path | None = None,
) -> tuple[str, Vocabulary, list[int]]:
    """Load tinyshakespeare and build vocab dynamically.

    Returns (text, vocab, indices). Raises FileNotFoundError with a hint to
    `make dataset-tinyshakespeare` if the file is missing.
    """
    p = Path(path) if path is not None else _default_tinyshakespeare_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"tinyshakespeare corpus not found at {p}. "
            f"Run `make dataset-tinyshakespeare` from the repo root."
        )
    text = p.read_text()
    vocab = Vocabulary.from_text(text)
    indices = vocab.encode(text)
    return text, vocab, indices


def train_val_split(indices: list[int], val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    """Deterministic train/val split — last val_frac of corpus is held out."""
    n_val = int(len(indices) * val_frac)
    n_train = len(indices) - n_val
    return indices[:n_train], indices[n_train:]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_gpt_data(
    corpus: list[int],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    starts_log: list[int] | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Generate sliding-window training data from corpus.

    Each sample: random offset into corpus, extract seq_len+1 tokens.
    Input = tokens[0:seq_len] one-hot, Target = tokens[1:seq_len+1] indices.
    `starts_log`, when given, collects the drawn offsets in draw order —
    the step oracle records them so the Idris side can replay the batch.
    """
    max_start = len(corpus) - seq_len - 1
    data: list[tuple[Tensor, Tensor]] = []
    device = get_device()
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        if starts_log is not None:
            starts_log.append(start)
        window = corpus[start : start + seq_len + 1]
        inp = torch.tensor(window[:seq_len], dtype=torch.long, device=device)
        tgt = torch.tensor(window[1 : seq_len + 1], dtype=torch.long, device=device)
        inp_onehot = F.one_hot(inp, vocab_size).to(get_dtype())
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
    total_loss = torch.tensor(0.0, device=get_device())

    for inp_onehot, target_indices in data:
        logits = model(inp_onehot)  # [seq_len, vocab_size]
        loss = F.cross_entropy(logits, target_indices)
        total_loss = total_loss + loss

    avg_loss = total_loss / len(data)
    # torch's Tensor.backward stub leaves its params unannotated.
    avg_loss.backward()  # pyright: ignore[reportUnknownMemberType]
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
    vocab: Vocabulary | None = None,
) -> str:
    """Autoregressive text generation.

    Seeds with seed_text, generates `length` additional characters.
    Pass `vocab=` to use a dynamic Vocabulary; otherwise the embedded
    shared 65-char mapping is used.
    """
    seq_len = model.seq_len
    vocab_size = model.vocab_size

    if vocab is None:
        # Legacy embedded-vocab path
        encode = encode_corpus
        decode = idx_to_char
        pad_id = 26  # space
    else:
        encode = vocab.encode
        decode = vocab.decode_idx
        pad_id = vocab.char_to_idx.get(" ", 0)

    # Encode seed, pad/truncate to seq_len
    tokens = encode(seed_text)[-seq_len:]
    if len(tokens) < seq_len:
        tokens = [pad_id] * (seq_len - len(tokens)) + tokens

    result = list(seed_text)

    with torch.no_grad():
        device = get_device()
        for _ in range(length):
            inp = torch.tensor(tokens, dtype=torch.long, device=device)
            inp_onehot = F.one_hot(inp, vocab_size).to(get_dtype())
            logits = model(inp_onehot)  # [seq_len, vocab_size]

            # Take logits at last position
            last_logits = logits[-1] / temperature
            probs = F.softmax(last_logits, dim=0)
            next_token = multinomial_safe(probs, 1).item()

            result.append(decode(int(next_token)))
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
    # torch's manual_seed stub leaves `seed` unannotated.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
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
    vocab_size: int = VOCAB_SIZE,
) -> float:
    """Evaluate bits per character on random windows from corpus.

    Defaults to the shared 65-char vocab; pass `vocab_size=` for a
    dynamic Vocabulary. n_samples > available windows clips at the
    corpus's max start index.
    """
    total_loss = 0.0
    max_start = len(corpus) - seq_len - 1
    if max_start < 0:
        return float("nan")

    with torch.no_grad():
        device = get_device()
        for _ in range(n_samples):
            start = random.randint(0, max_start)
            window = corpus[start : start + seq_len + 1]
            inp = torch.tensor(window[:seq_len], dtype=torch.long, device=device)
            tgt = torch.tensor(window[1 : seq_len + 1], dtype=torch.long, device=device)
            inp_onehot = F.one_hot(inp, vocab_size).to(get_dtype())
            logits = model(inp_onehot)
            loss = F.cross_entropy(logits, tgt)
            total_loss += loss.item()

    avg_loss = total_loss / n_samples
    return avg_loss / math.log(2)  # nats -> bits
