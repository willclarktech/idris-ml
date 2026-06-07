"""Multi-head Transformer for sequence reversal.

Pre-LN architecture with learned embeddings, sinusoidal positional encoding,
multi-head causal self-attention, and feedforward layers. Supports N stacked
blocks (default 1).

Architecture matches the Idris implementation:
- Per-head separate Q/K/V weights (not one big projection split into heads)
- Sum-not-concat: per-head output projections summed instead of concatenating
  heads then projecting. Mathematically equivalent to concat @ Wo.
- Pre-LN: LayerNorm before attention and FFN (better training stability)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_ref.training.runner import get_device


class TransformerBlock(nn.Module):
    """Single Pre-LN transformer block: attention + FFN with residuals."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, causal_mask: Tensor) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Register the mask as a buffer so `model.to(device)` moves it
        # alongside the parameters. Without this, .to(mps) on the outer
        # model leaves this inner-block reference pointing at the
        # original CPU tensor.
        self.register_buffer("causal_mask", causal_mask)
        self.causal_mask: Tensor

        self.query_ws = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.key_ws = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.value_ws = nn.ModuleList(
            [nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.out_proj_ws = nn.ModuleList(
            [nn.Linear(self.head_dim, d_model, bias=False) for _ in range(num_heads)]
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim, bias=False)
        self.ff2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        # Pre-LN Multi-Head Attention
        normed = self.norm1(h)
        attn_out = torch.zeros_like(h)
        for i in range(self.num_heads):
            q = self.query_ws[i](normed)
            k = self.key_ws[i](normed)
            v = self.value_ws[i](normed)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.causal_mask, -1e20)
            attn = F.softmax(scores, dim=-1)
            head_out = torch.matmul(attn, v)
            attn_out = attn_out + self.out_proj_ws[i](head_out)
        h = h + attn_out

        # Pre-LN Feedforward
        normed = self.norm2(h)
        h = h + self.ff2(F.relu(self.ff1(normed)))
        return h


class MultiHeadTransformer(nn.Module):
    """Pre-LN Transformer with N stacked blocks, learned embeddings, and output projection.

    Input: token indices (LongTensor [batch, seqLen])
    Output: logits [batch, seqLen, vocabSize]
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_blocks: int = 1,
        ff_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.head_dim = d_model // num_heads
        self.ff_dim = ff_dim or 4 * d_model

        self.token_embed = nn.Linear(vocab_size, d_model, bias=False)
        self.register_buffer("pos_enc", self._sinusoidal_pe(seq_len, d_model))

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.causal_mask: Tensor

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, self.ff_dim, self.causal_mask)
                for _ in range(num_blocks)
            ]
        )

        self.norm_final = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
        """Standard sinusoidal positional encoding [seq_len, d_model]."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded tokens [batch, seqLen, vocabSize]
               or [seqLen, vocabSize] (unbatched)

        Returns:
            Logits [batch, seqLen, vocabSize] or [seqLen, vocabSize]
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)

        h = self.token_embed(x) + self.pos_enc

        for block in self.blocks:
            h = block(h)

        logits = self.vocab_proj(self.norm_final(h))

        if unbatched:
            logits = logits.squeeze(0)

        return logits


def generate_sorting_data(
    num_samples: int,
    input_len: int,
    vocab_size: int,
    sep_token: int,
    eos_token: int,
) -> list[tuple[Tensor, Tensor]]:
    """Generate sequence sorting training data.

    Each sample: [t0, t1, ..., t_{n-1}, SEP, sorted_0, ..., sorted_{n-1}, EOS]
    Target: shifted by one (predict next token at each position).
    Tokens are sampled from [0, vocab_size - 2) to leave room for SEP and EOS.

    Returns list of (input_onehot, target_indices) pairs.
    """
    data: list[tuple[Tensor, Tensor]] = []
    base_vocab = vocab_size - 2
    device = get_device()
    for _ in range(num_samples):
        tokens = torch.randint(0, base_vocab, (input_len,), device=device)
        sorted_tokens, _ = tokens.sort()
        seq = torch.cat(
            [
                tokens,
                torch.tensor([sep_token], device=device),
                sorted_tokens,
                torch.tensor([eos_token], device=device),
            ]
        )
        inp = seq[:-1]
        tgt = seq[1:]
        inp_onehot = F.one_hot(inp.long(), vocab_size).float()
        data.append((inp_onehot, tgt.long()))
    return data


def generate_reversal_data(
    num_samples: int,
    input_len: int,
    vocab_size: int,
    sep_token: int,
    eos_token: int,
) -> list[tuple[Tensor, Tensor]]:
    """Generate sequence reversal training data.

    Each sample: [t0, t1, ..., t_{n-1}, SEP, t_{n-1}, ..., t0, EOS]
    Target: shifted by one (predict next token at each position).
    Tokens are sampled from [0, vocab_size - 2) to leave room for SEP and EOS.

    Returns list of (input_onehot, target_indices) pairs.
    """
    data: list[tuple[Tensor, Tensor]] = []
    base_vocab = vocab_size - 2
    device = get_device()
    for _ in range(num_samples):
        tokens = torch.randint(0, base_vocab, (input_len,), device=device)
        seq = torch.cat(
            [
                tokens,
                torch.tensor([sep_token], device=device),
                tokens.flip(0),
                torch.tensor([eos_token], device=device),
            ]
        )
        inp = seq[:-1]
        tgt = seq[1:]
        inp_onehot = F.one_hot(inp.long(), vocab_size).float()
        data.append((inp_onehot, tgt.long()))
    return data


def train_reversal_epoch(
    model: MultiHeadTransformer,
    data: list[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    reversal_start: int = 0,
) -> float:
    """Train one epoch on reversal data.

    Loss is computed only on positions >= reversal_start (the reversal
    portion). Prefix positions are random and unpredictable from left
    context, so masking them out makes the loss meaningful.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=get_device())

    for inp_onehot, target_indices in data:
        logits = model(inp_onehot)
        loss = F.cross_entropy(logits[reversal_start:], target_indices[reversal_start:])
        total_loss = total_loss + loss

    avg_loss = total_loss / len(data)
    # torch's Tensor.backward stub leaves its params unannotated.
    avg_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return avg_loss.item()


def eval_reversal_accuracy(
    model: MultiHeadTransformer,
    data: list[tuple[Tensor, Tensor]],
    reversal_start: int,
) -> tuple[float, float]:
    """Evaluate accuracy on reversal portion.

    Args:
        reversal_start: index where the reversed tokens begin in the target
                        (= input_len, since target is shifted by 1)

    Returns:
        (full_accuracy, reversal_accuracy)
    """
    total_correct = 0
    total_positions = 0
    rev_correct = 0
    rev_positions = 0

    with torch.no_grad():
        for inp_onehot, target_indices in data:
            logits = model(inp_onehot)
            preds = logits.argmax(dim=-1)

            total_correct += (preds == target_indices).sum().item()
            total_positions += target_indices.numel()

            rev_preds = preds[reversal_start:]
            rev_targets = target_indices[reversal_start:]
            rev_correct += (rev_preds == rev_targets).sum().item()
            rev_positions += rev_targets.numel()

    full_acc = total_correct / total_positions if total_positions > 0 else 0.0
    rev_acc = rev_correct / rev_positions if rev_positions > 0 else 0.0
    return full_acc, rev_acc
