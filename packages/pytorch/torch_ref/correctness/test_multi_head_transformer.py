"""Correctness tests for multi-head transformer on sequence reversal and sorting."""

import torch

from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    eval_reversal_accuracy,
    generate_reversal_data,
    generate_sorting_data,
    train_reversal_epoch,
)

# Reversal task config
VOCAB_SIZE = 10  # 8 content tokens + SEP + EOS
INPUT_LEN = 5
SEQ_LEN = 2 * INPUT_LEN + 1  # 11
SEP_TOKEN = 8
EOS_TOKEN = 9
D_MODEL = 32
NUM_HEADS = 4

# Sorting task config
SORT_VOCAB = 8  # digits 0-5 + SEP + EOS
SORT_SEP = 6
SORT_EOS = 7


class TestMultiHeadTransformer:
    def test_output_shape(self) -> None:
        """Output should be [seqLen, vocabSize]."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        inp = torch.randn(SEQ_LEN, VOCAB_SIZE)
        out = model(inp)
        assert out.shape == (SEQ_LEN, VOCAB_SIZE)

    def test_output_shape_batched(self) -> None:
        """Output should be [batch, seqLen, vocabSize] for batched input."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        inp = torch.randn(4, SEQ_LEN, VOCAB_SIZE)
        out = model(inp)
        assert out.shape == (4, SEQ_LEN, VOCAB_SIZE)

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 epochs (reversal-only loss)."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        initial_loss = train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

        loss_val = initial_loss
        for _ in range(200):
            data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
            loss_val = train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

        assert loss_val < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {loss_val:.4f}"
        )

    def test_converges(self) -> None:
        """After sufficient training, should achieve high reversal accuracy."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(500):
            data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
            train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

        eval_data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        _, rev_acc = eval_reversal_accuracy(model, eval_data, INPUT_LEN)
        assert rev_acc >= 0.9, f"Reversal accuracy too low: {rev_acc:.2%}"

    def test_data_generation(self) -> None:
        """Verify reversal data structure."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        data = generate_reversal_data(4, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        assert len(data) == 4

        for inp_onehot, target_indices in data:
            assert inp_onehot.shape == (SEQ_LEN, VOCAB_SIZE)
            assert target_indices.shape == (SEQ_LEN,)
            # All target indices should be valid
            assert target_indices.min() >= 0
            assert target_indices.max() < VOCAB_SIZE

    def test_causal_masking(self) -> None:
        """Verify causal mask prevents attending to future positions."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)

        # Two inputs identical except at last position
        inp1 = torch.randn(SEQ_LEN, VOCAB_SIZE)
        inp2 = inp1.clone()
        inp2[-1] = torch.randn(VOCAB_SIZE)  # change last position only

        out1 = model(inp1)
        out2 = model(inp2)

        # All positions except the last should have identical output
        # (causal mask means they can't see the last position)
        assert torch.allclose(out1[:-1], out2[:-1], atol=1e-5), (
            "Causal masking broken: earlier positions affected by change at last position"
        )

    def test_sorting_converges(self) -> None:
        """Sorting task should converge with 2 blocks."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]  # seed param untyped
        model = MultiHeadTransformer(SORT_VOCAB, SEQ_LEN, D_MODEL, NUM_HEADS, num_blocks=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(500):
            data = generate_sorting_data(16, INPUT_LEN, SORT_VOCAB, SORT_SEP, SORT_EOS)
            train_reversal_epoch(model, data, optimizer, reversal_start=INPUT_LEN)

        eval_data = generate_sorting_data(16, INPUT_LEN, SORT_VOCAB, SORT_SEP, SORT_EOS)
        _, acc = eval_reversal_accuracy(model, eval_data, INPUT_LEN)
        assert acc >= 0.9, f"Sorting accuracy too low: {acc:.2%}"
