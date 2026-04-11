"""Correctness tests for multi-head transformer on sequence reversal."""

import torch

from torch_ref.models.multi_head_transformer import (
    MultiHeadTransformer,
    eval_reversal_accuracy,
    generate_reversal_data,
    train_reversal_epoch,
)

# Task config matching Idris
VOCAB_SIZE = 10  # 8 content tokens + SEP + EOS
INPUT_LEN = 5
SEQ_LEN = 2 * INPUT_LEN + 1  # input + SEP + reversed + EOS, minus 1 for shift = 11
SEP_TOKEN = 8
EOS_TOKEN = 9
D_MODEL = 32
NUM_HEADS = 4


class TestMultiHeadTransformer:
    def test_output_shape(self) -> None:
        """Output should be [seqLen, vocabSize]."""
        torch.manual_seed(42)
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        inp = torch.randn(SEQ_LEN, VOCAB_SIZE)
        out = model(inp)
        assert out.shape == (SEQ_LEN, VOCAB_SIZE)

    def test_output_shape_batched(self) -> None:
        """Output should be [batch, seqLen, vocabSize] for batched input."""
        torch.manual_seed(42)
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        inp = torch.randn(4, SEQ_LEN, VOCAB_SIZE)
        out = model(inp)
        assert out.shape == (4, SEQ_LEN, VOCAB_SIZE)

    def test_loss_decreases(self) -> None:
        """Loss should decrease over 200 epochs."""
        torch.manual_seed(42)
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Measure initial loss
        initial_loss = train_reversal_epoch(model, data, optimizer)

        # Train 200 more epochs
        loss_val = initial_loss
        for _ in range(200):
            loss_val = train_reversal_epoch(model, data, optimizer)

        assert loss_val < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {loss_val:.4f}"
        )

    def test_converges(self) -> None:
        """After sufficient training, should achieve high reversal accuracy."""
        torch.manual_seed(42)
        model = MultiHeadTransformer(VOCAB_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS)
        data = generate_reversal_data(16, INPUT_LEN, VOCAB_SIZE, SEP_TOKEN, EOS_TOKEN)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(2000):
            train_reversal_epoch(model, data, optimizer)

        full_acc, rev_acc = eval_reversal_accuracy(model, data, INPUT_LEN)
        assert rev_acc >= 0.9, f"Reversal accuracy too low: {rev_acc:.2%}"

    def test_data_generation(self) -> None:
        """Verify reversal data structure."""
        torch.manual_seed(42)
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
        torch.manual_seed(42)
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
