"""Correctness tests for MaskedDropout (the explicit-mask nn.Dropout twin)."""

import torch

from torch_ref.models.masked_dropout import MaskedDropout


class TestMaskedDropout:
    def test_recorded_bits_reproduce_output(self) -> None:
        """The recorded '0'/'1' string rebuilds the output exactly:
        out == x * bits * 1/(1-p), element for element. This is the
        contract the Idris replay side consumes."""
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
        drop = MaskedDropout(0.5)
        drop.recorder = []
        x = torch.randn(8, 16, dtype=torch.float64)
        out = drop(x)
        assert len(drop.recorder) == 1
        bits = drop.recorder[0]
        assert len(bits) == x.numel()
        mask = torch.tensor([float(c) for c in bits], dtype=torch.float64).reshape(x.shape)
        assert torch.equal(out, x * mask * 2.0)

    def test_drop_fraction_tracks_p(self) -> None:
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
        drop = MaskedDropout(0.5)
        out = drop(torch.ones(100_000, dtype=torch.float64))
        zero_frac = (out == 0.0).double().mean().item()
        assert abs(zero_frac - 0.5) < 0.01

    def test_survivors_scaled(self) -> None:
        torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType] - untyped `seed` param in torch stubs
        drop = MaskedDropout(0.25)
        out = drop(torch.ones(1000, dtype=torch.float64))
        kept = out[out != 0.0]
        assert torch.equal(kept, torch.full_like(kept, 1.0 / 0.75))

    def test_eval_is_identity_and_records_nothing(self) -> None:
        drop = MaskedDropout(0.5).eval()
        drop.recorder = []
        x = torch.randn(4, 4, dtype=torch.float64)
        assert torch.equal(drop(x), x)
        assert drop.recorder == []
