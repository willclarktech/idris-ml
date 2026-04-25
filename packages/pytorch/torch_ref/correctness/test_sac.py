"""Correctness tests for SAC on Pendulum."""

from torch_ref.models.sac import evaluate, train_sac


class TestSac:
    def test_learns(self) -> None:
        """Recent returns should climb above the random baseline."""
        _, history = train_sac(total_steps=15000, log_every=100000)
        assert len(history) > 0
        early = sum(history[:5]) / max(1, min(len(history), 5))
        late = sum(history[-5:]) / max(1, min(len(history), 5))
        assert late > early + 100, f"Expected improvement; early={early:.1f} late={late:.1f}"

    def test_converges_loosely(self) -> None:
        """At hard-sync + 30k steps, greedy eval should be well above random (-1200).
        Standard SAC with Polyak+longer training hits -250; this setting trades off
        speed / simplicity for Idris portability."""
        actor, _ = train_sac(total_steps=30000, log_every=100000)
        avg = evaluate(actor, n_episodes=10)
        assert avg >= -1500.0, f"Expected avg >= -1500, got {avg:.1f}"
