"""Correctness tests for REINFORCE on CartPole."""

import torch

from torch_ref.models.reinforce import PolicyNetwork, evaluate, reinforce_epoch, train_reinforce


class TestReinforce:
    def test_returns_improve(self) -> None:
        """Average returns should increase over 200 epochs."""
        torch.manual_seed(42)
        policy = PolicyNetwork()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

        early_returns: list[float] = []
        for _ in range(50):
            avg_ret = reinforce_epoch(policy, optimizer, batch_size=10)
            early_returns.append(avg_ret)

        late_returns: list[float] = []
        for _ in range(50):
            avg_ret = reinforce_epoch(policy, optimizer, batch_size=10)
            late_returns.append(avg_ret)

        assert sum(late_returns) / len(late_returns) > sum(early_returns) / len(early_returns)

    def test_converges(self) -> None:
        """REINFORCE should solve CartPole (avg return >= 195) within 3000 epochs."""
        policy, history = train_reinforce(epochs=3000, seed=42)
        assert len(history) >= 100
        recent_avg = sum(history[-100:]) / 100
        assert recent_avg >= 150.0, f"Expected >= 150, got {recent_avg:.1f}"

    def test_greedy_eval(self) -> None:
        """Trained policy should achieve high greedy eval score."""
        policy, _ = train_reinforce(epochs=3000, seed=42)
        avg = evaluate(policy, n_episodes=50)
        assert avg >= 150.0, f"Expected >= 150, got {avg:.1f}"
