"""Correctness tests for LR-range test (lr_find).

Mirrors `packages/idris-ml/test/src/Test/Hpo/LrFinder.idr` — same
fixtures, same expected values. The two backends should agree on the
algorithm's pure-Python/Idris portions exactly (modulo floating-point
rounding); the integration `lr_find` call agrees within 2× on the
recommended LR for a fixed-seed synthetic problem.
"""

import math

from torch_ref.training.lr_finder import (
    LrFindConfig,
    has_diverged,
    recommend_from_curve,
    sweep_lr,
)


class TestSweepLr:
    def test_endpoint_min(self) -> None:
        assert sweep_lr(1e-7, 10.0, 100, 0) == 1e-7

    def test_endpoint_max(self) -> None:
        assert math.isclose(sweep_lr(1e-7, 10.0, 100, 99), 10.0, rel_tol=1e-9)

    def test_midpoint_log_uniform(self) -> None:
        # frac=0.5 with lrMin=1e-7, lrMax=10 → lr = 1e-7 * sqrt(1e8) = 1e-3
        assert math.isclose(sweep_lr(1e-7, 10.0, 3, 1), 1e-3, rel_tol=1e-9)


class TestRecommendFromCurve:
    # Mirror of `syntheticUCurve` in Test.Hpo.LrFinder
    CURVE = [
        (1e-4, 1.00),
        (1e-3, 0.95),
        (1e-2, 0.50),
        (1e-1, 0.10),
        (1.0, 2.00),
    ]

    def test_picks_steepest_descent_div_10(self) -> None:
        # Steepest descent at lr=1e-3 (slope -0.45 to next point), ÷10 = 1e-4
        assert math.isclose(recommend_from_curve(10.0, self.CURVE), 1e-4, rel_tol=1e-9)

    def test_div_one_returns_steepest_lr(self) -> None:
        # With div=1, returns the steepest LR itself (1e-3)
        assert math.isclose(recommend_from_curve(1.0, self.CURVE), 1e-3, rel_tol=1e-9)


class TestDefaults:
    def test_fastai_aligned_defaults(self) -> None:
        cfg = LrFindConfig()
        assert cfg.lr_min == 1e-7
        assert cfg.lr_max == 10.0
        assert cfg.num_iters == 100
        assert cfg.smooth_beta == 0.98
        assert cfg.recommend_div == 10.0


class TestHasDiverged:
    """Sign-stable divergence check.

    Mirror of the Idris `Test.Hpo.LrFinder` cases for `hasDiverged`. The
    classical `corrected > divergeFactor * best` rule is sign-broken for
    negative losses (RL examples reporting `-avg_return`); these tests
    pin the sign-stable behavior.
    """

    def test_positive_4x_diverged(self) -> None:
        assert has_diverged(4.0, 1.0, 5.0)

    def test_positive_3x_not_diverged(self) -> None:
        assert not has_diverged(4.0, 1.0, 3.0)

    def test_negative_best_corrected_above_not_diverged(self) -> None:
        # best=-100, divergeFactor=4 → diverged when corrected > 200.
        # corrected=-50 is improvement (not diverged).
        assert not has_diverged(4.0, -100.0, -50.0)

    def test_negative_best_corrected_at_threshold_diverged(self) -> None:
        assert has_diverged(4.0, -100.0, 200.0 + 1e-9)

    def test_negative_best_corrected_below_threshold_not_diverged(self) -> None:
        assert not has_diverged(4.0, -100.0, 199.0)

    def test_zero_best_no_change_not_diverged(self) -> None:
        # |best|<1e-8 floor avoids divide-by-zero; tiny absolute jump
        # still counts as a divergence.
        assert not has_diverged(4.0, 0.0, 0.0)

    def test_zero_best_unit_jump_diverged(self) -> None:
        assert has_diverged(4.0, 0.0, 1.0)
