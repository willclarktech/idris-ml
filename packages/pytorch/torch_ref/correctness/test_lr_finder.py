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
