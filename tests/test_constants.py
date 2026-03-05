"""Tests for shared constants."""

import math

from waxmorph.constants import EPS_DEN, EPS_DIST, EPS_NORM, FOUR_THIRDS_PI, RAND_EPS


class TestConstants:
    def test_four_thirds_pi(self):
        assert FOUR_THIRDS_PI == math.pi * 4.0 / 3.0

    def test_eps_dist_positive(self):
        assert EPS_DIST > 0

    def test_eps_den_positive(self):
        assert EPS_DEN > 0

    def test_eps_norm_positive(self):
        assert EPS_NORM > 0

    def test_rand_eps_positive(self):
        assert RAND_EPS > 0

    def test_stability_constants_small(self):
        """Numerical stability epsilons should be very small."""
        assert EPS_DEN < 1e-6
        assert EPS_NORM < 1e-6
        assert RAND_EPS < 1e-4
