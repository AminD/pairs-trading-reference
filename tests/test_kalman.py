"""Kalman filter for the hedge ratio."""

import numpy as np
import pytest

from pairs import kalman_hedge


def test_constant_relation_is_recovered():
    rng = np.random.default_rng(0)
    log_x = 4.0 + np.cumsum(0.01 * rng.standard_normal(3000))
    log_y = 0.7 + 1.5 * log_x + 0.02 * rng.standard_normal(3000)
    result = kalman_hedge(log_y, log_x, observation_variance=0.02**2, adaptation_days=np.inf,
                          initial_variance=10.0)
    assert result.beta[-1] == pytest.approx(1.5, abs=0.05)
    assert result.alpha[-1] == pytest.approx(0.7, abs=0.25)


def test_infinite_adaptation_time_is_recursive_least_squares():
    """With no state noise and a diffuse start, the filter reproduces ordinary least squares."""
    rng = np.random.default_rng(1)
    log_x = 4.0 + np.cumsum(0.01 * rng.standard_normal(800))
    log_y = 0.3 + 0.9 * log_x + 0.03 * rng.standard_normal(800)
    result = kalman_hedge(log_y, log_x, observation_variance=1.0, adaptation_days=np.inf, initial_variance=1e8)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    assert result.beta[-1] == pytest.approx(slope, abs=1e-4)
    assert result.alpha[-1] == pytest.approx(intercept, abs=1e-3)


def test_filter_tracks_a_drifting_hedge_ratio():
    rng = np.random.default_rng(2)
    n = 3000
    log_x = 4.0 + np.cumsum(0.01 * rng.standard_normal(n))
    true_beta = np.linspace(1.0, 1.6, n)
    log_y = true_beta * log_x + 0.01 * rng.standard_normal(n)
    adaptive = kalman_hedge(log_y, log_x, observation_variance=1e-4, adaptation_days=30.0, initial_state=(1.0, 0.0))
    frozen = kalman_hedge(log_y, log_x, observation_variance=1e-4, adaptation_days=np.inf, initial_state=(1.0, 0.0))
    fitted_adaptive = adaptive.beta * log_x + adaptive.alpha
    fitted_frozen = frozen.beta * log_x + frozen.alpha
    # The adaptive filter explains recent prices far better than a constant relation.
    assert np.abs(log_y - fitted_adaptive)[-500:].mean() < 0.2 * np.abs(log_y - fitted_frozen)[-500:].mean()


def test_forecast_uses_only_past_data():
    """Changing prices from day 500 on must not change anything the filter reported before."""
    rng = np.random.default_rng(3)
    log_x = 4.0 + np.cumsum(0.01 * rng.standard_normal(1000))
    log_y = 1.2 * log_x + 0.02 * rng.standard_normal(1000)
    base = kalman_hedge(log_y, log_x)
    altered_y = log_y.copy()
    altered_y[500:] += 0.5
    altered = kalman_hedge(altered_y, log_x)
    for name in ("beta", "alpha", "beta_prior", "error", "error_std"):
        assert np.array_equal(getattr(base, name)[:500], getattr(altered, name)[:500])
    # The hedge ratio available on day t is the one estimated on day t - 1.
    assert np.array_equal(base.beta_prior[1:], base.beta[:-1])


def test_forecast_errors_are_calibrated():
    """When the model is right, errors divided by their predicted std have unit variance."""
    rng = np.random.default_rng(4)
    log_x = 4.0 + np.cumsum(0.01 * rng.standard_normal(5000))
    log_y = 0.2 + 1.1 * log_x + 0.02 * rng.standard_normal(5000)
    result = kalman_hedge(log_y, log_x, observation_variance=0.02**2, adaptation_days=np.inf,
                          initial_state=(1.1, 0.2), initial_variance=1e-6)
    assert result.z_score[500:].std() == pytest.approx(1.0, abs=0.05)
