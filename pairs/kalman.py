"""A time-varying hedge ratio estimated with a Kalman filter.

A regression gives one hedge ratio for the whole sample, but relations
between companies drift. The Kalman filter treats the hedge ratio and the
intercept as hidden states that follow a random walk, and updates them every
day from the new prices:

    state:        theta_t = theta_{t-1} + w_t,          theta = (beta, alpha)
    observation:  log Y_t = beta_t * log X_t + alpha_t + v_t

Before seeing day t, the filter predicts log Y_t from yesterday's state. The
prediction error and its predicted standard deviation are known without any
information from day t onwards, which makes their ratio a signal free of
look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanResult:
    """Filtered path of the hedge ratio and the one-step-ahead forecast errors."""

    beta: np.ndarray              # hedge ratio after observing day t
    alpha: np.ndarray             # intercept after observing day t
    beta_prior: np.ndarray        # hedge ratio known *before* day t (use this to trade on day t)
    error: np.ndarray             # forecast error of log Y on day t: the spread
    error_std: np.ndarray         # its standard deviation, predicted before day t

    @property
    def z_score(self) -> np.ndarray:
        """Forecast error in units of its predicted standard deviation."""
        return self.error / self.error_std


def kalman_hedge(
    log_y: np.ndarray,
    log_x: np.ndarray,
    observation_variance: float = 1e-3,
    adaptation_days: float = 120.0,
    initial_state: tuple[float, float] = (0.0, 0.0),
    initial_variance: float = 1.0,
) -> KalmanResult:
    """Run the filter over two log-price series.

    Parameters
    ----------
    observation_variance : float
        Variance of the observation noise v_t, i.e. of the spread around its
        mean. Typically the residual variance of a regression on past data.
    adaptation_days : float
        Time scale over which the states follow the data. The state noise
        variance is observation_variance / adaptation_days^2, which makes the
        filter absorb a persistent deviation in roughly that many days. It
        must be clearly longer than the half-life of the spread: a filter that
        adapts as fast as the spread reverts explains every deviation away as
        a change of hedge ratio and leaves nothing to trade. Infinity gives a
        constant hedge ratio (recursive least squares).
    initial_state, initial_variance :
        Starting guess for (beta, alpha) and its uncertainty.
    """
    log_y, log_x = np.asarray(log_y, float), np.asarray(log_x, float)
    n = len(log_y)
    state_noise = observation_variance / adaptation_days**2 * np.eye(2)

    theta = np.array(initial_state, dtype=float)
    covariance = initial_variance * np.eye(2)
    beta, alpha, beta_prior = np.empty(n), np.empty(n), np.empty(n)
    error, error_std = np.empty(n), np.empty(n)

    for t in range(n):
        # Predict: the state is a random walk, so only its uncertainty grows.
        covariance = covariance + state_noise
        observation = np.array([log_x[t], 1.0])
        beta_prior[t] = theta[0]

        # Forecast error and its variance, both known before the update.
        error[t] = log_y[t] - observation @ theta
        variance = observation @ covariance @ observation + observation_variance
        error_std[t] = np.sqrt(variance)

        # Update: move the state towards the value that explains today's price.
        gain = covariance @ observation / variance
        theta = theta + gain * error[t]
        covariance = covariance - np.outer(gain, observation) @ covariance
        beta[t], alpha[t] = theta

    return KalmanResult(beta, alpha, beta_prior, error, error_std)
