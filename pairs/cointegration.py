"""Cointegration: testing whether two prices are tied together in the long run.

Two stock prices are each (close to) random walks: they wander without
returning to any level. They are *cointegrated* when some combination

    spread = log Y - beta * log X - alpha

does return to its mean. That spread is then predictable, which is the whole
basis of pairs trading. The Engle-Granger test checks it in two steps:
estimate beta by regression, then test whether the residual is stationary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp


@dataclass(frozen=True)
class EngleGrangerResult:
    """Outcome of an Engle-Granger test of ``y`` on ``x`` (both in logs)."""

    beta: float         # hedge ratio: units of log X per unit of log Y
    alpha: float        # intercept
    t_stat: float       # ADF statistic of the residual (more negative = more stationary)
    p_value: float      # probability of such a statistic if there is no cointegration
    half_life: float    # days for a deviation of the spread to halve (inf if not mean-reverting)
    spread_std: float   # standard deviation of the residual spread


def ols_hedge_ratio(log_y: np.ndarray, log_x: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the regression of log Y on log X."""
    x_centered = log_x - log_x.mean()
    beta = float(x_centered @ (log_y - log_y.mean()) / (x_centered @ x_centered))
    return beta, float(log_y.mean() - beta * log_x.mean())


def adf_statistic(series: np.ndarray, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic, without constant.

    Regression:  d s_t = gamma * s_{t-1} + sum_i phi_i * d s_{t-i} + e_t

    If the series is a random walk, gamma = 0. If it reverts to zero,
    gamma < 0. The statistic is gamma divided by its standard error. Lagged
    differences absorb short-term autocorrelation. No constant is included
    because the series tested here is a regression residual, of mean zero.
    """
    difference = np.diff(series)
    target = difference[lags:]
    columns = [series[lags:-1]] + [difference[lags - i:-i] for i in range(1, lags + 1)]
    design = np.column_stack(columns)

    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    variance = residual @ residual / (len(target) - design.shape[1])
    standard_error = np.sqrt(variance * np.linalg.inv(design.T @ design)[0, 0])
    return float(coefficients[0] / standard_error)


def half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) fit, in observations.

    s_t - s_{t-1} = a + b * s_{t-1} + e_t.  A deviation decays by a factor
    (1 + b) per day, so it halves after -ln 2 / ln(1 + b) days.
    """
    lagged, change = spread[:-1], np.diff(spread)
    lagged_centered = lagged - lagged.mean()
    slope = float(lagged_centered @ (change - change.mean()) / (lagged_centered @ lagged_centered))
    if not -1.0 < slope < 0.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + slope))


def engle_granger(log_y: np.ndarray, log_x: np.ndarray, lags: int = 1) -> EngleGrangerResult:
    """Engle-Granger cointegration test of log Y on log X.

    The residual of a regression is, by construction, the most stationary
    looking combination of the two series. Its ADF statistic therefore cannot
    be compared with the usual Dickey-Fuller tables: MacKinnon's tables for
    cointegration residuals are used instead (through statsmodels).
    """
    log_y, log_x = np.asarray(log_y, float), np.asarray(log_x, float)
    beta, alpha = ols_hedge_ratio(log_y, log_x)
    spread = log_y - beta * log_x - alpha
    t_stat = adf_statistic(spread, lags)
    return EngleGrangerResult(
        beta=beta,
        alpha=alpha,
        t_stat=t_stat,
        p_value=float(mackinnonp(t_stat, regression="c", N=2)),
        half_life=half_life(spread),
        spread_std=float(spread.std(ddof=1)),
    )


def scan_pairs(log_prices: pd.DataFrame, pairs: list[tuple[str, str]], lags: int = 1) -> pd.DataFrame:
    """Engle-Granger test on every pair, in both directions.

    The test is not symmetric: regressing Y on X or X on Y gives different
    residuals. Both are run and the more significant one is kept, with the
    dependent variable reported as ``y``. Keeping the better of two tests makes
    the p-values somewhat optimistic; the out-of-sample checks elsewhere in the
    project do not rely on them being exact.

    Returns one row per pair, sorted from the most to the least significant.
    """
    values = {ticker: log_prices[ticker].to_numpy() for ticker in log_prices.columns}
    rows = []
    for a, b in pairs:
        forward = engle_granger(values[a], values[b], lags)
        backward = engle_granger(values[b], values[a], lags)
        y, x, best = (a, b, forward) if forward.t_stat <= backward.t_stat else (b, a, backward)
        rows.append({"y": y, "x": x, "beta": best.beta, "alpha": best.alpha, "t_stat": best.t_stat,
                     "p_value": best.p_value, "half_life": best.half_life, "spread_std": best.spread_std})
    return pd.DataFrame(rows).sort_values("p_value", ignore_index=True)


def benjamini_hochberg(p_values, false_discovery_rate: float = 0.05) -> np.ndarray:
    """Which of many tests remain significant once the number of tests is accounted for.

    Testing 600 pairs at the 5% level yields about 30 "significant" pairs even
    if none is truly cointegrated. The Benjamini-Hochberg procedure keeps the
    k smallest p-values, where k is the largest rank with
    p_(k) <= k / m * rate. Among the tests it keeps, at most ``rate`` are
    expected to be false discoveries. Returns a boolean mask.
    """
    p_values = np.asarray(p_values, float)
    m = len(p_values)
    order = np.argsort(p_values)
    below = p_values[order] <= false_discovery_rate * np.arange(1, m + 1) / m
    keep = np.zeros(m, dtype=bool)
    if below.any():
        keep[order[: np.flatnonzero(below).max() + 1]] = True
    return keep
