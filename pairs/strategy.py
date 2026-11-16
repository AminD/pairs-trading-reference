"""From a spread to positions and returns.

The trading rule is the standard one: when the spread is unusually far from
its mean, bet on its return; close the bet when it has come back; give up if
it keeps diverging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cointegration import ols_hedge_ratio
from .kalman import kalman_hedge


@dataclass(frozen=True)
class Rule:
    """Thresholds of the trading rule, in standard deviations of the spread."""

    entry: float = 2.0    # open a position when |z| exceeds this
    exit: float = 0.5     # close it when |z| falls back below this
    stop: float = 4.0     # abandon it when |z| exceeds this (the relation may have broken)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit < self.entry < self.stop:
            raise ValueError("thresholds must satisfy 0 <= exit < entry < stop")


def positions_from_zscore(z_score: np.ndarray, rule: Rule = Rule()) -> np.ndarray:
    """Position in the spread at the close of each day: +1 long, -1 short, 0 flat.

    Long the spread (z very negative) means long Y and short X. After a
    stop-loss the pair is left alone until the spread has returned inside the
    exit band, so that a spread that keeps diverging is not re-entered.
    """
    position = np.zeros(len(z_score))
    current, stopped = 0.0, False
    for t, z in enumerate(z_score):
        if np.isnan(z):
            current = 0.0
        elif stopped:
            stopped = abs(z) > rule.exit
        elif current == 0.0:
            if rule.entry < abs(z) < rule.stop:
                current = -np.sign(z)
        elif abs(z) > rule.stop:
            current, stopped = 0.0, True
        elif abs(z) < rule.exit or np.sign(z) == current:
            # Back to the mean, or through it to the other side.
            current = 0.0
        position[t] = current
    return position


def pair_returns(
    returns_y: np.ndarray,
    returns_x: np.ndarray,
    hedge_ratio: np.ndarray | float,
    position: np.ndarray,
    cost_bps: float = 0.0,
) -> np.ndarray:
    """Daily return of the pair trade, per unit of gross capital.

    One unit of spread is +1 of Y and -beta of X in value, scaled so that the
    gross exposure |w_y| + |w_x| is one. The position decided at the close of
    day t earns the returns of day t + 1: a signal never trades on the price
    that produced it. Costs are charged on the traded value of each leg.
    """
    beta = np.broadcast_to(np.asarray(hedge_ratio, float), position.shape)
    gross = 1.0 + np.abs(beta)
    weight_y = position / gross
    weight_x = -position * beta / gross

    held_y = np.concatenate([[0.0], weight_y[:-1]])   # weights carried into each day
    held_x = np.concatenate([[0.0], weight_x[:-1]])
    profit = held_y * returns_y + held_x * returns_x
    traded = np.abs(np.diff(held_y, prepend=0.0)) + np.abs(np.diff(held_x, prepend=0.0))
    return profit - traded * cost_bps / 1e4


@dataclass(frozen=True)
class PairTrade:
    """Signal, positions and returns of one pair over a trading period."""

    z_score: pd.Series
    hedge_ratio: pd.Series
    position: pd.Series
    returns: pd.Series

    @property
    def n_trades(self) -> int:
        """Number of positions opened."""
        position = self.position.to_numpy()
        previous = np.concatenate([[0.0], position[:-1]])
        return int(((position != 0) & (previous == 0)).sum())

    @property
    def days_in_market(self) -> float:
        return float((self.position != 0).mean())


def trade_pair(
    formation: pd.DataFrame,
    trading: pd.DataFrame,
    method: str = "static",
    rule: Rule = Rule(),
    cost_bps: float = 0.0,
    adaptation_days: float = 120.0,
    unit_hedge: bool = False,
) -> PairTrade:
    """Trade one pair over ``trading`` using only ``formation`` to set it up.

    Both frames hold prices with the dependent stock first: columns (Y, X).

    Parameters
    ----------
    method : {"static", "kalman"}
        ``"static"``: hedge ratio, mean and standard deviation of the spread
        are estimated once on the formation period and frozen.
        ``"kalman"``: the hedge ratio is updated every day by a Kalman filter
        started on the formation period; the signal is the filter's forecast
        error divided by its predicted standard deviation.
    adaptation_days : float
        Time scale of the Kalman filter (see :func:`pairs.kalman.kalman_hedge`).
    unit_hedge : bool
        Force a hedge ratio of one (equal amounts of the two stocks), as in the
        classical distance method. Only for ``method="static"``.
    """
    if method not in ("static", "kalman"):
        raise ValueError("method must be 'static' or 'kalman'")
    log_formation, log_trading = np.log(formation.to_numpy()), np.log(trading.to_numpy())
    index = trading.index

    if unit_hedge:
        beta = 1.0
        alpha = float((log_formation[:, 0] - log_formation[:, 1]).mean())
    else:
        beta, alpha = ols_hedge_ratio(log_formation[:, 0], log_formation[:, 1])
    formation_spread = log_formation[:, 0] - beta * log_formation[:, 1] - alpha

    if method == "static":
        z_score = (log_trading[:, 0] - beta * log_trading[:, 1] - alpha) / formation_spread.std(ddof=1)
        hedge = np.full(len(index), beta)
    else:
        # Start from the regression estimate and let the filter run through the
        # formation period, so that it is warmed up on the first trading day.
        combined = np.vstack([log_formation, log_trading])
        variance = float(formation_spread.var(ddof=1))
        result = kalman_hedge(combined[:, 0], combined[:, 1], observation_variance=variance,
                              adaptation_days=adaptation_days, initial_state=(beta, alpha),
                              initial_variance=variance)
        z_score = result.z_score[len(log_formation):]
        hedge = result.beta_prior[len(log_formation):]

    position = positions_from_zscore(z_score, rule)
    simple = trading.pct_change().fillna(0.0).to_numpy()
    returns = pair_returns(simple[:, 0], simple[:, 1], hedge, position, cost_bps)
    return PairTrade(pd.Series(z_score, index), pd.Series(hedge, index), pd.Series(position, index),
                     pd.Series(returns, index))
