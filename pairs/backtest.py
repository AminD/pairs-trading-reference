"""Walk-forward backtest of a pairs-trading portfolio.

The calendar is cut into consecutive trading periods. Before each one, pairs
are selected and calibrated on the preceding formation window only, then
traded during the period with their parameters frozen (or updated by a filter
that only ever sees the past). This is the design of Gatev, Goetzmann and
Rouwenhorst (2006), the reference study on pairs trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cointegration import scan_pairs
from .data import candidate_pairs
from .strategy import Rule, trade_pair

SELECTIONS = ("cointegration", "distance", "random")


@dataclass(frozen=True)
class Settings:
    """Everything that defines a backtest."""

    formation_days: int = 504        # window used to select and calibrate pairs (2 years)
    trading_days: int = 126          # period during which they are traded (6 months)
    n_pairs: int = 20                # pairs traded per period
    selection: str = "cointegration" # "cointegration", "distance" or "random"
    method: str = "static"           # "static" or "kalman" hedge ratio
    same_sector: bool = True         # only pair stocks of the same sector
    max_p_value: float = 0.05        # cointegration: significance required
    min_half_life: float = 3.0       # cointegration: fastest mean reversion accepted (days)
    max_half_life: float = 60.0      # cointegration: slowest mean reversion accepted (days)
    rule: Rule = field(default_factory=Rule)
    cost_bps: float = 5.0            # cost per unit of traded value, in basis points
    adaptation_days: float = 120.0   # Kalman filter time scale
    seed: int = 0                    # for the random selection

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample results of the portfolio."""

    returns: pd.Series          # daily portfolio return on committed capital
    pairs: pd.DataFrame         # one row per pair and period: selection statistics and outcome
    settings: Settings

    @property
    def n_periods(self) -> int:
        return int(self.pairs["period"].nunique()) if len(self.pairs) else 0


def select_pairs(formation: pd.DataFrame, settings: Settings, rng: np.random.Generator,
                 scan_cache: dict | None = None) -> pd.DataFrame:
    """Choose the pairs to trade from the formation window.

    * ``cointegration``: Engle-Granger test on every candidate pair; keep the
      most significant ones whose half-life is in the accepted range.
    * ``distance``: the pairs whose normalised prices stayed closest (smallest
      sum of squared differences), traded one for one. No statistical test.
    * ``random``: pairs drawn at random among the candidates. A placebo: if the
      selected pairs do no better than these, the selection adds nothing.

    A stock may appear in several pairs. Returns a table with columns
    ``y``, ``x`` and the statistics available for the method.
    """
    candidates = candidate_pairs(formation.columns, settings.same_sector)
    log_prices = np.log(formation)

    if settings.selection == "cointegration":
        # The scan depends only on the window and the candidate set, not on the
        # trading rule: cache it so that parameter studies do not repeat it.
        key = (formation.index[0], formation.index[-1], settings.same_sector)
        if scan_cache is not None and key in scan_cache:
            scan = scan_cache[key]
        else:
            scan = scan_pairs(log_prices, candidates)
            if scan_cache is not None:
                scan_cache[key] = scan
        eligible = scan[
            (scan["p_value"] <= settings.max_p_value)
            & scan["half_life"].between(settings.min_half_life, settings.max_half_life)
            & (scan["beta"] > 0)
        ]
        return eligible.head(settings.n_pairs).reset_index(drop=True)

    if settings.selection == "distance":
        normalised = (log_prices - log_prices.iloc[0]).to_numpy()
        column = {ticker: i for i, ticker in enumerate(formation.columns)}
        distance = [float(((normalised[:, column[a]] - normalised[:, column[b]]) ** 2).sum()) for a, b in candidates]
        order = np.argsort(distance)[: settings.n_pairs]
        return pd.DataFrame({"y": [candidates[i][0] for i in order], "x": [candidates[i][1] for i in order],
                             "distance": [distance[i] for i in order]})

    chosen = rng.choice(len(candidates), size=min(settings.n_pairs, len(candidates)), replace=False)
    return pd.DataFrame({"y": [candidates[i][0] for i in chosen], "x": [candidates[i][1] for i in chosen]})


def run_backtest(prices: pd.DataFrame, settings: Settings = Settings(), scan_cache: dict | None = None) -> BacktestResult:
    """Walk-forward backtest of the pairs portfolio.

    The capital is split equally between ``n_pairs`` slots. A pair that is
    flat, or a slot left empty because too few pairs qualified, earns nothing:
    returns are measured on *committed* capital, the conservative convention.

    ``scan_cache`` is an optional dictionary reused between calls to avoid
    repeating the cointegration scans when only the trading rule changes.
    """
    n_days = len(prices)
    if n_days < settings.formation_days + settings.trading_days:
        raise ValueError("not enough data for one formation window and one trading period")
    rng = np.random.default_rng(settings.seed)
    unit_hedge = settings.selection == "distance"

    daily, records = [], []
    starts = range(settings.formation_days, n_days - 1, settings.trading_days)
    for period, start in enumerate(starts):
        formation = prices.iloc[start - settings.formation_days:start]
        # The trading window starts one day early so that the first return is defined.
        trading = prices.iloc[start - 1:start + settings.trading_days]
        selected = select_pairs(formation, settings, rng, scan_cache)

        period_returns = pd.Series(0.0, index=trading.index[1:])
        for _, pair in selected.iterrows():
            columns = [pair["y"], pair["x"]]
            trade = trade_pair(formation[columns], trading[columns], settings.method, settings.rule,
                               settings.cost_bps, settings.adaptation_days, unit_hedge)
            pair_returns = trade.returns.iloc[1:]
            period_returns += pair_returns / settings.n_pairs
            records.append({
                "period": period, "start": trading.index[1], "end": trading.index[-1],
                **pair.to_dict(),
                "profit": float(pair_returns.sum()), "n_trades": trade.n_trades,
                "days_in_market": float((trade.position.iloc[:-1] != 0).mean()),
            })
        daily.append(period_returns)

    return BacktestResult(pd.concat(daily).rename("returns"), pd.DataFrame(records), settings)
