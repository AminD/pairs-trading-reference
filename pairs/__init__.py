"""Pairs trading: cointegration tests, adaptive hedge ratios and an honest backtest.

Quick start
-----------
>>> from pairs import load_prices, split_benchmark, run_backtest, Settings
>>> prices, benchmark = split_benchmark(load_prices("data/prices.csv"))   # doctest: +SKIP
>>> result = run_backtest(prices, Settings(selection="cointegration"))      # doctest: +SKIP
>>> result.returns, result.pairs                                            # doctest: +SKIP
"""

from . import backtest, cointegration, kalman, metrics, strategy
from .backtest import BacktestResult, Settings, run_backtest, select_pairs
from .cointegration import benjamini_hochberg, engle_granger, scan_pairs
from .data import candidate_pairs, load_prices, sector_of, simulate_pair, split_benchmark
from .kalman import kalman_hedge
from .strategy import Rule, positions_from_zscore, trade_pair

__version__ = "1.0.0"

__all__ = [
    "BacktestResult", "Rule", "Settings", "backtest", "benjamini_hochberg", "candidate_pairs",
    "cointegration", "engle_granger", "kalman", "kalman_hedge", "load_prices", "metrics",
    "positions_from_zscore", "run_backtest", "scan_pairs", "sector_of", "select_pairs",
    "simulate_pair", "split_benchmark", "strategy", "trade_pair",
]
