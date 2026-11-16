"""Performance statistics of a series of daily returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def annualised_return(returns: pd.Series) -> float:
    """Arithmetic mean return per year (appropriate for a self-financing long-short book)."""
    return float(returns.mean() * TRADING_DAYS)


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """Mean over volatility, annualised. No risk-free rate: the book is dollar neutral."""
    deviation = returns.std(ddof=1)
    return float(returns.mean() / deviation * np.sqrt(TRADING_DAYS)) if deviation > 0 else 0.0


def sharpe_standard_error(returns: pd.Series) -> float:
    """Approximate standard error of the annualised Sharpe ratio: sqrt((1 + S^2 / 2) / years)."""
    years = len(returns) / TRADING_DAYS
    return float(np.sqrt((1.0 + 0.5 * sharpe_ratio(returns) ** 2) / years))


def drawdowns(returns: pd.Series) -> pd.Series:
    """Loss from the previous peak of cumulative profit, at each date (<= 0)."""
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdowns(returns).min())


def summary(returns: pd.DataFrame) -> pd.DataFrame:
    """One row of statistics per strategy (one column of ``returns`` each)."""
    return pd.DataFrame(
        {
            "Annual return": returns.apply(annualised_return),
            "Volatility": returns.apply(annualised_volatility),
            "Sharpe ratio": returns.apply(sharpe_ratio),
            "Sharpe std error": returns.apply(sharpe_standard_error),
            "Max drawdown": returns.apply(max_drawdown),
        }
    )
