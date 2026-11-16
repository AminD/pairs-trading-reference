"""Price data: loading, cleaning, sectors and simulated pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

BENCHMARK = "SPY"
TRADING_DAYS = 252

# GICS sectors of the tickers downloaded by scripts/download_prices.py.
SECTORS = {
    "Technology": "AAPL MSFT CSCO INTC ADBE CRM ACN AVGO TXN QCOM ORCL IBM ADP",
    "Communication": "GOOGL DIS T VZ CMCSA NFLX",
    "Consumer Discretionary": "AMZN HD NKE MCD LOW SBUX BKNG TGT TJX F",
    "Consumer Staples": "PG KO PEP WMT COST MO PM CL KMB GIS MDLZ",
    "Health Care": "JNJ UNH PFE MRK ABT TMO LLY AMGN MDT GILD CVS BMY DHR ISRG SYK CI",
    "Financials": "JPM V MA BAC GS MS BLK AXP C WFC USB SCHW SPGI PNC MMC CB",
    "Industrials": "HON UNP UPS CAT BA MMM GE DE LMT RTX EMR ITW FDX NSC CSX WM",
    "Energy": "XOM CVX COP SLB EOG",
    "Utilities": "DUK SO NEE D AEP",
    "Materials": "ECL APD SHW NEM",
    "Real Estate": "PLD AMT SPG",
}
SECTOR_OF = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers.split()}

# Price jumps that are artefacts of the data source, not market moves. Yahoo's
# adjusted prices mishandle some spin-offs: Danaher appears to gain 61% on the
# day it spun off Fortive, when shareholders in fact received Fortive shares.
KNOWN_DATA_ERRORS = [("DHR", "2016-07-05")]


def load_prices(path, fix_known_errors: bool = True) -> pd.DataFrame:
    """Read a price table (dates in rows, tickers in columns) and clean it.

    Missing prices are carried forward and tickers without a full history are
    dropped. Jumps listed in ``KNOWN_DATA_ERRORS`` are removed by rescaling the
    earlier part of the series, so that the return on that day equals the
    median return of the other stocks.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if prices.empty or prices.shape[1] < 2:
        raise ValueError(f"{path} does not look like a price table (dates in rows, tickers in columns)")
    prices = prices.ffill().dropna(axis=1, how="any")

    if fix_known_errors:
        returns = prices.pct_change()
        for ticker, day in KNOWN_DATA_ERRORS:
            day = pd.Timestamp(day)
            if ticker in prices.columns and day in prices.index:
                expected = returns.loc[day].drop(ticker).median()
                factor = (1.0 + returns.loc[day, ticker]) / (1.0 + expected)
                prices.loc[prices.index < day, ticker] *= factor
    return prices


def split_benchmark(prices: pd.DataFrame, benchmark: str = BENCHMARK) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate the stocks from the benchmark column, if present."""
    if benchmark in prices.columns:
        return prices.drop(columns=benchmark), prices[benchmark]
    return prices, None


def sector_of(tickers) -> pd.Series:
    """Sector of each ticker ("Other" when unknown)."""
    return pd.Series([SECTOR_OF.get(t, "Other") for t in tickers], index=list(tickers), name="sector")


def candidate_pairs(tickers, same_sector: bool = True) -> list[tuple[str, str]]:
    """All unordered pairs of tickers, optionally restricted to pairs within one sector.

    Restricting to a sector is an economic prior: two utilities share the same
    drivers, so a stable relation between them is plausible. It also divides
    the number of statistical tests, and therefore of false discoveries, by ten.
    """
    tickers = list(tickers)
    sectors = sector_of(tickers)
    return [
        (a, b)
        for i, a in enumerate(tickers)
        for b in tickers[i + 1:]
        if not same_sector or sectors[a] == sectors[b]
    ]


def simulate_pair(
    n_days: int = 1500,
    beta: float = 1.3,
    half_life: float = 15.0,
    spread_vol: float = 0.03,
    cointegrated: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Two price series with a known relation, for testing.

    ``X`` is a random walk in log price. With ``cointegrated=True``,
    log Y = beta * log X + spread, where the spread is a mean-reverting
    (Ornstein-Uhlenbeck) process with the given half-life in days and
    stationary volatility. With ``cointegrated=False`` the spread is itself a
    random walk, so nothing ties the two prices together.
    """
    rng = np.random.default_rng(seed)
    log_x = np.log(50.0) + np.cumsum(0.015 * rng.standard_normal(n_days))
    if cointegrated:
        phi = 0.5 ** (1.0 / half_life)                      # AR(1) coefficient
        shocks = spread_vol * np.sqrt(1.0 - phi**2) * rng.standard_normal(n_days)
        spread = np.zeros(n_days)
        for t in range(1, n_days):
            spread[t] = phi * spread[t - 1] + shocks[t]
    else:
        spread = np.cumsum(0.01 * rng.standard_normal(n_days))
    log_y = 0.5 + beta * log_x + spread
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame({"Y": np.exp(log_y), "X": np.exp(log_x)}, index=dates)
