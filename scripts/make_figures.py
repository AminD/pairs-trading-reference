"""Regenerate the figures and the result tables of the README.

Run from the repository root (takes a few minutes)::

    python scripts/make_figures.py

Figures are written to ``docs/figures`` and tables to ``docs/results``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pairs import (Rule, Settings, benjamini_hochberg, candidate_pairs, engle_granger, load_prices, metrics,  # noqa: E402
                   run_backtest, scan_pairs, split_benchmark, trade_pair)
from pairs.plotting import AQUA, BLUE, GRID, INK_SOFT, MUTED, ORANGE, YELLOW, use_style  # noqa: E402

FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results"
STRATEGIES = {
    "Distance": dict(selection="distance", method="static"),
    "Cointegration": dict(selection="cointegration", method="static"),
    "Cointegration + Kalman": dict(selection="cointegration", method="kalman"),
}
COLORS = {"Distance": BLUE, "Cointegration": ORANGE, "Cointegration + Kalman": AQUA, "Random pairs": MUTED}


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def null_rejection_rate(n_days: int = 504, n_pairs: int = 2000, seed: int = 0) -> float:
    """Share of *unrelated* random walks that the scan declares cointegrated at 5%.

    The scan keeps the better of two regressions per pair, so its false-positive
    rate is above the nominal 5%. Simulating independent random walks measures it.
    """
    rng = np.random.default_rng(seed)
    walks = np.cumsum(0.015 * rng.standard_normal((n_days, 2 * n_pairs)), axis=0)
    rejected = 0
    for i in range(n_pairs):
        a, b = walks[:, 2 * i], walks[:, 2 * i + 1]
        rejected += min(engle_granger(a, b).p_value, engle_granger(b, a).p_value) <= 0.05
    return rejected / n_pairs


def discovery_study(prices: pd.DataFrame, window: int = 504) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How many pairs test as cointegrated in each window, and whether they still do in the next."""
    log_prices = np.log(prices)
    pairs = candidate_pairs(prices.columns)
    scans = []
    for start in range(0, len(prices) - window + 1, window):
        scan = scan_pairs(log_prices.iloc[start:start + window], pairs)
        scan["window"] = f"{prices.index[start]:%Y}-{prices.index[start + window - 1]:%Y}"
        scans.append(scan)

    counts, persistence = [], []
    for k, scan in enumerate(scans):
        significant = scan["p_value"] <= 0.05
        counts.append({"window": scan["window"].iloc[0], "tested": len(scan), "p < 5%": int(significant.sum()),
                       "after Benjamini-Hochberg": int(benjamini_hochberg(scan["p_value"]).sum())})
        if k + 1 < len(scans):
            following = log_prices.iloc[(k + 1) * window:(k + 2) * window]
            # Same pair, same direction, next window.
            later = np.array([engle_granger(following[y].to_numpy(), following[x].to_numpy()).p_value
                              for y, x in zip(scan["y"], scan["x"])])
            persistence.append({
                "window": scan["window"].iloc[0],
                "selected": int(significant.sum()),
                "selected, still p < 5% next": float((later[significant.to_numpy()] <= 0.05).mean()),
                "all pairs, p < 5% next": float((later <= 0.05).mean()),
                "rank correlation of t-stats": float(pd.Series(scan["t_stat"].to_numpy()).corr(
                    pd.Series([engle_granger(following[y].to_numpy(), following[x].to_numpy()).t_stat
                               for y, x in zip(scan["y"], scan["x"])]), method="spearman")),
            })
    return pd.DataFrame(counts), pd.DataFrame(persistence)


def discovery_figure(counts: pd.DataFrame, persistence: pd.DataFrame, null_rate: float) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(counts))
    left.bar(x - 0.2, counts["p < 5%"], width=0.38, color=BLUE, label="Pairs with p < 5%")
    left.bar(x + 0.2, counts["after Benjamini-Hochberg"], width=0.38, color=ORANGE, label="Still significant after correction")
    expected = null_rate * counts["tested"].iloc[0]
    left.axhline(expected, color=MUTED, linestyle="--", linewidth=1.2)
    left.text(-0.45, expected * 1.03, f"expected by pure chance: {expected:.0f}", ha="left", va="bottom",
              color=INK_SOFT, fontsize=9)
    left.set_xticks(x, counts["window"], rotation=30, ha="right")
    left.set_ylabel(f"Number of pairs (out of {counts['tested'].iloc[0]})")
    left.set_title("Pairs that test as cointegrated")
    left.grid(axis="x", visible=False)
    left.set_ylim(0, 1.35 * counts["p < 5%"].max())   # headroom so the legend clears the bars
    left.legend(loc="upper left")

    x = np.arange(len(persistence))
    right.bar(x - 0.2, 100 * persistence["selected, still p < 5% next"], width=0.38, color=BLUE,
              label="Pairs significant in this window")
    right.bar(x + 0.2, 100 * persistence["all pairs, p < 5% next"], width=0.38, color=GRID, edgecolor=MUTED,
              linewidth=0.6, label="All pairs")
    right.set_xticks(x, persistence["window"], rotation=30, ha="right")
    right.set_ylabel("Significant in the following window (%)")
    right.set_title("Cointegration rarely persists")
    right.grid(axis="x", visible=False)
    right.set_ylim(0, 135 * persistence[["selected, still p < 5% next", "all pairs, p < 5% next"]].to_numpy().max())
    right.legend(loc="upper right")
    save(fig, "discovery.png")


def example_figure(prices: pd.DataFrame, y: str, x: str, start: int, formation: int = 504, trading: int = 126) -> None:
    """One pair: prices, signal and positions over a formation window and the trading period after it."""
    columns = [y, x]
    formation_prices = prices[columns].iloc[start - formation:start]
    trading_prices = prices[columns].iloc[start - 1:start + trading]
    trade = trade_pair(formation_prices, trading_prices, "static", unit_hedge=True)
    both = prices[columns].iloc[start - formation:start + trading]
    normalised = both / both.iloc[0]
    log_spread = np.log(both[y]) - np.log(both[x])
    reference = log_spread.iloc[:formation]
    z_all = (log_spread - reference.mean()) / reference.std(ddof=1)
    split = trading_prices.index[1]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    top.plot(normalised.index, normalised[y], color=BLUE, label=y)
    top.plot(normalised.index, normalised[x], color=ORANGE, label=x)
    top.axvline(split, color=MUTED, linestyle=":", linewidth=1.2)
    top.text(split, top.get_ylim()[1], "  trading starts", va="top", color=INK_SOFT, fontsize=9)
    top.set_ylabel("Price (start = 1)")
    top.set_title(f"{y} and {x}: formation window, then the trading period")
    top.legend(loc="upper left")

    bottom.plot(z_all.index, z_all, color=INK_SOFT, linewidth=1.2)
    for level, style in [(2.0, "--"), (-2.0, "--"), (0.5, ":"), (-0.5, ":")]:
        bottom.axhline(level, color=MUTED, linestyle=style, linewidth=1.0)
    held = trade.position.iloc[:-1]
    for sign, color, label in [(1.0, AQUA, f"Long {y}, short {x}"), (-1.0, YELLOW, f"Short {y}, long {x}")]:
        mask = (held == sign).to_numpy()
        bottom.fill_between(held.index, -4, 4, where=mask, color=color, alpha=0.25, linewidth=0, label=label)
    bottom.axvline(split, color=MUTED, linestyle=":", linewidth=1.2)
    bottom.set_ylim(-4, 4)
    bottom.set_ylabel("Spread (standard deviations)")
    bottom.set_title("The spread, with entry (dashed) and exit (dotted) levels")
    bottom.legend(loc="lower left", ncol=2)
    save(fig, "example_pair.png")


def backtest_study(prices: pd.DataFrame, cache: dict):
    """Main table: the three strategies and a placebo of random same-sector pairs."""
    base = Settings()
    results = {name: run_backtest(prices, replace(base, **options), cache) for name, options in STRATEGIES.items()}
    placebo = [run_backtest(prices, replace(base, selection="random", seed=seed)) for seed in range(30)]
    returns = pd.DataFrame({name: result.returns for name, result in results.items()})
    table = metrics.summary(returns)
    table["Profitable pairs"] = [float((r.pairs["profit"] > 0).mean()) for r in results.values()]
    table["Trades per pair"] = [float(r.pairs["n_trades"].mean()) for r in results.values()]

    placebo_sharpe = np.array([metrics.sharpe_ratio(r.returns) for r in placebo])
    placebo_returns = pd.DataFrame({i: r.returns for i, r in enumerate(placebo)})
    table.loc["Random pairs (mean of 30 draws)"] = [
        np.mean([metrics.annualised_return(r.returns) for r in placebo]),
        np.mean([metrics.annualised_volatility(r.returns) for r in placebo]),
        placebo_sharpe.mean(), placebo_sharpe.std(ddof=1),
        np.mean([metrics.max_drawdown(r.returns) for r in placebo]),
        np.mean([(r.pairs["profit"] > 0).mean() for r in placebo]),
        np.mean([r.pairs["n_trades"].mean() for r in placebo]),
    ]
    return results, returns, placebo_returns, placebo_sharpe, table


def backtest_figure(returns: pd.DataFrame, placebo_returns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cumulative_placebo = 100 * placebo_returns.cumsum()
    ax.fill_between(cumulative_placebo.index, cumulative_placebo.quantile(0.05, axis=1), cumulative_placebo.quantile(0.95, axis=1),
                    color=MUTED, alpha=0.22, linewidth=0, label="Random pairs, 5% to 95% of 30 draws")
    for name in returns.columns:
        ax.plot(returns.index, 100 * returns[name].cumsum(), color=COLORS[name], linewidth=1.6, label=name)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Cumulative return on committed capital (%)")
    ax.set_title("Out-of-sample performance after 5 bp costs")
    ax.legend(loc="upper left")
    save(fig, "backtest.png")


def robustness_study(prices: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Sharpe ratio when one design choice at a time is changed."""
    base = Settings()
    variants = {
        "Base case": {},
        "Entry at 1.5 std": dict(rule=Rule(entry=1.5)),
        "Entry at 2.5 std": dict(rule=Rule(entry=2.5)),
        "No stop-loss": dict(rule=Rule(stop=50.0)),
        "10 pairs": dict(n_pairs=10),
        "40 pairs": dict(n_pairs=40),
        "1-year formation": dict(formation_days=252),
        "3-year formation": dict(formation_days=756),
        "3-month trading": dict(trading_days=63),
        "Pairs across sectors": dict(same_sector=False),
        "No costs": dict(cost_bps=0.0),
        "10 bp costs": dict(cost_bps=10.0),
        "20 bp costs": dict(cost_bps=20.0),
    }
    rows = {}
    for label, change in variants.items():
        rows[label] = {}
        for name, options in STRATEGIES.items():
            if label == "Pairs across sectors" and options["selection"] == "cointegration":
                rows[label][name] = np.nan   # 5,356 pairs per window: left out to keep the run short
                continue
            settings = replace(base, **options, **change)
            rows[label][name] = metrics.sharpe_ratio(run_backtest(prices, settings, cache).returns)
    return pd.DataFrame(rows).T


def robustness_figure(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(table))[::-1]
    for offset, name in zip([0.26, 0.0, -0.26], table.columns):
        ax.barh(y + offset, table[name], height=0.24, color=COLORS[name], label=name)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, table.index)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("Changing one design choice at a time")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, "robustness.png")


def yearly_table(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.groupby(returns.index.year).sum()


def main() -> None:
    use_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220

    prices, _ = split_benchmark(load_prices(ROOT / "data" / "prices.csv"))
    print(f"{prices.shape[1]} stocks, {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}, "
          f"{len(candidate_pairs(prices.columns))} same-sector pairs\n")

    null_rate = null_rejection_rate()
    counts, persistence = discovery_study(prices)
    counts.to_csv(RESULTS / "discovery_counts.csv", index=False)
    persistence.to_csv(RESULTS / "persistence.csv", index=False)
    discovery_figure(counts, persistence, null_rate)
    print(f"false-positive rate of the scan on unrelated random walks: {null_rate:.3f}")
    print(counts.to_string(), "\n")
    print(persistence.to_string(), "\n")

    cache: dict = {}
    results, returns, placebo_returns, placebo_sharpe, table = backtest_study(prices, cache)
    table.to_csv(RESULTS / "backtest_summary.csv")
    backtest_figure(returns, placebo_returns)
    print(table.to_string(), "\n")
    print("placebo Sharpe: min %.2f max %.2f" % (placebo_sharpe.min(), placebo_sharpe.max()), "\n")

    yearly = yearly_table(returns)
    yearly.to_csv(RESULTS / "yearly_returns.csv")
    print((100 * yearly).round(2).to_string(), "\n")

    robustness = robustness_study(prices, cache)
    robustness.to_csv(RESULTS / "robustness.csv")
    robustness_figure(robustness)
    print(robustness.round(2).to_string(), "\n")

    # Illustration: the pair selected most often by the distance method, in the
    # period where it traded the most. Chosen to show the mechanics, not as a
    # representative outcome.
    selected = results["Distance"].pairs
    name = (selected["y"] + "/" + selected["x"]).value_counts().index[0]
    y, x = name.split("/")
    periods = selected[(selected["y"] == y) & (selected["x"] == x)]
    chosen = periods.sort_values("n_trades").iloc[-1]
    example_figure(prices, y, x, start=int(prices.index.get_loc(chosen["start"])))
    print(f"example pair: {name}, period starting {chosen['start']:%Y-%m-%d}, "
          f"{chosen['n_trades']} trades, profit {chosen['profit']:.3f}")


if __name__ == "__main__":
    main()
