"""Walk-forward backtest."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pairs import Settings, load_prices, metrics, run_backtest, select_pairs

BASE = Settings(formation_days=600, trading_days=100, n_pairs=3, same_sector=False, cost_bps=5)


def test_cointegration_selection_finds_the_true_pairs(market):
    selected = select_pairs(market.iloc[:600], BASE, np.random.default_rng(0))
    found = {frozenset((row.y, row.x)) for row in selected.itertuples()}
    assert found == {frozenset((f"Y{k}", f"X{k}")) for k in range(3)}
    assert (selected["p_value"] <= 0.05).all()


def test_backtest_on_true_pairs_is_profitable(market):
    result = run_backtest(market, BASE)
    assert len(result.returns) == len(market) - 600
    assert not result.returns.isna().any()
    assert result.n_periods == 12
    assert metrics.sharpe_ratio(result.returns) > 1.0
    assert {"period", "y", "x", "profit", "n_trades"} <= set(result.pairs.columns)


def test_selection_beats_random_pairs_when_real_pairs_exist(market):
    selected = metrics.sharpe_ratio(run_backtest(market, BASE).returns)
    random = np.mean([metrics.sharpe_ratio(run_backtest(market, replace(BASE, selection="random", seed=s)).returns)
                      for s in range(5)])
    assert selected > random + 0.5


def test_distance_selection(market):
    result = run_backtest(market, replace(BASE, selection="distance"))
    assert len(result.pairs) == 3 * result.n_periods
    assert "distance" in result.pairs.columns


def test_no_look_ahead(market):
    """Altering prices after a date must not change any return before it."""
    altered = market.copy()
    cut = 1200
    altered.iloc[cut:] = altered.iloc[cut:] * np.linspace(1.0, 2.0, altered.shape[1])
    for selection in ("cointegration", "distance"):
        settings = replace(BASE, selection=selection)
        base, changed = run_backtest(market, settings), run_backtest(altered, settings)
        before = market.index[cut - 1]
        assert np.allclose(base.returns.loc[:before], changed.returns.loc[:before])


def test_costs_reduce_returns(market):
    free = run_backtest(market, replace(BASE, cost_bps=0.0)).returns.sum()
    costly = run_backtest(market, replace(BASE, cost_bps=30.0)).returns.sum()
    assert costly < free


def test_scan_cache_gives_identical_results(market):
    cache = {}
    first = run_backtest(market, BASE, cache)
    assert len(cache) == first.n_periods
    second = run_backtest(market, BASE, cache)
    assert first.returns.equals(second.returns)


def test_empty_slots_earn_nothing(market):
    """Asking for more pairs than qualify leaves capital idle; returns scale down accordingly."""
    few = run_backtest(market, replace(BASE, n_pairs=3, max_p_value=1e-4))
    many = run_backtest(market, replace(BASE, n_pairs=30, max_p_value=1e-4))
    assert len(few.pairs) == len(many.pairs)
    assert np.allclose(few.returns * 3, many.returns * 30)


def test_validation(market):
    with pytest.raises(ValueError):
        Settings(selection="astrology")
    with pytest.raises(ValueError):
        run_backtest(market.iloc[:500], BASE)


def test_metrics():
    daily = pd.Series([0.01, -0.01] * 126)
    assert metrics.annualised_return(daily) == pytest.approx(0.0)
    assert metrics.annualised_volatility(daily) == pytest.approx(0.01 * np.sqrt(252), rel=1e-2)
    assert metrics.max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)
    assert metrics.sharpe_standard_error(pd.Series(np.zeros(252) + [0.01, -0.01] * 126)) == pytest.approx(1.0)


def test_known_price_error_is_removed(tmp_path):
    dates = pd.to_datetime(["2016-06-30", "2016-07-01", "2016-07-05", "2016-07-06"])
    raw = pd.DataFrame({"DHR": [41.4, 41.9, 67.6, 67.8], "AAA": [10.0, 10.1, 10.0, 10.2],
                        "BBB": [20.0, 20.0, 20.2, 20.1]}, index=dates)
    raw.index.name = "date"
    path = tmp_path / "prices.csv"
    raw.to_csv(path)
    fixed = load_prices(path)
    assert load_prices(path, fix_known_errors=False)["DHR"].pct_change().max() > 0.6
    assert abs(fixed["DHR"].pct_change().loc["2016-07-05"]) < 0.02
    # Prices after the event are untouched, and earlier returns are unchanged.
    assert fixed["DHR"].iloc[-1] == 67.8
    assert fixed["DHR"].pct_change().iloc[1] == pytest.approx(41.9 / 41.4 - 1)
