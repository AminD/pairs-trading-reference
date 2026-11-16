"""Trading rule and pair returns."""

import numpy as np
import pandas as pd
import pytest

from pairs import Rule, positions_from_zscore, trade_pair
from pairs.strategy import pair_returns


def test_positions_follow_the_rule():
    z = np.array([0.0, 1.0, 2.5, 1.5, 0.4, 0.0, -2.2, -1.0, -0.3, 0.0])
    #             flat flat short short exit flat long  long  exit flat
    assert positions_from_zscore(z, Rule(2.0, 0.5, 4.0)).tolist() == [0, 0, -1, -1, 0, 0, 1, 1, 0, 0]


def test_stop_loss_and_no_reentry_until_back_to_normal():
    z = np.array([2.5, 3.0, 4.5, 3.0, 2.5, 0.2, 2.5])
    #             short short stop  out  out  reset short
    assert positions_from_zscore(z, Rule(2.0, 0.5, 4.0)).tolist() == [-1, -1, 0, 0, 0, 0, -1]


def test_no_entry_beyond_the_stop_level():
    assert positions_from_zscore(np.array([5.0, 5.0]), Rule(2.0, 0.5, 4.0)).tolist() == [0, 0]


def test_position_closes_when_the_spread_crosses_to_the_other_side():
    assert positions_from_zscore(np.array([2.5, -1.0]), Rule(2.0, 0.5, 4.0)).tolist() == [-1, 0]


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule(entry=1.0, exit=1.5)
    with pytest.raises(ValueError):
        Rule(entry=2.0, exit=0.5, stop=1.5)


def test_pair_returns_by_hand():
    """Long the spread from the close of day 0, hedge ratio 1: half long Y, half short X."""
    returns_y = np.array([0.00, 0.02, -0.01])
    returns_x = np.array([0.00, 0.01, 0.01])
    position = np.array([1.0, 1.0, 0.0])
    result = pair_returns(returns_y, returns_x, 1.0, position)
    assert result == pytest.approx([0.0, 0.5 * 0.02 - 0.5 * 0.01, -0.5 * 0.01 - 0.5 * 0.01])
    # Costs: one unit of value traded when the position is opened (0.5 + 0.5), 10 bp.
    with_costs = pair_returns(returns_y, returns_x, 1.0, position, cost_bps=10)
    assert with_costs[1] == pytest.approx(result[1] - 0.001)


def test_a_position_never_earns_the_return_of_the_day_it_is_opened():
    returns_y = np.array([0.10, 0.0])   # a large move on the day the signal fires
    returns_x = np.array([0.0, 0.0])
    assert pair_returns(returns_y, returns_x, 1.0, np.array([1.0, 1.0])).tolist() == [0.0, 0.0]


def test_gross_exposure_is_one():
    position = np.ones(3)
    for beta in (0.5, 1.0, 2.0):
        # Y up 1%, X up 1%: the result is (1 - beta) / (1 + beta) percent.
        result = pair_returns(np.full(3, 0.01), np.full(3, 0.01), beta, position)
        assert result[1] == pytest.approx(0.01 * (1 - beta) / (1 + beta))


@pytest.mark.parametrize("method", ["static", "kalman"])
def test_mean_reverting_pair_is_profitable(cointegrated, method):
    trade = trade_pair(cointegrated.iloc[:750], cointegrated.iloc[750:], method, cost_bps=5)
    assert trade.n_trades >= 3
    assert trade.returns.sum() > 0
    assert 0 < trade.days_in_market < 1


def test_unrelated_pair_is_not_reliably_profitable():
    """Across many independent random-walk pairs the average profit is about zero (minus costs)."""
    from pairs import simulate_pair

    profits = []
    for seed in range(40):
        pair = simulate_pair(n_days=1000, cointegrated=False, seed=100 + seed)
        profits.append(trade_pair(pair.iloc[:500], pair.iloc[500:], "static", cost_bps=5).returns.sum())
    assert abs(np.mean(profits)) < 3 * np.std(profits) / np.sqrt(len(profits))


@pytest.mark.parametrize("method", ["static", "kalman"])
def test_no_look_ahead(cointegrated, method):
    """Positions up to day t depend only on prices up to day t."""
    formation, trading = cointegrated.iloc[:750], cointegrated.iloc[750:]
    base = trade_pair(formation, trading, method)
    altered = trading.copy()
    altered.iloc[300:] = altered.iloc[300:] * [1.3, 0.8]
    changed = trade_pair(formation, altered, method)
    assert base.position.iloc[:300].equals(changed.position.iloc[:300])
    assert np.allclose(base.returns.iloc[:300], changed.returns.iloc[:300])
    assert np.allclose(base.z_score.iloc[:300], changed.z_score.iloc[:300])


def test_unit_hedge(cointegrated):
    trade = trade_pair(cointegrated.iloc[:750], cointegrated.iloc[750:], "static", unit_hedge=True)
    assert (trade.hedge_ratio == 1.0).all()


def test_unknown_method_raises(cointegrated):
    with pytest.raises(ValueError):
        trade_pair(cointegrated.iloc[:750], cointegrated.iloc[750:], "magic")
