"""Cointegration tests."""

import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller, coint

from pairs import benjamini_hochberg, engle_granger, scan_pairs
from pairs.cointegration import adf_statistic, half_life, ols_hedge_ratio


def logs(pair):
    return np.log(pair["Y"].to_numpy()), np.log(pair["X"].to_numpy())


def test_hedge_ratio_recovers_the_truth(cointegrated):
    beta, alpha = ols_hedge_ratio(*logs(cointegrated))
    assert beta == pytest.approx(1.3, abs=0.03)
    assert alpha == pytest.approx(0.5, abs=0.12)


@pytest.mark.parametrize("lags", [0, 1, 4])
def test_adf_statistic_matches_statsmodels(cointegrated, lags):
    log_y, log_x = logs(cointegrated)
    beta, alpha = ols_hedge_ratio(log_y, log_x)
    spread = log_y - beta * log_x - alpha
    reference = adfuller(spread, maxlag=lags, regression="n", autolag=None, result_object=False)[0]
    assert adf_statistic(spread, lags) == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("fixture", ["cointegrated", "unrelated"])
def test_engle_granger_matches_statsmodels(fixture, request):
    log_y, log_x = logs(request.getfixturevalue(fixture))
    result = engle_granger(log_y, log_x)
    t_stat, p_value, _ = coint(log_y, log_x, trend="c", maxlag=1, autolag=None)
    assert result.t_stat == pytest.approx(t_stat, rel=1e-8)
    assert result.p_value == pytest.approx(p_value, rel=1e-6)


def test_cointegrated_pair_is_detected_and_unrelated_pair_is_not(cointegrated, unrelated):
    assert engle_granger(*logs(cointegrated)).p_value < 0.001
    assert engle_granger(*logs(unrelated)).p_value > 0.10


def test_half_life_recovers_the_truth(cointegrated):
    result = engle_granger(*logs(cointegrated))
    assert result.half_life == pytest.approx(15.0, rel=0.25)
    assert result.spread_std == pytest.approx(0.03, rel=0.2)


def test_half_life_of_a_random_walk_is_long():
    walk = np.cumsum(np.random.default_rng(0).standard_normal(2000))
    assert half_life(walk) > 100


def test_false_positive_rate_is_close_to_nominal():
    """On independent random walks, a 5% test should reject about 5% of the time."""
    rng = np.random.default_rng(3)
    walks = np.cumsum(rng.standard_normal((500, 800)), axis=0)
    rejections = np.mean([engle_granger(walks[:, 2 * i], walks[:, 2 * i + 1]).p_value < 0.05 for i in range(400)])
    assert 0.02 < rejections < 0.09


def test_scan_finds_the_true_pairs(market):
    pairs = [(a, b) for i, a in enumerate(market.columns) for b in market.columns[i + 1:]]
    scan = scan_pairs(np.log(market), pairs)
    assert len(scan) == len(pairs) == 66
    assert scan["p_value"].is_monotonic_increasing
    top = {frozenset((row.y, row.x)) for row in scan.head(3).itertuples()}
    assert top == {frozenset((f"Y{k}", f"X{k}")) for k in range(3)}


def test_benjamini_hochberg():
    # Classic example: only the two smallest p-values survive at a 5% false discovery rate.
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert benjamini_hochberg(p_values, 0.05).tolist() == [True, True] + [False] * 8
    assert not benjamini_hochberg(np.full(20, 0.5)).any()
    assert benjamini_hochberg(np.full(20, 1e-6)).all()
    # The result does not depend on the order of the inputs.
    shuffled = np.random.default_rng(0).permutation(p_values)
    assert benjamini_hochberg(shuffled).sum() == 2
