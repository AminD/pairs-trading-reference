"""Shared fixtures: simulated pairs with a known relation, and a small simulated market."""

import numpy as np
import pandas as pd
import pytest

from pairs import simulate_pair


@pytest.fixture(scope="session")
def cointegrated():
    """log Y = 0.5 + 1.3 log X + mean-reverting spread (half-life 15 days)."""
    return simulate_pair(n_days=1500, beta=1.3, half_life=15.0, cointegrated=True, seed=1)


@pytest.fixture(scope="session")
def unrelated():
    return simulate_pair(n_days=1500, cointegrated=False, seed=1)


@pytest.fixture(scope="session")
def market():
    """Twelve stocks: three truly cointegrated pairs and six independent random walks."""
    frames = []
    for k in range(3):
        pair = simulate_pair(n_days=1800, beta=1.0 + 0.1 * k, half_life=5.0 + 2 * k, seed=10 + k)
        frames.append(pair.rename(columns={"Y": f"Y{k}", "X": f"X{k}"}))
    rng = np.random.default_rng(99)
    walks = np.exp(np.log(40.0) + np.cumsum(0.015 * rng.standard_normal((1800, 6)), axis=0))
    frames.append(pd.DataFrame(walks, index=frames[0].index, columns=[f"W{i}" for i in range(6)]))
    return pd.concat(frames, axis=1)
