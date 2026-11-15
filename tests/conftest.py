import matplotlib

matplotlib.use("Agg")  # headless backend for plotting tests

import numpy as np
import pytest

GRID = np.linspace(0, 1, 51)
PHI1 = np.sqrt(2) * np.sin(2 * np.pi * GRID)
PHI2 = np.sqrt(2) * np.cos(2 * np.pi * GRID)
LAM = np.array([4.0, 1.0])
NOISE_SD = 0.2


def mean_fn(t):
    return 1.0 + t


def simulate(n_curves=150, n_obs=None, seed=0):
    """Curves X_i(t) = mu(t) + xi_i1 phi1(t) + xi_i2 phi2(t) + noise.

    ``n_obs=None`` gives dense curves on GRID; an int keeps that many random
    grid points per curve (sparse, irregular design). Returns the curves and the
    true scores.
    """
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=(n_curves, 2)) * np.sqrt(LAM)
    curves = []
    for i in range(n_curves):
        idx = np.arange(GRID.size) if n_obs is None else np.sort(rng.choice(GRID.size, n_obs, replace=False))
        t = GRID[idx]
        y = mean_fn(t) + xi[i, 0] * PHI1[idx] + xi[i, 1] * PHI2[idx] + rng.normal(0, NOISE_SD, idx.size)
        curves.append((t, y))
    return curves, xi


@pytest.fixture(scope="session")
def dense_data():
    return simulate(seed=0)


@pytest.fixture(scope="session")
def sparse_data():
    return simulate(n_curves=250, n_obs=8, seed=1)
