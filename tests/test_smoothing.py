"""The smoothers in fpca._smoothing (adapted from FDApy) must reproduce FDApy exactly.

The comparison tests run only where FDApy is installed (it requires numpy<2); the
smoke tests run everywhere, including under NumPy 2.
"""
import numpy as np
import pytest

from fpca._smoothing import LocalPolynomial, PSplines


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, 300))
    y = np.sin(x) + rng.normal(0, 0.3, x.size)
    grid = np.linspace(0, 10, 41)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    surface = np.sin(gx / 3) * np.cos(gy / 4) + rng.normal(0, 0.05, gx.shape)
    weights = rng.uniform(0.5, 2.0, gx.shape)
    np.fill_diagonal(weights, 0.0)       # as PACE excludes the covariance diagonal
    return x, y, grid, surface, weights


def _fits(lp_cls, ps_cls, seed=0):
    """The four calls PACE makes: LP mean (1-D), LP covariance (2-D), PS mean, PS covariance."""
    x, y, grid, surface, weights = _data(seed)
    out = {}
    out["lp_1d"] = lp_cls(kernel_name="epanechnikov", bandwidth=0.8, degree=1).predict(y=y, x=x, x_new=grid)
    gx, gy = np.meshgrid(grid[::4], grid[::4], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vals = surface[::4, ::4].ravel()
    out["lp_2d"] = lp_cls(kernel_name="epanechnikov", bandwidth=2.0, degree=2).predict(y=vals, x=pts, x_new=pts)
    ps1 = ps_cls(n_segments=10, degree=3, order_penalty=2)
    ps1.fit(x=grid, y=np.interp(grid, x, y), sample_weights=np.ones_like(grid), penalty=(5.0,))
    out["ps_1d"] = ps1.predict(grid)
    ps2 = ps_cls(n_segments=12, degree=3, order_penalty=2)
    ps2.fit(x=[grid, grid], y=surface, sample_weights=weights, penalty=(2.0, 2.0))
    out["ps_2d"] = ps2.predict([grid, grid])
    return out


def test_smoothers_run_and_are_finite():
    for name, values in _fits(LocalPolynomial, PSplines).items():
        assert np.all(np.isfinite(values)), name


def test_smoothers_reproduce_fdapy_exactly():
    pytest.importorskip("FDApy")
    from FDApy.preprocessing.smoothing.local_polynomial import LocalPolynomial as FDApyLP
    from FDApy.preprocessing.smoothing.psplines import PSplines as FDApyPS

    for seed in (0, 1):
        ours, theirs = _fits(LocalPolynomial, PSplines, seed), _fits(FDApyLP, FDApyPS, seed)
        for name in ours:
            np.testing.assert_array_equal(ours[name], theirs[name], err_msg=name)
