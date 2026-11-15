"""Recovery and API tests for the univariate PACE estimator."""
import numpy as np
import pytest
from conftest import GRID, LAM, NOISE_SD, PHI1, PHI2, mean_fn

from fpca import PACE


def _abs_corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])


def _check_recovery(model, xi_true, scores, min_corr):
    phi1 = np.interp(GRID, model.grid_, model.eigenfunction_values_[0])
    phi2 = np.interp(GRID, model.grid_, model.eigenfunction_values_[1])
    assert _abs_corr(phi1, PHI1) > min_corr
    assert _abs_corr(phi2, PHI2) > min_corr
    assert _abs_corr(scores[:, 0], xi_true[:, 0]) > min_corr
    assert _abs_corr(scores[:, 1], xi_true[:, 1]) > min_corr - 0.1


@pytest.mark.parametrize("method", ["PS", "LP"])
def test_dense_recovery(dense_data, method):
    curves, xi = dense_data
    # A narrow noise bandwidth: the default (0.3 of the domain) over-smooths the
    # strongly varying raw diagonal of this simulation and inflates sigma^2.
    model = PACE(n_components=2, method_smoothing=method, noise_bandwidth_fraction=0.05).fit(curves)
    scores = model.transform(curves)
    assert scores.shape == (len(curves), 2)
    assert np.max(np.abs(model.mean_values_ - mean_fn(model.grid_))) < 0.5
    _check_recovery(model, xi, scores, min_corr=0.95)
    # eigenvalues within a generous factor of the truth
    assert 0.6 < model.eigenvalues_[0] / LAM[0] < 1.5
    assert 0.5 < model.eigenvalues_[1] / LAM[1] < 1.6
    assert model.noise_variance_ == pytest.approx(NOISE_SD ** 2, rel=0.5)


def test_sparse_recovery(sparse_data):
    curves, xi = sparse_data
    model = PACE(n_components=2, method_smoothing="LP").fit(curves)
    assert not model.is_dense_
    _check_recovery(model, xi, model.transform(curves), min_corr=0.85)


def test_component_counts_and_clamping(dense_data):
    curves, _ = dense_data
    model = PACE(n_components=50, variance_threshold=0.95)
    with pytest.warns(UserWarning):
        model.fit(curves)
    assert model.n_components == 50  # never mutated
    assert model.n_components_used_ == model.n_components_available_ == len(model.eigenvalues_)
    with pytest.warns(UserWarning):
        model.transform(curves[:3], n_components=99)


def test_reconstruct_curve(dense_data):
    curves, _ = dense_data
    model = PACE(n_components=2).fit(curves)
    t, y = curves[0]
    scores = model.transform([(t, y)])[0]
    values, t_eval = model.reconstruct_curve(t, scores)
    assert values.shape == t_eval.shape
    rmse = np.sqrt(np.mean((np.interp(t, t_eval, values) - y) ** 2))
    assert rmse < 2 * NOISE_SD


def test_input_validation():
    with pytest.raises(ValueError):
        PACE(variance_threshold=0)
    with pytest.raises(ValueError):
        PACE(method_smoothing="spline")
    with pytest.raises(ValueError):
        PACE().fit([(np.arange(3.0), np.ones(3))])  # a single curve
    with pytest.raises(ValueError):
        PACE().transform([(np.arange(3.0), np.ones(3))])  # not fitted


def test_unsorted_and_duplicate_times_are_corrected(dense_data):
    curves, _ = dense_data
    model = PACE(n_components=2).fit(curves)
    t, y = curves[0]
    t2 = np.concatenate([t[::-1], t[:1]])
    y2 = np.concatenate([y[::-1], y[:1]])
    with pytest.warns(UserWarning):
        s = model.transform([(t2, y2)])
    np.testing.assert_allclose(s, model.transform([(t, y)]), atol=1e-8)


def test_extrapolation_guard(dense_data):
    curves, _ = dense_data
    model = PACE(n_components=2).fit(curves)
    with pytest.raises(ValueError):
        model.transform([(np.array([0.5, 1.5]), np.array([1.0, 2.0]))])
