"""Tests for fpca.utils, fpca.stats and smoke tests for fpca.plotting."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import simulate

import fpca
from fpca import (
    MFPCA,
    PACE,
    explained_variance_ratio,
    n_components_for_variance,
    permanova,
    plotting,
    reconstruction_rmse,
)


@pytest.fixture(scope="module")
def fitted():
    curves, xi = simulate(n_curves=80, seed=3)
    model = PACE(n_components=2, variance_threshold=0.999).fit(curves)
    return model, curves, xi


def test_public_api():
    for name in fpca.__all__:
        assert hasattr(fpca, name)


def test_mfpca_default_method_is_happ_greven(fitted):
    model, _, _ = fitted
    assert MFPCA({"a": model}).method == "happ_greven"


def test_variance_helpers():
    ev = np.array([6.0, 3.0, 1.0])
    np.testing.assert_allclose(explained_variance_ratio(ev), [0.6, 0.3, 0.1])
    assert n_components_for_variance(ev, 0.5) == 1
    assert n_components_for_variance(ev, 0.9) == 2
    assert n_components_for_variance(ev, 1.0) == 3
    with pytest.raises(ValueError):
        n_components_for_variance(ev, 0)


def test_reconstruction_rmse_decreases(fitted):
    model, curves, _ = fitted
    err = reconstruction_rmse(model, curves, max_components=2)
    assert err.shape == (len(curves), 3)
    means = err.mean(axis=0)
    assert means[0] > means[1] > means[2]


def test_permanova_detects_shift_and_not_null():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 3))
    labels = np.repeat(["a", "b", "c"], 30)
    null = permanova(X, labels, n_perm=199)
    X_shift = X.copy()
    X_shift[labels == "a", 0] += 2.0
    alt = permanova(X_shift, labels, n_perm=199)
    assert null["p"] > 0.01
    assert alt["p"] < 0.01 and alt["R2"] > null["R2"]
    for key in ("n", "groups", "F", "R2", "p", "PERMDISP_F", "PERMDISP_p"):
        assert key in alt


def test_permanova_block_schemes_need_blocks():
    X = np.random.default_rng(1).normal(size=(20, 2))
    labels = np.repeat(["a", "b"], 10)
    with pytest.raises(ValueError):
        permanova(X, labels, scheme="within")
    res = permanova(X, labels, blocks=np.tile(np.arange(10), 2), scheme="within", n_perm=49)
    assert 0 < res["p"] <= 1


def test_univariate_plots_run(fitted):
    model, curves, xi = fitted
    scores = model.transform(curves)
    err = reconstruction_rmse(model, curves, max_components=2, scores=scores)
    calls = [
        lambda: plotting.plot_curves(curves, n_curves=10),
        lambda: plotting.plot_modes_of_variation(model, n_components=2),
        lambda: plotting.plot_cumulative_variance({"a": model, "b": model}),
        lambda: plotting.plot_reconstruction(model, *curves[0], n_components=(0, 1, 2)),
        lambda: plotting.plot_reconstruction_error(err),
        lambda: plotting.plot_scores(scores, labels=np.where(xi[:, 0] > 0, "pos", "neg")),
        lambda: model.plot_mean(),
        lambda: model.plot_eigenfunctions(),
    ]
    for call in calls:
        out = call()
        assert out is not None
        plt.close("all")


@pytest.mark.parametrize("method", ["happ_greven", "berrendero"])
def test_mfpca_plots_run(method):
    curves_a, _ = simulate(n_curves=40, seed=4)
    curves_b, _ = simulate(n_curves=40, seed=5)
    data = {"a": curves_a, "b": curves_b}
    models = {k: PACE(n_components=2).fit(v) for k, v in data.items()}
    mfpca = MFPCA(models, method=method, n_components=2, n_components_per_feature=2)
    scores = mfpca.fit_transform(data)
    plotting.plot_mfpca_eigenfunctions(mfpca)
    plotting.plot_mfpca_reconstruction_comparison(mfpca, data, scores, subject_indices=[0, 1])
    plotting.plot_mfpca_extreme_subjects(mfpca, data, scores, component=0)
    plt.close("all")
