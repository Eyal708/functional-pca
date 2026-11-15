"""Tests for single-observation curves, MFPCA.fit_from_scores with two-level models,
eigenfunction orientation, permanova_table, curves_from_long and the new plots."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from conftest import GRID, simulate

from fpca import (
    MFPCA,
    PACE,
    TwoLevelPACE,
    curves_from_long,
    orient_eigenfunctions,
    permanova,
    permanova_table,
    plotting,
)

PS_KW = dict(method_smoothing="PS", mean_penalty=1.0, covariance_penalty=1.0, n_components=2)


# ----------------------------------------------------------------------------- single observations

def test_single_observation_curves_warn_and_inform_the_mean():
    curves, _ = simulate(n_curves=80, n_obs=6, seed=1)
    singles = [(t[:1], y[:1]) for t, y in simulate(n_curves=40, n_obs=6, seed=2)[0]]
    with pytest.warns(UserWarning, match="40 of 120 curves have a single observation"):
        model = PACE(**PS_KW).fit(curves + singles)
    assert np.all(np.isfinite(model.mean_values_)) and model.noise_variance_ >= 0
    # Single-observation curves change the mean (they are used) ...
    base = PACE(**PS_KW).fit(curves)
    assert not np.allclose(np.interp(GRID, model.grid_, model.mean_values_),
                           np.interp(GRID, base.grid_, base.mean_values_))
    # ... and can be scored.
    assert model.transform(singles).shape == (40, 2)


def test_all_single_observation_curves_raise():
    with pytest.raises(ValueError, match="at least one curve with 2 or more observations"):
        PACE(**PS_KW).fit([(np.array([0.1 * i]), np.array([1.0])) for i in range(5)])


def _two_level(n_subjects=40, n_visits=3, seed=0, n_single=0, features=("a", "b")):
    """Sparse two-level curves for several features sharing subjects and times."""
    rng = np.random.default_rng(seed)
    phi_b, psi_w = np.sqrt(2) * np.sin(2 * np.pi * GRID), np.sqrt(2) * np.cos(2 * np.pi * GRID)
    data = {f: [] for f in features}
    subjects = []
    for s in range(n_subjects):
        b = rng.normal(0, 1.4)
        for v in range(n_visits):
            w = rng.normal(0, 0.7)
            n_obs = 1 if (s < n_single and v == 0) else 8
            idx = np.sort(rng.choice(GRID.size, n_obs, replace=False))
            for j, f in enumerate(features):
                y = (j + 1) * (b * phi_b[idx] + w * psi_w[idx]) + rng.normal(0, 0.2, n_obs)
                data[f].append((GRID[idx], y))
            subjects.append(s)
    return data, np.array(subjects)


def test_two_level_accepts_single_observation_curves_without_pace_warning():
    data, subjects = _two_level(n_single=10, features=("a",))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*single observation")
        model = TwoLevelPACE(variance_threshold=2, n_segments=8).fit(data["a"], subjects)
    scores = model.transform(data["a"], subjects)
    assert scores["within"].shape == (len(subjects), 2)


# ----------------------------------------------------------------------------- MFPCA from scores

@pytest.fixture(scope="module")
def two_feature_pace():
    rng = np.random.default_rng(5)
    curves_a, xi = simulate(n_curves=90, n_obs=10, seed=5)
    # Feature b shares the scores of feature a (strong joint structure).
    curves_b = [(t, 2 * (y - (1 + t)) + rng.normal(0, 0.1, t.size)) for t, y in curves_a]
    data = {"a": curves_a, "b": curves_b}
    models = {f: PACE(**PS_KW).fit(c) for f, c in data.items()}
    return data, models


def test_fit_from_scores_matches_fit(two_feature_pace):
    data, models = two_feature_pace
    via_curves = MFPCA(models, n_components_per_feature=2)
    s1 = via_curves.fit_transform(data)
    univariate = {f: m.transform(data[f]) for f, m in models.items()}
    via_scores = MFPCA(models, n_components_per_feature=2).fit_from_scores(univariate)
    s2 = via_scores.transform_from_scores(univariate)
    np.testing.assert_allclose(via_scores.eigenvalues_, via_curves.eigenvalues_, rtol=1e-10)
    np.testing.assert_allclose(np.abs(s2), np.abs(s1), atol=1e-8)
    np.testing.assert_allclose(np.abs(via_scores.weight_functions_), np.abs(via_curves.weight_functions_),
                               atol=1e-10)


def test_inverse_transform_scores_recovers_univariate_scores(two_feature_pace):
    data, models = two_feature_pace
    univariate = {f: m.transform(data[f]) for f, m in models.items()}
    mv = MFPCA(models, n_components_per_feature=2).fit_from_scores(univariate)
    back = mv.inverse_transform_scores(mv.transform_from_scores(univariate))
    for f in models:
        np.testing.assert_allclose(back[f], univariate[f], atol=1e-8)
    one = mv.inverse_transform_scores(mv.transform_from_scores(univariate)[0], which_components=1)
    assert one["a"].shape == (2,)


def test_fit_from_scores_validates_input(two_feature_pace):
    _, models = two_feature_pace
    with pytest.raises(ValueError, match="Missing scores"):
        MFPCA(models).fit_from_scores({"a": np.zeros((5, 2))})
    with pytest.raises(ValueError, match="same number of units"):
        MFPCA(models).fit_from_scores({"a": np.zeros((5, 2)), "b": np.zeros((6, 2))})
    with pytest.raises(ValueError, match="happ_greven"):
        MFPCA(models, method="berrendero").fit_from_scores({"a": np.zeros((5, 2)), "b": np.zeros((5, 2))})


@pytest.fixture(scope="module")
def two_level_models():
    data, subjects = _two_level(seed=3)
    models, scores = {}, {}
    for f, curves in data.items():
        models[f] = TwoLevelPACE(variance_threshold=2, n_segments=8).fit(curves, subjects)
        scores[f] = models[f].transform(curves, subjects)
    return data, subjects, models, scores


def test_level_model_attributes(two_level_models):
    _, _, models, _ = two_level_models
    m = models["a"]
    between, within = m.level_model("between"), m.level_model("within")
    np.testing.assert_allclose(between.mean_values_, m.mean_values_)
    np.testing.assert_allclose(within.mean_values_, 0.0)
    np.testing.assert_allclose(within.eigenfunction_values_, m.within_eigenfunctions_)
    assert between.n_components_used_ == len(m.between_eigenvalues_)
    with pytest.raises(NotImplementedError):
        between.transform([])
    with pytest.raises(ValueError, match="level must be"):
        m.level_model("visit")


def test_multilevel_mfpca_per_level(two_level_models):
    data, subjects, models, scores = two_level_models
    for level, key, n_rows in [("between", "between", 40), ("within", "within", len(subjects))]:
        level_models = {f: m.level_model(level) for f, m in models.items()}
        mv = MFPCA(level_models).fit_from_scores({f: scores[f][key] for f in models})
        S = mv.transform_from_scores({f: scores[f][key] for f in models})
        assert S.shape[0] == n_rows
        # The two features share one latent score per level: one joint component dominates.
        assert mv.explained_variance_ratio_[0] > 0.9
        values, grid = mv.reconstruct_feature(S[0], "b", which_components=1)
        assert values.shape == grid.shape
        fig, _ = plotting.plot_mfpca_eigenfunctions(mv, n_components=2, n_cols=1)
        plt.close(fig)


# ----------------------------------------------------------------------------- orientation

def test_orient_pace_keeps_reconstructions(two_feature_pace):
    data, _ = two_feature_pace
    model = PACE(**PS_KW).fit(data["a"])
    before = model.transform(data["a"][:5])
    rec_before = [model.reconstruct_curve(t, s)[0] for (t, _), s in zip(data["a"][:5], before)]
    signs = np.sign(model.eigenfunction_values_.mean(axis=1))[: before.shape[1]]
    orient_eigenfunctions(model)
    assert np.all(model.eigenfunction_values_.mean(axis=1) > 0)
    after = model.transform(data["a"][:5])
    np.testing.assert_allclose(after, before * signs, atol=1e-10)
    rec_after = [model.reconstruct_curve(t, s)[0] for (t, _), s in zip(data["a"][:5], after)]
    np.testing.assert_allclose(np.concatenate(rec_after), np.concatenate(rec_before), atol=1e-10)


def test_orient_two_level_and_mfpca(two_feature_pace, two_level_models):
    _, _, models, _ = two_level_models
    m = orient_eigenfunctions(models["a"])
    assert np.all(m.between_eigenfunctions_.mean(axis=1) > 0)
    assert np.all(m.within_eigenfunctions_.mean(axis=1) > 0)

    data, pace_models = two_feature_pace
    mv = MFPCA(pace_models, n_components_per_feature=2).fit(data)
    S = mv.transform(data)
    rec = mv.reconstruct_feature(S[:3], "a")[0]
    orient_eigenfunctions(mv)
    assert np.all(mv.weight_functions_.mean(axis=1).sum(axis=1) > 0)
    S2 = mv.transform(data)
    np.testing.assert_allclose(np.abs(S2), np.abs(S), atol=1e-10)
    np.testing.assert_allclose(mv.reconstruct_feature(S2[:3], "a")[0], rec, atol=1e-8)
    with pytest.raises(TypeError):
        orient_eigenfunctions(object())


# ----------------------------------------------------------------------------- stats and utils

def test_permanova_table_matches_permanova():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    g = np.repeat(["a", "b", "c"], 20)
    X[g == "a"] += 1.0
    blocks = np.tile(np.arange(20), 3)
    table = permanova_table(X, {"free": g, "within": (g, blocks, "within"),
                                "dict": {"labels": g, "blocks": blocks, "scheme": "within"}}, n_perm=199)
    assert list(table.index) == ["free", "within", "dict"]
    ref = permanova(X, g, n_perm=199, seed=0)
    assert table.loc["free", "pseudo-F"] == pytest.approx(ref["F"])
    assert table.loc["free", "p (PERMANOVA)"] == pytest.approx(ref["p"])
    assert table.loc["within", "p (PERMANOVA)"] == table.loc["dict", "p (PERMANOVA)"]


def test_curves_from_long():
    frame = pd.DataFrame({"id": ["b", "a", "a", "b", "a"], "t": [2.0, 3.0, 1.0, 1.0, 2.0],
                          "y": [20.0, 3.0, 1.0, 10.0, np.nan]})
    ids, curves = curves_from_long(frame, value_col="y", curve_col="id", time_col="t")
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(curves[0][0], [1.0, 3.0])
    np.testing.assert_array_equal(curves[1][1], [10.0, 20.0])


def test_new_plots_smoke(two_feature_pace, two_level_models):
    data, models = two_feature_pace
    fig, axes = plotting.plot_feature_eigenfunctions(models, n_components=2, labels={"a": "A"})
    assert axes.shape == (2, 2)
    plt.close(fig)
    _, _, ml, _ = two_level_models
    fig, _ = plotting.plot_feature_eigenfunctions({f: m.level_model("within") for f, m in ml.items()},
                                                  n_components=3)
    plt.close(fig)
    S = models["a"].transform(data["a"])
    fig, _ = plotting.plot_scores(S, labels=np.arange(len(S)), cmap="viridis", label_name="index",
                                  xlabel="x", ylabel="y", title="t")
    plt.close(fig)
    fig, axes = plotting.plot_scores_by(S, {"group": np.arange(len(S)) % 3, "index": np.arange(len(S))},
                                        continuous=["index"], xlabel="x", title="scores")
    assert len(axes) == 2
    plt.close(fig)


# ----------------------------------------------------------------------------- evaluation grid

def test_default_grid_is_union_of_observed_times():
    curves, _ = simulate(n_curves=60, n_obs=6, seed=7)
    rng = np.random.default_rng(7)
    curves = [(t + rng.uniform(0, 0.004, t.size), y) for t, y in curves]   # off the simulation grid
    union = np.unique(np.concatenate([t for t, _ in curves]))
    model = PACE(**PS_KW).fit(curves)
    assert model.n_grid_points is None                     # the parameter is not rewritten
    np.testing.assert_array_equal(model.grid_, union)
    capped = PACE(**PS_KW, n_grid_points=40).fit(curves)
    assert capped.grid_.size == 40
    np.testing.assert_allclose(np.diff(capped.grid_), np.diff(capped.grid_)[0])
    lp = PACE(method_smoothing="LP", n_components=2).fit(curves)
    assert lp.grid_.size == 50                             # LP keeps its 50-point default cap


def test_off_grid_times_are_interpolated_and_out_of_domain_rejected():
    curves, _ = simulate(n_curves=80, n_obs=8, seed=8)
    model = PACE(**PS_KW).fit(curves)
    g = model.grid_
    t_off = (g[10:15] + g[11:16]) / 2                      # midpoints between grid points
    assert not np.isin(t_off, g).any()
    score = np.array([1.0, -0.5])
    values, t_eval = model.reconstruct_curve(t_off, score)
    on_grid = model.mean_values_ + score @ model.eigenfunction_values_[:2]
    np.testing.assert_allclose(values[np.isin(t_eval, t_off)], np.interp(t_off, g, on_grid))
    assert model.transform([(t_off, np.zeros_like(t_off))]).shape == (1, 2)
    outside = np.array([g[-1] + 0.5])
    with pytest.raises(ValueError, match="outside the fitted domain"):
        model.transform([(outside, np.zeros(1))])
    lenient = PACE(**PS_KW, allow_extrapolation=True).fit(curves)
    with pytest.warns(UserWarning, match="flat-extrapolating"):
        lenient.transform([(outside, np.zeros(1))])


def test_modes_of_variation_grid_layout(two_feature_pace):
    _, models = two_feature_pace
    fig, axes = plotting.plot_modes_of_variation(models["a"], n_components=2, n_cols=1)
    assert len(axes) == 2 and axes[0].get_subplotspec().get_geometry()[:2] == (2, 1)
    plt.close(fig)
