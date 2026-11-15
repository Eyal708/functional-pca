"""Tests for the two-level (multilevel) FPCA estimator."""
import numpy as np
import pytest

from fpca import TwoLevelPACE

GRID = np.linspace(0, 1, 30)
PHI_B = np.sqrt(2) * np.sin(2 * np.pi * GRID)   # between-subject eigenfunction
PSI_W = np.sqrt(2) * np.cos(4 * np.pi * GRID)   # within-subject eigenfunction


def _simulate(n_subjects=60, n_visits=4, lam_b=2.0, lam_w=0.5, seed=0):
    rng = np.random.default_rng(seed)
    curves, subjects, visits, xi = [], [], [], []
    for s in range(n_subjects):
        b = rng.normal(0, np.sqrt(lam_b))
        xi.append(b)
        for v in range(n_visits):
            w = rng.normal(0, np.sqrt(lam_w))
            y = 0.5 + b * PHI_B + w * PSI_W + rng.normal(0, 0.1, GRID.size)
            curves.append((GRID, y))
            subjects.append(s)
            visits.append(v)
    return curves, np.array(subjects), np.array(visits), np.array(xi)


@pytest.fixture(scope="module")
def ml_data():
    return _simulate()


@pytest.fixture(scope="module")
def ml_model(ml_data):
    curves, subjects, _, _ = ml_data
    return TwoLevelPACE(variance_threshold=1, n_segments=8).fit(curves, subjects)


def test_recovers_between_and_within_components(ml_model):
    assert abs(np.corrcoef(ml_model.between_eigenfunctions_[0], PHI_B)[0, 1]) > 0.95
    assert abs(np.corrcoef(ml_model.within_eigenfunctions_[0], PSI_W)[0, 1]) > 0.95
    assert ml_model.between_eigenvalues_[0] > ml_model.within_eigenvalues_[0]


def test_transform_and_reconstruct_shapes(ml_data, ml_model):
    curves, subjects, _, xi = ml_data
    scores = ml_model.transform(curves, subjects)
    assert scores["between"].shape == (len(np.unique(subjects)), 1)
    assert scores["within"].shape == (len(curves), 1)
    assert abs(np.corrcoef(scores["between"][:, 0], xi)[0, 1]) > 0.95
    recon = ml_model.reconstruct(scores, subjects)
    assert recon.shape == (len(curves), GRID.size)
    resid = recon - np.vstack([y for _, y in curves])
    assert np.sqrt(np.mean(resid ** 2)) < 0.3


def test_visit_effects_require_visit_ids(ml_data):
    curves, subjects, visits, _ = ml_data
    model = TwoLevelPACE(variance_threshold=1, n_segments=8, visit_effects=True)
    with pytest.raises(ValueError):
        model.fit(curves, subjects)
    model.fit(curves, subjects, visits)
    assert model.visit_effects_.shape == (len(np.unique(visits)), GRID.size)


def test_needs_repeated_curves_per_subject():
    curves = [(GRID, np.sin(GRID) + i) for i in range(5)]
    with pytest.raises(ValueError):
        TwoLevelPACE().fit(curves, np.arange(5))
