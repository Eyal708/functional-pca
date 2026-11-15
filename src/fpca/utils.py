"""
Small helpers for preparing data for, and working with, fitted FPCA models.

Most functions accept any fitted :class:`fpca.PACE` model (and, where noted,
any object exposing an ``eigenvalues_`` array) and answer the questions that
come up after almost every fit: how much variance does each component carry,
how many components are needed to reach a given share, and how well do the
first ``k`` components reconstruct the observed curves.
:func:`orient_eigenfunctions` fixes the arbitrary signs of eigenfunctions, and
:func:`curves_from_long` builds the ``(times, values)`` input from a long table.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "curves_from_long",
    "explained_variance_ratio",
    "n_components_for_variance",
    "orient_eigenfunctions",
    "reconstruction_rmse",
]


def curves_from_long(
    frame,
    value_col: str,
    curve_col: str,
    time_col: str,
) -> Tuple[List, List[Tuple[np.ndarray, np.ndarray]]]:
    """Build ``(times, values)`` curves from a long table with one row per observation.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long-format data.
    value_col : str
        Column with the observed values.
    curve_col : str
        Column identifying the curve each row belongs to.
    time_col : str
        Column with the observation times.

    Returns
    -------
    ids : list
        Curve labels, sorted.
    curves : list of (times, values)
        One pair per label in ``ids``, sorted by time, in the format of
        ``PACE.fit``. Rows with a missing time or value are dropped.
    """
    sub = frame[[curve_col, time_col, value_col]].dropna().sort_values([curve_col, time_col])
    ids, curves = [], []
    for cid, g in sub.groupby(curve_col, sort=True):
        ids.append(cid)
        curves.append((g[time_col].to_numpy(dtype=float), g[value_col].to_numpy(dtype=float)))
    return ids, curves


def _flip_to_positive_mean(eigenfunctions: np.ndarray) -> np.ndarray:
    """Flip, in place, every row whose mean is negative. Returns the signs used."""
    signs = np.where(eigenfunctions.mean(axis=1) < 0, -1.0, 1.0)
    eigenfunctions *= signs[:, None]
    return signs


def orient_eigenfunctions(model):
    """Fix the arbitrary sign of each eigenfunction so that its mean over the grid is positive.

    Eigenfunctions are identified only up to sign. After orientation a positive
    score means "above the mean on average over the domain", which makes plots
    and scores easier to read and comparable across models. Components are
    flipped in place, so call this right after ``fit`` and before computing any
    scores or reconstructions; everything computed afterwards is consistent with
    the flipped components.

    Parameters
    ----------
    model : PACE, TwoLevelPACE, LevelModel or MFPCA
        A fitted model.

        - ``PACE`` / ``LevelModel``: each row of ``eigenfunction_values_``.
        - ``TwoLevelPACE``: the between- and within-subject eigenfunctions.
        - ``MFPCA``: each multivariate component, using the mean over the grid
          summed across features of ``weight_functions_``. For the Happ-Greven
          method the PCA loadings are flipped with it, so ``transform`` and the
          reconstruction methods stay consistent.

    Returns
    -------
    model
        The same object, for chaining.
    """
    if hasattr(model, "between_eigenfunctions_"):
        if model.between_eigenfunctions_ is None:
            raise ValueError("The model has not been fitted yet; call fit() first.")
        _flip_to_positive_mean(model.between_eigenfunctions_)
        _flip_to_positive_mean(model.within_eigenfunctions_)
    elif hasattr(model, "weight_functions_"):
        if model.weight_functions_ is None:
            raise ValueError("The model has not been fitted yet; call fit() first.")
        wf = model.weight_functions_  # (K, T, F)
        signs = np.where(wf.mean(axis=1).sum(axis=1) < 0, -1.0, 1.0)
        wf *= signs[:, None, None]
        if getattr(model, "pca_model_", None) is not None:
            model.pca_model_.components_ *= signs[:, None]
            model.score_pca_components_ *= signs[:, None]
    elif hasattr(model, "eigenfunction_values_"):
        if model.eigenfunction_values_ is None:
            raise ValueError("The model has not been fitted yet; call fit() first.")
        _flip_to_positive_mean(model.eigenfunction_values_)
    else:
        raise TypeError(f"Cannot orient the eigenfunctions of a {type(model).__name__}.")
    return model


def _eigenvalues_of(model_or_eigenvalues) -> np.ndarray:
    """Return a 1-D eigenvalue array from a fitted model or an array-like."""
    ev = getattr(model_or_eigenvalues, "eigenvalues_", model_or_eigenvalues)
    if ev is None:
        raise ValueError("The model has not been fitted yet; call fit() first.")
    ev = np.asarray(ev, dtype=float).ravel()
    if ev.size == 0:
        raise ValueError("No eigenvalues are available.")
    return ev


def explained_variance_ratio(model_or_eigenvalues) -> np.ndarray:
    """Share of the retained variance carried by each component.

    Parameters
    ----------
    model_or_eigenvalues : fitted model or array-like
        A fitted model with an ``eigenvalues_`` attribute (``PACE``, ``MFPCA``)
        or a 1-D array of eigenvalues in descending order.

    Returns
    -------
    ndarray, shape (n_components,)
        ``eigenvalues / eigenvalues.sum()``. For ``PACE`` the denominator is
        the variance retained by ``variance_threshold``, not the total
        variance of the process.
    """
    ev = _eigenvalues_of(model_or_eigenvalues)
    return ev / ev.sum()


def n_components_for_variance(model_or_eigenvalues, share: float = 0.95) -> int:
    """Smallest number of leading components whose cumulative share reaches ``share``.

    Parameters
    ----------
    model_or_eigenvalues : fitted model or array-like
        As in :func:`explained_variance_ratio`.
    share : float in (0, 1], default=0.95
        Target cumulative share of the retained variance.

    Returns
    -------
    int
        Number of components, between 1 and ``len(eigenvalues)``.
    """
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share!r}.")
    cum = np.cumsum(explained_variance_ratio(model_or_eigenvalues))
    # Tolerance so that share=1.0 is reached despite floating-point round-off.
    k = int(np.searchsorted(cum, share - 1e-12) + 1)
    return min(k, cum.size)


def reconstruction_rmse(
    model,
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    max_components: int = 10,
    scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-curve RMSE of PACE reconstructions with 0, 1, ..., K components.

    Each curve is scored once from all of its observed points (PACE conditional
    expectation). Its reconstruction ``mean + sum_{k<=K} score_k * phi_k`` is
    then evaluated at the curve's own observation times and compared with the
    observed values. Column 0 uses the mean function alone, so the columns
    show how quickly the error falls as components are added.

    This is an in-sample measure of fit; for model selection prefer held-out
    curves or held-out points.

    Parameters
    ----------
    model : PACE
        A fitted :class:`fpca.PACE` model.
    curves : list of (times, values)
        Curves to evaluate, in the same format as ``PACE.fit``.
    max_components : int, default=10
        Largest number of components ``K`` to evaluate. Clamped to the number
        of components the model retains.
    scores : ndarray, shape (n_curves, >= K), optional
        Pre-computed scores (rows aligned with ``curves``), for example from
        ``model.transform``. Saves re-scoring large datasets.

    Returns
    -------
    rmse : ndarray, shape (n_curves, K + 1)
        ``rmse[i, k]`` is the RMSE of curve ``i`` reconstructed with ``k``
        components.
    """
    if getattr(model, "grid_", None) is None:
        raise ValueError("The model has not been fitted yet; call fit() first.")
    K = int(min(max_components, model.n_components_available_))
    if scores is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scores = model.transform(curves, n_components=K)
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] != len(curves) or scores.shape[1] < K:
        raise ValueError(
            f"scores must have one row per curve and at least {K} columns, "
            f"got shape {scores.shape} for {len(curves)} curves."
        )

    out = np.empty((len(curves), K + 1))
    for i, (t, v) in enumerate(curves):
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        resid = v - np.interp(t, model.grid_, model.mean_values_)
        out[i, 0] = np.sqrt(np.mean(resid ** 2))
        if K == 0:
            continue
        phi = model._interp_eigenfunctions(t, K)  # (K, len(t))
        fitted = np.cumsum(scores[i, :K, None] * phi, axis=0)  # row k-1: first k components
        out[i, 1:] = np.sqrt(np.mean((resid[None, :] - fitted) ** 2, axis=1))
    return out
