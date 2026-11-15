"""
Plotting helpers for fitted FPCA models.

Every function draws with Matplotlib, never calls ``plt.show()``, and returns
the figure (and axes) so the caller can restyle, save or embed it. Colours
follow the active Matplotlib colour cycle unless a ``colors`` mapping is
given, and a feature or group keeps the same colour across panels.

Univariate (``PACE``)
    :func:`plot_curves`, :func:`plot_modes_of_variation`,
    :func:`plot_cumulative_variance`, :func:`plot_reconstruction`,
    :func:`plot_reconstruction_error`, :func:`plot_scores`,
    :func:`plot_scores_by`, :func:`plot_feature_eigenfunctions`

Multivariate (``MFPCA``)
    :func:`plot_mfpca_eigenfunctions`,
    :func:`plot_mfpca_reconstruction_comparison`,
    :func:`plot_mfpca_extreme_subjects`

``PACE`` and ``TwoLevelPACE`` also provide ``plot_mean`` and
``plot_eigenfunctions`` methods of their own.
"""

from __future__ import annotations

import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .utils import explained_variance_ratio

__all__ = [
    "plot_curves",
    "plot_modes_of_variation",
    "plot_cumulative_variance",
    "plot_reconstruction",
    "plot_reconstruction_error",
    "plot_scores",
    "plot_scores_by",
    "plot_feature_eigenfunctions",
    "plot_mfpca_eigenfunctions",
    "plot_mfpca_reconstruction_comparison",
    "plot_mfpca_extreme_subjects",
]

_GRID_ALPHA = 0.25


def _resolve_colors(names: Sequence[str], colors: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Map each name to a colour: explicit entries first, then the colour cycle in order."""
    resolved = dict(colors) if colors else {}
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    unassigned = [n for n in names if n not in resolved]
    for i, n in enumerate(unassigned):
        resolved[n] = cycle[i % len(cycle)]
    return resolved


def _style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=_GRID_ALPHA)
    ax.spines[["top", "right"]].set_visible(False)


def _new_axes(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


# ---------------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------------

def plot_curves(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_curves: Optional[int] = None,
    ax=None,
    color: Optional[str] = None,
    alpha: float = 0.4,
    markers: bool = True,
    xlabel: str = "t",
    ylabel: str = "value",
    title: Optional[str] = None,
    figsize=(8, 4),
):
    """Spaghetti plot of raw ``(times, values)`` curves.

    Parameters
    ----------
    curves : list of (times, values)
        Curves in the ``PACE.fit`` input format.
    n_curves : int, optional
        Plot only the first ``n_curves`` curves.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if None.
    color : str, optional
        One colour for every curve. Defaults to the first colour of the cycle.
    alpha : float, default=0.4
        Line transparency.
    markers : bool, default=True
        Mark each observation. Useful for sparse data.

    Returns
    -------
    fig, ax
    """
    fig, ax = _new_axes(ax, figsize)
    color = color or plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    shown = curves if n_curves is None else curves[:n_curves]
    for t, v in shown:
        order = np.argsort(t)
        ax.plot(np.asarray(t)[order], np.asarray(v)[order], "-o" if markers else "-",
                color=color, alpha=alpha, lw=1, ms=3)
    _style(ax, xlabel, ylabel, title or f"{len(shown)} observed curves")
    return fig, ax


def plot_modes_of_variation(
    model,
    n_components: int = 3,
    scale: float = 2.0,
    xlabel: str = "t",
    ylabel: str = "value",
    figsize=None,
    n_cols: Optional[int] = None,
):
    """Mean function plus and minus ``scale * sqrt(eigenvalue_k) * phi_k(t)``.

    The standard way to read a functional principal component: each panel
    shows how a curve with a score ``scale`` standard deviations above (``+``)
    or below (``-``) the mean on component ``k`` differs from the mean curve.

    Parameters
    ----------
    model : PACE
        A fitted :class:`fpca.PACE` model (or any fitted model with ``grid_``,
        ``mean_values_``, ``eigenvalues_`` and ``eigenfunction_values_``).
    n_components : int, default=3
        Number of leading components to show (clamped to those retained).
    scale : float, default=2.0
        Multiple of the component's standard deviation ``sqrt(eigenvalue)``.
    n_cols : int, optional
        Panels per row. Defaults to all panels in one row.

    Returns
    -------
    fig, axes : axes is a flat array with one entry per component
    """
    if getattr(model, "grid_", None) is None:
        raise ValueError("The model has not been fitted yet; call fit() first.")
    n = int(min(n_components, len(model.eigenvalues_)))
    ratio = explained_variance_ratio(model)
    n_cols = n if n_cols is None else max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (4 * n_cols, 3.6 * n_rows), sharey=True,
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    axes = axes[:n]
    grid, mean = model.grid_, model.mean_values_
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, ax in enumerate(axes):
        delta = scale * np.sqrt(model.eigenvalues_[k]) * model.eigenfunction_values_[k]
        ax.plot(grid, mean, color="0.25", lw=2, label="mean")
        ax.plot(grid, mean + delta, color=cycle[0], lw=2, label=f"+{scale:g} SD")
        ax.plot(grid, mean - delta, color=cycle[1], lw=2, ls="--", label=f"-{scale:g} SD")
        _style(ax, xlabel, ylabel if k % n_cols == 0 else None, f"PC{k + 1} ({100 * ratio[k]:.1f}%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Modes of variation", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_cumulative_variance(
    models,
    max_components: int = 10,
    ax=None,
    colors: Optional[Mapping[str, str]] = None,
    figsize=(7, 4.5),
):
    """Cumulative share of retained variance explained by the first 1..K components.

    Parameters
    ----------
    models : fitted model or dict of str to fitted model
        One model (``PACE``, ``MFPCA``) or several, keyed by a display name.
    max_components : int, default=10
        Largest number of components on the x-axis.
    colors : dict, optional
        Name -> colour mapping.

    Returns
    -------
    fig, ax
    """
    if not isinstance(models, Mapping):
        models = {"model": models}
    colors = _resolve_colors(list(models), colors)
    fig, ax = _new_axes(ax, figsize)
    for name, m in models.items():
        cum = 100 * np.cumsum(explained_variance_ratio(m))[:max_components]
        ax.plot(np.arange(1, cum.size + 1), cum, "-o", lw=2, ms=6, color=colors[name], label=name)
    ax.set_xticks(range(1, max_components + 1))
    ax.set_ylim(top=101)
    _style(ax, "Number of components", "Cumulative variance explained (%)",
           "Cumulative variance explained")
    if len(models) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_reconstruction(
    model,
    times: np.ndarray,
    values: np.ndarray,
    n_components: Sequence[int] = (0, 1, 2, 3),
    scores: Optional[np.ndarray] = None,
    true_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    xlabel: str = "t",
    ylabel: str = "value",
    panel_size=(3.6, 3.0),
):
    """Reconstruction of one curve with an increasing number of components.

    One panel per entry of ``n_components``; each shows the observed points
    and ``mean + sum_{k<K} score_k * phi_k`` on the model's grid. ``K = 0`` is
    the mean function alone.

    Parameters
    ----------
    model : PACE
        A fitted :class:`fpca.PACE` model.
    times, values : ndarray
        The curve's observations.
    n_components : sequence of int, default=(0, 1, 2, 3)
        Numbers of components to show; clamped to those retained.
    scores : ndarray, optional
        Pre-computed PACE scores of this curve.
    true_curve : (times, values), optional
        Noise-free curve to overlay (for simulations).

    Returns
    -------
    fig, axes
    """
    ks = sorted({int(min(k, model.n_components_available_)) for k in n_components})
    k_max = max(ks)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if scores is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scores = model.transform([(times, values)], n_components=max(k_max, 1))[0]
    scores = np.asarray(scores, dtype=float)
    cum = np.concatenate([[0.0], 100 * np.cumsum(explained_variance_ratio(model))])
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, len(ks), figsize=(panel_size[0] * len(ks), panel_size[1]),
                             sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        fit = model.mean_values_ + scores[:k] @ model.eigenfunction_values_[:k]
        if true_curve is not None:
            ax.plot(*true_curve, color="0.55", lw=1.5, ls=":", label="true curve")
        ax.scatter(times, values, s=16, color=cycle[0], alpha=0.7, label="observed", zorder=2)
        ax.plot(model.grid_, fit, color=cycle[1], lw=2.5, label="reconstruction", zorder=3)
        _style(ax, xlabel, None, f"K = {k} ({cum[k]:.1f}% of variance)")
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes[0]


def plot_reconstruction_error(
    errors,
    normalize: Optional[Mapping[str, float]] = None,
    statistic: str = "mean",
    ax=None,
    colors: Optional[Mapping[str, str]] = None,
    figsize=(7, 4.5),
):
    """Reconstruction error against the number of components.

    Parameters
    ----------
    errors : ndarray or dict of str to ndarray
        Output of :func:`fpca.utils.reconstruction_rmse` (shape
        ``(n_curves, K + 1)``), or several such arrays keyed by a display name.
    normalize : dict of str to float, optional
        Divide each model's errors by this value (e.g. the pooled standard
        deviation of the observations) to plot a normalised RMSE that is
        comparable across signals measured in different units.
    statistic : {'mean', 'median'}, default='mean'
        How per-curve errors are summarised.

    Returns
    -------
    fig, ax
    """
    if not isinstance(errors, Mapping):
        errors = {"model": errors}
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'.")
    agg = np.mean if statistic == "mean" else np.median
    colors = _resolve_colors(list(errors), colors)
    fig, ax = _new_axes(ax, figsize)
    for name, e in errors.items():
        y = agg(np.asarray(e), axis=0)
        if normalize is not None:
            y = y / normalize[name]
        ax.plot(np.arange(y.size), y, "-o", lw=2, ms=6, color=colors[name], label=name)
    ax.set_xticks(range(max(np.asarray(e).shape[1] for e in errors.values())))
    ylabel = f"{statistic.capitalize()} {'normalised ' if normalize is not None else ''}RMSE"
    _style(ax, "Number of components (0 = mean only)", ylabel, "Reconstruction error")
    if len(errors) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_scores(
    scores: np.ndarray,
    labels: Optional[Sequence] = None,
    components: Tuple[int, int] = (0, 1),
    ax=None,
    colors: Optional[Mapping[str, str]] = None,
    annotate: Optional[Sequence[str]] = None,
    figsize=(6, 5),
    cmap: Optional[str] = None,
    label_name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    **scatter_kwargs,
):
    """Scatter plot of two score columns, optionally coloured by group or by a numeric value.

    Parameters
    ----------
    scores : ndarray, shape (n_samples, n_components)
        Scores from ``PACE.transform`` or ``MFPCA.transform``.
    labels : array-like, optional
        Group label of each row: one colour per group, with a legend. With
        ``cmap``, a numeric value per row instead, drawn on a colour scale with a
        colorbar.
    components : (int, int), default=(0, 1)
        0-based score columns for the x and y axes.
    annotate : sequence of str, optional
        Text placed next to each point (e.g. sample names).
    cmap : str, optional
        Matplotlib colormap name. If given, ``labels`` is treated as continuous.
    label_name : str, optional
        Legend title (categorical labels) or colorbar label (continuous labels).
    xlabel, ylabel, title : str, optional
        Axis labels and title. Default to ``"PC<i> score"`` and ``"Component scores"``.

    Returns
    -------
    fig, ax
    """
    scores = np.asarray(scores)
    i, j = components
    fig, ax = _new_axes(ax, figsize)
    kw = {"s": 30, "alpha": 0.8, "edgecolors": "white", "linewidths": 0.5}
    kw.update(scatter_kwargs)
    if labels is None:
        ax.scatter(scores[:, i], scores[:, j], **kw)
    elif cmap is not None:
        sc = ax.scatter(scores[:, i], scores[:, j], c=np.asarray(labels, dtype=float), cmap=cmap, **kw)
        fig.colorbar(sc, ax=ax, label=label_name)
    else:
        labels = np.asarray(labels).astype(str)
        groups = list(dict.fromkeys(labels.tolist()))
        group_colors = _resolve_colors(groups, colors)
        for g in groups:
            m = labels == g
            ax.scatter(scores[m, i], scores[m, j], color=group_colors[g], label=g, **kw)
        ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", title=label_name)
    if annotate is not None:
        for x, y, text in zip(scores[:, i], scores[:, j], annotate):
            ax.annotate(text, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color="0.3")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    _style(ax, xlabel or f"PC{i + 1} score", ylabel or f"PC{j + 1} score", title or "Component scores")
    fig.tight_layout()
    return fig, ax


def plot_scores_by(
    scores: np.ndarray,
    labels: Mapping[str, Sequence],
    continuous: Sequence[str] = (),
    cmap: str = "viridis",
    components: Tuple[int, int] = (0, 1),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    panel_size=(6.4, 5.5),
    **scatter_kwargs,
):
    """The same score scatter coloured by several labels, one panel per label.

    Parameters
    ----------
    scores : ndarray, shape (n_samples, n_components)
        Score matrix, as for :func:`plot_scores`.
    labels : dict of str to array-like
        Panel name -> one label per row. Panels are drawn in this order and
        titled ``"Coloured by <name>"``.
    continuous : sequence of str, default=()
        Names in ``labels`` whose values are numeric and drawn on the colour
        scale ``cmap`` with a colorbar; the others get one colour per group.
    components, xlabel, ylabel, **scatter_kwargs
        As for :func:`plot_scores`.
    title : str, optional
        Figure title.

    Returns
    -------
    fig, axes : axes has one entry per label
    """
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (name, values) in zip(axes, labels.items()):
        plot_scores(scores, labels=values, components=components, ax=ax,
                    cmap=cmap if name in continuous else None, label_name=name,
                    xlabel=xlabel, ylabel=ylabel, title=f"Coloured by {name}", **scatter_kwargs)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_feature_eigenfunctions(
    models: Mapping[str, object],
    n_components: int = 2,
    labels: Optional[Mapping[str, str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    xlabel: str = "t",
    title: Optional[str] = None,
    figsize=None,
):
    """Leading eigenfunctions of several univariate models side by side.

    One column per model (feature) and one row per component, so the same
    component can be compared across features at a glance. Each panel title
    gives the component's share of that model's retained variance.

    Parameters
    ----------
    models : dict of str to fitted model
        Feature name -> fitted model with ``grid_``, ``eigenvalues_`` and
        ``eigenfunction_values_`` (``PACE``, or one level of a ``TwoLevelPACE``
        from ``level_model``).
    n_components : int, default=2
        Number of leading components (rows). A model with fewer components
        leaves its extra panels empty.
    labels : dict of str to str, optional
        Feature -> display name.
    colors : dict, optional
        Feature -> colour mapping.

    Returns
    -------
    fig, axes : axes has shape (n_components, n_features)
    """
    names = list(models)
    labels = {**{n: n for n in names}, **(labels or {})}
    colors = _resolve_colors(names, colors)
    n_f = len(names)
    fig, axes = plt.subplots(n_components, n_f, figsize=figsize or (3.4 * n_f, 2.8 * n_components),
                             sharex=True, squeeze=False)
    for c, name in enumerate(names):
        model = models[name]
        ratio = explained_variance_ratio(model)
        for k in range(n_components):
            ax = axes[k, c]
            if k >= len(ratio):
                ax.set_visible(False)
                continue
            ax.axhline(0, color="0.35", lw=0.9, zorder=1)
            ax.plot(model.grid_, model.eigenfunction_values_[k], color=colors[name], lw=2)
            _style(ax, xlabel if k == n_components - 1 else None, f"PC{k + 1}" if c == 0 else None,
                   f"{labels[name]} ({100 * ratio[k]:.1f}%)")
            ax.title.set_fontsize(9)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------

def plot_mfpca_eigenfunctions(
    mfpca,
    features: Optional[Sequence[str]] = None,
    n_components: Optional[int] = None,
    colors: Optional[Mapping[str, str]] = None,
    descale: bool = False,
    xlabel: str = "t",
    figsize=None,
    n_cols: int = 2,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
):
    """Multivariate eigenfunctions: one panel per component, features overlaid.

    Parameters
    ----------
    mfpca : MFPCA
        A fitted :class:`fpca.MFPCA` model (either method, fitted with ``fit`` or
        ``fit_from_scores``).
    features : list of str, optional
        Features to draw. Defaults to ``mfpca.feature_names_``.
    n_components : int, optional
        Number of components to show. Defaults to all.
    colors : dict, optional
        Feature -> colour mapping.
    descale : bool, default=False
        If True, show each feature's eigenfunction in its original units
        (see ``MFPCA.get_feature_weight_function``). Otherwise they are shown
        on the standardised scale used for fitting, which makes the relative
        contribution of each feature comparable.
    n_cols : int, default=2
        Panels per row. With ``n_cols=1`` the legend is placed to the right.
    labels : dict of str to str, optional
        Feature -> display name used in the legend.
    title : str, optional
        Figure title. Defaults to ``"Multivariate eigenfunctions (<method>)"``.

    Returns
    -------
    fig, axes
    """
    features = list(features or mfpca.feature_names_)
    colors = _resolve_colors(features, colors)
    labels = {**{f: f for f in features}, **(labels or {})}
    n = mfpca.n_components_ if n_components is None else min(n_components, mfpca.n_components_)
    n_cols = max(1, min(n, n_cols))
    n_rows = int(np.ceil(n / n_cols))
    width = 13 if n_cols == 1 else 6 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (width, 3.6 * n_rows), squeeze=False)
    axes = axes.ravel()
    grid = mfpca.get_time_grid()
    ratio = 100 * mfpca.get_explained_variance_ratio()
    cum = np.cumsum(ratio)
    ylabel = "loading" if descale else "loading (scaled)"
    for k in range(n):
        ax = axes[k]
        for feat in features:
            wf = mfpca.get_feature_weight_function(feat, descale=descale)
            ax.plot(grid, wf[k], color=colors[feat], lw=2, label=labels[feat])
        ax.axhline(0, color="0.35", lw=0.9, zorder=1)
        _style(ax, xlabel, ylabel, f"MV-PC{k + 1}  ({ratio[k]:.1f}%, cum {cum[k]:.1f}%)")
        if k == 0 and len(features) > 1:
            if n_cols == 1:
                ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
            else:
                ax.legend(frameon=False, fontsize=9)
    for ax in axes[n:]:
        ax.set_visible(False)
    method = {"happ_greven": "Happ-Greven", "berrendero": "Berrendero"}.get(mfpca.method, mfpca.method)
    fig.suptitle(title or f"Multivariate eigenfunctions ({method})", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_mfpca_reconstruction_comparison(
    mfpca,
    data: Mapping[str, Sequence[Tuple[np.ndarray, np.ndarray]]],
    mfpca_scores: np.ndarray,
    subject_indices: Sequence[int],
    features: Optional[Sequence[str]] = None,
    xlabel: str = "t",
    figsize=None,
):
    """Observed data vs. univariate PACE vs. multivariate MFPCA reconstructions.

    Rows are subjects and columns are features. Each panel shows the observed
    points, the feature's own univariate PACE reconstruction, the MFPCA
    reconstruction from the joint scores, and the feature's mean function.

    Parameters
    ----------
    mfpca : MFPCA
        A fitted :class:`fpca.MFPCA` model; its ``pace_models_`` are used for
        the univariate reconstructions.
    data : dict of str to list of (times, values)
        Multivariate sample, in the ``MFPCA.fit`` format.
    mfpca_scores : ndarray, shape (n_subjects, n_components)
        Output of ``mfpca.transform(data)``.
    subject_indices : sequence of int
        Rows of ``data`` / ``mfpca_scores`` to display.
    features : list of str, optional
        Features to display. Defaults to ``mfpca.feature_names_``.

    Returns
    -------
    fig, axes
    """
    features = list(features or mfpca.feature_names_)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_s, n_f = len(subject_indices), len(features)
    fig, axes = plt.subplots(n_s, n_f, figsize=figsize or (4.8 * n_f, 3.4 * n_s), squeeze=False)
    for r, s in enumerate(subject_indices):
        for c, feat in enumerate(features):
            ax = axes[r, c]
            pace = mfpca.pace_models_[feat]
            t, v = data[feat][s]
            order = np.argsort(t)
            t, v = np.asarray(t, dtype=float)[order], np.asarray(v, dtype=float)[order]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                uni_scores = pace.transform([(t, v)])[0]
            uni_recon, uni_t = pace.reconstruct_curve(t, uni_scores)
            mv_recon, mv_t = mfpca.reconstruct_feature(mfpca_scores[s], feat)
            ax.plot(pace.grid_, pace.mean_values_, color="0.6", lw=1.5, ls=":", label="mean")
            ax.scatter(t, v, s=14, color=cycle[0], alpha=0.7, label="observed", zorder=2)
            ax.plot(uni_t, uni_recon, color=cycle[1], lw=2, label="univariate PACE", zorder=3)
            ax.plot(mv_t, mv_recon, color=cycle[2], lw=2, ls="--", label="MFPCA", zorder=4)
            _style(ax, xlabel if r == n_s - 1 else None, feat if c == 0 else None,
                   f"Subject {s} | {feat}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Univariate PACE vs. multivariate MFPCA reconstruction", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_mfpca_extreme_subjects(
    mfpca,
    data: Mapping[str, Sequence[Tuple[np.ndarray, np.ndarray]]],
    mfpca_scores: np.ndarray,
    component: int = 0,
    features: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    xlabel: str = "t",
    figsize=None,
):
    """The highest- and lowest-scoring subjects on one multivariate component.

    One row per feature, two columns (highest | lowest). Each panel shows the
    subject's observations and MFPCA reconstruction as deviations from the
    feature's mean function, which makes the component's joint pattern across
    features visible on real subjects.

    Parameters
    ----------
    mfpca : MFPCA
        A fitted :class:`fpca.MFPCA` model.
    data : dict of str to list of (times, values)
        Multivariate sample, in the ``MFPCA.fit`` format.
    mfpca_scores : ndarray, shape (n_subjects, n_components)
        Output of ``mfpca.transform(data)``.
    component : int, default=0
        0-based index of the multivariate component.

    Returns
    -------
    fig, axes
    """
    features = list(features or mfpca.feature_names_)
    colors = _resolve_colors(features, colors)
    scores = np.asarray(mfpca_scores)
    picks = [
        (int(np.argmax(scores[:, component])), "Highest"),
        (int(np.argmin(scores[:, component])), "Lowest"),
    ]
    n_f = len(features)
    fig, axes = plt.subplots(n_f, 2, figsize=figsize or (12, 3.4 * n_f), sharey="row", squeeze=False)
    for f_idx, feat in enumerate(features):
        pace = mfpca.pace_models_[feat]
        for col, (s, label) in enumerate(picks):
            ax = axes[f_idx, col]
            t, v = data[feat][s]
            t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
            order = np.argsort(t)
            t, v = t[order], v[order]
            ax.plot(t, v - np.interp(t, pace.grid_, pace.mean_values_), "o", ms=3,
                    color=colors[feat], alpha=0.6, label="observed")
            recon, rt = mfpca.reconstruct_feature(scores[s], feat)
            ax.plot(rt, recon - np.interp(rt, pace.grid_, pace.mean_values_), lw=2.5,
                    color=colors[feat], label="MFPCA reconstruction")
            ax.axhline(0, color="0.6", ls="--", lw=1, label="mean")
            _style(ax, xlabel if f_idx == n_f - 1 else None,
                   f"{feat}\n(deviation from mean)" if col == 0 else None,
                   f"{label} | subject {s} | MV-PC{component + 1} = {scores[s, component]:.2f}")
            if f_idx == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Extreme subjects on MV-PC{component + 1}", fontweight="bold")
    fig.tight_layout()
    return fig, axes
