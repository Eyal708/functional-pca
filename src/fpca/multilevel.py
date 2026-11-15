"""Two-level functional principal component analysis (multilevel FPCA).

Implements the two-level FPCA model of

    Di, C.-Z., Crainiceanu, C. M., Caffo, B. S. and Punjabi, N. M. (2009).
    "Multilevel functional principal component analysis."
    Annals of Applied Statistics, 3(1), 458-488.

Model
-----
Each curve Y_ij(t) (subject i, visit j) is decomposed as

    Y_ij(t) = mu(t) + eta_j(t) + xi_i(t) + zeta_ij(t) + eps_ij(t),

where mu(t) is the population mean, eta_j(t) the visit effect, xi_i(t) the
between-subject (level-1) signal, zeta_ij(t) the within-subject (level-2)
deviation and eps_ij(t) measurement error. The visit effect is estimated only
when ``visit_effects=True``; otherwise it is omitted from the model.

Input
-----
The input convention follows ``PACE.fit``. ``curves`` is a sequence of
``(times, values)`` pairs, one per curve, where each curve is one subject at one
visit. Curves may be observed at different times. ``subject_ids`` gives the
subject label of each curve, and ``visit_ids`` gives the visit number of each
curve (required when ``visit_effects=True``).

Estimation
----------
1. Common grid: the sorted union of all observation times.
2. Mean: the pointwise average of the observed values. Visit effects (optional):
   the visit-specific average minus the mean.
3. Raw covariance surfaces by the method of moments: the total surface K_T from
   pairs of observations on the same curve, and the between-subject surface K_B
   from pairs of observations on distinct curves of the same subject. The
   within-subject surface is K_W = K_T - K_B.
4. Smoothing: each surface is smoothed with a tensor-product cubic P-spline, with
   the smoothing parameter chosen by REML. The diagonal of the total surface is
   excluded from its fit (Di et al. 2009, Sec. 2.3).
5. Noise variance: sigma^2 is the integral of G_T(t, t) - K_T(t, t) over the grid
   domain, divided by the domain length.
6. Eigendecomposition of the between-subject and within-subject surfaces through
   ``PACE._eigendecompose`` (Yao, Muller and Wang 2005).
7. Component counts, per level: the threshold rule of Di et al. (2009, Sec. 2.2)
   when ``variance_threshold`` is a float, or a fixed count when it is an int.
8. Scores: Gaussian posterior means (BLUPs) of the full model (Di et al. 2009,
   Sec. 3.1), computed per subject from each curve's own observation times.

Dependencies
------------
The mean function and the eigendecomposition step are delegated to
:class:`fpca.pace.PACE` (``PACE._fit_mean`` and ``PACE._eigendecompose``), so
both levels use exactly the same quadrature-weighted eigenproblem as the
univariate estimator. Everything else (the multilevel moment estimates, the
REML-tuned P-spline surface smoother and the BLUP scores) is implemented in this
module.

Example
-------
>>> from fpca import TwoLevelPACE
>>> model = TwoLevelPACE(variance_threshold=0.9).fit(curves, subject_ids)
>>> scores = model.transform(curves, subject_ids)
>>> scores["between"].shape, scores["within"].shape
((n_subjects, N1), (n_curves, N2))

References
----------
Di, C.-Z., Crainiceanu, C. M., Caffo, B. S. and Punjabi, N. M. (2009).
    Multilevel functional principal component analysis.
    Annals of Applied Statistics, 3(1), 458-488.
    https://doi.org/10.1214/08-AOAS206
Yao, F., Muller, H.-G. and Wang, J.-L. (2005).
    Functional data analysis for sparse longitudinal data.
    Journal of the American Statistical Association, 100(470), 577-590.
    https://doi.org/10.1198/016214504000001745
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import BSpline

from .pace import PACE

__all__ = ["TwoLevelPACE", "LevelModel"]


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Composite trapezoid rule on the grid (np.trapz is deprecated in NumPy 2.x).

    Used for the noise-variance integral (Di et al. 2009, Sec. 2.3, step 4).
    """
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) * 0.5))


def _curves_to_matrix(curves: Sequence, grid: np.ndarray) -> np.ndarray:
    """Place each curve's (times, values) on `grid`; NaN where not observed.

    Non-finite values are dropped. Duplicate times within a curve are averaged,
    with a warning, as in ``PACE.fit``.

    Returns
    -------
    ndarray, shape (n_curves, len(grid))
    """
    L = len(grid)
    Y = np.full((len(curves), L), np.nan)
    n_dup = 0
    for i, (t, v) in enumerate(curves):
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        idx = np.searchsorted(grid, t)
        if np.any(idx >= L) or not np.array_equal(grid[np.minimum(idx, L - 1)], t):
            raise ValueError(f"curve {i}: observation times are not on the fitted grid")
        uniq, inv = np.unique(idx, return_inverse=True)
        n_dup += len(idx) - len(uniq)
        Y[i, uniq] = np.bincount(inv, weights=v) / np.bincount(inv)
    if n_dup:
        warnings.warn(f"{n_dup} duplicate (curve, time) observations were averaged")
    return Y


def _raw_covariance_total(y_centered: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw total covariance K_T on the grid, by the method of moments (Di et al. 2009, Sec. 2.3).

    Averages y(s) y(t) over all pairs of grid points observed on the same curve.
    Returns the surface (NaN cells set to 0) and the matrix of pair counts, which
    serves as the fitting weight.
    """
    a = np.where(mask, y_centered, 0.0)
    total_sum = a.T @ a
    total_count = mask.T.astype(float) @ mask.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = total_sum / total_count
    return np.nan_to_num(cov, nan=0.0), total_count


def _raw_covariance_between(
    y_centered: np.ndarray, mask: np.ndarray, subject_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw between-subject covariance K_B on the grid (Di et al. 2009, Sec. 2.3).

    Averages y(s) y(t) over ordered pairs of distinct curves belonging to the same
    subject, pooled over subjects. Visit labels are not inspected. Cells with no
    such pair are set to 0 with zero weight. Returns the surface and the matrix of
    pair counts.

    Raises
    ------
    ValueError
        If no subject contributes at least two curves.

    Notes
    -----
    The sum over ordered pairs of distinct curves is computed without an explicit
    double loop: sum_{j != j'} outer(A_j, A_j') = outer(S, S) - sum_j outer(A_j, A_j),
    where S = sum_j A_j.
    """
    L = y_centered.shape[1]
    sum_total = np.zeros((L, L))
    count_total = np.zeros((L, L))
    for s in np.unique(subject_ids):
        idx = np.where(subject_ids == s)[0]
        if len(idx) < 2:
            continue
        a = np.where(mask[idx], y_centered[idx], 0.0)
        m = mask[idx].astype(float)
        a_sum = a.sum(axis=0)
        m_sum = m.sum(axis=0)
        sum_total += np.outer(a_sum, a_sum) - (a.T @ a)
        count_total += np.outer(m_sum, m_sum) - (m.T @ m)
    if not np.any(count_total > 0):
        raise ValueError("No subject has >=2 curves -- cannot estimate a between-subject surface.")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sum_total / count_total
    return np.nan_to_num(cov, nan=0.0), count_total


def _bspline_basis(grid: np.ndarray, n_segments: int, degree: int) -> np.ndarray:
    """Equally spaced B-spline design matrix on the grid, shape (L, n_segments + degree).

    Knots extend `degree` segments beyond each end of the grid, so each row sums to 1
    (partition of unity). Used as the P-spline basis of Di et al. (2009, Sec. 2.3).
    """
    a, b = float(grid[0]), float(grid[-1])
    dx = (b - a) / n_segments
    # The first and last interior knots are set to a and b exactly: computing them as
    # a + dx * k can round just inside the grid's end and make design_matrix fail.
    inner = np.linspace(a, b, n_segments + 1)
    knots = np.concatenate([a - dx * np.arange(degree, 0, -1), inner, b + dx * np.arange(1, degree + 1)])
    return BSpline.design_matrix(np.asarray(grid, dtype=float), knots, degree).toarray()


def _difference_matrix(k: int, order: int) -> np.ndarray:
    """Matrix of the `order`-th finite differences on k coefficients, shape (k - order, k).

    Its Gram matrix D'D defines the P-spline difference penalty (Di et al. 2009, Sec. 2.3).
    """
    D = np.eye(k)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def _smooth_surface(
    raw: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    n_segments: int = 15,
    degree: int = 3,
    order: int = 2,
    n_lambda: int = 61,
) -> tuple[np.ndarray, float]:
    """Penalized tensor-product P-spline smoother for a covariance surface (Di et al. 2009, Sec. 2.3).

    Minimizes sum_ij w_ij (raw_ij - F_ij)^2 + lambda theta' P theta, with F = B Theta B' and
    P = kron(D'D, I) + kron(I, D'D). Cells with weight 0 are dropped from the fit. Lambda is
    chosen on a log grid by the REML criterion (profile form, penalty null space removed).

    Parameters
    ----------
    raw : ndarray, shape (L, L)
        Raw covariance estimate on the grid.
    weights : ndarray, shape (L, L)
        Fitting weights (pair counts); zero excludes a cell.
    grid : ndarray, shape (L,)
        Evaluation grid.
    n_segments, degree, order, n_lambda
        B-spline segments, B-spline degree, difference order of the penalty, and the
        number of candidate smoothing parameters.

    Returns
    -------
    surface : ndarray, shape (L, L)
        Symmetric smoothed surface.
    lam : float
        Selected smoothing parameter.
    """
    B = _bspline_basis(grid, n_segments, degree)
    _, k = B.shape
    D = _difference_matrix(k, order)
    DtD = D.T @ D
    Ik = np.eye(k)
    P = np.kron(DtD, Ik) + np.kron(Ik, DtD)  # acts on row-major vec(Theta)
    null_dim = order * order

    W = np.where(weights > 0, weights, 0.0)
    Y = np.where(W > 0, raw, 0.0)

    Z = np.einsum("jb,ij,jd->ibd", B, W, B, optimize=True)  # Z_i = B' diag(W_i) B
    XtWX = np.einsum("ia,ic,ibd->abcd", B, B, Z, optimize=True).reshape(k * k, k * k)
    XtWy = (B.T @ ((W * Y) @ B)).reshape(k * k)

    scale = np.trace(XtWX) / np.trace(P)
    lambdas = scale * np.logspace(-8, 4, n_lambda)
    n_eff = int(np.count_nonzero(W))
    rank_P = k * k - null_dim
    ridge = 1e-12 * np.trace(XtWX) / (k * k)

    best = None
    for lam in lambdas:
        M = XtWX + lam * P + ridge * np.eye(k * k)
        theta = np.linalg.solve(M, XtWy)
        F = B @ theta.reshape(k, k) @ B.T
        rss = float(np.sum(W * (Y - F) ** 2) + lam * theta @ P @ theta)
        _, logdet = np.linalg.slogdet(M)
        crit = (n_eff - null_dim) * np.log(rss / (n_eff - null_dim)) + logdet - rank_P * np.log(lam)
        if best is None or crit < best[0]:
            best = (crit, lam, theta)

    _, lam_sel, theta_sel = best
    surface = B @ theta_sel.reshape(k, k) @ B.T
    return 0.5 * (surface + surface.T), float(lam_sel)


def _select_n_components(eigenvalues: np.ndarray, p1: float, p2: float) -> int:
    """Number of components by the threshold rule of Di et al. (2009, Sec. 2.2).

    N = min{k : rho_k >= P1 and f_k < P2} (1-based), where f_k = lambda_k / sum(lambda)
    is the variance share of component k and rho_k = cumsum(f)_k its cumulative share.
    If no k qualifies, all components are kept.

    Parameters
    ----------
    eigenvalues : ndarray
        Positive eigenvalues in descending order.
    p1 : float
        Cumulative-variance threshold P1.
    p2 : float
        Per-component share cap P2 (1/T in the paper).

    Returns
    -------
    int
        Number of components (0 if `eigenvalues` is empty).
    """
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.size == 0:
        return 0
    f = lam / lam.sum()
    rho = np.cumsum(f)
    ok = np.where((rho >= p1) & (f < p2))[0]
    return int(ok[0] + 1) if ok.size else int(lam.size)


def _check_one_visit_per_subject(subject_ids: np.ndarray, visit_ids: np.ndarray) -> None:
    """Raise ValueError if any subject has two curves with the same visit number."""
    seen = set()
    for subject, visit in zip(np.asarray(subject_ids).tolist(), np.asarray(visit_ids).tolist()):
        if (subject, visit) in seen:
            raise ValueError(
                f"subject {subject!r} has more than one curve at visit {visit!r}; "
                "each subject may have at most one curve per visit."
            )
        seen.add((subject, visit))


def _visit_labels_to_index(visit_ids: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Map per-curve visit numbers to indices 0..J-1 of the fitted visit levels.

    Raises
    ------
    ValueError
        If a visit number does not appear in `levels`.
    """
    visit_ids = np.asarray(visit_ids)
    idx = np.clip(np.searchsorted(levels, visit_ids), 0, len(levels) - 1)
    unseen = levels[idx] != visit_ids
    if np.any(unseen):
        raise ValueError(f"visit numbers not seen in fit: {np.unique(visit_ids[unseen]).tolist()}")
    return idx


@dataclass
class TwoLevelPACE:
    """Two-level functional principal component analysis (Di et al. 2009) of one signal.

    The estimator is fitted from raw curves with ``fit``, scored with ``transform``
    and reconstructed on the fitted grid with ``reconstruct``. Method details are in
    the module docstring. The attributes below are None until ``fit`` is called.

    Parameters
    ----------
    variance_threshold : float in (0, 1] or int, default 0.9
        If float, the threshold P1 of Di et al. (2009, Sec. 2.2), applied separately
        to the between-subject and within-subject eigenvalues. The cap P2 is 1/T, with
        T the number of grid points. If int, the number of components kept at each
        level (capped at the number of positive eigenvalues).
    n_segments : int, default 15
        Number of equally spaced P-spline segments in each dimension of the covariance
        surfaces.
    visit_effects : bool, default False
        If True, estimate the visit effects eta_j(t) of the model. Then ``visit_ids``
        (the visit number of each curve) must be passed to ``fit``, ``transform`` and
        ``reconstruct``. If False, no visit effects are estimated and ``visit_ids`` is
        ignored.
    mean_kwargs : dict, optional
        Keyword arguments for the ``PACE`` instance that estimates the mean function, so
        that the mean is estimated exactly as a univariate ``PACE`` fit estimates it (the
        pooled raw observations are smoothed, with the same smoothing options). Keys
        override the defaults ``{"method_smoothing": "PS", "smooth_covariance": False}``,
        so the PACE defaults apply (``mean_penalty=1.0``). Pass, for example,
        ``{"mean_penalty": 5.0}`` to change the penalty, or ``{"method_smoothing": "LP"}``
        for local-linear smoothing. The configured PACE instance is kept as ``mean_pace_``.

    Attributes
    ----------
    grid_ : ndarray, shape (L,)
        Sorted union of all observation times. Every surface is evaluated on this grid.
    mean_values_ : ndarray, shape (L,)
        Population mean function mu(t) on ``grid_``.
    mean_pace_ : PACE or None
        The PACE instance that estimated the mean function, configured by ``mean_kwargs``.
        Kept so the smoothing settings and the mean can be inspected after fitting.
    visit_levels_ : ndarray, shape (J,), or None
        Sorted distinct visit numbers seen in ``fit``. None if ``visit_effects`` is False.
    visit_effects_ : ndarray, shape (J, L)
        Visit effects eta_j(t) on ``grid_``, one row per entry of ``visit_levels_``. If
        ``visit_effects`` is False, a single row of zeros (J = 1).
    noise_variance_ : float
        Measurement-error variance sigma^2, either supplied to ``fit`` or estimated.
    between_eigenvalues_ : ndarray, shape (N1,)
        Retained eigenvalues of the between-subject covariance, in descending order.
    between_eigenfunctions_ : ndarray, shape (N1, L)
        L2-orthonormal between-subject eigenfunction values on ``grid_``; row k is phi_k.
    within_eigenvalues_ : ndarray, shape (N2,)
        Retained eigenvalues of the within-subject covariance, in descending order.
    within_eigenfunctions_ : ndarray, shape (N2, L)
        L2-orthonormal within-subject eigenfunction values on ``grid_``; row k is psi_k.
    total_covariance_ : ndarray, shape (L, L)
        Smoothed total covariance surface G_T on ``grid_``. Symmetric.
    between_covariance_ : ndarray, shape (L, L)
        Smoothed between-subject covariance surface K_B on ``grid_``. Symmetric.
    within_covariance_ : ndarray, shape (L, L)
        Within-subject covariance surface K_W = total_covariance_ - between_covariance_.
    total_penalty_ : float
        Smoothing parameter selected by REML for ``total_covariance_``.
    between_penalty_ : float
        Smoothing parameter selected by REML for ``between_covariance_``.
    """

    variance_threshold: float | int = 0.9
    n_segments: int = 15
    visit_effects: bool = False
    mean_kwargs: Optional[dict] = None
    mean_pace_: Optional[object] = field(default=None, init=False)

    grid_: Optional[np.ndarray] = field(default=None, init=False)
    mean_values_: Optional[np.ndarray] = field(default=None, init=False)
    visit_levels_: Optional[np.ndarray] = field(default=None, init=False)
    visit_effects_: Optional[np.ndarray] = field(default=None, init=False)
    noise_variance_: Optional[float] = field(default=None, init=False)
    between_eigenvalues_: Optional[np.ndarray] = field(default=None, init=False)
    between_eigenfunctions_: Optional[np.ndarray] = field(default=None, init=False)
    within_eigenvalues_: Optional[np.ndarray] = field(default=None, init=False)
    within_eigenfunctions_: Optional[np.ndarray] = field(default=None, init=False)
    total_covariance_: Optional[np.ndarray] = field(default=None, init=False)
    between_covariance_: Optional[np.ndarray] = field(default=None, init=False)
    within_covariance_: Optional[np.ndarray] = field(default=None, init=False)
    total_penalty_: Optional[float] = field(default=None, init=False)
    between_penalty_: Optional[float] = field(default=None, init=False)

    def fit(
        self,
        curves: Sequence,
        subject_ids: np.ndarray,
        visit_ids: Optional[np.ndarray] = None,
        noise_variance: Optional[float] = None,
    ) -> "TwoLevelPACE":
        """Fit the two-level decomposition from raw curves.

        Parameters
        ----------
        curves : sequence of (times, values)
            One ``(times, values)`` pair per curve (one subject at one visit). ``times``
            and ``values`` are one-dimensional arrays of equal length; non-finite values
            are ignored. Curves may be observed at different times.
        subject_ids : array-like, shape (n_curves,)
            Subject label of each curve.
        visit_ids : array-like, shape (n_curves,), optional
            Visit number of each curve, aligned with ``curves``. Required when
            ``visit_effects=True`` and ignored otherwise. Visit numbers need not be
            consecutive; ``visit_levels_`` holds the distinct values.
        noise_variance : float, optional
            If given, used as sigma^2 in place of the estimate of Di et al. (2009,
            Sec. 2.3, step 4).

        Returns
        -------
        self : TwoLevelPACE
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``subject_ids`` does not have one entry per curve; if ``visit_effects`` is
            True and ``visit_ids`` is None; if no subject has at least two curves, so that
            no between-subject surface can be estimated; or if a covariance surface has no
            positive eigenvalue.

        Warns
        -----
        UserWarning
            If a curve has duplicate observation times; their values are averaged.

        Notes
        -----
        The between-subject surface is pooled over all pairs of distinct curves of the
        same subject. The diagonal of the total covariance surface is excluded from its
        P-spline fit, since it carries measurement error.
        """
        subject_ids = np.asarray(subject_ids)
        n_curves = len(curves)
        if len(subject_ids) != n_curves:
            raise ValueError("subject_ids must have one entry per curve")
        if visit_ids is not None:
            visit_ids = np.asarray(visit_ids)
            if len(visit_ids) != n_curves:
                raise ValueError("visit_ids must have one entry per curve")
            _check_one_visit_per_subject(subject_ids, visit_ids)

        # The grid holds only times whose values are finite; non-finite observations
        # are dropped and counted.
        finite_times = []
        clean_curves = []
        n_nonfinite = 0
        for t, v in curves:
            t = np.asarray(t, dtype=float)
            v = np.asarray(v, dtype=float)
            ok = np.isfinite(t) & np.isfinite(v)
            n_nonfinite += int(np.sum(~ok))
            finite_times.append(t[ok])
            clean_curves.append((t[ok], v[ok]))
        if n_nonfinite:
            warnings.warn(
                f"{n_nonfinite} observation(s) with a non-finite time or value were dropped.",
                UserWarning,
                stacklevel=2,
            )
        self.grid_ = np.unique(np.concatenate(finite_times))
        if self.grid_.size == 0:
            raise ValueError("No finite observations were supplied.")
        Y = _curves_to_matrix(curves, self.grid_)
        mask = ~np.isnan(Y)
        L = len(self.grid_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_values_ = self._estimate_mean(clean_curves)

            if self.visit_effects:
                if visit_ids is None:
                    raise ValueError("visit_effects=True requires visit_ids (visit number per curve)")
                self.visit_levels_ = np.unique(np.asarray(visit_ids))
                visit_idx = _visit_labels_to_index(visit_ids, self.visit_levels_)
                J = len(self.visit_levels_)
            else:
                self.visit_levels_ = None
                visit_idx = np.zeros(n_curves, dtype=int)
                J = 1
            self.visit_effects_ = np.zeros((J, L))
            if self.visit_effects:
                for j in range(J):
                    self.visit_effects_[j] = np.nan_to_num(
                        np.nanmean(Y[visit_idx == j], axis=0) - self.mean_values_, nan=0.0
                    )

        y_centered = np.where(mask, Y - self.mean_values_ - self.visit_effects_[visit_idx], 0.0)
        total_cov_raw, total_count = _raw_covariance_total(y_centered, mask)
        between_cov_raw, between_count = _raw_covariance_between(y_centered, mask, subject_ids)

        # Paper Sec. 2.3 step 3: diagonal excluded from the total surface's fit.
        total_weights = total_count.copy()
        np.fill_diagonal(total_weights, 0.0)
        self.total_covariance_, self.total_penalty_ = _smooth_surface(
            total_cov_raw, total_weights, self.grid_, n_segments=self.n_segments
        )
        self.between_covariance_, self.between_penalty_ = _smooth_surface(
            between_cov_raw, between_count, self.grid_, n_segments=self.n_segments
        )
        self.within_covariance_ = self.total_covariance_ - self.between_covariance_

        # Paper Sec. 2.3 step 4: sigma^2 = integral of {G_T(t,t) - K_T(t,t)} dt, per unit length.
        if noise_variance is not None:
            self.noise_variance_ = float(noise_variance)
        else:
            gap = np.diag(total_cov_raw) - np.diag(self.total_covariance_)
            span = self.grid_[-1] - self.grid_[0]
            self.noise_variance_ = max(_trapz(gap, self.grid_) / span, 0.0)

        # Paper Sec. 2.2 steps 2-3: eigenanalysis of K_B (level 1) and K_W (level 2).
        lam_B, phi_B = self._pace_eigendecompose(self.between_covariance_, self.grid_)
        lam_W, phi_W = self._pace_eigendecompose(self.within_covariance_, self.grid_)
        self.between_eigenvalues_, self.between_eigenfunctions_ = self._truncate(lam_B, phi_B)
        self.within_eigenvalues_, self.within_eigenfunctions_ = self._truncate(lam_W, phi_W)
        return self

    def _estimate_mean(self, curves: Sequence) -> np.ndarray:
        """Pooled mean function on ``grid_``, estimated as ``PACE`` estimates its mean.

        Validates the curves and runs the mean step of a ``PACE`` instance configured
        with ``mean_kwargs``. The covariance and noise steps are not run. The result is
        interpolated from the PACE evaluation grid onto ``grid_``.
        """
        kwargs = {"method_smoothing": "PS", "smooth_covariance": False}
        kwargs.update(self.mean_kwargs or {})
        pace = PACE(variance_threshold=1.0, **kwargs)
        with warnings.catch_warnings():
            # PACE warns that single-observation curves add no within-curve pairs to
            # its covariance surface. Here only the mean is taken from PACE, and such
            # curves still enter the between-subject surface through pairs with the
            # subject's other curves, so the warning does not apply.
            warnings.filterwarnings("ignore", message=r".*have a single observation", category=UserWarning)
            validated = pace._validate_and_prepare(curves, data_type="auto")
        pace._fit_mean(validated)
        self.mean_pace_ = pace
        return np.interp(self.grid_, pace.grid_, pace.mean_values_)

    def plot_mean(self, ax=None, **kwargs):
        """Plot the estimated mean function mu(t).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if None.
        **kwargs
            Passed to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.grid_ is None:
            raise RuntimeError("TwoLevelPACE is not fitted yet.")
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.grid_, self.mean_values_, **kwargs)
        ax.set_xlabel("t")
        ax.set_ylabel("mu(t)")
        ax.set_title("Mean function")
        ax.grid(True, alpha=0.3)
        return ax

    def _variance_split(self):
        """Percentages of the retained variance carried by each level."""
        lam_b, lam_w = self.between_eigenvalues_, self.within_eigenvalues_
        grand = lam_b.sum() + lam_w.sum()
        return 100 * lam_b.sum() / grand, 100 * lam_w.sum() / grand

    def plot_eigenfunctions(self, n_between=3, n_within=3, figsize=None):
        """Plot between- and within-subject eigenfunctions, one row per level.

        The top row (red) shows the first ``n_between`` between-subject eigenfunctions
        and the bottom row (blue) the first ``n_within`` within-subject ones. Each panel
        title gives the component's share of the grand total variance (between plus
        within). The figure title gives the split of retained variance between the two
        levels.

        Parameters
        ----------
        n_between : int, default 3
            Number of leading between-subject eigenfunctions to show. Clamped to the
            number retained.
        n_within : int, default 3
            Number of leading within-subject eigenfunctions to show. Clamped to the
            number retained.
        figsize : tuple, optional
            Figure size passed to ``plt.subplots``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes, shape (2, max(n_between, n_within))
            Unused panels are hidden, not removed.
        """
        import matplotlib.pyplot as plt

        if self.grid_ is None:
            raise RuntimeError("TwoLevelPACE is not fitted yet.")
        n_b = min(int(n_between), len(self.between_eigenvalues_))
        n_w = min(int(n_within), len(self.within_eigenvalues_))
        ncols = max(n_b, n_w, 1)
        grand = self.between_eigenvalues_.sum() + self.within_eigenvalues_.sum()
        pct_b, pct_w = self._variance_split()
        if figsize is None:
            figsize = (3.6 * ncols, 6.5)

        fig, axes = plt.subplots(2, ncols, figsize=figsize, sharex=True, squeeze=False)
        rows = [
            ("Between-subject", self.between_eigenvalues_, self.between_eigenfunctions_, n_b, "tab:red"),
            ("Within-subject", self.within_eigenvalues_, self.within_eigenfunctions_, n_w, "tab:blue"),
        ]
        for r, (label, lam, phi, n, color) in enumerate(rows):
            for k in range(ncols):
                ax = axes[r, k]
                if k >= n:
                    ax.set_visible(False)
                    continue
                ax.axhline(0, color="0.35", lw=0.9, zorder=1)
                ax.plot(self.grid_, phi[k], color=color, lw=2)
                ax.set_title(f"PC{k + 1} ({100 * lam[k] / grand:.1f}% of grand total)", fontsize=9)
                ax.grid(True, alpha=0.2)
                if k == 0:
                    ax.set_ylabel(label)
        for ax in axes[1]:
            if ax.get_visible():
                ax.set_xlabel("t")
        fig.suptitle(
            "Eigenfunctions: between-subject vs within-subject\n"
            f"(variance split: {pct_b:.1f}% between / {pct_w:.1f}% within, of retained variance)",
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, axes

    def plot_eigenvalues(self, figsize=(8, 4)):
        """Bar plot of the retained eigenvalues, between and within level side by side.

        Parameters
        ----------
        figsize : tuple, default (8, 4)
            Figure size passed to ``plt.subplots``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.grid_ is None:
            raise RuntimeError("TwoLevelPACE is not fitted yet.")
        lam_b, lam_w = self.between_eigenvalues_, self.within_eigenvalues_
        pct_b, pct_w = self._variance_split()
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(np.arange(1, len(lam_b) + 1) - 0.2, lam_b, width=0.4, color="tab:red",
               label=f"between ({pct_b:.1f}%)")
        ax.bar(np.arange(1, len(lam_w) + 1) + 0.2, lam_w, width=0.4, color="tab:blue",
               label=f"within ({pct_w:.1f}%)")
        ax.set_xticks(np.arange(1, max(len(lam_b), len(lam_w)) + 1))
        ax.set_xlabel("component")
        ax.set_ylabel("eigenvalue")
        ax.set_title("Eigenvalues by level (share of retained variance)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _pace_eigendecompose(surface: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All positive eigenpairs of a covariance surface (Yao, Muller and Wang 2005).

        Uses the eigen step of ``PACE._eigendecompose`` (quadrature-weighted symmetric
        eigenproblem), with no truncation by variance threshold.

        Returns
        -------
        eigenvalues : ndarray, shape (K,)
            Positive eigenvalues in descending order.
        eigenfunctions : ndarray, shape (K, L)
            L2-orthonormal eigenfunction values on `grid`.
        """
        pace = PACE(variance_threshold=1.0)
        pace.grid_ = np.asarray(grid, dtype=float)
        pace._eigendecompose(surface)
        return pace._all_eigenvalues_.copy(), pace._all_eigenfunctions_.copy()

    def _truncate(self, eigenvalues: np.ndarray, eigenfunctions: np.ndarray):
        """Select the retained components of one level (Di et al. 2009, Sec. 2.2).

        With a float ``variance_threshold``, the count comes from ``_select_n_components``;
        with an int, it is that fixed number. Components with eigenvalues at numerical
        zero (relative to the largest) are then dropped.

        Returns
        -------
        (eigenvalues, eigenfunctions) : tuple of ndarrays
            Retained eigenvalues (N,) and eigenfunction values (N, L).
        """
        if isinstance(self.variance_threshold, (int, np.integer)):
            if self.variance_threshold < 1:
                raise ValueError("An integer variance_threshold must be at least 1.")
            npc = min(int(self.variance_threshold), len(eigenvalues))
        else:
            positive = eigenvalues[eigenvalues > 0]
            npc = _select_n_components(positive, p1=float(self.variance_threshold), p2=1.0 / len(self.grid_))
        lam, phi = eigenvalues[:npc], eigenfunctions[:npc]
        if lam.size == 0:
            raise ValueError(
                f"No positive eigenvalue is retained at variance_threshold={self.variance_threshold!r}; "
                "the covariance surface has no positive variance to decompose."
            )
        keep = lam > 1e-10 * lam[0]
        return lam[keep], phi[keep]

    def _curve_design(self, times, values, visit_index: int):
        """Eigenfunction design and centered residuals for one curve, at its own times.

        Mean, visit effect and eigenfunctions are linearly interpolated from the fitted
        grid (as in ``PACE``). Non-finite values are dropped. Returns the between-subject
        design (n_obs, N1), the within-subject design (n_obs, N2) and the residuals
        Y(t) - mu(t) - eta_j(t) (n_obs,), which are the inputs of the score equations
        (Di et al. 2009, Sec. 3.1).
        """
        t = np.asarray(times, dtype=float)
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        grid = self.grid_
        mu = np.interp(t, grid, self.mean_values_)
        eta = np.interp(t, grid, self.visit_effects_[visit_index])
        B1 = np.stack([np.interp(t, grid, f) for f in self.between_eigenfunctions_], axis=1)
        B2 = np.stack([np.interp(t, grid, f) for f in self.within_eigenfunctions_], axis=1)
        return B1, B2, v - mu - eta

    def _row_visit_index(self, subject_ids: np.ndarray, visit_ids: Optional[np.ndarray], n: int) -> np.ndarray:
        """Indices into ``visit_levels_`` for n curves; all zeros when visit_effects is False.

        Raises ValueError if the input lengths do not match n curves, if a subject has
        two curves at one visit, if visit_effects is True and visit_ids is None, or if a
        visit number was not seen in ``fit``.
        """
        if len(subject_ids) != n:
            raise ValueError("subject_ids must have one entry per curve")
        if visit_ids is None:
            if self.visit_effects:
                raise ValueError("visit_effects=True requires visit_ids (visit number per curve)")
            return np.zeros(n, dtype=int)
        visit_ids = np.asarray(visit_ids)
        if len(visit_ids) != n:
            raise ValueError("visit_ids must have one entry per curve")
        _check_one_visit_per_subject(subject_ids, visit_ids)
        if not self.visit_effects:
            return np.zeros(n, dtype=int)
        return _visit_labels_to_index(visit_ids, self.visit_levels_)

    def transform(self, curves: Sequence, subject_ids: np.ndarray, visit_ids: Optional[np.ndarray] = None) -> dict:
        """Gaussian posterior-mean (BLUP) scores of the full model (Di et al. 2009, Sec. 3.1).

        For subject i with curves r = 1, ..., n_i, let U_i stack the eigenfunction designs
        of the subject's curves (between-subject block first) and y_i stack their residuals
        from ``_curve_design``. The scores are

            theta_i = (U_i' U_i + sigma^2 D_i^-1)^-1 U_i' y_i,

        where theta_i = (xi_i, zeta_i1, ..., zeta_in_i), with xi_i the between-subject
        coefficients and zeta_ir the within-subject coefficients of curve r, and
        D_i = blockdiag(Lambda_B, Lambda_W, ..., Lambda_W) holds the retained eigenvalues.

        Parameters
        ----------
        curves : sequence of (times, values)
            Curves to score, one ``(times, values)`` pair each. They need not be the curves
            used in ``fit``, and their observation times may differ from the fitted grid
            (values outside the grid range are held at the boundary values).
        subject_ids : array-like, shape (n_curves,)
            Subject label of each curve. All curves of one subject should be passed in the
            same call, since the between-subject score uses all of them jointly.
        visit_ids : array-like, shape (n_curves,), optional
            Visit number of each curve, aligned with ``curves``. Required when
            ``visit_effects=True``; every value must be one of ``visit_levels_``.

        Returns
        -------
        dict
            With keys:

            ``'between'`` : ndarray, shape (n_subjects, N1)
                Between-subject scores xi_i, one row per subject in ``'subject_order'``.
            ``'within'`` : ndarray, shape (n_curves, N2)
                Within-subject scores zeta_ir; row r belongs to ``curves[r]``.
            ``'subject_order'`` : ndarray
                Distinct subject labels in order of first appearance in ``subject_ids``.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If ``visit_effects`` is True and ``visit_ids`` is None, or a visit number was
            not seen in ``fit``.
        """
        if self.grid_ is None:
            raise RuntimeError("TwoLevelPACE is not fitted yet.")
        subject_ids = np.asarray(subject_ids)
        n = len(curves)
        visit_idx = self._row_visit_index(subject_ids, visit_ids, n)
        N1, N2 = len(self.between_eigenvalues_), len(self.within_eigenvalues_)
        sigma2 = self.noise_variance_
        lam_B, lam_W = self.between_eigenvalues_, self.within_eigenvalues_

        subj_order = list(dict.fromkeys(subject_ids.tolist()))
        between = np.zeros((len(subj_order), N1))
        within = np.full((n, N2), np.nan)
        for si, s in enumerate(subj_order):
            rows = np.where(subject_ids == s)[0]
            n_i = len(rows)
            K = N1 + n_i * N2
            UtU = np.zeros((K, K))
            Uty = np.zeros(K)
            for q, r in enumerate(rows):
                B1, B2, y = self._curve_design(*curves[r], visit_idx[r])
                lo, hi = N1 + q * N2, N1 + (q + 1) * N2
                UtU[:N1, :N1] += B1.T @ B1
                Uty[:N1] += B1.T @ y
                cross = B1.T @ B2
                UtU[:N1, lo:hi] = cross
                UtU[lo:hi, :N1] = cross.T
                UtU[lo:hi, lo:hi] = B2.T @ B2
                Uty[lo:hi] = B2.T @ y
            d_diag = np.concatenate([lam_B, np.tile(lam_W, n_i)])
            theta = np.linalg.solve(UtU + sigma2 * np.diag(1.0 / d_diag), Uty)
            between[si] = theta[:N1]
            within[rows] = theta[N1:].reshape(n_i, N2)
        return {"between": between, "within": within, "subject_order": np.array(subj_order)}

    def reconstruct(self, scores: dict, subject_ids: np.ndarray, visit_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct curves on the fitted grid from their scores.

        For each curve r of subject i and visit j, the fitted curve is

            Y_hat_ir(t) = mu(t) + eta_j(t) + sum_k xi_ik phi_k(t) + sum_k zeta_irk psi_k(t),

        where phi_k and psi_k are the retained between-subject and within-subject
        eigenfunctions. The term eta_j(t) is zero when ``visit_effects`` is False.

        Parameters
        ----------
        scores : dict
            Output of ``transform``.
        subject_ids : array-like, shape (n_curves,)
            Subject label of each row of ``scores['within']``. Each label must appear in
            ``scores['subject_order']``.
        visit_ids : array-like, shape (n_curves,), optional
            Visit number of each row, aligned with ``scores['within']``. Required when
            ``visit_effects=True``.

        Returns
        -------
        ndarray, shape (n_curves, L)
            Reconstructed curves evaluated on ``grid_``, one row per row of
            ``scores['within']``.

        Raises
        ------
        ValueError
            If ``visit_effects`` is True and ``visit_ids`` is None, or a visit number was
            not seen in ``fit``.
        KeyError
            If a subject label is not in ``scores['subject_order']``.
        """
        subject_ids = np.asarray(subject_ids)
        n = scores["within"].shape[0]
        visit_idx = self._row_visit_index(subject_ids, visit_ids, n)
        subj_index = {s: i for i, s in enumerate(scores["subject_order"].tolist())}
        out = np.empty((n, len(self.grid_)))
        for r in range(n):
            si = subj_index[subject_ids[r]]
            out[r] = (
                self.mean_values_
                + self.visit_effects_[visit_idx[r]]
                + scores["between"][si] @ self.between_eigenfunctions_
                + scores["within"][r] @ self.within_eigenfunctions_
            )
        return out

    def level_model(self, level: str) -> "LevelModel":
        """One level of the fitted decomposition, in the form ``MFPCA`` reads.

        ``MFPCA`` combines univariate models through their mean function,
        eigenvalues and eigenfunctions. This method returns those quantities for
        one level, so that a multivariate analysis can be run separately at each
        level with ``MFPCA.fit_from_scores``, using the level's columns of
        ``transform``'s output as scores.

        Parameters
        ----------
        level : {"between", "within"}
            ``"between"`` (level 1): eigenfunctions phi_k with mean mu(t).
            ``"within"`` (level 2): eigenfunctions psi_k with a zero mean, since
            level-2 curves are deviations from the subject's own trajectory.

        Returns
        -------
        LevelModel
            A copy of the level's quantities; later changes to this estimator do
            not affect it.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If ``level`` is not ``"between"`` or ``"within"``.
        """
        if self.grid_ is None:
            raise RuntimeError("TwoLevelPACE is not fitted yet.")
        if level == "between":
            lam, phi, mean = self.between_eigenvalues_, self.between_eigenfunctions_, self.mean_values_
        elif level == "within":
            lam, phi, mean = self.within_eigenvalues_, self.within_eigenfunctions_, np.zeros_like(self.grid_)
        else:
            raise ValueError(f"level must be 'between' or 'within', got {level!r}.")
        return LevelModel(level=level, grid_=self.grid_.copy(), mean_values_=mean.copy(),
                          eigenvalues_=lam.copy(), eigenfunction_values_=phi.copy())


@dataclass
class LevelModel:
    """One level of a fitted ``TwoLevelPACE``, with the attributes of a fitted ``PACE``.

    Returned by ``TwoLevelPACE.level_model``. It carries the level's mean function,
    eigenvalues and eigenfunctions under the names ``MFPCA`` reads
    (``grid_``, ``mean_values_``, ``eigenvalues_``, ``eigenfunction_values_``,
    ``n_components_used_``), and reconstructs curves from level scores with
    ``reconstruct_curve``. It cannot score raw curves: level scores come from
    ``TwoLevelPACE.transform``, so combine levels with ``MFPCA.fit_from_scores``.
    """

    level: str
    grid_: np.ndarray
    mean_values_: np.ndarray
    eigenvalues_: np.ndarray
    eigenfunction_values_: np.ndarray

    @property
    def n_components_used_(self) -> int:
        return len(self.eigenvalues_)

    @property
    def n_components_available_(self) -> int:
        return len(self.eigenvalues_)

    def transform(self, data):
        raise NotImplementedError(
            "A LevelModel cannot score raw curves. Use TwoLevelPACE.transform for the "
            "level scores and MFPCA.fit_from_scores / transform_from_scores."
        )

    def reconstruct_curve(self, times: np.ndarray, scores: np.ndarray, n_components: Optional[int] = None):
        """Mean plus ``scores`` times the level's eigenfunctions.

        Same signature and return value as ``PACE.reconstruct_curve``: the values
        on the sorted union of ``grid_`` and ``times``, and that grid. Values
        outside ``grid_`` are held at the boundary values.
        """
        n = len(self.eigenvalues_) if n_components is None else min(n_components, len(self.eigenvalues_))
        scores = np.asarray(scores, dtype=float)
        n = min(n, len(scores))
        t_eval = np.unique(np.concatenate([self.grid_, np.asarray(times, dtype=float)]))
        phi = np.stack([np.interp(t_eval, self.grid_, f) for f in self.eigenfunction_values_[:n]]) \
            if n else np.zeros((0, len(t_eval)))
        values = np.interp(t_eval, self.grid_, self.mean_values_) + scores[:n] @ phi
        return values, t_eval
