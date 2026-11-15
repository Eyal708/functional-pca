"""
PACE -- Principal Analysis by Conditional Expectation (PACE)
==============================================================

A general, self-contained implementation of the PACE method of

    Yao, F., Mueller, H.-G. and Wang, J.-L. (2005). "Functional Data
    Analysis for Sparse Longitudinal Data." Journal of the American
    Statistical Association, 100(470), 577-590.

PACE estimates a population mean function and covariance surface from
a collection of curves that may be observed densely, sparsely, or at
different (irregular) sets of points per curve, possibly contaminated
with independent measurement noise. It then diagonalizes the smoothed
covariance surface to obtain functional principal components
(eigenfunctions/eigenvalues) and computes, for every curve, "PACE
scores" -- best linear unbiased predictors of that curve's functional
principal component scores obtained via the conditional-expectation
formula (Yao et al. 2005, eq. 5), which remains well defined even when
a curve is observed at only a handful of points.

This module has no notion of what a "curve" represents: it can be a
subject followed over time, a repeated measurement observed at
irregular time points, or any other one-dimensional functional object
observed (possibly noisily, possibly sparsely) at a set of "argvals"
(observation locations) with corresponding "values". Throughout this
module and its docstrings, "curve" / "subject" refers to one such unit
and "argvals" / "times" refers to its observation locations, on a
shared one-dimensional domain across all curves.

Component-count model (read this before using ``n_components``)
------------------------------------------------------------------
Two different, independently controlled quantities determine "how many
eigencomponents" this class deals with. This mirrors PACE's own
description (and every practical implementation of FPCA) but is spelled
out here explicitly because it is a common source of confusion:

1. ``variance_threshold`` (constructor argument, default 0.99) controls
   how many eigencomponents are *computed and retained internally* when
   the covariance surface is diagonalized: components are added, in
   decreasing order of eigenvalue, until the requested fraction of
   total estimated variance is reached. This full retained set is
   always exposed after fitting as ``self.eigenvalues_`` and
   ``self.eigenfunction_values_`` -- these two arrays are **never**
   truncated to match anything the user asked for; their length is
   ``self.n_components_available_``. Reaching *exactly* 100% of the
   variance is generally impossible for any finite discretization of a
   continuous covariance surface (it would require as many components
   as there are grid points), which is why this is a genuine,
   separately-named "retain until 99% [or whatever threshold]"
   parameter rather than a request for "all of them".

2. ``n_components`` (constructor argument, default ``None``) is the
   user-facing "how many components do I want to *use*, by default,
   for scores / reconstruction / plotting" request. ``None`` means "use
   everything that is available" (i.e. ``n_components_available_``).
   This attribute is **never mutated in place** after fitting -- it
   always reflects exactly what was passed to the constructor. The
   number actually used by default after fitting is stored separately,
   in ``self.n_components_used_``.

   Every method that produces components-dependent output
   (``transform``, ``reconstruct_curve``, ``plot_eigenfunctions``) also
   accepts its own, optional, per-call ``n_components`` argument that
   overrides the model's default for that call only. If more components
   are requested (at construction or per-call) than are actually
   available, this is **not an error**: the request is silently
   clamped to what is available, but a ``UserWarning`` is always raised
   stating both the requested and the available count, so the caller
   is never left guessing why they got fewer components than asked.

Implementation approach
------------------------------------------------------------------
This module works directly with low-level, general-purpose smoothing
primitives -- the ``LocalPolynomial`` and ``PSplines`` smoothers of FDApy
(Golovkine), included in :mod:`fpca._smoothing` under FDApy's MIT License --
rather than a higher-level functional-data API.
Raw mean/covariance pooling, diagonal exclusion, sample weighting, the
trapezoidal-quadrature eigendecomposition, and component-count
bookkeeping are all implemented directly in this module, which keeps
each estimation step fully explicit and independently verifiable:

- The diagonal of a raw covariance estimate is inflated by measurement
  noise rather than reflecting the smooth signal covariance (Yao et
  al. 2005, eq. 2), and must therefore be excluded from a covariance
  smoother's fit. For the P-spline path, this means excluding it from
  both the *values* being fit and the *fitting weights* attached to
  those values -- omitting either one alone would bias the fitted
  surface near the diagonal, since a zeroed-out value with a nonzero
  weight is fit as if it were a trustworthy zero observation.
  ``_smooth_covariance_ps`` builds its weight matrix from the same
  diagonal-exclusion mask used for the data, so the two always stay in
  sync. The local-polynomial ('LP') path achieves the same exclusion
  by simply omitting diagonal points from the regression entirely.
- The mean and covariance surfaces are always smoothed with
  independent configuration: ``mean_bandwidth``/``mean_penalty`` and
  ``covariance_bandwidth``/``covariance_penalty`` are never shared or
  conflated, and a P-spline covariance fit always constructs an
  explicit, correctly-shaped penalty for each of its two axes.
- Eigendecomposition uses ``numpy.linalg.eigh`` (the symmetric-matrix
  solver) followed by an explicit descending sort of eigenvalues and
  their corresponding eigenvectors, so cumulative-variance-based
  component selection never silently depends on any particular
  ordering convention of the underlying linear-algebra routine.
- Pooling per-curve observations onto the shared evaluation grid
  (``_pooled_raw_covariance``) accumulates each curve's contribution
  with a single vectorized ``np.add.at`` call against a pre-computed
  grid index, so the per-curve cost of pooling does not grow with the
  grid's own resolution.

Dense vs. irregular data
------------------------------------------------------------------
There is exactly one estimation pipeline in this module -- "dense"
data (every curve observed at the same set of argvals) is simply the
special case where the pooled raw mean/covariance happens to have full
support at every grid point. Smoothing parameters (bandwidth/penalty)
therefore always apply, on both dense and irregular input, unless
explicitly disabled via ``smooth_mean=False`` / ``smooth_covariance=
False``. This differs from a common convention of silently skipping
all smoothing purely because the input happens to share one common
grid across curves; that behavior can silently hide genuine noise in a
"dense" signal, so it is not adopted here. If you have exactly-
observed, effectively noiseless dense curves and want the raw
empirical mean/covariance with no smoothing bias at all (e.g. for a
closed-form numerical check), set ``smooth_mean=False,
smooth_covariance=False`` explicitly.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import eigh, solve

from ._smoothing import LocalPolynomial, PSplines

__all__ = ["PACE"]

CurveData = List[Tuple[np.ndarray, np.ndarray]]

# Default grid cap for method_smoothing='LP' (whose covariance smoothing costs
# roughly O(len(grid)^4)), and the size above which the default union-of-times
# grid triggers a memory/run-time warning.
_LP_DEFAULT_GRID_POINTS = 50
_LARGE_GRID_WARNING = 2000


class PACE:
    """
    Principal Analysis by Conditional Expectation (PACE).

    Fits a population mean function and covariance surface to a
    collection of (possibly sparse, possibly irregularly-observed,
    possibly noisy) curves, diagonalizes the covariance surface to
    obtain functional principal components, and estimates, for each
    curve, functional principal component scores via the PACE
    conditional-expectation formula. See the module docstring for the
    full method description, the component-count semantics
    (``n_components`` vs. ``variance_threshold`` vs.
    ``n_components_available_``), and this implementation's approach.

    Parameters
    ----------
    n_components : int, optional
        Default number of components to use for scores, reconstruction
        and plotting. ``None`` (default) means "use every component
        retained by ``variance_threshold``". Never mutated after
        construction; see ``n_components_used_`` for what was actually
        used after fitting, and the per-call ``n_components`` argument
        of ``transform``/``reconstruct_curve``/``plot_eigenfunctions``
        to override it for a single call.

    variance_threshold : float, default=0.99
        Fraction of total estimated variance (of the covariance
        surface's eigenvalues, after clipping negative numerical noise
        to zero) to retain when selecting how many eigencomponents to
        compute. Must be in (0, 1]. Because a continuous covariance
        surface admits infinitely many eigencomponents (and a finite
        discretization admits as many as there are grid points),
        requesting exactly 1.0 will generally retain the full
        discretized rank, not an idealized "100% of the true process
        variance" -- this parameter is deliberately a stopping
        criterion, not a promise of exhaustiveness.

    method_smoothing : {'LP', 'PS'}, default='PS'
        Smoothing method used for the mean and (if ``smooth_covariance``
        is True) the covariance surface:

        - ``'PS'`` (P-splines, Eilers & Marx; the default): global,
          roughness-penalized regression splines, fit jointly over the
          whole domain rather than locally around each evaluation
          point. Uses ``mean_penalty`` and ``covariance_penalty``
          (both default to 1.0; see below for tuning advice). The diagonal of
          the raw covariance is excluded from the P-spline fit by
          zeroing both its value *and* its fitting weight together, so
          the smoother is never told a fabricated zero is a trusted
          observation (see the module docstring). Its computational
          cost is governed by the spline basis size
          (``ps_n_segments``/``ps_mean_n_segments``) rather than by the
          evaluation-grid resolution, so it scales well to a fine grid
          or a large number of curves (see "Computational cost"
          below).
        - ``'LP'`` (local polynomial): local-linear smoothing (degree
          1) for the mean and, following Yao et al. (2005, Sec. 2.2),
          local-quadratic smoothing (degree 2) for the covariance
          surface, both with an Epanechnikov kernel, evaluated with the
          diagonal of the raw covariance excluded from the fit
          (measurement noise inflates only the diagonal; see Yao et al.
          2005, eq. 2 and surrounding discussion). Covariance-surface
          smoothing cost under ``'LP'`` scales sharply with evaluation-
          grid resolution (see ``n_grid_points`` and "Computational
          cost" below), which is why ``'LP'`` defaults to a much
          coarser grid than ``'PS'``.

    mean_bandwidth, covariance_bandwidth : float, optional
        Bandwidths (in the curves' own argvals units, i.e. NOT assumed
        normalized to [0, 1]) for the 'LP' mean and covariance
        smoothers respectively. If ``None`` (default), a domain-scaled
        default is used: ``mean_bandwidth = mean_bandwidth_fraction *
        R`` and ``covariance_bandwidth = covariance_bandwidth_fraction
        * R``, where ``R`` is the observed range (max - min) of all
        pooled argvals across all curves. Ignored when
        ``method_smoothing='PS'`` or the corresponding
        ``smooth_mean``/``smooth_covariance`` flag is False.

    mean_bandwidth_fraction : float, default=0.05
    covariance_bandwidth_fraction : float, default=0.10
        Domain-range fractions used to build the default
        ``mean_bandwidth``/``covariance_bandwidth`` when not supplied
        explicitly. These specific constants (5% / 10% of the observed
        range) are adopted directly from ``fdapace``, the reference
        implementation shipped by Yao/Mueller/Wang's own research group
        (github.com/functionaldata/tPACE, `R/SetOptions.R`): its own
        shipped, non-cross-validated *default* bandwidth is exactly
        ``0.05 * diff(range(t))`` for the mean and
        ``0.10 * diff(range(t))`` for the covariance surface (the 2x
        ratio reflecting that covariance-surface smoothing needs a
        wider neighbourhood than the simpler 1-D mean). The original
        Yao et al. (2005) paper itself specifies no domain-scaling
        rule -- it recommends one-curve-leave-out cross-validation for
        the covariance bandwidth and says "subjective choices are often
        adequate" for the mean -- so these fractions are best
        understood as a well-tested, field-proven *starting point*, not
        a substitute for cross-validation on any specific dataset.

    noise_bandwidth : float, optional
    noise_bandwidth_fraction : float, default=0.3
        Bandwidth (or domain-range fraction, if ``noise_bandwidth`` is
        None) for the local-linear smoother used to estimate the
        measurement-noise variance from the gap between the raw and
        smoothed covariance-surface diagonal (Yao et al. 2005, eq. 2).
        0.3 is a wider fraction than the mean/covariance defaults
        because this step smooths a much noisier 1-D signal (squared
        residuals) and only needs to recover a slowly-varying diagonal
        trend, not fine local structure; a bandwidth that is too
        narrow here under-smooths the noisy residual signal and biases
        the resulting noise-variance estimate. `fdapace` does not
        expose an equivalent domain-scaled default for this step, so
        this fraction is calibrated directly for this purpose rather
        than adopted from an external reference.

    mean_penalty, covariance_penalty : float, optional
        P-spline roughness penalties for the mean (1-D) and covariance
        surface (2-D; applied identically along both axes) when
        ``method_smoothing='PS'``. Both default to 1.0. The right
        magnitude depends on the domain's scale and the data's noise
        level, so a penalty that suits one dataset can be too weak or
        too strong for another (on one small synthetic benchmark with
        domain range ~18, a penalty of 1 retained more components than
        the data contained, while a penalty around 500 recovered the
        true number). Tune the penalties for each new dataset by
        inspecting the fitted mean and eigenfunctions. Setting one to
        None raises a ``ValueError`` when the matching ``smooth_mean``
        or ``smooth_covariance`` flag is True.

    ps_n_segments, ps_degree, ps_order_penalty : int, default=30, 3, 2
        B-spline basis size, spline degree, and difference-penalty
        order used to fit the *covariance* surface under
        ``method_smoothing='PS'``. ``ps_n_segments`` sets how many
        basis segments span the domain along each axis (more segments
        allow finer local structure, at a computational cost that
        scales with basis size roughly independently of
        ``n_grid_points`` -- see "Computational cost" below);
        ``ps_degree`` and ``ps_order_penalty`` set the B-spline degree
        and the order of the finite-difference roughness penalty
        applied between adjacent basis coefficients (Eilers & Marx
        P-splines). ``ps_degree``/``ps_order_penalty`` are shared
        between the mean and covariance fits.

    ps_mean_n_segments : int, default=10
        B-spline basis size for the *mean* function under
        ``method_smoothing='PS'``. Deliberately smaller than
        ``ps_n_segments`` (30): the mean is a one-dimensional curve
        summarizing the pooled observations of every fitted curve,
        while the covariance surface is a two-dimensional object that
        typically needs a finer basis to represent local curvature
        well; sharing one basis size between the two would force a
        tradeoff between over-parameterizing the mean (risking
        overfitting what is usually a smooth population-level trend)
        and under-parameterizing the covariance surface.
        ``ps_degree``/``ps_order_penalty`` are shared between mean and
        covariance.

    n_grid_points : int or None, default=None
        Cap on the size of the evaluation grid (``self.grid_``) on which
        the mean, covariance surface and eigenfunctions are represented.

        - ``None`` (the default): the grid is the **union of all observed
          times** (the sorted distinct pooled argvals), so no observation
          is binned. A ``UserWarning`` is raised when this grid has more
          than 2000 points, since the covariance surface is a dense
          ``len(grid) x len(grid)`` matrix. Exception: with
          ``method_smoothing='LP'`` the default cap is 50 points, because
          LP covariance smoothing cost grows roughly as the fourth power
          of the grid size (see "Computational cost" below).
        - an int: the union of observed times is used if it has at most
          this many points; otherwise a uniform grid of ``n_grid_points``
          points spanning the observed range, and observations are binned
          to their nearest grid point for the mean (P-splines) and the raw
          covariance.

        The cap exists purely for computational scalability and does not
        materially change what is being *estimated*; it changes the
        resolution at which the mean, covariance and eigenfunctions are
        represented and, when it binds, the granularity at which raw
        observations are aggregated before smoothing.

        Times not on the grid (in ``transform`` and ``reconstruct_curve``)
        are handled by linear interpolation of the mean and eigenfunctions
        between neighbouring grid points. Times outside the fitted range
        raise a ``ValueError`` unless ``allow_extrapolation=True``.

    Computational cost (read this before fitting large datasets)
    ------------------------------------------------------------------
    Mean smoothing costs O(n_grid_points * n_total_observations) --
    cheap, and does not depend on ``n_grid_points`` being large.
    Covariance smoothing pools all curves' own within-curve observation
    pairs onto the (possibly binned) evaluation grid first -- an O(sum
    over curves of n_i^2) step, unavoidable and identical in spirit to
    any raw covariance computation over irregularly-observed curves --
    and then smooths that grid. For ``method_smoothing='PS'``, the
    P-spline basis size (``ps_n_segments``) determines cost, essentially
    independent of ``n_grid_points`` or the number of curves/points --
    P-splines therefore scale well to large problems. For
    ``method_smoothing='LP'``, the underlying local-polynomial smoother
    evaluates every query point against every training point with no
    spatial indexing -- cost scales roughly as O(n_grid_points^2 *
    n_populated_grid_cells), and when curves substantially overlap (as
    is typical when many curves are observed over a shared or
    near-shared range) the number of populated grid cells itself grows
    like O(n_grid_points^2), making LP covariance smoothing cost scale
    roughly like O(n_grid_points^4) overall. This is why the default
    ``n_grid_points=50`` for ``'LP'`` is deliberately modest: doubling
    the grid resolution under ``'LP'`` can easily increase covariance-
    fitting time by an order of magnitude or more. If covariance
    smoothing is a bottleneck, or a finer evaluation grid is needed
    than the ``'LP'`` default resolves cleanly, prefer
    ``method_smoothing='PS'`` (with an explicit ``covariance_penalty``)
    -- its cost is essentially independent of ``n_grid_points``,
    governed only by ``ps_n_segments`` -- rather than raising
    ``n_grid_points`` under ``'LP'``.

    smooth_mean, smooth_covariance : bool, default=True
        Whether to smooth the mean / covariance surface at all. Set to
        False to use the raw pooled empirical mean / covariance
        directly (e.g. for exact-recovery checks on dense, noiseless
        synthetic data, or if the input is already smooth). See the
        module docstring's "Dense vs. irregular data" section: unlike
        some other implementations, this class does not silently skip
        smoothing just because the input happens to share one common
        grid -- smoothing is applied (or not) uniformly, controlled
        only by these two flags.

    regularization : float, default=1e-6
        A small *relative* jitter added to the diagonal of the
        per-curve score-covariance matrix solved in the PACE
        conditional-expectation formula, purely for numerical
        stability (not a substitute for the measurement-noise variance
        term, which is always included separately and exactly). It is
        expressed as a fraction of a data-dependent scale (the
        estimated noise variance, or, if that is numerically
        negligible, the mean of the retained eigenvalues) computed once
        at fit time, rather than as an absolute constant -- so the same
        default behaves sensibly whether the fitted data has variance
        of order 1e-3 or 1e6.

    allow_extrapolation : bool, default=False
        If False (default), ``transform``/``reconstruct_curve`` raise a
        ``ValueError`` if asked to evaluate at an argval outside the
        range spanned by the fitted grid (rather than silently
        flat-extrapolating, which ``numpy.interp`` would otherwise do).
        Set to True to explicitly opt into flat extrapolation (a
        ``UserWarning`` is then raised instead, on every call that
        actually extrapolates, naming the offending value and the
        fitted domain).

    Attributes set after calling ``fit``
    -------------------------------------
    grid_ : ndarray
        Sorted evaluation grid on which the mean, eigenfunctions, and
        covariance surface are represented: the exact sorted, unique
        pooled argvals (union of every curve's own observation
        locations) by default, or a uniform grid when an
        ``n_grid_points`` cap binds (see the ``n_grid_points`` parameter).
    mean_values_ : ndarray
        Fitted mean function, evaluated on ``grid_``.
    eigenvalues_ : ndarray
        *All* eigenvalues retained by ``variance_threshold``, sorted
        descending. Length equals ``n_components_available_``. Never
        truncated to ``n_components``.
    eigenfunction_values_ : ndarray, shape (n_components_available_, len(grid_))
        Corresponding eigenfunctions, evaluated on ``grid_``, one row
        per component, same ordering as ``eigenvalues_``, sign
        arbitrary (as for any eigendecomposition).
    n_components_available_ : int
        Number of components retained by ``variance_threshold`` --
        i.e. ``len(eigenvalues_)``.
    n_components_used_ : int
        Number of components actually used, by default, for
        ``transform``/``reconstruct_curve``/``plot_eigenfunctions``
        calls that do not override ``n_components`` themselves. Equal
        to ``min(n_components, n_components_available_)`` if
        ``n_components`` was set at construction, else equal to
        ``n_components_available_``.
    noise_variance_ : float
        Estimated measurement-error variance.
    is_dense_ : bool
        Whether the fitted input was auto/user-detected as sharing one
        common grid across every curve (informational only -- does not
        change which estimation pipeline is used; see module
        docstring).
    raw_operator_eigenvalues_ : ndarray
        Full eigenvalue spectrum of the (weighted) covariance operator,
        as returned by the eigensolver and sorted descending, *before*
        clipping negative values to zero or dropping components below
        the numerical-tolerance cutoff. Diagnostic only.
    positive_variance_, negative_variance_ : float
        Sum of the positive, respectively sum of the magnitude of the
        negative, entries of ``raw_operator_eigenvalues_``. A properly
        estimated covariance operator is positive semi-definite, so
        ``negative_variance_`` reflects only estimation/discretization
        noise; it is expected to be small relative to
        ``positive_variance_``.
    negative_variance_ratio_ : float
        ``negative_variance_ / (negative_variance_ + positive_variance_)``
        (0 if both are zero). A substantial ratio signals inadequate
        smoothing, sparse covariance support, or a boundary artifact
        worth investigating -- diagnostic only, not used by any other
        computation in this class.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.99,
        method_smoothing: str = "PS",
        mean_bandwidth: Optional[float] = None,
        covariance_bandwidth: Optional[float] = None,
        mean_bandwidth_fraction: float = 0.05,
        covariance_bandwidth_fraction: float = 0.10,
        noise_bandwidth: Optional[float] = None,
        noise_bandwidth_fraction: float = 0.3,
        mean_penalty: Optional[float] = 1.0,
        covariance_penalty: Optional[float] = 1.0,
        ps_n_segments: int = 30,
        ps_mean_n_segments: int = 10,
        ps_degree: int = 3,
        ps_order_penalty: int = 2,
        n_grid_points: Optional[int] = None,
        smooth_mean: bool = True,
        smooth_covariance: bool = True,
        regularization: float = 1e-6,
        allow_extrapolation: bool = False,
        mean_degree: int = 1,
        covariance_degree: int = 2,
        noise_degree: int = 1,
        kernel_name: str = "epanechnikov",
    ) -> None:
        """
        Construct a PACE estimator.

        Stores and validates all hyperparameters as instance attributes;
        no estimation is performed here. See the class docstring for the
        full description of every parameter. Raises ``ValueError`` if
        any parameter fails a basic sanity check (e.g. an out-of-range
        ``variance_threshold``, an unrecognized ``method_smoothing``, a
        non-positive ``n_components``/``n_grid_points``, or a negative
        ``regularization``).
        """
        if variance_threshold <= 0 or variance_threshold > 1:
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {variance_threshold!r}."
            )
        if method_smoothing not in ("LP", "PS"):
            raise ValueError(
                f"method_smoothing must be 'LP' or 'PS', got {method_smoothing!r}."
            )
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components!r}.")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization!r}.")
        if n_grid_points is not None and n_grid_points < 10:
            raise ValueError(f"n_grid_points must be >= 10, got {n_grid_points!r}.")

        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.method_smoothing = method_smoothing
        self.mean_bandwidth = mean_bandwidth
        self.covariance_bandwidth = covariance_bandwidth
        self.mean_bandwidth_fraction = mean_bandwidth_fraction
        self.covariance_bandwidth_fraction = covariance_bandwidth_fraction
        self.noise_bandwidth = noise_bandwidth
        self.noise_bandwidth_fraction = noise_bandwidth_fraction
        self.mean_penalty = mean_penalty
        self.covariance_penalty = covariance_penalty
        self.ps_n_segments = ps_n_segments
        self.ps_mean_n_segments = ps_mean_n_segments
        self.ps_degree = ps_degree
        self.ps_order_penalty = ps_order_penalty
        self.n_grid_points = n_grid_points
        self.smooth_mean = smooth_mean
        self.smooth_covariance = smooth_covariance
        self.regularization = regularization
        self.allow_extrapolation = allow_extrapolation
        self.mean_degree = mean_degree
        self.covariance_degree = covariance_degree
        self.noise_degree = noise_degree
        self.kernel_name = kernel_name

        # learned quantities (set by fit())
        self.grid_: Optional[np.ndarray] = None
        self.mean_values_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenfunction_values_: Optional[np.ndarray] = None
        self.n_components_available_: Optional[int] = None
        self.n_components_used_: Optional[int] = None
        self.noise_variance_: Optional[float] = None
        self.is_dense_: Optional[bool] = None
        self._score_regularization_scale_: Optional[float] = None
        self._grid_is_uniform_: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: CurveData, data_type: str = "auto") -> "PACE":
        """
        Fit the PACE model.

        Parameters
        ----------
        data : list of (times, values)
            One ``(times, values)`` pair per curve. ``times`` and
            ``values`` must be equal-length, one-dimensional,
            finite-valued arrays (curves may of course use different, or
            overlapping, argvals from each other). Within a single
            curve, unsorted times are sorted automatically (with
            ``values`` reordered to match) and duplicate times are
            collapsed by averaging their ``values``; both corrections
            raise a ``UserWarning`` rather than an error (see
            ``_validate_one_curve``).
        data_type : {'auto', 'dense', 'irregular'}, default='auto'
            'auto' detects whether every curve shares exactly the same
            argvals ('dense') or not ('irregular'); this is purely
            informational (``self.is_dense_``) and does not change the
            estimation pipeline (see module docstring).

        Returns
        -------
        self
        """
        curves = self._validate_and_prepare(data, data_type=data_type)
        self._fit_mean(curves)
        self._fit_covariance(curves)
        self._fit_noise_variance(curves)
        self._resolve_component_counts()
        self._prepare_score_regularization()
        return self

    def transform(
        self,
        data: CurveData,
        n_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute PACE scores for a set of curves.

        Parameters
        ----------
        data : list of (times, values)
            Curves to score. Need not be the curves used to ``fit``.
        n_components : int, optional
            Number of components to return. Defaults to
            ``self.n_components_used_``. If larger than
            ``self.n_components_available_``, clamped with a
            ``UserWarning`` (see class docstring).

        Returns
        -------
        scores : ndarray, shape (n_curves, n_components)
        """
        self._check_is_fitted()
        n_used = self._resolve_n_components(n_components)
        scores = np.vstack(
            [
                self._project_individual(
                    *self._validate_one_curve(t, v, context="transform", index=i), n_used
                )
                for i, (t, v) in enumerate(data)
            ]
        )
        return scores

    def reconstruct_curve(
        self,
        times: np.ndarray,
        scores: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct a curve's trajectory from PACE scores.

        Parameters
        ----------
        times : ndarray
            Argvals at which to reconstruct. Merged with the model's
            own fitted grid so the returned curve is smooth while still
            including the requested points exactly.
        scores : ndarray
            PACE scores for one curve (e.g. one row of ``transform``'s
            output).
        n_components : int, optional
            Number of components of ``scores`` to use. Defaults to
            ``self.n_components_used_``, clamped (with a warning) to
            ``min(len(scores), self.n_components_available_)``.

        Returns
        -------
        values : ndarray
            Reconstructed values on ``times_eval``.
        times_eval : ndarray
            Sorted, unique evaluation grid actually used (fitted grid
            union ``times``).
        """
        self._check_is_fitted()
        n_requested = self._resolve_n_components(n_components)
        n_used = min(n_requested, len(scores))
        if n_used < n_requested:
            warnings.warn(
                f"n_components resolved to {n_requested}, but scores has only "
                f"{len(scores)} entries; using {n_used} components instead.",
                UserWarning,
                stacklevel=2,
            )

        times = np.asarray(times, dtype=float)
        self._check_in_domain(times, context="reconstruct_curve")
        times_eval = np.unique(np.concatenate([self.grid_, times]))

        phi = self._interp_eigenfunctions(times_eval, n_used)
        mean = np.interp(times_eval, self.grid_, self.mean_values_)
        values = mean + np.asarray(scores[:n_used]) @ phi
        return values, times_eval

    def plot_mean(self, ax=None, **kwargs):
        """
        Plot the fitted mean function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not
            given.
        **kwargs
            Additional keyword arguments passed through to ``ax.plot``.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        self._check_is_fitted()
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.grid_, self.mean_values_, **kwargs)
        ax.set_xlabel("Argval")
        ax.set_ylabel("Value")
        ax.set_title("Mean function")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_eigenfunctions(self, figsize=None, n_components: Optional[int] = None, **kwargs):
        """
        Plot fitted eigenfunctions, one subplot per component, each
        annotated with its share of the total estimated variance.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size passed to ``plt.subplots``. Defaults to a size
            that grows with the number of components plotted.
        n_components : int, optional
            Number of leading eigenfunctions to plot. Defaults to
            ``self.n_components_used_``. If larger than
            ``self.n_components_available_``, clamped with a
            ``UserWarning`` (see class docstring).
        **kwargs
            Additional keyword arguments passed through to each
            subplot's ``ax.plot``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes
            Flattened array of subplot axes, including any unused
            trailing axes (hidden, not removed) when the number of
            components plotted is odd.
        """
        import matplotlib.pyplot as plt

        self._check_is_fitted()
        n_used = self._resolve_n_components(n_components)
        n_rows = int(np.ceil(n_used / 2))
        n_cols = 2
        if figsize is None:
            figsize = (12, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        total_variance = np.sum(self.eigenvalues_)
        variance_explained = (self.eigenvalues_[:n_used] / total_variance) * 100

        for i in range(n_used):
            ax = axes[i]
            ax.axhline(0, color="0.35", lw=0.9, zorder=1)
            ax.plot(self.grid_, self.eigenfunction_values_[i], **kwargs)
            ax.set_xlabel("Argval")
            ax.set_ylabel("Value")
            ax.set_title(f"Eigenfunction {i + 1} ({variance_explained[i]:.2f}%)")
            ax.grid(True, alpha=0.3)
        for i in range(n_used, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_one_curve(self, times, values, context: str, index=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and coerce a single curve's ``(times, values)`` pair.

        Converts both inputs to 1-D float ``ndarray``s and checks equal
        length, finiteness, and non-emptiness. Two further conditions are
        auto-corrected rather than rejected:

        - **Unsorted times**: if ``times`` is not already ascending, both
          ``times`` and ``values`` are reordered by the same permutation
          that sorts ``times`` (so each value stays paired with its own
          time), and a ``UserWarning`` is raised naming the curve.
        - **Duplicate times**: if, after sorting, two or more observations
          share the exact same time, they are collapsed into a single
          observation whose value is the mean of the duplicates' values,
          and a ``UserWarning`` is raised naming the curve and the
          duplicated time(s).

        Sorting is always applied before duplicate detection, so a curve
        with both issues at once is handled correctly by both corrections
        together.

        Parameters
        ----------
        times, values : array-like
            One curve's observation locations and observed values.
        context : str
            Name of the calling public method; used only to make error
            and warning messages more informative.
        index : int, optional
            Curve index within a batch; used only to make error and
            warning messages more informative (e.g. "curve 3" instead of
            "curve").

        Returns
        -------
        times, values : ndarray
            Validated, float-typed, 1-D arrays: sorted ascending, with
            any duplicate observation times averaged into one.

        Raises
        ------
        ValueError
            If ``times``/``values`` are not 1-D, have mismatched lengths,
            contain NaN/infinite values, or are empty.
        """
        label = f"curve {index}" if index is not None else "curve"
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)

        if times.ndim != 1 or values.ndim != 1:
            raise ValueError(
                f"[{context}] {label}: times and values must be 1-D arrays, "
                f"got shapes {times.shape} and {values.shape}."
            )
        if len(times) != len(values):
            raise ValueError(
                f"[{context}] {label}: times and values must have the same length, "
                f"got {len(times)} and {len(values)}."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError(f"[{context}] {label}: times contains NaN or infinite values.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"[{context}] {label}: values contains NaN or infinite values.")
        if len(times) == 0:
            raise ValueError(f"[{context}] {label}: has zero observations.")

        if np.any(np.diff(times) < 0):
            order = np.argsort(times, kind="stable")
            times = times[order]
            values = values[order]
            warnings.warn(
                f"[{context}] {label}: times were not sorted ascending; sorted "
                "automatically and reordered values to match.",
                UserWarning,
                stacklevel=3,
            )

        unique_times, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
        if len(unique_times) != len(times):
            dup_times = unique_times[counts > 1]
            averaged_values = np.zeros(len(unique_times))
            np.add.at(averaged_values, inverse, values)
            averaged_values /= counts
            warnings.warn(
                f"[{context}] {label}: duplicate observation time(s) "
                f"{dup_times.tolist()} found; values at each duplicated time were "
                "averaged into a single observation.",
                UserWarning,
                stacklevel=3,
            )
            times = unique_times
            values = averaged_values

        return times, values

    def _validate_and_prepare(self, data: CurveData, data_type: str) -> CurveData:
        """
        Validate an entire dataset passed to ``fit`` and prepare it for
        estimation.

        Validates every curve via ``_validate_one_curve``, requires at
        least two curves and at least one curve with two or more
        observations (needed to estimate a covariance surface), warns
        about curves with a single observation (kept for the mean and the
        noise variance, but contributing no pairs to the covariance
        surface), resolves
        ``data_type='auto'`` into a concrete 'dense'/'irregular'
        classification (stored on ``self.is_dense_``), and, when
        ``method_smoothing='PS'``, checks that the required penalty
        argument(s) were supplied for whichever of the mean/covariance
        surface(s) will actually be smoothed.

        Parameters
        ----------
        data : list of (times, values)
            Raw input as passed to ``fit``.
        data_type : {'auto', 'dense', 'irregular'}
            As passed to ``fit``.

        Returns
        -------
        curves : list of (times, values)
            Validated curves, ready for mean/covariance estimation.

        Raises
        ------
        ValueError
            On an invalid ``data_type``, fewer than two curves, no curve with
            at least two observations, or a missing required penalty.

        Warns
        -----
        UserWarning
            If some curves have a single observation.
        """
        if data_type not in ("auto", "dense", "irregular"):
            raise ValueError(
                f"data_type must be 'auto', 'dense' or 'irregular', got {data_type!r}."
            )
        if len(data) < 2:
            raise ValueError(
                f"fit() requires at least 2 curves to estimate a covariance surface, "
                f"got {len(data)}."
            )

        curves = [
            self._validate_one_curve(t, v, context="fit", index=i)
            for i, (t, v) in enumerate(data)
        ]

        n_single = sum(len(t) < 2 for t, _ in curves)
        if n_single == len(curves):
            raise ValueError(
                "fit() requires at least one curve with 2 or more observations: "
                "the covariance surface is estimated from pairs of observations "
                "within a curve."
            )
        if n_single:
            # A single observation forms no within-curve pair, so it only reaches
            # the raw covariance diagonal, which both smoothers exclude. It still
            # informs the mean and the raw diagonal used for the noise variance.
            warnings.warn(
                f"{n_single} of {len(curves)} curves have a single observation. They "
                "are used to estimate the mean function and the noise variance, but "
                "contribute no pairs to the covariance surface.",
                UserWarning,
                stacklevel=3,
            )

        if data_type == "auto":
            times_tuples = {tuple(np.round(t, decimals=10)) for t, _ in curves}
            data_type = "dense" if len(times_tuples) == 1 else "irregular"
        self.is_dense_ = data_type == "dense"

        if self.method_smoothing == "PS":
            if self.smooth_mean and self.mean_penalty is None:
                raise ValueError(
                    "method_smoothing='PS' with smooth_mean=True requires a numeric "
                    "mean_penalty (it is None here). Either supply mean_penalty, "
                    "or use method_smoothing='LP', or set smooth_mean=False."
                )
            if self.smooth_covariance and self.covariance_penalty is None:
                raise ValueError(
                    "method_smoothing='PS' with smooth_covariance=True requires a "
                    "numeric covariance_penalty (it is None here). Either supply "
                    "covariance_penalty, or use method_smoothing='LP', or set "
                    "smooth_covariance=False."
                )

        return curves

    def _check_is_fitted(self) -> None:
        """Raise ``ValueError`` if ``fit`` has not been called yet."""
        if self.grid_ is None:
            raise ValueError("This PACE instance has not been fitted yet; call fit() first.")

    def _check_in_domain(self, times: np.ndarray, context: str) -> None:
        """
        Check that requested evaluation times fall within the fitted
        grid's domain (up to a small relative tolerance).

        If any time falls outside ``[grid_[0], grid_[-1]]``, either
        raises ``ValueError`` (the default, ``allow_extrapolation=
        False``) or emits a ``UserWarning`` and lets the caller proceed
        with flat extrapolation (``allow_extrapolation=True``).

        Parameters
        ----------
        times : ndarray
            Requested evaluation times.
        context : str
            Name of the calling public method; used only in the raised
            error/warning message.
        """
        lo, hi = self.grid_[0], self.grid_[-1]
        span = hi - lo if hi > lo else 1.0
        tol = 1e-8 * span
        out_lo = times < (lo - tol)
        out_hi = times > (hi + tol)
        out = out_lo | out_hi
        if not np.any(out):
            return
        offenders = times[out]
        if not self.allow_extrapolation:
            raise ValueError(
                f"[{context}] {len(offenders)} requested time(s) fall outside the "
                f"fitted domain [{lo:.6g}, {hi:.6g}] (e.g. {offenders[0]:.6g}). "
                "This implementation does not silently flat-extrapolate; pass only "
                "times within the fitted domain, or construct PACE(allow_extrapolation"
                "=True) to explicitly opt into flat extrapolation."
            )
        else:
            warnings.warn(
                f"[{context}] {len(offenders)} requested time(s) fall outside the "
                f"fitted domain [{lo:.6g}, {hi:.6g}] (e.g. {offenders[0]:.6g}); "
                "flat-extrapolating because allow_extrapolation=True.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Mean estimation
    # ------------------------------------------------------------------

    def _make_grid(self, pooled_t: np.ndarray) -> None:
        """
        Build the fixed evaluation/binning grid ``self.grid_``.

        Uses the exact set of distinct pooled argvals when there are at
        most ``n_grid_points`` of them (no approximation for small/dense
        problems), otherwise a uniform grid of ``n_grid_points`` points
        spanning the observed range. See the ``n_grid_points`` parameter
        docstring for why this matters for computational cost.
        """
        unique_t = np.unique(pooled_t)
        domain_range = unique_t[-1] - unique_t[0]
        if domain_range <= 0:
            raise ValueError(
                "All pooled observation times are identical; cannot fit a mean "
                "function or covariance surface over a degenerate (zero-width) domain."
            )
        cap = self.n_grid_points
        if cap is None:
            # Default: the union of all observed times, so no observation is
            # binned. 'LP' is the exception: its covariance smoothing costs
            # roughly O(len(grid)^4), so it keeps a 50-point cap by default.
            if self.method_smoothing == "LP":
                cap = _LP_DEFAULT_GRID_POINTS
            else:
                cap = len(unique_t)
                if cap > _LARGE_GRID_WARNING:
                    warnings.warn(
                        f"The evaluation grid is the union of all observation times and "
                        f"has {cap} points; the covariance surface is a {cap} x {cap} "
                        "matrix. Set n_grid_points to bin the times onto a coarser "
                        "uniform grid if memory or run time becomes a problem.",
                        UserWarning,
                        stacklevel=4,
                    )
        if len(unique_t) <= cap:
            self.grid_ = unique_t
            self._grid_is_uniform_ = False
        else:
            self.grid_ = np.linspace(unique_t[0], unique_t[-1], cap)
            self._grid_is_uniform_ = True
        self._domain_range_ = float(domain_range)

    def _grid_index(self, times: np.ndarray) -> np.ndarray:
        """Nearest-grid-point index for each value in ``times`` (exact match
        when ``self.grid_`` holds the exact distinct observed argvals)."""
        grid = self.grid_
        if self._grid_is_uniform_:
            step = grid[1] - grid[0]
            idx = np.round((times - grid[0]) / step).astype(int)
        else:
            idx = np.searchsorted(grid, times)
        return np.clip(idx, 0, len(grid) - 1)

    def _fit_mean(self, curves: CurveData) -> None:
        """
        Fit the population mean function and store it, together with the
        evaluation grid, on the instance.

        Pools every curve's observations, builds the evaluation grid
        (``_make_grid``), and either uses the raw pointwise average
        directly (``smooth_mean=False``) or smooths it with the
        configured method: local-linear regression on the raw pooled
        scatter for ``method_smoothing='LP'``, or a P-spline fit of the
        grid-aggregated pointwise average for ``method_smoothing='PS'``
        (see the inline comments below for why the P-spline path
        aggregates onto the grid before fitting). Sets
        ``self.mean_values_``.

        Parameters
        ----------
        curves : list of (times, values)
            Validated curves, as returned by ``_validate_and_prepare``.
        """
        pooled_t = np.concatenate([t for t, _ in curves])
        pooled_y = np.concatenate([y for _, y in curves])
        self._make_grid(pooled_t)
        grid = self.grid_
        domain_range = self._domain_range_

        if not self.smooth_mean:
            self.mean_values_ = self._pointwise_average(pooled_t, pooled_y)
            return

        if self.method_smoothing == "LP":
            bandwidth = self.mean_bandwidth
            if bandwidth is None:
                bandwidth = self.mean_bandwidth_fraction * domain_range
            lp = LocalPolynomial(
                kernel_name=self.kernel_name, bandwidth=bandwidth, degree=self.mean_degree
            )
            # Not binned: mean smoothing is a cheap O(n_grid_points *
            # n_total_observations) operation regardless of how large the
            # pooled observation count is (the expensive-per-query-point
            # LocalPolynomial loop only iterates over `grid`, not over the
            # training data), so no binning approximation is needed here.
            self.mean_values_ = lp.predict(y=pooled_y, x=pooled_t, x_new=grid)
        else:  # PS
            # A P-spline fit's own working weight matrix scales with the
            # number of points it is given, so handing it the full pooled
            # scatter directly (every curve's every observation,
            # concatenated) is both wasteful and, at sufficient scale,
            # memory-prohibitive. Aggregating the pooled scatter onto
            # `grid` first (a per-unique-argval mean) is lossless for the
            # mean *shape*: points sharing an exact argval carry no shape
            # information beyond their average, and their scatter around
            # that average is noise information, handled separately by
            # noise_variance_. This keeps the P-spline fit's working size
            # bounded by len(grid) regardless of how many curves or
            # observations were pooled.
            #
            # Each aggregated grid point is given equal fitting weight
            # (1), not a weight proportional to how many raw observations
            # backed it. Weighting by count is arguably the more literal
            # choice for an aggregated weighted-least-squares fit, but the
            # fitting objective here has no 1/n normalization, so it would
            # make the effective amount of smoothing for a given nominal
            # `mean_penalty` depend on how unevenly observations are
            # distributed across the domain -- an extra, hard-to-reason-
            # about source of variation. Equal weighting keeps the
            # relationship between `mean_penalty` and the resulting
            # smoothness predictable.
            binned_mean = self._pointwise_average(pooled_t, pooled_y)
            ps = PSplines(
                n_segments=self.ps_mean_n_segments,
                degree=self.ps_degree,
                order_penalty=self.ps_order_penalty,
            )
            ps.fit(
                x=grid,
                y=binned_mean,
                sample_weights=np.ones_like(grid),
                penalty=(self.mean_penalty,),
            )
            self.mean_values_ = ps.predict(grid)

    def _pointwise_average(self, pooled_t: np.ndarray, pooled_y: np.ndarray) -> np.ndarray:
        """
        Average pooled observations falling in the same evaluation-grid
        cell, filling any grid point with no observations by linear
        interpolation from its nearest populated neighbors.

        Parameters
        ----------
        pooled_t, pooled_y : ndarray
            Concatenated observation times/values across all curves.

        Returns
        -------
        result : ndarray, shape (len(self.grid_),)
            Per-grid-point average value.
        """
        grid = self.grid_
        idx = self._grid_index(pooled_t)
        sums = np.zeros_like(grid)
        counts = np.zeros_like(grid)
        np.add.at(sums, idx, pooled_y)
        np.add.at(counts, idx, 1.0)
        empty = counts == 0
        counts[empty] = 1.0
        result = sums / counts
        if np.any(empty):
            # fall back to interpolation for any grid point with no binned
            # observations at all (can only happen on the coarser, binned
            # grid branch)
            filled_idx = np.where(~empty)[0]
            result[empty] = np.interp(grid[empty], grid[filled_idx], result[filled_idx])
        return result

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def _pooled_raw_covariance(self, curves: CurveData):
        """
        Pool mean-centered cross products of every curve's own observations
        onto the shared grid.

        For each curve, every pair of its own observation times
        contributes one mean-centered cross product to the corresponding
        cell of the pooled covariance grid; the per-curve contribution is
        computed and accumulated with a single vectorized ``np.add.at``
        call against a pre-computed grid index, so pooling cost per curve
        does not depend on the grid's own resolution.

        Returns
        -------
        cov_sum, cov_count : ndarray, shape (m, m)
            Running sum / count of centered cross products at each pair of
            grid indices. ``cov_sum / max(cov_count, 1)`` is the raw
            (unsmoothed) empirical covariance estimate.
        """
        grid = self.grid_
        m = len(grid)
        cov_sum = np.zeros((m, m))
        cov_count = np.zeros((m, m))
        # Raw diagonal accumulated separately, from un-binned squared
        # residuals (mean of e_j^2), not the squared mean of residuals that
        # the (possibly binned) off-diagonal path below would otherwise
        # imply for the diagonal -- those differ whenever independent
        # measurement noise is present, with the squared-mean version
        # systematically smaller (noise partially cancels before squaring),
        # which would bias the noise-variance estimate downward whenever a
        # curve's own observations collide into a shared grid cell.
        diag_sum = np.zeros(m)
        diag_count = np.zeros(m)
        for t, y in curves:
            mu = np.interp(t, grid, self.mean_values_)
            resid = y - mu
            full_idx = self._grid_index(t)
            np.add.at(diag_sum, full_idx, resid ** 2)
            np.add.at(diag_count, full_idx, 1.0)

            idx = np.unique(full_idx)
            if len(idx) != len(t):
                # This curve's own points collide into shared grid cells
                # (happens when n_grid_points is coarser than this curve's
                # own point spacing) -- re-aggregate its residuals per cell
                # first, so each contributing grid cell is used once per
                # curve for this curve's contribution to cov_sum/cov_count.
                cell_sum = np.zeros(len(grid))
                cell_count = np.zeros(len(grid))
                np.add.at(cell_sum, full_idx, resid)
                np.add.at(cell_count, full_idx, 1.0)
                resid_binned = cell_sum[idx] / cell_count[idx]
            else:
                resid_binned = resid
            outer = np.outer(resid_binned, resid_binned)
            cov_sum[np.ix_(idx, idx)] += outer
            cov_count[np.ix_(idx, idx)] += 1.0
        return cov_sum, cov_count, diag_sum, diag_count

    def _fit_covariance(self, curves: CurveData) -> None:
        """
        Fit the smoothed covariance surface and diagonalize it.

        Pools raw, mean-centered covariance observations across curves
        (``_pooled_raw_covariance``), optionally smooths the resulting
        surface with the configured method (the raw pooled covariance is
        used unchanged if ``smooth_covariance=False``), symmetrizes the
        result to remove floating-point asymmetry, and calls
        ``_eigendecompose`` to obtain the functional principal
        components. Also stores the raw covariance diagonal for later
        use by ``_fit_noise_variance``.

        Parameters
        ----------
        curves : list of (times, values)
            Validated curves, as returned by ``_validate_and_prepare``.
        """
        cov_sum, cov_count, diag_sum, diag_count = self._pooled_raw_covariance(curves)
        has_data = cov_count > 0
        raw_cov = np.divide(cov_sum, cov_count, out=np.zeros_like(cov_sum), where=has_data)
        diag_has_data = diag_count > 0
        self._raw_diag_ = np.divide(
            diag_sum, diag_count, out=np.zeros_like(diag_sum), where=diag_has_data
        )
        self._raw_diag_has_data_ = diag_has_data

        if not self.smooth_covariance:
            covariance_smooth = raw_cov.copy()
        elif self.method_smoothing == "LP":
            covariance_smooth = self._smooth_covariance_lp(raw_cov, has_data)
        else:
            covariance_smooth = self._smooth_covariance_ps(raw_cov, cov_count, has_data)

        covariance_smooth = 0.5 * (covariance_smooth + covariance_smooth.T)
        self._covariance_surface_ = covariance_smooth
        self._eigendecompose(covariance_smooth)

    def _smooth_covariance_lp(self, raw_cov: np.ndarray, has_data: np.ndarray) -> np.ndarray:
        """
        Smooth the raw covariance surface with local-polynomial
        regression (degree ``covariance_degree``, Epanechnikov kernel by
        default), excluding the diagonal from the fit.

        Parameters
        ----------
        raw_cov : ndarray, shape (m, m)
            Raw pooled covariance estimate on the evaluation grid.
        has_data : ndarray of bool, shape (m, m)
            Which grid cells have at least one pooled observation.

        Returns
        -------
        smoothed : ndarray, shape (m, m)
            Covariance surface evaluated on the full grid x grid mesh.

        Raises
        ------
        ValueError
            If no off-diagonal grid cell has any pooled observation
            (nothing to fit a covariance surface to).
        """
        grid = self.grid_
        m = len(grid)
        fit_mask = has_data.copy()
        np.fill_diagonal(fit_mask, False)  # exclude diagonal: measurement-noise inflated
        if not np.any(fit_mask):
            raise ValueError(
                "No off-diagonal covariance observations are available to fit a "
                "covariance surface (every curve may have too few overlapping "
                "observation locations)."
            )
        tt, ss = np.meshgrid(grid, grid, indexing="ij")
        x = np.column_stack([tt[fit_mask], ss[fit_mask]])
        y = raw_cov[fit_mask]

        bandwidth = self.covariance_bandwidth
        if bandwidth is None:
            bandwidth = self.covariance_bandwidth_fraction * self._domain_range_
        lp = LocalPolynomial(
            kernel_name=self.kernel_name, bandwidth=bandwidth, degree=self.covariance_degree
        )
        query = np.column_stack([tt.ravel(), ss.ravel()])
        smoothed = lp.predict(y=y, x=x, x_new=query)
        return smoothed.reshape(m, m)

    def _smooth_covariance_ps(self, raw_cov: np.ndarray, cov_count: np.ndarray, has_data: np.ndarray) -> np.ndarray:
        """
        Smooth the raw covariance surface with a 2-D P-spline fit,
        excluding the diagonal from both the fitted values and the
        fitting weights.

        Parameters
        ----------
        raw_cov : ndarray, shape (m, m)
            Raw pooled covariance estimate on the evaluation grid.
        cov_count : ndarray, shape (m, m)
            Number of pooled observation pairs contributing to each grid
            cell. Used here only to help build the ``has_data``/weights
            mask; see the inline comment below for why the fitting
            weights themselves are not scaled by this count.
        has_data : ndarray of bool, shape (m, m)
            Which grid cells have at least one pooled observation.

        Returns
        -------
        smoothed : ndarray, shape (m, m)
            Covariance surface evaluated on the full grid x grid mesh.
        """
        grid = self.grid_
        data_for_fit = raw_cov.copy()
        np.fill_diagonal(data_for_fit, 0.0)
        # The diagonal of a raw covariance estimate is inflated by
        # measurement noise (Yao et al. 2005, eq. 2) and must be excluded
        # from the fit -- both the *value* at the diagonal and its
        # *fitting weight* are zeroed together, so the smoother is never
        # told a fabricated zero is a trusted observation. Off-diagonal
        # weights are equal (1) for every populated cell and zero where no
        # curve pair contributed data, rather than weighted by the raw
        # pooled observation count: as in the P-spline branch of
        # `_fit_mean`, weighting by count would make the effective amount
        # of smoothing for a given nominal `covariance_penalty` depend on
        # how unevenly curve-pair overlap is distributed across the grid
        # (which it generally is for real, non-dense data) rather than
        # depending only on the penalty itself. Equal weighting keeps that
        # relationship predictable.
        weights = np.ones_like(cov_count)
        np.fill_diagonal(weights, 0.0)
        weights[~has_data] = 0.0

        ps = PSplines(
            n_segments=self.ps_n_segments,
            degree=self.ps_degree,
            order_penalty=self.ps_order_penalty,
        )
        ps.fit(
            x=[grid, grid],
            y=data_for_fit,
            sample_weights=weights,
            penalty=(self.covariance_penalty, self.covariance_penalty),
        )
        return ps.predict([grid, grid])

    # ------------------------------------------------------------------
    # Eigendecomposition (trapezoidal-quadrature, explicitly sorted)
    # ------------------------------------------------------------------

    @staticmethod
    def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
        """Nodal (composite trapezoid) quadrature weights for a 1-D grid."""
        w = np.zeros_like(grid)
        diffs = np.diff(grid)
        w[0] = diffs[0] / 2.0
        w[-1] = diffs[-1] / 2.0
        w[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
        return w

    def _eigendecompose(self, covariance_smooth: np.ndarray) -> None:
        """
        Diagonalize the smoothed covariance surface to obtain functional
        principal components.

        Converts the discretized covariance surface into a genuine
        trapezoidal-quadrature eigenproblem (via a symmetric similarity
        transform by the square root of the quadrature weights), solves
        it with ``numpy.linalg.eigh``, explicitly sorts eigenvalues (and
        their corresponding eigenvectors) in descending order, maps the
        eigenvectors back to L2-orthonormal eigenfunction values on
        ``self.grid_``, clips negative numerical noise in the eigenvalues
        to zero, drops eigenvalues at or below numerical tolerance, and
        retains as many leading components as are needed to reach
        ``self.variance_threshold`` of total estimated variance. Sets
        ``self.eigenvalues_`` and ``self.eigenfunction_values_``. The
        sign of each returned eigenfunction is arbitrary, as for any
        eigendecomposition.

        Parameters
        ----------
        covariance_smooth : ndarray, shape (m, m)
            Smoothed (or raw, if ``smooth_covariance=False``) covariance
            surface on ``self.grid_``.

        Raises
        ------
        ValueError
            If no eigenvalue survives the numerical-tolerance cutoff
            (e.g. a degenerate or badly oversmoothed covariance surface).
        """
        grid = self.grid_
        w = self._trapezoid_weights(grid)
        sqrt_w = np.sqrt(w)
        A = sqrt_w[:, None] * covariance_smooth * sqrt_w[None, :]
        A = 0.5 * (A + A.T)

        eigvals, eigvecs = eigh(A)  # symmetric solver: real eigenvalues, but
        # no documented cross-version guarantee on their ordering.
        order = np.argsort(eigvals)[::-1]  # explicit descending sort, so
        # downstream cumulative-variance component selection never
        # silently depends on eigh's own ordering convention.
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # PSD diagnostics, captured before clipping: how much of the
        # spectrum (if any) was negative prior to the standard
        # positive-semidefinite projection below. A few numerically-tiny
        # negative eigenvalues are expected; a substantial negative share
        # signals inadequate smoothing, sparse covariance support, or a
        # boundary artifact worth investigating.
        self.raw_operator_eigenvalues_ = eigvals.copy()
        _neg = eigvals[eigvals < 0]
        _pos = eigvals[eigvals > 0]
        self.negative_variance_ = float(-_neg.sum()) if _neg.size else 0.0
        self.positive_variance_ = float(_pos.sum()) if _pos.size else 0.0
        _total_abs = self.negative_variance_ + self.positive_variance_
        self.negative_variance_ratio_ = (
            self.negative_variance_ / _total_abs if _total_abs > 0 else 0.0
        )

        eigvals = np.clip(eigvals, 0.0, None)

        # Map back from the sqrt(w)-weighted eigenbasis to function values, and
        # normalize so that each eigenfunction has unit L2 norm under the
        # trapezoid-quadrature inner product (eigh already gives eigvecs that
        # are orthonormal in the *Euclidean* sense on the weighted problem,
        # which is exactly equivalent to L2-orthonormal after this mapping).
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenfunctions = eigvecs / sqrt_w[:, None]
        eigenfunctions = np.nan_to_num(eigenfunctions, nan=0.0, posinf=0.0, neginf=0.0)
        eigenfunctions = eigenfunctions.T  # shape (m_components, m_grid)

        # Scale-invariant tolerance (standard numerical-linear-algebra
        # convention, akin to numpy.linalg.matrix_rank's default): relative
        # to the largest eigenvalue and matrix dimension, rather than a
        # fixed absolute floor -- a fixed floor would wrongly discard
        # legitimate components whenever the data's natural covariance
        # scale is much smaller than 1.
        largest = eigvals[0] if len(eigvals) else 0.0
        tolerance = np.finfo(float).eps * len(eigvals) * max(largest, np.finfo(float).tiny)
        valid = eigvals > tolerance
        eigvals = eigvals[valid]
        eigenfunctions = eigenfunctions[valid]

        if len(eigvals) == 0:
            raise ValueError(
                "The fitted covariance surface has no positive eigenvalues above "
                "numerical tolerance; the data may be degenerate (e.g. all curves "
                "identical) or oversmoothed."
            )

        # Full positive-eigenvalue spectrum, kept separately from the
        # variance_threshold-truncated set below -- used to build the PACE
        # score-covariance matrix K in _project_individual, so that signal
        # variance in components beyond variance_threshold is not
        # misattributed to noise in the K^-1 solve. Reporting/reconstruction
        # still use the truncated set.
        self._all_eigenvalues_ = eigvals.copy()
        self._all_eigenfunctions_ = eigenfunctions.copy()

        cum_frac = np.cumsum(eigvals) / eigvals.sum()
        k = int(np.searchsorted(cum_frac, self.variance_threshold) + 1)
        k = min(k, len(eigvals))

        self.eigenvalues_ = eigvals[:k]
        self.eigenfunction_values_ = eigenfunctions[:k]

    # ------------------------------------------------------------------
    # Noise variance
    # ------------------------------------------------------------------

    def _fit_noise_variance(self, curves: CurveData) -> None:
        """
        Estimate measurement-error variance sigma^2 from the gap between the
        raw and smoothed covariance-surface diagonal (Yao, Mueller & Wang,
        2005, eq. 2): the raw diagonal (per-grid-point average squared
        residual from the fitted mean, pooled across curves) is smoothed
        with a local-linear fit and compared to the already-fitted
        covariance surface's own diagonal, integrated over the interior
        half of the domain (trimming a quarter off each end, as in the
        source paper) using a domain-length-normalized trapezoidal
        integral -- NOT an unweighted mean over ``grid_``, which is
        generically non-uniformly spaced (the sorted union of every
        curve's own observation times) and would over/under-weight
        differently-sampled regions of the domain.
        """
        grid = self.grid_
        raw_diag = self._raw_diag_
        has_diag = self._raw_diag_has_data_
        if not np.any(has_diag):
            raise ValueError(
                "No curve has repeated observations at any single grid point; "
                "cannot estimate a raw covariance diagonal for noise-variance "
                "estimation."
            )

        bandwidth = self.noise_bandwidth
        if bandwidth is None:
            bandwidth = self.noise_bandwidth_fraction * self._domain_range_
        lp = LocalPolynomial(kernel_name=self.kernel_name, bandwidth=bandwidth, degree=self.noise_degree)
        v_hat = lp.predict(y=raw_diag[has_diag], x=grid[has_diag], x_new=grid)

        # Use the already-fitted, full smoothed covariance surface's own
        # diagonal directly, rather than reconstructing it from the
        # variance_threshold-truncated eigenset -- the latter necessarily
        # omits whatever signal variance lives past the truncation point,
        # which would otherwise be misattributed to noise here.
        g_diag = np.diag(self._covariance_surface_).copy()

        domain_len = self._domain_range_
        lo = grid[0] + domain_len / 4
        hi = grid[-1] - domain_len / 4
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() < 2:
            # domain too narrow to trim a quarter off each side and still
            # integrate; fall back to the full domain rather than failing.
            mask = np.ones_like(grid, dtype=bool)

        gap = v_hat[mask] - g_diag[mask]
        # Composite trapezoid rule, written out so it works on NumPy 1.x and 2.x
        # (np.trapz is deprecated in NumPy 2).
        x = grid[mask]
        integral = float(np.sum(np.diff(x) * (gap[1:] + gap[:-1]) * 0.5))
        span = grid[mask][-1] - grid[mask][0]
        estimate = integral / span if span > 0 else float(np.mean(gap))
        self.noise_variance_ = float(np.maximum(estimate, 0.0))

    # ------------------------------------------------------------------
    # Component-count bookkeeping
    # ------------------------------------------------------------------

    def _resolve_component_counts(self) -> None:
        """
        Resolve the constructor's ``n_components`` request against
        ``self.n_components_available_`` (the number of components the
        covariance eigendecomposition actually retained) once, right
        after fitting, and store the result in
        ``self.n_components_used_``.

        Warns (rather than raising) and clamps to the available count if
        more components were requested than are available.
        """
        self.n_components_available_ = len(self.eigenvalues_)
        requested = self.n_components if self.n_components is not None else self.n_components_available_
        if requested > self.n_components_available_:
            warnings.warn(
                f"n_components={requested} was requested but only "
                f"{self.n_components_available_} component(s) are available at "
                f"variance_threshold={self.variance_threshold}; using "
                f"{self.n_components_available_} instead.",
                UserWarning,
                stacklevel=3,
            )
            requested = self.n_components_available_
        self.n_components_used_ = requested

    def _resolve_n_components(self, requested: Optional[int]) -> int:
        """
        Resolve a per-call ``n_components`` override (as accepted by
        ``transform``, ``reconstruct_curve``, ``plot_eigenfunctions``)
        into a concrete integer.

        Parameters
        ----------
        requested : int, optional
            Per-call override. ``None`` means "use
            ``self.n_components_used_``".

        Returns
        -------
        n : int
            Resolved component count, clamped (with a ``UserWarning``)
            to ``self.n_components_available_`` if the request exceeds
            it.

        Raises
        ------
        ValueError
            If ``fit`` has not been called, or if ``requested`` is less
            than 1.
        """
        self._check_is_fitted()
        if requested is None:
            return self.n_components_used_
        if requested > self.n_components_available_:
            warnings.warn(
                f"n_components={requested} was requested but only "
                f"{self.n_components_available_} component(s) are available; "
                f"using {self.n_components_available_} instead.",
                UserWarning,
                stacklevel=3,
            )
            return self.n_components_available_
        if requested < 1:
            raise ValueError(f"n_components must be >= 1, got {requested!r}.")
        return requested

    def _prepare_score_regularization(self) -> None:
        """
        Precompute the data-dependent scale used to convert
        ``self.regularization`` (a relative jitter fraction) into an
        absolute jitter added to the per-curve score-covariance matrix
        solved in ``_project_individual``.

        Uses ``self.noise_variance_`` when it is numerically non-
        negligible, otherwise falls back to the mean of the retained
        eigenvalues, so the same relative default behaves sensibly
        regardless of the fitted data's absolute scale. Sets
        ``self._score_regularization_scale_``.
        """
        eig_scale = float(np.mean(self.eigenvalues_)) if len(self.eigenvalues_) else 0.0
        scale = self.noise_variance_ if self.noise_variance_ > 1e-12 else eig_scale
        self._score_regularization_scale_ = max(scale, 1e-12)

    # ------------------------------------------------------------------
    # PACE conditional-expectation scores
    # ------------------------------------------------------------------

    def _interp_eigenfunctions(self, times: np.ndarray, n_components: int) -> np.ndarray:
        """
        Linearly interpolate the leading ``n_components`` fitted
        eigenfunctions from ``self.grid_`` onto arbitrary evaluation
        times.

        Parameters
        ----------
        times : ndarray
            Evaluation times (assumed already checked to lie within the
            fitted domain by the caller).
        n_components : int
            Number of leading eigenfunctions to interpolate.

        Returns
        -------
        phi_t : ndarray, shape (n_components, len(times))
            Interpolated eigenfunction values.
        """
        phi_t = np.zeros((n_components, len(times)))
        for k in range(n_components):
            phi_t[k] = np.interp(times, self.grid_, self.eigenfunction_values_[k])
        return phi_t

    def _project_individual(self, times: np.ndarray, values: np.ndarray, n_used: int) -> np.ndarray:
        """
        Compute PACE scores for one curve (Yao, Mueller & Wang, 2005, eq. 5).

        The score-covariance matrix K (and the intermediate solve) always
        uses the *full* available eigencomponent set
        (``n_components_available_``), not just the requested output count
        -- otherwise any real signal variance from omitted components would
        be misattributed to residual/measurement noise in the K^-1 solve,
        systematically distorting the returned scores whenever more
        components are available than requested. Only the final returned
        vector is truncated to ``n_used``.
        """
        self._check_in_domain(times, context="transform")
        mu = np.interp(times, self.grid_, self.mean_values_)
        y_centered = values - mu

        # K is built from the *full* positive eigenspectrum
        # (_all_eigenvalues_/_all_eigenfunctions_, set in _eigendecompose),
        # not just the variance_threshold-truncated set -- omitting the
        # truncated tail here would misattribute real signal variance to
        # noise in the K^-1 solve, distorting every returned score.
        n_all = len(self._all_eigenvalues_)
        phi_t_all = np.zeros((n_all, len(times)))
        for k in range(n_all):
            phi_t_all[k] = np.interp(times, self.grid_, self._all_eigenfunctions_[k])

        K = (phi_t_all.T * self._all_eigenvalues_) @ phi_t_all
        jitter = self.regularization * self._score_regularization_scale_
        K += (self.noise_variance_ + jitter) * np.eye(len(times))
        K_inv_y = solve(K, y_centered)

        # The returned scores are still restricted to the
        # variance_threshold-retained set, matching this class's existing
        # n_components/n_components_used_ semantics -- only K's
        # construction above uses the full positive eigenspectrum.
        n_full = self.n_components_available_
        phi_t = self._interp_eigenfunctions(times, n_full)
        scores_full = self.eigenvalues_ * (phi_t @ K_inv_y)
        return scores_full[:n_used]
