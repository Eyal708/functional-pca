"""
Multivariate functional principal component analysis (MFPCA).

The ``MFPCA`` class extends functional principal component analysis to
vector-valued functional data. A subject is described by several
features (for example, several signals observed over time), and the
model finds a small set of multivariate components that describe joint
variation across all features.

``MFPCA`` does not model raw trajectories directly. Each feature is
first modelled by its own fitted univariate ``PACE`` estimator
(:class:`fpca.pace.PACE`), which implements principal analysis by conditional
expectation (Yao, Mueller and Wang, 2005). ``MFPCA`` then combines the
per-feature mean functions, eigenfunctions and scores.

Methods
-------
Two estimation strategies are available, selected with ``method``.

``"happ_greven"`` (default)
    Multivariate FPCA for features that may be observed on different
    time grids (Happ and Greven, 2018). The univariate PACE scores of all
    features are concatenated into one score matrix, optionally rescaled
    per feature, and principal component analysis is applied to that
    matrix. Multivariate eigenfunctions are obtained by mapping the PCA
    loadings back to each feature's univariate eigenfunctions.

``"berrendero"``
    The time-wise covariance decomposition of Berrendero, Justel and
    Svarc (2011). Each feature's PACE trajectories are reconstructed on
    one common time grid. At every grid point a cross-feature covariance
    matrix is estimated and eigendecomposed. The time-varying
    eigenvectors are aligned in sign across time and normalized to form
    global eigenfunctions. All features must share an identical time
    grid.

Input format
------------
All public methods take the same data structure: a dictionary that maps
feature names to lists of subjects::

    {feature_name: [(times_1, values_1), (times_2, values_2), ...], ...}

Each ``(times, values)`` pair holds the observations of one subject for
one feature. Subjects are aligned by list position, so the i-th entry of
every feature refers to the same subject, and all features must contain
the same number of subjects.

Example
-------
>>> from fpca import PACE, MFPCA
>>> data = {"x": x_curves, "y": y_curves}
>>> models = {
...     name: PACE(n_components=4, method_smoothing="LP").fit(curves)
...     for name, curves in data.items()
... }
>>> mfpca = MFPCA(models, method="happ_greven", n_components=3)
>>> scores = mfpca.fit_transform(data)  # shape (n_subjects, 3)

References
----------
Berrendero, J. R., Justel, A., and Svarc, M. (2011). Principal components
for multivariate functional data. Computational Statistics and Data
Analysis, 55(9), 2619-2634.

Happ, C., and Greven, S. (2018). Multivariate functional principal
component analysis for data observed on different (dimensional) domains.
Journal of the American Statistical Association, 113(522), 649-659.
https://doi.org/10.1080/01621459.2016.1273115

Yao, F., Mueller, H.-G., and Wang, J.-L. (2005). Functional data analysis
for sparse longitudinal data. Journal of the American Statistical
Association, 100(470), 577-590.
https://doi.org/10.1198/016214504000001745
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["MFPCA"]


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Composite trapezoid rule of ``y`` over the grid ``x``.

    Written out explicitly so the module works with NumPy 1.x and 2.x.
    """
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) * 0.5))


class MFPCA:
    """
    Multivariate functional principal component analysis.

    Decomposes a vector-valued functional sample into a shared set of
    multivariate principal components, using one fitted univariate ``PACE``
    model per feature. See the module docstring for the two estimation
    methods and the input format.

    Parameters
    ----------
    pace_models : dict of str to PACE
        Fitted univariate PACE models, keyed by feature name. Each model must
        be fitted (``fit`` called) before it is passed in. With
        ``method="berrendero"`` all models must share an identical ``grid_``;
        with ``method="happ_greven"`` they may use different grids.
    feature_names : list of str, optional
        Ordered feature names. This order defines the feature axis of every
        multi-feature output. Every name must be a key of ``pace_models``.
        If None, the sorted keys of ``pace_models`` are used.
    n_components : int, optional
        Number of multivariate components to retain. If None, the maximum
        available is used: the number of features for ``"berrendero"``, and
        ``min(n_subjects, total retained univariate components)`` for
        ``"happ_greven"``. A larger value is reduced to the maximum available.
    method : {"happ_greven", "berrendero"}, default="happ_greven"
        Estimation method (see the module docstring). Other values are
        accepted by the constructor and rejected by ``fit``.
    n_components_per_feature : int or dict of str to int, optional
        Happ-Greven only. Number of univariate PACE components per feature
        that enter the joint PCA. An int applies to every feature. A dict maps
        feature names to counts; features absent from the dict use 5. If None,
        5 is used for every feature. Any count that exceeds the number of
        components a feature's PACE model retains is reduced to that number. An
        explicit request is reduced with a ``UserWarning`` that names the
        feature and the counts; the default of 5 is reduced silently. Ignored by
        ``"berrendero"``.
    center : bool, default=True
        Berrendero only. If True, the PACE mean functions are subtracted from
        the reconstructed trajectories before covariances are estimated, and
        added back during reconstruction. ``"happ_greven"`` always works with
        centred PACE scores and always adds the mean functions back, so this
        flag has no effect for it.
    scale_features : bool, default=True
        If True, feature j is divided by s_j = sqrt(sum_l lambda_l^(j)), the
        square root of the total variance of feature j implied by its PACE
        eigenvalues. This places features measured in different units or
        amplitudes on a comparable scale. For ``"happ_greven"`` this is the
        weighted variant of Happ and Greven (2018, Sections 3.2 and 5), with
        weights w_j = 1 / s_j^2. If False, the unweighted algorithm is used.
        For ``"berrendero"`` each feature's reconstructed trajectory is divided
        by s_j before covariance estimation and multiplied back during
        reconstruction.
    smooth_cov : float, optional
        Berrendero only. Smoothing parameter ``s`` passed to
        ``scipy.interpolate.UnivariateSpline`` (cubic) when each element of the
        time-wise covariance matrices is smoothed along the time axis. If None
        (default), no smoothing is applied. ``s`` is a budget on squared
        residuals, so a useful value depends on the scale of the covariances:
        larger values give smoother covariance functions, and 0 interpolates
        them exactly. Ignored by ``"happ_greven"``.
    name : str, default="MFPCA"
        Label used in the message printed by ``fit`` when ``verbose`` is True.
    verbose : bool, default=False
        If True, ``fit`` prints one summary line with the number of subjects
        used and dropped. Independently of this flag, a ``UserWarning`` is
        raised whenever subjects are dropped.

    Attributes
    ----------
    The constructor parameters are stored under their own names (``method``,
    ``n_components``, ``n_components_per_feature``, ``center``,
    ``scale_features``, ``smooth_cov``, ``name``). The following attributes
    are set by the constructor or by ``fit``.

    pace_models_ : dict of str to PACE
        The PACE models passed to the constructor.
    feature_names_ : list of str
        Feature order used throughout the model.
    n_components_ : int
        Number of multivariate components retained by ``fit``.
    grid_ : ndarray, shape (n_timepoints,)
        Display grid. For ``"berrendero"`` this is the common grid of all PACE
        models. For ``"happ_greven"`` it is the grid of the first feature's
        PACE model, and the eigenfunctions and mean functions of the other
        features are linearly interpolated onto it. ``weight_functions_``,
        ``mean_functions_`` and ``reconstruct_from_scores`` all use this grid.
    mean_functions_ : dict of str to ndarray
        Mean function of each feature, evaluated on ``grid_``.
    weight_functions_ : ndarray, shape (n_components_, n_timepoints, n_features)
        Multivariate eigenfunctions on ``grid_``. ``weight_functions_[k, t, f]``
        is the loading of feature f at time t on component k. When
        ``scale_features`` is True the stored value equals the loading in
        original units divided by ``scale_stds_[f]``;
        ``get_feature_weight_function(..., descale=True)`` returns original
        units.
    eigenvalues_ : ndarray, shape (n_components_,)
        Global eigenvalues. For ``"berrendero"`` these are the time-integrated
        local eigenvalues (trapezoidal rule on ``grid_``). For
        ``"happ_greven"`` they are the variances of the joint PCA scores.
        Both are expressed in rescaled units when ``scale_features`` is True.
    explained_variance_ratio_ : ndarray, shape (n_components_,)
        Proportion of variance explained by each component. For
        ``"berrendero"`` the denominator is the sum of the retained eigenvalues.
        For ``"happ_greven"`` the denominator is the total variance of all
        concatenated univariate score columns, so the ratios of the retained
        components need not sum to 1.
    scale_stds_ : ndarray, shape (n_features,) or None
        The factors s_j. None when ``scale_features`` is False.
    score_pca_components_ : ndarray, shape (n_components_, total_univariate_components) or None
        Happ-Greven only. Loading vectors of the joint PCA, one row per
        multivariate component. Columns are grouped by feature in
        ``feature_names_`` order.
    pca_model_ : sklearn.decomposition.PCA or None
        Happ-Greven only. The joint PCA fitted to the rescaled score matrix.
    pca_mean_offset_ : ndarray, shape (n_timepoints, n_features) or None
        Happ-Greven only. The PCA column mean mapped to function space. It is
        added back during reconstruction.
    n_components_per_feature_ : list of int or None
        Happ-Greven only. Requested number of univariate components per
        feature, in ``feature_names_`` order.

    Notes
    -----
    Fitting proceeds as follows. Subjects with missing observations for any
    feature are dropped. The common grid is established and the PACE mean
    functions are collected. For ``"berrendero"``, trajectories are
    reconstructed by PACE on ``grid_``, centred and rescaled as configured,
    used to estimate time-wise covariance matrices (optionally smoothed), and
    decomposed at each time point. The local eigenvectors are sign-aligned
    over time and integrated into global eigenfunctions. For
    ``"happ_greven"``, the univariate scores of each feature are truncated to
    ``n_components_per_feature``, rescaled, concatenated, and passed to a PCA.
    The loading vectors are then mapped to eigenfunctions in function space.

    ``transform`` applies the fitted model to new data without dropping
    subjects. Every subject must have observations for every feature;
    otherwise a ``ValueError`` is raised. For ``"berrendero"`` the subject's
    trajectories are reconstructed and projected onto ``weight_functions_``
    by time-integrated inner products. For ``"happ_greven"`` the fitted PCA
    is applied to the rescaled score matrix. The centring and scaling
    settings used at fit time are reused.

    Limitations
    -----------
    - The settings are read again at transform time, so they must not be
      changed after ``fit``; refit the model instead.
    - ``"berrendero"`` requires identical grids. Its estimates are pointwise:
      where two eigenvalues swap order, the sign alignment can still produce
      a discontinuity in the eigenfunctions.
    - ``"happ_greven"``: if a PACE model retains fewer components than the
      requested ``n_components_per_feature``, the request is reduced to the
      available number (with a ``UserWarning`` for an explicit request); the
      reduced counts are stored in
      ``n_components_per_feature_`` and reused by ``transform``.
    - ``reconstruct_from_scores`` evaluates all features on ``grid_``. For
      ``"happ_greven"`` the values of features other than the first are
      interpolated onto it. Use ``reconstruct_feature`` to obtain values on
      each feature's own PACE grid.
    - Scores and eigenfunctions are identified only up to sign.
    """

    def __init__(
        self,
        pace_models: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        n_components: Optional[int] = None,
        method: str = "happ_greven",
        n_components_per_feature: Optional[Union[int, Dict[str, int]]] = None,
        center: bool = True,
        scale_features: bool = True,
        smooth_cov: Optional[float] = None,
        name: str = "MFPCA",
        verbose: bool = False,
    ) -> None:
        """
        Construct an MFPCA estimator.

        Stores the configuration and the PACE models. No estimation is performed
        here. All parameters are described in the class docstring.

        Raises
        ------
        ValueError
            If ``pace_models`` is not a non-empty dict, or if ``feature_names``
            contains names that are not keys of ``pace_models``.
        """
        # Validate pace_models
        if not isinstance(pace_models, dict) or len(pace_models) == 0:
            raise ValueError("pace_models must be a non-empty dictionary")

        # Store PACE models
        self.pace_models_ = pace_models

        # Determine feature names
        if feature_names is None:
            self.feature_names_ = sorted(pace_models.keys())
        else:
            self.feature_names_ = feature_names
            # Validate all features exist in pace_models
            missing = set(feature_names) - set(pace_models.keys())
            if missing:
                raise ValueError(f"Features in feature_names not found in pace_models: {missing}")

        # Store parameters
        self.n_components = n_components
        self.method = method
        self.n_components_per_feature = n_components_per_feature
        self.center = center
        self.scale_features = scale_features
        self.smooth_cov = smooth_cov
        self.name = name
        self.verbose = verbose

        # Initialize result attributes
        self.grid_ = None
        self.mean_functions_ = None
        self.weight_functions_ = None
        self.score_pca_components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.scale_stds_ = None
        self.n_components_per_feature_ = None
        self.pca_model_ = None
        self.pca_mean_offset_ = None

    def fit(self, data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> 'MFPCA':
        """
        Fit the multivariate model.

        Parameters
        ----------
        data : dict of str to list of (times, values)
            Multivariate sample in the format described in the module docstring.
            Every feature in ``feature_names_`` must be present, and all features
            must contain the same number of subjects.

        Returns
        -------
        self : MFPCA
            The fitted instance.

        Raises
        ------
        ValueError
            If ``data`` is not a dict, a feature is missing, the features contain
            different numbers of subjects, no subject is complete, a PACE model is
            not fitted, the method is unknown, or (for ``"berrendero"``) the PACE
            models do not share an identical grid.

        Notes
        -----
        Subjects without observations for every feature are dropped before
        fitting. A message reports how many subjects were used and dropped. See
        the class Notes for the steps performed by each method.
        """
        # Validate and filter data
        filtered_data, n_valid, n_dropped = self._validate_and_filter_subjects(data)

        if self.verbose:
            print(f"Fitting {self.name} with {n_valid} subjects "
                  f"(filtered {n_dropped} subjects with incomplete feature observations)")
        if n_dropped:
            warnings.warn(
                f"{self.name}: {n_dropped} subject(s) with incomplete feature "
                f"observations were dropped; fitting on {n_valid} subject(s).",
                UserWarning,
                stacklevel=2,
            )

        if n_valid == 0:
            raise ValueError(
                "No subjects have complete observations for all features. "
                f"Required features: {self.feature_names_}"
            )

        # Establish common time grid
        self._establish_common_grid()

        # Compute multivariate mean functions from PACE models
        self._compute_multivariate_means()

        method = self._validate_method()

        if method == "berrendero":
            # Reconstruct multivariate trajectories on common grid
            trajectories = self._reconstruct_multivariate_trajectories(filtered_data)

            # Center trajectories if requested
            if self.center:
                trajectories = self._center_trajectories(trajectories)

            # Scale trajectories if requested
            if self.scale_features:
                self._compute_feature_scaling(trajectories)
                trajectories = self._apply_scaling(trajectories)

            # Estimate time-wise covariances
            time_covariances = self._estimate_time_wise_covariances(trajectories)
            if self.smooth_cov is not None:
                time_covariances = self._smooth_covariances(time_covariances)
            self._extract_global_eigenfunctions(time_covariances)

            # Compute explained variance ratio
            total_variance = np.sum(self.eigenvalues_)
            if total_variance > 0:
                self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
            else:
                self.explained_variance_ratio_ = np.zeros(self.n_components_)
        else:
            self._prepare_happ_greven()
            scores_matrix = self._build_scores_matrix(filtered_data, self.n_components_per_feature_)
            self._fit_happ_greven_pca(scores_matrix)

        return self

    def fit_from_scores(self, scores: Dict[str, np.ndarray]) -> 'MFPCA':
        """
        Fit the Happ-Greven model from pre-computed univariate scores.

        ``fit`` computes each feature's univariate scores from raw curves with
        ``pace_models_[feature].transform``. This method takes the scores directly
        and then runs the same steps: per-feature scaling, a PCA of the
        concatenated scores, and the mapping of the PCA loadings to multivariate
        eigenfunctions. It is the fitting counterpart of ``transform_from_scores``.

        Any estimator that supplies a univariate decomposition can be combined
        this way. In particular, the levels of a two-level model are combined with
        ``TwoLevelPACE.level_model``, which exposes one level's mean, eigenvalues
        and eigenfunctions with the attributes ``MFPCA`` reads, so that the
        multivariate analysis is run separately at each level::

            level1 = {f: m.level_model("between") for f, m in two_level_models.items()}
            mv1 = MFPCA(level1).fit_from_scores(
                {f: two_level_scores[f]["between"] for f in level1})

        Parameters
        ----------
        scores : dict of str to ndarray
            Maps each feature in ``feature_names_`` to an array of shape
            (n_units, n_scores) of raw, unscaled univariate scores, aligned by
            row across features. Column l must belong to the l-th eigenfunction of
            ``pace_models_[feature]``. Only ``n_components_per_feature_[f]``
            leading columns are used.

        Returns
        -------
        self : MFPCA
            The fitted instance. ``transform_from_scores``,
            ``inverse_transform_scores``, ``reconstruct_feature``,
            ``get_feature_weight_function`` and the plotting functions work as
            after ``fit``. ``transform`` needs models that can score raw curves.

        Raises
        ------
        ValueError
            If ``method`` is not ``"happ_greven"``, a feature is missing from
            ``scores``, the arrays do not share the same number of rows, a model
            is not fitted, or the scores are not finite.
        """
        if self._validate_method() != "happ_greven":
            raise ValueError(
                "fit_from_scores() is only supported for method='happ_greven'; the "
                "'berrendero' method needs reconstructed trajectories."
            )
        missing = [f for f in self.feature_names_ if f not in scores]
        if missing:
            raise ValueError(f"Missing scores for feature(s): {missing}")
        scores = {f: np.atleast_2d(np.asarray(scores[f], dtype=float)) for f in self.feature_names_}
        n_units_set = {scores[f].shape[0] for f in self.feature_names_}
        if len(n_units_set) > 1:
            raise ValueError(f"All features must have the same number of units; got {n_units_set}")
        if not all(np.all(np.isfinite(scores[f])) for f in self.feature_names_):
            raise ValueError("scores must be finite.")

        self._establish_common_grid()
        self._compute_multivariate_means()
        self._prepare_happ_greven(available={f: scores[f].shape[1] for f in self.feature_names_})
        self._fit_happ_greven_pca(self._assemble_scores_matrix(scores, self.n_components_per_feature_))
        return self

    def _prepare_happ_greven(self, available: Optional[Dict[str, int]] = None) -> None:
        """Set the feature scaling and the number of univariate components per feature.

        Each requested count is clamped to the components the feature's model
        retains (and, if given, to ``available[feature]`` score columns), with a
        ``UserWarning`` when a request is reduced.
        """
        if self.scale_features:
            self._compute_feature_scaling(np.empty((0, 0, 0)))
        requested_counts = self._resolve_n_components_per_feature()
        self.n_components_per_feature_ = []
        for feat, req in zip(self.feature_names_, requested_counts):
            avail = self.pace_models_[feat].n_components_used_
            if available is not None:
                avail = min(avail, available[feat])
            if req > avail:
                # The default of 5 is a cap, not a request: reduce it silently.
                if self.n_components_per_feature is not None:
                    warnings.warn(
                        f"Requested {req} components for '{feat}', but only {avail} are "
                        f"available. Using {avail}.",
                        UserWarning, stacklevel=3,
                    )
                req = avail
            self.n_components_per_feature_.append(req)

    def _fit_happ_greven_pca(self, scores_matrix: np.ndarray) -> None:
        """PCA of the concatenated, scaled univariate scores (Happ and Greven 2018, Sec. 3.2)."""
        n_subjects, total_components = scores_matrix.shape
        if total_components == 0:
            raise ValueError("No univariate components are available for scores PCA.")

        if self.n_components is None:
            n_components = min(n_subjects, total_components)
        else:
            n_components = min(self.n_components, n_subjects, total_components)

        self.pca_model_ = PCA(n_components=n_components)
        self.pca_model_.fit(scores_matrix)

        self.n_components_ = self.pca_model_.n_components_
        self.score_pca_components_ = self.pca_model_.components_.copy()
        self.eigenvalues_ = self.pca_model_.explained_variance_.copy()
        self.explained_variance_ratio_ = self.pca_model_.explained_variance_ratio_.copy()

        # Compute multivariate eigenfunctions in function space
        self._compute_happ_greven_weight_functions()

    def inverse_transform_scores(
        self,
        scores: np.ndarray,
        which_components: Optional[Union[int, List[int]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Map multivariate scores back to univariate scores of each feature.

        Inverts the joint PCA of the Happ-Greven method: the scaled univariate
        scores are ``pca_mean + sum_k rho_k c_k``, which are then multiplied back
        by each feature's scale ``s_j``. With all components this recovers the
        univariate scores used in fitting; with fewer it gives their projection
        onto the leading multivariate components.

        Parameters
        ----------
        scores : ndarray, shape (n_units, n_components_) or (n_components_,)
            Multivariate scores.
        which_components : int or list of int, optional
            Components to use: an int ``m`` selects the first ``m``, a list selects
            the given indices, and None uses all.

        Returns
        -------
        dict of str to ndarray
            Feature -> univariate scores, shape (n_units, n_components_per_feature_[f]),
            or one-dimensional for one-dimensional ``scores``.

        Raises
        ------
        ValueError
            If the model has not been fitted or ``method`` is not ``"happ_greven"``.
        """
        if self.pca_model_ is None or self.n_components_per_feature_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if self._validate_method() != "happ_greven":
            raise ValueError("inverse_transform_scores() is only supported for method='happ_greven'.")
        single = np.asarray(scores).ndim == 1
        scores = np.atleast_2d(np.asarray(scores, dtype=float))
        if which_components is None:
            comp = np.arange(self.n_components_)
        elif isinstance(which_components, int):
            comp = np.arange(min(which_components, self.n_components_))
        else:
            comp = np.asarray(which_components)

        scaled = self.pca_model_.mean_ + scores[:, comp] @ self.score_pca_components_[comp]
        offsets = np.cumsum([0] + list(self.n_components_per_feature_))
        out = {}
        for f, feat in enumerate(self.feature_names_):
            block = scaled[:, offsets[f]:offsets[f + 1]]
            if self.scale_features:
                block = block * self.scale_stds_[f]
            out[feat] = block[0] if single else block
        return out

    def transform(self, data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
        """
        Compute multivariate scores for new subjects.

        Parameters
        ----------
        data : dict of str to list of (times, values)
            Multivariate sample in the format described in the module docstring.
            Every subject must have observations for every feature.

        Returns
        -------
        scores : ndarray, shape (n_subjects, n_components_)
            Multivariate scores. ``scores[i, k]`` is the score of subject i on
            component k.

        Raises
        ------
        ValueError
            If the model has not been fitted, a feature is missing, the features
            contain different numbers of subjects, or any subject lacks
            observations for some feature.

        Notes
        -----
        Unlike ``fit``, ``transform`` does not drop subjects. If a subject lies
        outside the domain of a PACE model, the PACE model raises ``ValueError``
        unless it was constructed with ``allow_extrapolation=True``.
        """
        # Check if fitted
        if self.grid_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Validate data completeness (raises error if any subject incomplete)
        self._validate_transform_data(data)

        method = self._validate_method()

        if method == "berrendero":
            # Reconstruct multivariate trajectories
            trajectories = self._reconstruct_multivariate_trajectories(data)

            # Center trajectories if model was fitted with centering
            if self.center:
                trajectories = self._center_trajectories(trajectories)

            # Scale trajectories if model was fitted with scaling
            if self.scale_features:
                trajectories = self._apply_scaling(trajectories)

            # Compute scores via inner products with weight functions
            scores = self._compute_multivariate_scores(trajectories)
        else:
            if self.pca_model_ is None or self.n_components_per_feature_ is None:
                raise ValueError("Model has not been fitted yet. Call fit() first.")

            scores_matrix = self._build_scores_matrix(data, self.n_components_per_feature_)
            scores = self.pca_model_.transform(scores_matrix)

        return scores

    def fit_transform(self, data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
        """
        Fit the model and return the scores of the fitting data.

        Equivalent to ``fit(data)`` followed by ``transform(data)``.

        Parameters
        ----------
        data : dict of str to list of (times, values)
            Multivariate sample (see ``fit``).

        Returns
        -------
        scores : ndarray, shape (n_subjects, n_components_)
            Multivariate scores of the subjects in ``data``.

        Raises
        ------
        ValueError
            If no subject in ``data`` has observations for every feature.

        Notes
        -----
        Subjects with incomplete features are dropped, as in ``fit``, and only the
        complete subjects are scored. The returned array therefore has one row per
        complete subject.
        """
        self.fit(data)
        # fit drops subjects with incomplete features; transform requires complete
        # subjects, so score only the subjects that fit used.
        filtered_data, _, _ = self._validate_and_filter_subjects(data)
        return self.transform(filtered_data)

    def reconstruct_from_scores(
        self,
        scores: np.ndarray,
        which_components: Optional[Union[int, List[int]]] = None
    ) -> np.ndarray:
        """
        Reconstruct multivariate trajectories from multivariate scores.

        For subject i, feature f and grid point t the reconstruction is::

            X_hat[i, t, f] = m_f(t) + s_f * (o_f(t) + sum_k scores[i, k] * W[k, t, f])

        where ``W`` is ``weight_functions_``, ``m_f`` is the PACE mean function of
        feature f (``mean_functions_``), ``s_f`` is ``scale_stds_[f]``, and ``o_f``
        is the PCA mean offset (``pca_mean_offset_``, Happ-Greven only). A term is
        omitted when it does not apply: the factor ``s_f`` when ``scale_features``
        is False, the mean term for ``"berrendero"`` when ``center`` is False, and
        the offset for ``"berrendero"``.

        Parameters
        ----------
        scores : ndarray, shape (n_subjects, n_components_) or (n_components_,)
            Multivariate scores, for example from ``transform``. A one-dimensional
            array is treated as one subject. The array must have at least as many
            columns as the components used.
        which_components : int or list of int, optional
            Components to use. An int ``m`` selects the first ``m`` components, a
            list selects the given indices, and None uses all components. Unselected
            components contribute nothing to the reconstruction.

        Returns
        -------
        reconstructed : ndarray
            Shape (n_subjects, n_timepoints, n_features), or (n_timepoints,
            n_features) when ``scores`` has one row. Values are on ``grid_``.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Notes
        -----
        For ``"happ_greven"``, features observed on other grids are interpolated
        onto ``grid_``. Use ``reconstruct_feature`` to obtain values on each
        feature's own PACE grid.
        """
        # Check if fitted
        if self.grid_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if self.weight_functions_ is None:
            raise ValueError("Weight functions are not available. Call fit() first.")

        # Handle single subject (1D scores)
        scores = np.atleast_2d(scores)
        n_subjects = scores.shape[0]

        # Determine which components to use
        if which_components is None:
            comp_indices = np.arange(self.n_components_)
        elif isinstance(which_components, int):
            comp_indices = np.arange(min(which_components, self.n_components_))
        else:
            comp_indices = np.array(which_components)

        method = self._validate_method()

        # Initialize reconstruction with zeros
        n_timepoints = len(self.grid_)
        n_features = len(self.feature_names_)
        reconstructed = np.zeros((n_subjects, n_timepoints, n_features))

        # Add weighted eigenfunctions: sum_k rho_ik * psi_k(t)
        for k in comp_indices:
            # weight_functions_[k] has shape (n_timepoints, n_features)
            # scores[:, k] has shape (n_subjects,)
            # Broadcasting: (n_subjects, 1, 1) * (n_timepoints, n_features)
            reconstructed += scores[:, k:k+1, np.newaxis] * self.weight_functions_[k]

        # For Happ-Greven: add the PCA mean offset in function space
        # This accounts for sklearn PCA's internal centering of the score matrix
        if method == "happ_greven" and self.pca_mean_offset_ is not None:
            reconstructed += self.pca_mean_offset_

        # Undo scaling if enabled (multiply by s_q for each feature)
        if self.scale_features:
            reconstructed = self._undo_scaling(reconstructed)

        # Add PACE mean functions
        # For Berrendero: only if centering was used
        # For Happ-Greven: always (PACE means are the feature means)
        if method == "happ_greven" or self.center:
            mean_array = self._build_mean_array()  # shape: (n_timepoints, n_features)
            reconstructed += mean_array

        # Return 2D array if single subject
        if n_subjects == 1:
            return reconstructed[0]

        return reconstructed

    def reconstruct_feature(
        self,
        scores: np.ndarray,
        feature_name: str,
        which_components: Optional[Union[int, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the trajectory of one feature from multivariate scores.

        Parameters
        ----------
        scores : ndarray, shape (n_subjects, n_components_) or (n_components_,)
            Multivariate scores (see ``reconstruct_from_scores``).
        feature_name : str
            Name of the feature to reconstruct. Must be in ``feature_names_``.
        which_components : int or list of int, optional
            Components to use, as in ``reconstruct_from_scores``.

        Returns
        -------
        values : ndarray
            Shape (n_subjects, n_timepoints), or (n_timepoints,) for one subject.
            For ``"happ_greven"`` the values lie on the feature's own PACE grid;
            for ``"berrendero"`` they lie on ``grid_``.
        times : ndarray, shape (n_timepoints,)
            The grid on which ``values`` are evaluated.

        Raises
        ------
        ValueError
            If the model has not been fitted, or ``feature_name`` is not in
            ``feature_names_``.

        Notes
        -----
        For ``"happ_greven"`` the univariate scores of the feature are recovered
        from the multivariate scores by inverting the PCA (adding back the PCA
        mean and undoing the rescaling). The feature's PACE model then evaluates
        the univariate reconstruction. For ``"berrendero"`` the feature is sliced
        from ``reconstruct_from_scores``.
        """
        # Check if fitted
        if self.grid_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if feature_name not in self.feature_names_:
            raise ValueError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {self.feature_names_}"
            )

        # Handle single subject (1D scores)
        single_subject = scores.ndim == 1
        scores = np.atleast_2d(scores)
        n_subjects = scores.shape[0]

        # Determine which components to use
        if which_components is None:
            comp_indices = np.arange(self.n_components_)
        elif isinstance(which_components, int):
            comp_indices = np.arange(min(which_components, self.n_components_))
        else:
            comp_indices = np.array(which_components)

        f_idx = self.feature_names_.index(feature_name)
        method = self._validate_method()

        if method == "happ_greven":
            # Score inversion: recover univariate PACE scores for this feature
            # from multivariate scores via the PCA eigenvectors
            pace = self.pace_models_[feature_name]
            n_univ = self.n_components_per_feature_[f_idx]
            # z_l^(q) = s_q * (pca_mean_l^(q) + sum_k rho_k c_kl^(q))
            univ_scores = self.inverse_transform_scores(scores, list(comp_indices))[feature_name]

            # Reconstruct each subject's curve on the feature's own native grid
            feature_grid = pace.grid_
            n_timepoints = len(feature_grid)
            values = np.zeros((n_subjects, n_timepoints))

            for i in range(n_subjects):
                recon_vals, _ = pace.reconstruct_curve(
                    feature_grid,
                    univ_scores[i],
                    n_components=n_univ
                )
                values[i, :] = recon_vals

            if single_subject:
                return values[0], feature_grid.copy()
            return values, feature_grid.copy()

        else:
            # Berrendero: slice the full multivariate reconstruction.
            # reconstruct_from_scores returns (T, F) for a single subject
            # and (N, T, F) for multiple subjects — index the feature axis directly.
            full_recon = self.reconstruct_from_scores(scores, which_components)
            if single_subject:
                values = full_recon[:, f_idx]       # (T, F) → (T,)
            else:
                values = full_recon[:, :, f_idx]    # (N, T, F) → (N, T)

            if single_subject:
                return values.ravel(), self.grid_.copy()
            return values, self.grid_.copy()

    # Getter methods for model outputs

    def get_time_grid(self) -> np.ndarray:
        """
        Return the display grid.

        Returns
        -------
        grid : ndarray, shape (n_timepoints,)
            The grid ``grid_``.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.grid_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.grid_

    def get_mean_functions(self) -> Dict[str, np.ndarray]:
        """
        Return the mean function of each feature.

        Returns
        -------
        mean_functions : dict of str to ndarray
            Mean function of each feature, evaluated on ``grid_``.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.mean_functions_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.mean_functions_

    def get_weight_functions(self) -> np.ndarray:
        """
        Return the multivariate eigenfunctions.

        Returns
        -------
        weight_functions : ndarray, shape (n_components_, n_timepoints, n_features)
            ``weight_functions[k, t, f]`` is the loading of feature f at time t on
            component k, in the stored (rescaled) units. See the class docstring.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.weight_functions_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.weight_functions_

    def get_eigenvalues(self) -> np.ndarray:
        """
        Return the global eigenvalues of the components.

        Returns
        -------
        eigenvalues : ndarray, shape (n_components_,)
            Eigenvalues in descending order, as described in the class docstring.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.eigenvalues_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.eigenvalues_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Return the proportion of variance explained by each component.

        Returns
        -------
        explained_variance_ratio : ndarray, shape (n_components_,)
            Proportions of variance explained. For ``"berrendero"`` they sum to 1
            over the retained components. For ``"happ_greven"`` they are relative
            to the total variance of all univariate score columns, so they sum to
            1 only when every available component is retained.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.explained_variance_ratio_

    def get_feature_weight_function(
        self,
        feature_name: str,
        descale: bool = False
    ) -> np.ndarray:
        """
        Return the eigenfunction values of one feature.

        For component k this returns the loading of feature ``feature_name`` at
        each grid point, that is, the values of the k-th multivariate
        eigenfunction restricted to this feature, on ``grid_``. For
        ``"happ_greven"`` these values are a linear combination of the feature's
        univariate eigenfunctions, with coefficients taken from the joint PCA.

        Parameters
        ----------
        feature_name : str
            Name of the feature. Must be in ``feature_names_``.
        descale : bool, default=False
            If True and ``scale_features`` was used, multiply by s_j to return the
            values to original units. Ignored when ``scale_features`` is False.

        Returns
        -------
        feature_weights : ndarray, shape (n_components_, n_timepoints)
            ``feature_weights[k, t]`` is the value of eigenfunction k for this
            feature at time t on ``grid_``.

        Raises
        ------
        ValueError
            If the model has not been fitted, or ``feature_name`` is not in
            ``feature_names_``.
        """
        if self.weight_functions_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if feature_name not in self.feature_names_:
            raise ValueError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {self.feature_names_}"
            )

        f_idx = self.feature_names_.index(feature_name)
        # weight_functions_ shape: (n_components_, n_timepoints, n_features)
        feature_weights = self.weight_functions_[:, :, f_idx].copy()

        if descale and self.scale_features and self.scale_stds_ is not None:
            feature_weights *= self.scale_stds_[f_idx]

        return feature_weights

    # Private helper methods

    def _compute_happ_greven_weight_functions(self) -> None:
        """
        Compute multivariate eigenfunctions in function space for Happ-Greven.

        Maps PCA eigenvectors ĉₖ from the concatenated (scaled) score space back
        to function space. For feature j with M_j retained univariate components:

            weight_functions_[k, :, j] = Σₗ [ĉₖ]ₗ^(j) φₗ^(j)(t)

        This is stored in the wⱼ^{1/2}-scaled space. During reconstruction,
        _undo_scaling multiplies by sⱼ = wⱼ^{-1/2} to recover the true weighted
        multivariate eigenfunction from Happ & Greven (2018) Section 3.2 Extensions:

            ψ̂ₖ^(j)(t) = wⱼ^{-1/2} Σₗ [ĉₖ]ₗ^(j) φₗ^(j)(t)

        with wⱼ = (Σₗ λₗ^(j))⁻¹ from Section 5.

        Eigenfunctions of features whose PACE grid differs from self.grid_ are
        interpolated to the display grid before projection.

        Also computes pca_mean_offset_: the PCA column-mean mapped to function
        space for exact reconstruction (accounts for sklearn PCA's internal centering).

        Sets
        ----
        self.weight_functions_ : np.ndarray
            Shape (n_components_, n_timepoints, n_features). Stored in the
            wⱼ^{1/2}-scaled space; multiply by scale_stds_[f] to get true ψ̂ₖ^(j).
        self.pca_mean_offset_ : np.ndarray
            Shape (n_timepoints, n_features). PCA mean offset in function space.
        """
        n_timepoints = len(self.grid_)
        n_features = len(self.feature_names_)

        # Initialize weight functions and PCA mean offset
        weight_functions = np.zeros((self.n_components_, n_timepoints, n_features))
        pca_mean_offset = np.zeros((n_timepoints, n_features))

        # Build cumulative offsets for splitting the concatenated score vector
        # into per-feature blocks
        offsets = np.cumsum([0] + self.n_components_per_feature_)

        for f, feat in enumerate(self.feature_names_):
            pace = self.pace_models_[feat]
            n_univ = self.n_components_per_feature_[f]

            # Univariate eigenfunctions: interpolate to display grid if needed.
            # Each PACE model may have its own grid (H-G allows different domains;
            # Happ & Greven 2018 Section 1). The display grid self.grid_ comes from
            # the first PACE model; other features are interpolated here.
            phi_q_raw = pace.eigenfunction_values_[:n_univ, :]  # (n_univ, n_timepoints_q)
            if not np.array_equal(pace.grid_, self.grid_):
                n_display = len(self.grid_)
                phi_q = np.zeros((n_univ, n_display))
                for comp in range(n_univ):
                    phi_q[comp] = np.interp(self.grid_, pace.grid_, phi_q_raw[comp])
            else:
                phi_q = phi_q_raw

            # PCA mean block for this feature
            # pca_model_.mean_ shape: (total_univ_components,)
            mean_block = self.pca_model_.mean_[offsets[f]:offsets[f + 1]]
            # mean_block shape: (n_univ,)

            # Map PCA mean to function space: m_tilde_q(t) = sum_l z_bar_l^(q) * phi_ql(t)
            # (n_univ,) @ (n_univ, n_timepoints) -> (n_timepoints,)
            pca_mean_offset[:, f] = mean_block @ phi_q

            # For each multivariate component k, compute psi_qk(t)
            for k in range(self.n_components_):
                # Extract PCA eigenvector block for this feature
                # score_pca_components_ shape: (n_components_, total_univ_components)
                c_block = self.score_pca_components_[k, offsets[f]:offsets[f + 1]]
                # c_block shape: (n_univ,)

                # psi_qk(t) = sum_l c_kl^(q) * phi_ql(t)
                # (n_univ,) @ (n_univ, n_timepoints) -> (n_timepoints,)
                weight_functions[k, :, f] = c_block @ phi_q

        self.weight_functions_ = weight_functions
        self.pca_mean_offset_ = pca_mean_offset

    def _validate_and_filter_subjects(
        self,
        data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
    ) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], int, int]:
        """
        Validate data format and filter out subjects with incomplete observations.

        Subjects must have at least one observation for ALL features to be included.
        This ensures all subjects contribute to multivariate covariance estimation.

        Parameters
        ----------
        data : dict
            Raw multivariate data.

        Returns
        -------
        filtered_data : dict
            Data with only complete subjects.
        n_valid : int
            Number of valid subjects.
        n_dropped : int
            Number of dropped subjects.

        Raises
        ------
        ValueError
            If data is not a dict or missing required features.
        ValueError
            If features have different numbers of subjects.
        """
        # Validate data is a dict
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary mapping feature names to subject lists")

        # Check all required features are present
        missing_features = set(self.feature_names_) - set(data.keys())
        if missing_features:
            raise ValueError(
                f"data is missing required features: {missing_features}. "
                f"Expected features: {self.feature_names_}"
            )

        # Check all features have same number of subjects
        feature_lengths = {feat: len(data[feat]) for feat in self.feature_names_}
        if len(set(feature_lengths.values())) > 1:
            raise ValueError(
                f"All features must have the same number of subjects. "
                f"Got lengths: {feature_lengths}"
            )

        n_total = feature_lengths[self.feature_names_[0]]

        # Identify valid subjects (have observations for ALL features)
        valid_indices = []
        for i in range(n_total):
            is_valid = True
            for feat in self.feature_names_:
                if i >= len(data[feat]):
                    is_valid = False
                    break
                times, values = data[feat][i]
                if times is None or values is None or len(times) == 0 or len(values) == 0:
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(i)

        # Build filtered data
        filtered_data = {}
        for feat in self.feature_names_:
            filtered_data[feat] = [data[feat][i] for i in valid_indices]

        n_valid = len(valid_indices)
        n_dropped = n_total - n_valid

        return filtered_data, n_valid, n_dropped

    def _validate_transform_data(self, data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> None:
        """
        Validate that all subjects have complete observations for transform.

        Transform requires complete feature observations for all subjects because
        multivariate scores depend on all features. Unlike fit(), this method
        raises an error rather than filtering.

        Parameters
        ----------
        data : dict
            Multivariate data to validate.

        Raises
        ------
        ValueError
            If any subject is missing observations for any feature.
        """
        # Check all required features are present
        missing_features = set(self.feature_names_) - set(data.keys())
        if missing_features:
            raise ValueError(
                f"data is missing required features: {missing_features}. "
                f"Expected features: {self.feature_names_}"
            )

        # Check all features have same number of subjects
        feature_lengths = {feat: len(data[feat]) for feat in self.feature_names_}
        if len(set(feature_lengths.values())) > 1:
            raise ValueError(
                f"All features must have the same number of subjects. "
                f"Got lengths: {feature_lengths}"
            )

        n_subjects = feature_lengths[self.feature_names_[0]]

        # Check each subject has observations for all features
        invalid_indices = []
        for i in range(n_subjects):
            for feat in self.feature_names_:
                times, values = data[feat][i]
                if times is None or values is None or len(times) == 0 or len(values) == 0:
                    invalid_indices.append(i)
                    break

        if invalid_indices:
            # Show first 10 invalid indices
            shown_indices = invalid_indices[:10]
            msg = (
                f"Cannot transform subjects with missing feature observations. "
                f"Found {len(invalid_indices)} subjects missing data: {shown_indices}"
            )
            if len(invalid_indices) > 10:
                msg += "... (showing first 10). "
            msg += f"\nAll subjects must have observations for all features: {self.feature_names_}"
            raise ValueError(msg)

    def _validate_method(self) -> str:
        """Return ``method`` after checking that it is supported."""
        if self.method not in {"berrendero", "happ_greven"}:
            raise ValueError(
                f"Invalid method '{self.method}'. Use 'berrendero' or 'happ_greven'."
            )
        return self.method

    def _resolve_n_components_per_feature(self) -> List[int]:
        """Return the requested univariate component count for each feature.

        Features absent from a dict argument, and every feature when the
        argument is None, use a count of 5.
        """
        if self.n_components_per_feature is None:
            default_count = 5
            return [default_count for _ in self.feature_names_]

        if isinstance(self.n_components_per_feature, int):
            if self.n_components_per_feature <= 0:
                raise ValueError("n_components_per_feature must be positive.")
            return [self.n_components_per_feature for _ in self.feature_names_]

        if not isinstance(self.n_components_per_feature, dict):
            raise ValueError(
                "n_components_per_feature must be an int or a dict mapping feature names to counts."
            )

        counts = []
        for feat in self.feature_names_:
            count = self.n_components_per_feature.get(feat, 5)
            if count <= 0:
                raise ValueError(f"n_components_per_feature for '{feat}' must be positive.")
            counts.append(count)
        return counts

    def _build_scores_matrix(
        self,
        data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
        component_counts: List[int]
    ) -> np.ndarray:
        """Compute univariate scores with each PACE model and assemble them."""
        scores_by_feat = {
            feat: self.pace_models_[feat].transform(data[feat])
            for feat in self.feature_names_
        }
        return self._assemble_scores_matrix(scores_by_feat, component_counts)

    def _assemble_scores_matrix(
        self,
        scores_by_feat: Dict[str, np.ndarray],
        component_counts: List[int]
    ) -> np.ndarray:
        """Truncate each feature's univariate scores to component_counts[f],
        scale by scale_stds_ (if scale_features), and concatenate in
        feature_names_ order. Shared tail end of _build_scores_matrix (which
        computes scores_by_feat from raw trajectories) and
        transform_from_scores (which takes scores_by_feat directly)."""
        n_subjects = len(next(iter(scores_by_feat.values())))
        score_blocks = []

        for f, feat in enumerate(self.feature_names_):
            scores = scores_by_feat[feat]
            max_available = scores.shape[1]
            requested = component_counts[f]

            if requested > max_available:
                warnings.warn(
                    f"Requested {requested} components for '{feat}', but only {max_available} are available. "
                    f"Using {max_available}.",
                    UserWarning, stacklevel=3,
                )
                requested = max_available

            block = scores[:, :requested]
            if self.scale_features:
                block = block / self.scale_stds_[f]
            score_blocks.append(block)

        if not score_blocks:
            return np.zeros((n_subjects, 0))

        return np.concatenate(score_blocks, axis=1)

    def transform_from_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute multivariate scores from pre-computed univariate scores.

        This method bypasses the trajectory input of ``transform``. The caller
        supplies, for each feature, the univariate PACE scores directly. For
        example, these can be a per-unit average of several score vectors. Only
        ``method="happ_greven"`` is supported, because the ``"berrendero"`` method
        works on reconstructed trajectories and has no scores-only equivalent.

        Parameters
        ----------
        scores : dict of str to ndarray
            Maps each feature in ``feature_names_`` to an array of shape
            (n_units, n_scores). The arrays contain raw, unscaled univariate PACE
            scores, as ``pace_models_[feature].transform`` would return them. All
            arrays must have the same number of rows, and each must have at least
            ``n_components_per_feature_[f]`` columns; only that many leading
            columns are used.

        Returns
        -------
        scores : ndarray, shape (n_units, n_components_)
            Multivariate scores.

        Raises
        ------
        ValueError
            If the model has not been fitted, ``method`` is not
            ``"happ_greven"``, a feature is missing from ``scores``, or the arrays
            do not share the same number of rows.
        """
        if self.grid_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        method = self._validate_method()
        if method != "happ_greven":
            raise ValueError(
                "transform_from_scores() is only supported for method='happ_greven' "
                "(the 'berrendero' method requires trajectory reconstruction, not "
                "just per-feature scores)."
            )
        if self.pca_model_ is None or self.n_components_per_feature_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        missing = [f for f in self.feature_names_ if f not in scores]
        if missing:
            raise ValueError(f"Missing scores for feature(s): {missing}")
        n_units_set = {scores[f].shape[0] for f in self.feature_names_}
        if len(n_units_set) > 1:
            raise ValueError(f"All features must have the same number of units; got {n_units_set}")

        scores_matrix = self._assemble_scores_matrix(scores, self.n_components_per_feature_)
        return self.pca_model_.transform(scores_matrix)


    def _establish_common_grid(self) -> None:
        """
        Establish the time grid from PACE models.

        For Berrendero: all PACE models must share an identical time grid (required
        for pointwise covariance estimation). Raises ValueError with a message
        suggesting happ_greven if grids differ.

        For Happ-Greven: only checks that models are fitted. self.grid_ is set to
        the first model's grid and used as a display grid for weight_functions_
        visualization. Features with different grids have their eigenfunctions
        interpolated in _compute_happ_greven_weight_functions; the core PCA and
        scoring are fully grid-independent (Happ & Greven 2018, Section 1 and
        Section 3.2).

        Raises
        ------
        ValueError
            If any PACE model is not fitted, or if Berrendero is used with
            mismatched grids.
        """
        first_feature = self.feature_names_[0]
        first_pace = self.pace_models_[first_feature]

        if not hasattr(first_pace, 'grid_') or first_pace.grid_ is None:
            raise ValueError(f"PACE model for '{first_feature}' has not been fitted yet.")

        self.grid_ = first_pace.grid_.copy()

        method = self._validate_method()

        for feat in self.feature_names_[1:]:
            pace = self.pace_models_[feat]

            if not hasattr(pace, 'grid_') or pace.grid_ is None:
                raise ValueError(f"PACE model for '{feat}' has not been fitted yet.")

            if method == "berrendero":
                if len(pace.grid_) != len(self.grid_) or not np.allclose(pace.grid_, self.grid_):
                    raise ValueError(
                        f"Berrendero method requires all PACE models to share an identical "
                        f"time grid. '{first_feature}' and '{feat}' have different grids. "
                        f"For features on different time grids, use method='happ_greven'."
                    )
        # For happ_greven: self.grid_ serves as the display grid. Eigenfunctions
        # of features with different grids are interpolated to it only for
        # weight_functions_ storage; transform() and scoring use each model's own grid.

    def _compute_multivariate_means(self) -> None:
        """
        Compute multivariate mean functions from PACE models.

        Extracts the univariate mean function from each fitted PACE model.
        These are the feature-wise mean functions on the common time grid.

        Notes
        -----
        Mean functions are taken directly from fitted PACE models.
        """
        self.mean_functions_ = {}

        for feat in self.feature_names_:
            pace = self.pace_models_[feat]

            if not hasattr(pace, 'mean_values_') or pace.mean_values_ is None:
                raise ValueError(f"PACE model for '{feat}' does not have mean_values_.")

            # For H-G with different grids, interpolate to the display grid so that
            # _build_mean_array (which allocates shape (len(self.grid_), n_features))
            # can assign each feature's mean without a shape mismatch.
            if not np.array_equal(pace.grid_, self.grid_):
                self.mean_functions_[feat] = np.interp(
                    self.grid_, pace.grid_, pace.mean_values_
                )
            else:
                self.mean_functions_[feat] = pace.mean_values_.copy()

    def _reconstruct_multivariate_trajectories(
        self,
        data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
    ) -> np.ndarray:
        """
        Reconstruct smooth multivariate trajectories on common grid using PACE.

        Uses conditional expectation (PACE projection) to reconstruct each subject's
        trajectory for each feature on the common time grid.

        Parameters
        ----------
        data : dict
            Multivariate data (assumes validation already done).

        Returns
        -------
        trajectories : np.ndarray
            Shape (n_subjects, n_timepoints, n_features).
            trajectories[i, t, f] is the reconstructed value for subject i,
            time point t, feature f.

        Notes
        -----
        Uses PACE scores and reconstruct_curve() to generate smooth trajectories
        on the common grid.
        """
        n_subjects = len(data[self.feature_names_[0]])
        n_timepoints = len(self.grid_)
        n_features = len(self.feature_names_)

        # Initialize trajectory array
        trajectories = np.zeros((n_subjects, n_timepoints, n_features))

        # Reconstruct for each feature
        for f, feat in enumerate(self.feature_names_):
            pace = self.pace_models_[feat]
            subject_list = data[feat]

            # Get PACE scores for all subjects
            scores = pace.transform(subject_list)

            # Reconstruct each subject's trajectory on common grid using PACE
            for i in range(n_subjects):

                # Use PACE's reconstruct_curve method
                # This properly handles all components fitted by PACE
                reconstructed_values, reconstructed_times = pace.reconstruct_curve(
                    self.grid_,
                    scores[i],
                    n_components=None  # Use all fitted components
                )

                # Interpolate to common grid (reconstruct_curve returns combined grid)
                trajectories[i, :, f] = np.interp(
                    self.grid_,
                    reconstructed_times,
                    reconstructed_values
                )

        return trajectories

    def _center_trajectories(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Center multivariate trajectories by subtracting mean functions.

        Parameters
        ----------
        trajectories : np.ndarray
            Uncentered trajectories, shape (n_subjects, n_timepoints, n_features).

        Returns
        -------
        centered : np.ndarray
            Centered trajectories, same shape as input.
        """
        mean_array = self._build_mean_array()  # shape: (n_timepoints, n_features)
        # Broadcasting: (n_subjects, n_timepoints, n_features) - (n_timepoints, n_features)
        return trajectories - mean_array

    def _build_mean_array(self) -> np.ndarray:
        """
        Build mean array from mean_functions_ dict.

        Returns
        -------
        mean_array : np.ndarray
            Shape (n_timepoints, n_features), where mean_array[:, f] is the
            mean function for feature f.
        """
        n_timepoints = len(self.grid_)
        n_features = len(self.feature_names_)
        mean_array = np.zeros((n_timepoints, n_features))

        for f, feat in enumerate(self.feature_names_):
            mean_array[:, f] = self.mean_functions_[feat]

        return mean_array

    def _compute_feature_scaling(self, trajectories: np.ndarray) -> None:
        """
        Compute per-feature scaling factors from PACE eigenvalues.

        Sets scale_stds_[f] = sⱼ = √(Σₗ λₗ^(j)), where Σₗ λₗ^(j) is the total
        variance of feature j. This implements wⱼ^{1/2} = 1/sⱼ, the weight from
        Happ & Greven (2018) Section 5: wⱼ = (∫ var(X^(j)(t)) dt)⁻¹ = 1/sⱼ².

        Parameters
        ----------
        trajectories : np.ndarray
            Not used; kept for API consistency with the Berrendero path.
        """
        n_features = len(self.feature_names_)
        scale_stds = np.ones(n_features)

        for f, feat in enumerate(self.feature_names_):
            pace = self.pace_models_[feat]
            if not hasattr(pace, "eigenvalues_") or pace.eigenvalues_ is None:
                raise ValueError(f"PACE model for '{feat}' does not have eigenvalues_.")
            total_variance = float(np.sum(pace.eigenvalues_))
            scale_stds[f] = np.sqrt(total_variance) if total_variance > 1e-12 else 1.0

        self.scale_stds_ = scale_stds

    def _apply_scaling(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Apply feature-wise scaling to trajectories.

        Parameters
        ----------
        trajectories : np.ndarray
            Input trajectories, shape (n_subjects, n_timepoints, n_features).

        Returns
        -------
        scaled : np.ndarray
            Scaled trajectories, same shape as input.
        """
        if self.scale_stds_ is None:
            raise ValueError("Scaling parameters are not available. Fit the model first.")

        scaled = trajectories.copy()
        for f in range(len(self.feature_names_)):
            scaled[:, :, f] = scaled[:, :, f] / self.scale_stds_[f]
        return scaled

    def _undo_scaling(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Undo feature-wise scaling on trajectories.

        Parameters
        ----------
        trajectories : np.ndarray
            Scaled trajectories, shape (n_subjects, n_timepoints, n_features).

        Returns
        -------
        unscaled : np.ndarray
            Unscaled trajectories in original units.
        """
        if self.scale_stds_ is None:
            raise ValueError("Scaling parameters are not available. Fit the model first.")

        unscaled = trajectories.copy()
        for f in range(len(self.feature_names_)):
            unscaled[:, :, f] = unscaled[:, :, f] * self.scale_stds_[f]
        return unscaled

    def _estimate_time_wise_covariances(self, trajectories: np.ndarray) -> List[np.ndarray]:
        """
        Estimate time-wise covariance matrices.

        Following Berrendero method: estimates covariance at each discrete time
        point across features. This gives a sequence of covariance matrices
        indexed by time.

        Parameters
        ----------
        trajectories : np.ndarray
            Centered trajectories, shape (n_subjects, n_timepoints, n_features).

        Returns
        -------
        time_covariances : list of np.ndarray
            List of length n_timepoints, where time_covariances[t] is the
            empirical covariance matrix across features at time point t,
            shape (n_features, n_features).

        Notes
        -----
        Estimates covariance at each discrete time point across features.
        """
        n_subjects, n_timepoints, n_features = trajectories.shape
        time_covariances = []

        for t in range(n_timepoints):
            # Extract data at time t: shape (n_subjects, n_features)
            data_t = trajectories[:, t, :]

            # Compute empirical covariance: (n_features, n_features)
            # rowvar=False means each column is a variable (feature)
            cov_t = np.cov(data_t, rowvar=False)

            # Handle edge case: single feature results in scalar
            if n_features == 1:
                cov_t = np.atleast_2d(cov_t)

            time_covariances.append(cov_t)

        return time_covariances

    def _smooth_covariances(self, time_covariances: List[np.ndarray]) -> List[np.ndarray]:
        """
        Smooth time-wise covariance matrices using B-spline smoothing.

        For each element (i, j) in the covariance matrix, smooths the function
        Cov(feature_i, feature_j, t) across time using B-splines.

        Parameters
        ----------
        time_covariances : list of np.ndarray
            Raw time-wise covariance matrices.

        Returns
        -------
        smoothed_covariances : list of np.ndarray
            Smoothed time-wise covariance matrices, same structure as input.

        Notes
        -----
        Smooths each of the p² covariance functions independently along the time
        dimension. Uses scipy's UnivariateSpline with the smoothing parameter
        specified in self.smooth_cov.
        """
        from scipy.interpolate import UnivariateSpline

        n_timepoints = len(time_covariances)
        n_features = time_covariances[0].shape[0]

        # Convert list of matrices to 3D array for easier manipulation
        # Shape: (n_timepoints, n_features, n_features)
        cov_array = np.array(time_covariances)

        # Create smoothed array
        smoothed_array = np.zeros_like(cov_array)

        # Smooth each element (i, j) across time
        for i in range(n_features):
            for j in range(n_features):
                # Extract time series for this covariance element
                cov_ij_t = cov_array[:, i, j]

                # Fit B-spline with smoothing
                # s parameter controls smoothing (0 = interpolation, larger = more smoothing)
                spline = UnivariateSpline(self.grid_, cov_ij_t, s=self.smooth_cov, k=3)

                # Evaluate smoothed function at grid points
                smoothed_array[:, i, j] = spline(self.grid_)

        # Ensure symmetry is preserved (average (i,j) and (j,i) after smoothing)
        for i in range(n_features):
            for j in range(i+1, n_features):
                avg = (smoothed_array[:, i, j] + smoothed_array[:, j, i]) / 2
                smoothed_array[:, i, j] = avg
                smoothed_array[:, j, i] = avg

        # Convert back to list of matrices
        smoothed_covariances = [smoothed_array[t] for t in range(n_timepoints)]

        return smoothed_covariances

    def _extract_global_eigenfunctions(self, time_covariances: List[np.ndarray]) -> None:
        """
        Extract global multivariate eigenfunctions and eigenvalues (Berrendero method).

        Performs a local PCA at each time point, aligns eigenvector signs for
        temporal continuity, then assembles time-varying weight functions and
        computes global eigenvalues by time integration.

        Parameters
        ----------
        time_covariances : list of np.ndarray
            Time-wise covariance matrices, length n_timepoints.

        Notes
        -----
        ``np.linalg.eigh`` returns eigenvectors defined up to sign. A temporal
        sweep flips the sign of the eigenvector at time t when its dot product
        with the eigenvector at time t-1 is negative, which is the sign convention
        used by Happ and Greven (2018) for consistent eigenfunction estimation.

        Limitation: if two eigenvalues cross rank order at some time point, the
        alignment may still produce a discontinuity at the crossing, an inherent
        limitation of the pointwise PCA approach.
        """
        n_timepoints = len(time_covariances)
        n_features = len(self.feature_names_)

        # Determine number of components
        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = min(self.n_components, n_features)

        # Storage for local eigendecompositions
        local_eigenvectors = np.zeros((n_timepoints, n_features, n_features))
        local_eigenvalues = np.zeros((n_timepoints, n_features))

        # Perform eigen-decomposition at each time
        for t in range(n_timepoints):
            cov_t = time_covariances[t]

            # Eigen-decomposition (eigh assumes symmetric matrix)
            eigvals, eigvecs = np.linalg.eigh(cov_t)

            # Sort in descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            local_eigenvalues[t] = eigvals
            local_eigenvectors[t] = eigvecs

        # Sign alignment: eigh returns eigenvectors defined up to sign. Without
        # alignment, adjacent time points may have opposite signs, producing
        # discontinuous weight functions and near-zero time-integrated scores.
        # Flip sign at time t when the dot product with t-1 is negative
        # (the sign convention used by Happ and Greven (2018)).
        # Limitation: if two eigenvalues cross rank order at some t, alignment
        # may still produce a jump at the crossing — an inherent limit of
        # pointwise-PCA.
        for t in range(1, n_timepoints):
            for k in range(n_features):
                if np.dot(local_eigenvectors[t, :, k], local_eigenvectors[t - 1, :, k]) < 0:
                    local_eigenvectors[t, :, k] *= -1

        # Assemble weight functions: for each component, stack eigenvector over time
        # weight_functions[k, t, :] = local_eigenvectors[t, :, k]
        weight_functions = np.zeros((self.n_components_, n_timepoints, n_features))

        for k in range(self.n_components_):
            for t in range(n_timepoints):
                weight_functions[k, t, :] = local_eigenvectors[t, :, k]

        # Compute global eigenvalues via time integration
        # Use trapezoidal rule for discrete integration
        global_eigenvalues = np.zeros(self.n_components_)
        for k in range(self.n_components_):
            # Integrate local eigenvalues for component k over time
            local_eigs_k = local_eigenvalues[:, k]
            global_eigenvalues[k] = _trapezoid(local_eigs_k, self.grid_)

        # Normalize weight functions by time-integrated L2 norm
        for k in range(self.n_components_):
            # Compute L2 norm: integral of ||weight_function[k, t, :]||^2 dt
            norms_squared = np.sum(weight_functions[k]**2, axis=1)  # ||w[k,t]||^2 for each t
            norm_integrated = _trapezoid(norms_squared, self.grid_)

            if norm_integrated > 1e-10:  # avoid division by zero
                weight_functions[k] /= np.sqrt(norm_integrated)

        # Store results
        self.weight_functions_ = weight_functions
        self.eigenvalues_ = global_eigenvalues

    def _compute_multivariate_scores(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Compute multivariate scores via inner products with weight functions.

        Projects new data onto multivariate eigenfunctions via time-integrated
        inner product. For each subject and component:
            score[i, k] = ∫ <trajectory[i, t, :], weight[k, t, :]> dt

        Parameters
        ----------
        trajectories : np.ndarray
            Centered trajectories, shape (n_subjects, n_timepoints, n_features).

        Returns
        -------
        scores : np.ndarray
            Multivariate scores, shape (n_subjects, n_components_).

        Notes
        -----
        Projects data onto multivariate eigenfunctions via time integration.
        """
        n_subjects = trajectories.shape[0]
        scores = np.zeros((n_subjects, self.n_components_))

        for i in range(n_subjects):
            for k in range(self.n_components_):
                # Inner product at each time: dot product across features
                # trajectories[i, t, :] · weight_functions[k, t, :]
                inner_products = np.sum(
                    trajectories[i] * self.weight_functions_[k],
                    axis=1  # sum over features
                )

                # Integrate over time using trapezoidal rule
                scores[i, k] = _trapezoid(inner_products, self.grid_)

        return scores
