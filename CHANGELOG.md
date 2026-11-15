# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `MFPCA.fit_from_scores`: fit the Happ-Greven model from pre-computed univariate scores, the fitting counterpart of `transform_from_scores`.
- `MFPCA.inverse_transform_scores`: map multivariate scores back to each feature's univariate scores (used by `reconstruct_feature`).
- `TwoLevelPACE.level_model` and `LevelModel`: one level of a two-level model (mean, eigenvalues, eigenfunctions) in the form `MFPCA` reads, for multilevel MFPCA run separately at each level.
- `fpca.utils.orient_eigenfunctions`: fix the sign of each eigenfunction (positive mean) for `PACE`, `TwoLevelPACE`, `LevelModel` and `MFPCA`.
- `fpca.utils.curves_from_long`: build `(times, values)` curves from a long table.
- `fpca.stats.permanova_table`: several PERMANOVA/PERMDISP tests on the same scores, as a DataFrame.
- `fpca.plotting.plot_scores_by` and `plot_feature_eigenfunctions`; `plot_scores` colours by a numeric label with `cmap`; `plot_mfpca_eigenfunctions` takes `n_cols`, `labels` and `title` and shows cumulative variance; `plot_modes_of_variation` takes `n_cols`.
- Example notebook `examples/baboon_microbiome.ipynb` with its data (Björk et al., 2022).
- `pandas` as a runtime dependency.

### Changed
- `PACE(n_grid_points=None)` (the default) now uses the union of all observed times as the evaluation grid, with no cap, so no observation is binned; a `UserWarning` is raised above 2000 grid points. `method_smoothing="LP"` keeps its 50-point default cap. The constructor no longer rewrites `n_grid_points`. Times off the grid are handled by linear interpolation; times outside the fitted range raise unless `allow_extrapolation=True`.
- `PACE.fit` accepts curves with a single observation: they inform the mean and the noise variance, a `UserWarning` reports how many there are, and an error is raised only if no curve has two observations. `TwoLevelPACE` accepts them as well.
- `MFPCA` reduces the default of 5 univariate components per feature silently; only an explicit `n_components_per_feature` that exceeds the available components warns.

## [0.1.0] - 2026-11-11

First public release.

### Added
- `PACE`: univariate FPCA by principal analysis by conditional expectation (Yao, Müller & Wang, 2005), with P-spline and local-polynomial smoothing, diagonal-excluded covariance smoothing, quadrature-weighted eigendecomposition and conditional-expectation scores.
- `TwoLevelPACE`: two-level (multilevel) FPCA (Di et al., 2009) with REML-tuned P-spline covariance smoothing, optional visit effects and BLUP scores.
- `MFPCA`: multivariate FPCA on fitted `PACE` models, with the Happ-Greven (2018, default) and Berrendero et al. (2011) methods, `transform_from_scores`, and per-feature reconstruction.
- `fpca.plotting`: modes of variation, raw curves, cumulative variance, reconstructions, reconstruction error, score scatter plots and multivariate eigenfunction / reconstruction / extreme-subject plots.
- `fpca.utils`: `explained_variance_ratio`, `n_components_for_variance`, `reconstruction_rmse`.
- `fpca.stats`: `permanova` (PERMANOVA + PERMDISP with free, within-block and between-block permutation schemes).
