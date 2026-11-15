# functional-pca

**Functional principal component analysis in Python for sparse, irregular, multilevel and multivariate functional data.**

[![Tests](https://github.com/Eyal708/functional-pca/actions/workflows/tests.yml/badge.svg)](https://github.com/Eyal708/functional-pca/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eyal708/functional-pca/blob/main/examples/baboon_microbiome.ipynb)

`functional-pca` (imported as `fpca`) is a small, dependency-light library of FPCA estimators that work directly on raw `(times, values)` observations, with no pre-smoothing or interpolation onto a common grid:

| Estimator | What it does | Reference |
|---|---|---|
| `PACE` | Univariate FPCA by *Principal Analysis by Conditional Expectation*: smoothed mean and covariance, eigenfunctions, and conditional-expectation scores that stay well defined when a curve has only a handful of observations. | Yao, Müller & Wang (2005) |
| `TwoLevelPACE` | Multilevel FPCA for repeated curves per subject (e.g. several visits or days per person). Splits variation into a *between-subject* and a *within-subject* level, each with its own eigenfunctions and scores. | Di, Crainiceanu, Caffo & Punjabi (2009) |
| `MFPCA` | Multivariate FPCA for several signals observed on the same subjects. It finds joint components across signals, built from one fitted `PACE` model per signal, or from the scores of one level of a `TwoLevelPACE` model per signal (multilevel MFPCA). | Happ & Greven (2018); Berrendero, Justel & Svarc (2011) |

It also ships plotting helpers (modes of variation, scores, reconstructions, eigenfunctions of several features, multivariate eigenfunctions), utilities (variance shares, component selection, reconstruction error, sign orientation of eigenfunctions, long-table-to-curves conversion) and a fast PERMANOVA/PERMDISP permutation test for comparing groups of curves in score space.

The code is domain-agnostic: a "curve" can be a patient's lab values over time, a sensor signal over a day, a growth curve, a spectrum, or any other one-dimensional function observed with noise.

For an introduction to functional data analysis and functional PCA, see Ramsay & Silverman (2005); the method papers of each estimator are listed under [References](#references).

---

## Installation

The package is installed from this repository (it is not on PyPI). Python 3.10 or newer is required.

```bash
# Install straight from GitHub
pip install git+https://github.com/Eyal708/functional-pca.git

# Or clone it, e.g. to run the example notebooks or modify the code
git clone https://github.com/Eyal708/functional-pca.git
cd functional-pca
pip install -e ".[examples]"     # editable install + notebook dependencies
```

Runtime dependencies are NumPy (1.x or 2.x), SciPy, scikit-learn, pandas and Matplotlib (`requirements.txt` lists them for `pip install -r`). The `examples` extra adds Jupyter, for the example notebook. The `dev` extra adds pytest and ruff.

## Quick start

```python
import numpy as np
from fpca import PACE
from fpca.plotting import plot_modes_of_variation

# Sparse, irregular data: each curve has its own 5-15 observation times
rng = np.random.default_rng(0)
curves = []
for _ in range(200):
    t = np.sort(rng.uniform(0, 10, rng.integers(5, 16)))
    xi1, xi2 = rng.normal(0, [2.0, 1.0])
    y = np.sin(t / 2) + xi1 * np.cos(t / 5) + xi2 * np.sin(t / 3) + rng.normal(0, 0.3, t.size)
    curves.append((t, y))

model = PACE(n_components=3, method_smoothing="LP").fit(curves)

scores = model.transform(curves)                 # (200, 3) FPC scores
print(model.eigenvalues_ / model.eigenvalues_.sum())

values, grid = model.reconstruct_curve(curves[0][0], scores[0])   # denoised curve 0
plot_modes_of_variation(model, n_components=3)
```

## Data format

Every estimator takes the same plain-Python input, with no special container classes:

```python
curve  = (times, values)              # two 1-D arrays of equal length
curves = [curve_1, curve_2, ...]      # one entry per curve; times may differ per curve
```

| Estimator | Input |
|---|---|
| `PACE.fit(curves)` | list of curves |
| `TwoLevelPACE.fit(curves, subject_ids, visit_ids=None)` | list of curves + one subject label per curve (+ optional visit label) |
| `MFPCA.fit(data)` | `{feature_name: curves}`; the *i*-th curve of every feature belongs to the same subject |
| `MFPCA.fit_from_scores(scores)` | `{feature_name: score matrix}`; row *i* of every matrix belongs to the same unit |

Dense data (every curve on the same grid) is just a special case. Unsorted times are sorted, duplicate times are averaged, and both raise a warning rather than failing silently. A curve with a single observation is allowed: it informs the mean and the noise variance (a warning says it adds nothing to the covariance surface) and is scored like any other curve.

Data in a long table (one row per observation) is converted with `curves_from_long(frame, value_col, curve_col, time_col)`, which returns the curve labels and the list of curves.

## Usage

### Univariate FPCA: `PACE`

```python
from fpca import PACE

model = PACE(
    n_components=None,          # components used by default for scores/plots (None = all retained)
    variance_threshold=0.99,    # how many components are computed and retained
    method_smoothing="PS",      # "PS" (P-splines, default) or "LP" (local polynomial)
).fit(curves)

model.grid_, model.mean_values_            # mean function on the evaluation grid
model.eigenvalues_                         # retained eigenvalues (descending)
model.eigenfunction_values_                # (n_components, len(grid_))
model.noise_variance_                      # measurement-error variance

scores = model.transform(new_curves, n_components=3)
values, t = model.reconstruct_curve(times, scores[0])
model.plot_mean(); model.plot_eigenfunctions()
```

`variance_threshold` controls how many components are *computed*. `n_components` controls how many are *used* by default, and every method also accepts a per-call `n_components`. Requests above the number available are clamped with a warning, never silently. The class docstring (`help(PACE)`) explains every parameter, the smoothing choices and their computational cost in detail.

### Multilevel FPCA: `TwoLevelPACE`

```python
from fpca import TwoLevelPACE

ml = TwoLevelPACE(variance_threshold=0.9).fit(curves, subject_ids)
s = ml.transform(curves, subject_ids)
s["between"]        # (n_subjects, N1): one score vector per subject (stable "trait")
s["within"]         # (n_curves, N2):   one score vector per curve (curve-specific "state")
ml.plot_eigenfunctions(n_between=3, n_within=3)
ml.plot_eigenvalues()
```

Set `visit_effects=True` and pass `visit_ids` to model a visit-specific mean shift (e.g. a systematic Monday vs. Sunday effect).

### Multivariate FPCA: `MFPCA`

```python
from fpca import PACE, MFPCA

data = {"signal_a": curves_a, "signal_b": curves_b}
univariate = {name: PACE(n_components=5).fit(c) for name, c in data.items()}

mfpca = MFPCA(univariate, n_components=3)        # method="happ_greven" (default)
joint_scores = mfpca.fit_transform(data)                         # (n_subjects, 3)
values, t = mfpca.reconstruct_feature(joint_scores[0], "signal_a")
```

| `method` | When to use |
|---|---|
| `"happ_greven"` (default) | Features may live on **different grids or domains**. Works on concatenated univariate PACE scores. |
| `"berrendero"` | All features share **one identical grid**. Uses pointwise (time-wise) cross-feature covariances. |

With `scale_features=True` (default), each feature is standardised by its total PACE variance, so signals in different units contribute comparably.

**From scores, and per level of a two-level model.** `fit` computes the univariate scores itself from raw curves. `fit_from_scores` takes them directly and runs the same Happ-Greven steps, with `transform_from_scores` as its scoring counterpart. This is how multilevel MFPCA is done: fit one `TwoLevelPACE` per signal, then combine each level separately. `level_model("between" | "within")` exposes one level's mean (zero for the within level), eigenvalues and eigenfunctions in the form `MFPCA` reads:

```python
from fpca import TwoLevelPACE, MFPCA

two_level = {name: TwoLevelPACE().fit(c, subject_ids) for name, c in data.items()}
level_scores = {name: m.transform(data[name], subject_ids) for name, m in two_level.items()}

between = MFPCA({n: m.level_model("between") for n, m in two_level.items()})
between.fit_from_scores({n: s["between"] for n, s in level_scores.items()})       # one row per subject
joint_between = between.transform_from_scores({n: s["between"] for n, s in level_scores.items()})
between.inverse_transform_scores(joint_between, which_components=3)              # back to univariate scores
```

### Plotting, utilities and tests

```python
from fpca import (plotting, reconstruction_rmse, n_components_for_variance, orient_eigenfunctions,
                  permanova, permanova_table)

plotting.plot_curves(curves)                          # raw data
plotting.plot_modes_of_variation(model)               # mean ± 2 SD along each component
plotting.plot_cumulative_variance({"A": m1, "B": m2})
plotting.plot_reconstruction(model, *curves[0])       # K = 0, 1, 2, 3 components
plotting.plot_reconstruction_error(reconstruction_rmse(model, curves))
plotting.plot_scores(scores, labels=groups)            # or a numeric label with cmap="viridis"
plotting.plot_scores_by(scores, {"group": groups, "age": age}, continuous=["age"])
plotting.plot_feature_eigenfunctions(univariate)      # features x components grid
plotting.plot_mfpca_eigenfunctions(mfpca, n_cols=1)
plotting.plot_mfpca_reconstruction_comparison(mfpca, data, joint_scores, subject_indices=[0, 1])
plotting.plot_mfpca_extreme_subjects(mfpca, data, joint_scores, component=0)

n_components_for_variance(model, share=0.95)          # smallest K reaching 95%
orient_eigenfunctions(model)                          # fix signs (call right after fit)
permanova(scores[:, :3], labels=groups, n_perm=999)   # do groups differ in curve shape?
permanova_table(scores[:, :3], {"group": groups, "site": (sites, subject_ids, "block")})
```

All plotting functions return `(fig, axes)` and never call `plt.show()`, so figures can be restyled or saved.

## Examples

The [`examples/`](examples) folder contains Jupyter notebooks that walk through each estimator end to end, with figures and interpretation. [`baboon_microbiome.ipynb`](examples/baboon_microbiome.ipynb) applies `PACE`, `TwoLevelPACE`, `MFPCA` and multilevel MFPCA to the gut microbiome of wild baboons (Björk et al., 2022), with the data in `examples/data/`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eyal708/functional-pca/blob/main/examples/baboon_microbiome.ipynb) runs the notebook in the browser with nothing to install: its first cell installs `fpca` from GitHub and reads the data from this repository. To run it locally instead, clone the repository and install it with `pip install -e ".[examples]"`, then open the notebook in Jupyter or VS Code.

## Implementation notes

- **Diagonal exclusion.** The raw covariance diagonal carries measurement noise, so it is excluded from covariance smoothing (Yao et al., 2005). For P-splines, both the value *and* its fitting weight are dropped.
- **Two smoothers.** `"PS"` (P-splines; cost set by basis size, scales to large data) and `"LP"` (local polynomial with an Epanechnikov kernel; local-linear mean, local-quadratic covariance as in Yao et al., 2005; cost grows quickly with grid size). Default LP bandwidths are 5% / 10% of the domain range for mean / covariance, following `fdapace`.
- **Smoothing is always explicit.** Dense input is smoothed the same way as sparse input unless `smooth_mean=False` / `smooth_covariance=False` is set.
- **Quadrature-weighted eigenproblem.** Eigenfunctions are L²-orthonormal on non-uniform grids.
- **Multilevel smoothing.** `TwoLevelPACE` uses tensor-product P-splines with the smoothing parameter selected by REML, and Gaussian BLUP scores (Di et al., 2009).
- **No silent extrapolation.** Evaluating outside the fitted domain raises an error unless `allow_extrapolation=True`.
- **Tuning matters.** Defaults are sensible starting points, not substitutes for data-specific tuning. Check the fitted mean and eigenfunctions, and adjust `mean_penalty` / `covariance_penalty` (PS) or the bandwidths (LP) for your data. The noise-variance smoother (`noise_bandwidth_fraction`, default 0.3) can over-estimate σ² when the variance changes quickly along the domain. Use a smaller value if `noise_variance_` looks too large.

## Related software

- [`fdapace`](https://github.com/functionaldata/tPACE) (R): the reference PACE implementation by the original authors.
- [`MFPCA`](https://cran.r-project.org/package=MFPCA) (R): reference implementation of Happ & Greven (2018).
- [`refund`](https://cran.r-project.org/package=refund) (R): includes multilevel FPCA (`mfpca.sc`).
- [`FDApy`](https://github.com/StevenGolovkine/FDApy) and [`scikit-fda`](https://github.com/GAA-UAM/scikit-fda) (Python): general functional data analysis toolkits. This package includes FDApy's local-polynomial and P-spline smoothers (see Acknowledgements).

## Citing

If you use this package in published work, please cite the method papers below for the estimators you used, and this repository (GitHub's **"Cite this repository"** button uses [`CITATION.cff`](CITATION.cff)).

## References

- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer. https://doi.org/10.1007/b98888
- Yao, F., Müller, H.-G., & Wang, J.-L. (2005). Functional data analysis for sparse longitudinal data. *Journal of the American Statistical Association*, 100(470), 577–590. https://doi.org/10.1198/016214504000001745
- Di, C.-Z., Crainiceanu, C. M., Caffo, B. S., & Punjabi, N. M. (2009). Multilevel functional principal component analysis. *The Annals of Applied Statistics*, 3(1), 458–488. https://doi.org/10.1214/08-AOAS206
- Happ, C., & Greven, S. (2018). Multivariate functional principal component analysis for data observed on different (dimensional) domains. *Journal of the American Statistical Association*, 113(522), 649–659. https://doi.org/10.1080/01621459.2016.1273115
- Berrendero, J. R., Justel, A., & Svarc, M. (2011). Principal components for multivariate functional data. *Computational Statistics & Data Analysis*, 55(9), 2619–2634. https://doi.org/10.1016/j.csda.2011.03.011
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655
- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26(1), 32–46. https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x
- Anderson, M. J. (2006). Distance-based tests for homogeneity of multivariate dispersions. *Biometrics*, 62(1), 245–253. https://doi.org/10.1111/j.1541-0420.2005.00440.x
- Golovkine, S. FDApy: a Python package for functional data. https://github.com/StevenGolovkine/FDApy
- Björk, J. R., et al. (2022). Synchrony and idiosyncrasy in the gut microbiome of wild baboons. *Nature Ecology & Evolution*, 6. https://doi.org/10.1038/s41559-022-01773-4 (data of the example notebook; raw reads in ENA, study ERP119849)

## Acknowledgements

The local-polynomial and P-spline smoothers are taken from [FDApy](https://github.com/StevenGolovkine/FDApy) 1.0.3 by Steven Golovkine (MIT License). They are included in [`src/fpca/_smoothing.py`](src/fpca/_smoothing.py) with FDApy's copyright and license notice, so that `fpca` does not depend on FDApy's `numpy<2` requirement; the only change to the code is replacing the `np.float_` alias, removed in NumPy 2, by `np.float64`. Default PACE bandwidths follow [`fdapace`](https://github.com/functionaldata/tPACE). Numerical work relies on NumPy, SciPy and scikit-learn; plots use Matplotlib.

This package was developed with the assistance of [Claude Code](https://claude.com/claude-code) (Anthropic).

## Contributing and license

Bug reports, questions and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Released under the [MIT License](LICENSE).
