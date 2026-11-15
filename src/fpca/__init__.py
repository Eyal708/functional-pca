"""
fpca -- functional principal component analysis for sparse, irregular and
multivariate functional data.

Estimators
----------
PACE
    Univariate FPCA by Principal Analysis by Conditional Expectation
    (Yao, Mueller and Wang, 2005).
TwoLevelPACE
    Multilevel (two-level) FPCA separating between-subject and within-subject
    variation (Di, Crainiceanu, Caffo and Punjabi, 2009).
MFPCA
    Multivariate FPCA built on fitted ``PACE`` models (Happ and Greven, 2018;
    Berrendero, Justel and Svarc, 2011). With ``fit_from_scores`` and
    ``TwoLevelPACE.level_model`` it is also run separately at each level of a
    two-level model.

Submodules
----------
fpca.plotting
    Plots for fitted models (modes of variation, scores, reconstructions,
    multivariate eigenfunctions, ...).
fpca.utils
    Variance shares, component counts, reconstruction errors, sign orientation
    of eigenfunctions and conversion of long tables to curves.
fpca.stats
    PERMANOVA / PERMDISP permutation tests on score vectors, alone or as a table.
"""

from .multilevel import LevelModel, TwoLevelPACE
from .multivariate import MFPCA
from .pace import PACE
from .stats import permanova, permanova_table
from .utils import (
    curves_from_long,
    explained_variance_ratio,
    n_components_for_variance,
    orient_eigenfunctions,
    reconstruction_rmse,
)

__version__ = "0.1.0"

__all__ = [
    "PACE",
    "TwoLevelPACE",
    "LevelModel",
    "MFPCA",
    "permanova",
    "permanova_table",
    "curves_from_long",
    "explained_variance_ratio",
    "n_components_for_variance",
    "orient_eigenfunctions",
    "reconstruction_rmse",
    "__version__",
]
