"""
Permutation tests on FPCA score vectors.

FPCA scores turn each curve into a short vector, so classical multivariate
tests can ask whether groups of curves differ in shape. :func:`permanova`
tests for a difference in group *location* (centroids) and, alongside it,
PERMDISP tests for a difference in group *dispersion*. Reporting both is
recommended: a significant PERMANOVA together with a significant PERMDISP
can reflect unequal spread rather than shifted centroids (Anderson, 2006).

References
----------
Anderson, M. J. (2001). A new method for non-parametric multivariate analysis
    of variance. Austral Ecology, 26(1), 32-46.
    https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x
Anderson, M. J. (2006). Distance-based tests for homogeneity of multivariate
    dispersions. Biometrics, 62(1), 245-253.
    https://doi.org/10.1111/j.1541-0420.2005.00440.x
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["permanova", "permanova_table"]


def _group_stats(X: np.ndarray, codes: np.ndarray, n_groups: int):
    """Between-group sum of squares and group centroids for integer group codes.

    Vectorised over a leading permutation axis when ``codes`` is 2-D.
    """
    codes = np.atleast_2d(codes)
    P, n = codes.shape
    p = X.shape[1]
    counts = np.zeros((P, n_groups))
    sums = np.zeros((P, n_groups, p))
    rows = np.repeat(np.arange(P), n)
    np.add.at(counts, (rows, codes.ravel()), 1)
    np.add.at(sums, (rows, codes.ravel()), np.tile(X, (P, 1)))
    cent = sums / np.maximum(counts, 1)[..., None]
    grand = X.mean(axis=0)
    ssb = np.sum(counts * np.sum((cent - grand) ** 2, axis=-1), axis=1)
    return ssb, cent


def _permute(codes: np.ndarray, blocks: Optional[np.ndarray], scheme: str, rng, n_perm: int) -> np.ndarray:
    """``n_perm`` permutations of the group codes under the requested scheme."""
    n = len(codes)
    out = np.empty((n_perm, n), dtype=codes.dtype)
    if scheme == "free":
        for i in range(n_perm):
            out[i] = rng.permutation(codes)
    elif scheme == "within":
        base = np.argsort(blocks, kind="stable")
        for i in range(n_perm):
            order = np.lexsort((rng.random(n), blocks))
            out[i, base] = codes[order]
    elif scheme == "block":
        uniq, first = np.unique(blocks, return_index=True)
        block_codes = codes[first]
        block_idx = np.searchsorted(uniq, blocks)
        for i in range(n_perm):
            out[i] = rng.permutation(block_codes)[block_idx]
    else:
        raise ValueError(f"scheme must be 'free', 'within' or 'block', got {scheme!r}.")
    return out


def permanova(
    X: np.ndarray,
    labels,
    blocks=None,
    scheme: str = "free",
    n_perm: int = 999,
    seed: Optional[int] = 0,
) -> dict:
    """One-way PERMANOVA and PERMDISP with Euclidean distance, by permutation.

    With Euclidean distances the PERMANOVA pseudo-F equals the one-way
    MANOVA trace statistic computed from group centroids, so no distance
    matrix is formed and the test scales to many thousands of rows. PERMDISP
    is the one-way ANOVA F of each row's distance to its group centroid. Both
    statistics are evaluated on the same set of permutations.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Score vectors, e.g. the first few columns of ``PACE.transform``. Scale
        the columns first if they should contribute equally.
    labels : array-like, shape (n_samples,)
        Group label of each row. At least two groups are required.
    blocks : array-like, shape (n_samples,), optional
        Block (e.g. subject) of each row. Required for ``scheme='within'`` and
        ``scheme='block'``.
    scheme : {'free', 'within', 'block'}, default='free'
        How labels are permuted to build the null distribution:

        - ``'free'``: shuffle labels over all rows (independent rows).
        - ``'within'``: shuffle labels only among rows of the same block, for
          a within-subject factor in a repeated-measures design.
        - ``'block'``: keep each block's label constant and shuffle labels
          between blocks, for a between-subject factor when each subject
          contributes several rows.
    n_perm : int, default=999
        Number of permutations.
    seed : int or None, default=0
        Seed of the random number generator.

    Returns
    -------
    dict
        ``n`` (rows), ``groups`` (number of groups), ``F`` (pseudo-F), ``R2``
        (between-group share of the total sum of squares), ``p`` (permutation
        p-value of F), ``PERMDISP_F`` and ``PERMDISP_p``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    labels = np.asarray(labels).astype(str)
    if len(labels) != len(X):
        raise ValueError(f"labels has {len(labels)} entries but X has {len(X)} rows.")
    uniq, codes = np.unique(labels, return_inverse=True)
    g, n = len(uniq), len(labels)
    if g < 2:
        raise ValueError("permanova needs at least two groups.")
    if n <= g:
        raise ValueError("permanova needs more rows than groups.")
    if scheme in ("within", "block"):
        if blocks is None:
            raise ValueError(f"scheme={scheme!r} requires blocks.")
        blocks = np.unique(np.asarray(blocks).astype(str), return_inverse=True)[1]
    sst = np.sum((X - X.mean(axis=0)) ** 2)

    def _stats(code_mat):
        ssb, cent = _group_stats(X, code_mat, g)
        f = (ssb / (g - 1)) / ((sst - ssb) / (n - g))
        code_mat = np.atleast_2d(code_mat)
        P = len(code_mat)
        z = np.linalg.norm(X[None, :, :] - cent[np.arange(P)[:, None], code_mat], axis=-1)
        zbar = z.mean(axis=1, keepdims=True)
        zc = np.zeros((P, g))
        cnt = np.zeros((P, g))
        r = np.repeat(np.arange(P), n)
        np.add.at(zc, (r, code_mat.ravel()), z.ravel())
        np.add.at(cnt, (r, code_mat.ravel()), 1)
        zm = zc / np.maximum(cnt, 1)
        ssb_z = np.sum(cnt * (zm - zbar) ** 2, axis=1)
        ssw_z = np.sum((z - zm[np.arange(P)[:, None], code_mat]) ** 2, axis=1)
        fd = (ssb_z / (g - 1)) / (ssw_z / (n - g))
        return f, ssb / sst, fd

    f_obs, r2, fd_obs = (a[0] for a in _stats(codes))
    rng = np.random.default_rng(seed)
    f_null = np.empty(n_perm)
    fd_null = np.empty(n_perm)
    perms = _permute(codes, blocks, scheme, rng, n_perm)
    for start in range(0, n_perm, 50):
        f, _, fd = _stats(perms[start:start + 50])
        f_null[start:start + 50] = f
        fd_null[start:start + 50] = fd
    return {
        "n": n,
        "groups": g,
        "F": float(f_obs),
        "R2": float(r2),
        "p": float((1 + np.sum(f_null >= f_obs)) / (1 + n_perm)),
        "PERMDISP_F": float(fd_obs),
        "PERMDISP_p": float((1 + np.sum(fd_null >= fd_obs)) / (1 + n_perm)),
    }


def permanova_table(
    X: np.ndarray,
    tests: Mapping[str, Union[Sequence, Tuple, Mapping]],
    n_perm: int = 999,
    seed: Optional[int] = 0,
):
    """Run several :func:`permanova` tests on the same score vectors and tabulate them.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Score vectors, as in :func:`permanova`.
    tests : dict
        Maps a row name to one test, given as either

        - the labels alone (free permutation),
        - a tuple ``(labels, blocks, scheme)``, or
        - a dict with keys ``labels`` and optionally ``blocks`` and ``scheme``.

        See :func:`permanova` for ``labels``, ``blocks`` and ``scheme``.
    n_perm : int, default=999
        Number of permutations for every test.
    seed : int or None, default=0
        Seed of the random number generator, reused for every test.

    Returns
    -------
    pandas.DataFrame
        One row per test, indexed by name, with columns ``groups``,
        ``pseudo-F``, ``R2``, ``p (PERMANOVA)``, ``F (PERMDISP)`` and
        ``p (PERMDISP)``.
    """
    import pandas as pd

    rows = []
    for name, spec in tests.items():
        if isinstance(spec, Mapping):
            labels, blocks, scheme = spec["labels"], spec.get("blocks"), spec.get("scheme", "free")
        elif isinstance(spec, tuple):
            labels, blocks, scheme = spec
        else:
            labels, blocks, scheme = spec, None, "free"
        res = permanova(X, labels, blocks=blocks, scheme=scheme, n_perm=n_perm, seed=seed)
        rows.append({"test": name, "groups": res["groups"], "pseudo-F": res["F"], "R2": res["R2"],
                     "p (PERMANOVA)": res["p"], "F (PERMDISP)": res["PERMDISP_F"],
                     "p (PERMDISP)": res["PERMDISP_p"]})
    return pd.DataFrame(rows).set_index("test")
