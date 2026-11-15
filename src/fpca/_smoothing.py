"""
Local-polynomial and P-spline smoothers, adapted from FDApy.

This module contains the ``LocalPolynomial`` and ``PSplines`` smoothers and the
``_basis_bsplines`` helper of FDApy 1.0.3 (https://github.com/StevenGolovkine/FDApy),
by Steven Golovkine, which ``fpca.PACE`` uses for mean, covariance and noise
smoothing. They are included here, rather than imported, because FDApy 1.0.3
requires ``numpy<2``. The code is copied verbatim from
``FDApy/preprocessing/smoothing/local_polynomial.py``,
``FDApy/preprocessing/smoothing/psplines.py`` and ``FDApy/misc/basis.py``, with
these modifications:

- the three files are merged into one module, with their imports combined;
- ``np.float64`` (removed in NumPy 2) is replaced by the identical ``np.float64``
  in two type annotations.

FDApy is distributed under the MIT License, reproduced below.

MIT License

Copyright (c) 2018 Steven Golovkine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Union  # noqa: F401  (used in annotations)

import numpy as np
import numpy.typing as npt
from scipy.special import gamma
from sklearn.preprocessing import PolynomialFeatures

__all__ = ["LocalPolynomial", "PSplines"]


##############################################################################
# From FDApy/misc/basis.py

def _basis_bsplines(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 10,
    degree: int = 3,
    domain_min: float | None = None,
    domain_max: float | None = None,
) -> npt.NDArray[np.float64]:
    """Define B-splines basis of function.

    Build a basis of :math:`n_functions` functions using B-splines basis on the
    interval defined by ``argvals``. We assume that the knots are regularly spaced. The
    number of knots is equal to ``n_functions - degree``.

    Parameters
    ----------
    argvals
        The values on which evaluated the B-splines.
    n_functions
        Number of considered B-splines.
    degree
        Degree of the B-splines. The default gives cubic splines.
    domain_min
        Minimum number for the argvals.
    domain_max
        Maximum number for hte argvals.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of a B-splines
        basis.

    Notes
    -----
    This function is adapted from the `bbase` function in the R package `JOPS` _[2]. It
    computes a proper B-splines basis function (see _[1], Section 8.1).

    Examples
    --------
    >>> _basis_bsplines(argvals=np.arange(0, 1, 0.01), n_functions=10)

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X., (2023)
        JOPS: Practical Smoothing with P-Splines.

    """

    def _tpower(x, knots, p):
        res = np.zeros((len(x), len(knots)))
        for idx, knot in enumerate(knots):
            res[:, idx] = np.power(x - knot, p) * (x >= knot)
        return res

    if domain_min is None:
        domain_min = min(argvals)
    if domain_max is None:
        domain_max = max(argvals)

    # Compute the B-splines
    n_segments = n_functions - degree
    dx = (domain_max - domain_min) / n_segments
    knots = np.linspace(
        domain_min - degree * dx,
        domain_max + degree * dx,
        num=int(n_segments + 2 * degree) + 1,
        endpoint=True,
    )
    p_mat = _tpower(argvals, knots, degree)
    d_mat = np.diff(np.eye(p_mat.shape[1]), n=degree + 1, axis=0) / (
        gamma(degree + 1) * np.power(dx, degree)
    )
    basis_mat = np.power(-1, degree + 1) * p_mat @ d_mat.T

    # Make B-splines exactly zero beyond their end knots
    sk = knots[np.arange(basis_mat.shape[1]) + degree + 1]
    mask = np.zeros((len(argvals), len(sk)))
    for idx, val in enumerate(argvals):
        mask[idx, :] = val < sk
    return (basis_mat * mask).T


##############################################################################
# From FDApy/preprocessing/smoothing/local_polynomial.py

##############################################################################
# Inner functions for the LocalPolynomial class.


def _gaussian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the Gaussian density with mean 0 and standard deviation 1.

    The Gaussian density is given by

    .. math::
        K(x) = \frac{\exp(-x^2/2)}{\sqrt{2 \pi}}.

    Parameters
    ----------
    x
        Array at which computes the gaussian density.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    """
    return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)


def _epanechnikov(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the Epanechnikov kernel.

    The Epanechnikov kernel is given, in [1]_ equation 6.4, by

    .. math::
        K(x) = \frac{3}{4}(1 - x^2)  \text{if } |x| \leq 1.

    Parameters
    ----------
    x
        Array on which computes the Epanechnikov kernel.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) <= 1)
    kernel[idx] = 0.75 * (1 - np.square(x[idx]))
    return kernel


def _tri_cube(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the tri-cube kernel.

    The tri-cube kernel is given, in [1]_ equation 6.6, by

    .. math::
        K(x) = (1 - |x|^3)^3  \text{if } |x| \leq 1.

    Parameters
    ----------
    x
        Array on which computes the tri-cube kernel.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) < 1)
    kernel[idx] = np.power(1 - np.power(np.abs(x[idx]), 3), 3)
    return kernel


def _bi_square(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Compute the bi-square kernel.

    The bi-square kernel is given, in [1]_, by

    .. math::
        K(x) = (1 - x^2)^2  \text{if } |x| \leq 1.

    Parameters
    ----------
    x
        Array on which computes the bi-square kernel.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    ..[1] Cleveland W. (1979) Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association,
        74(368): 829--836.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) < 1)
    kernel[idx] = np.square(1 - np.square(x[idx]))
    return kernel


def _kernel(name: str) -> Callable:
    """Map between kernel names and functions.

    Parameters
    ----------
    name
        Name of the kernel.

    Returns
    -------
    Callable
        The kernel function.

    """
    if name == "gaussian":
        return _gaussian
    elif name == "epanechnikov":
        return _epanechnikov
    elif name == "tricube":
        return _tri_cube
    elif name == "bisquare":
        return _bi_square
    else:
        raise NotImplementedError(f"The kernel {name} is not implemented")


def _compute_kernel(
    x: npt.NDArray[np.float64],
    x0: float | npt.NDArray[np.float64],
    bandwidth: float,
    kernel: Callable = _epanechnikov,
) -> npt.NDArray[np.float64]:
    r"""Compute the weights at a particular query point.

    The weights are defined using a kernel function and are computed at points
    :math:`\lvert\lvert x - x0\rvert\rvert / \lambda` defined in [1]_, equation
    6.13. The used norm is the Euclidean norm. The function allows
    multidimensional inputs. The arguments `x`, `x0` must have a common
    dimension.

    Parameters
    ----------
    x
        Training data.
    x0
        Query point. For one-dimensional smoothing, `x0` must be passed as a
        `float`. For higher-dimensional smoothing, `x0` must be passed as
        a `npt.NDArray[np.float64]]`.
    bandwidth
        Width of the neighborhood of `x0`.
    kernel
        Kernel function to used.

    Returns
    -------
    npt.NDArray[np.float64] , shape=(n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    if x.ndim == 1:
        xx = np.abs(x - x0)
    else:
        xx = np.linalg.norm(x - x0, axis=1)
    return kernel(xx / bandwidth)


def _local_regression(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    dmat: npt.NDArray[np.float64],
    dmat_x0: npt.NDArray[np.float64],
    bandwidth: float = 0.05,
    kernel: Callable = _epanechnikov,
) -> float:
    r"""Local polynomial regression for one point.

    This function estimates the regression function :math:`f(x)` over the
    domain `\mathbb{R}^p` at a particular query point `x_0`. A generalized
    version of the equation (6.8) in [1]_ is implemented. It allows
    multidimensional inputs for `x` and `x_0`.

    Parameters
    ----------
    y
        Target values.
    x
        Training data.
    x0
        Query point. For one-dimensional smoothing, `x0` must be passed as a
        `float`. For higher-dimensional smoothing, `x0` must be passed as
        a `npt.NDArray[np.float64]]`.
    dmat
        Design matrix for the training data `x`. The dimension `n_features` is
        related to the degree of the fitted polynomials. It includes intercept
        and interaction in the case of multidimensional inputs.
    dmat_x0
        Design matrix for the query points `x_0`. The dimension `n_features`
        must be the same as the design matrix of `x`.
    bandwidth
        Width of the neighborhood of `x0`.
    kernel
        Kernel function to used.

    Returns
    -------
    float
        Estimation of `f(x_0)`.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel_values = _compute_kernel(x=x, x0=x0, bandwidth=bandwidth, kernel=kernel)
    temp = dmat.T * kernel_values
    beta = np.linalg.lstsq(np.dot(temp, dmat), np.dot(temp, y), rcond=1e-10)[0]

    return np.dot(dmat_x0, beta)


#############################################################################
# Class LocalPolynomial


class LocalPolynomial:
    r"""Local Polynomial regression.

    This module implements Local Polynomial Regression over different
    dimensional domain [2]_. The idea of local regression is to fit a
    (simple) different model separetely at each query point :math:`x_0`. Using
    only the observations close to :math:`x_0`, the resulting estimated
    function is smooth in the definition domain. Selecting observations close
    to :math:`x_0` is achieved via a weighted (kernel) function which assigned
    a weight to each observation based on its (euclidean) distance from the
    query point.

    Different kernels are defined (`gaussian`, `epanechnikov`, `tricube`,
    `bisquare`). Each of them has slightly different properties. Kernels are
    indexed by a parameter (`bandwith`) that controls the width of the
    neighborhood of :math:`x_0`. Note that the bandwidth can be adaptive and
    depend on :math:`x_0`.

    The degree of smoothing functions is controled using the `degree`
    parameter. A degree of 0 corresponds to locally constant, a degree of 1 to
    locally linear and a degree of 2 to locally quadratic, etc. High degrees
    can cause overfitting.

    The implementation is adapted from [3]_.

    Parameters
    ----------
    kernel_name
        Kernel name used as weight (`gaussian`, `epanechnikov`, `tricube`,
        `bisquare`).
    bandwidth
        Strictly positive. Control the size of the associated neighborhood.
    degree
        Degree of the local polynomial to fit. If ``degree = 0``, we fit the
        local constant estimator (equivalent to the Nadaraya-Watson estimator).
        If ``degree = 1``, we fit the local linear estimator. If
        ``degree = 2``, we fit the local quadratic estimator.
    robust
        Whether to apply the robustification procedure from [1]_, page 831.

    Attributes
    ----------
    kernel: Callable
        Function associated to the kernel name.
    poly_features: PolynomialFeatures
        An instance of ``sklearn.preprocessing.PolynomialFeatures`` used to
        create design matrices. It includes an intercept and interactions for
        multidimensional inputs.

    Notes
    -----
    This methods is *memory-based* and thus require no training; all the work
    is performed at evaluation time [2]_. For now, no ``fit`` function is
    necessary and only a ``predict`` is implemented.

    References
    ----------
    .. [1] Cleveland W. (1979) Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association,
        74(368): 829--836.
    .. [2] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.
    .. [3] https://github.com/arokem/lowess/blob/master/lowess/lowess.py

    """

    def __init__(
        self,
        kernel_name: str = "epanechnikov",
        bandwidth: float = 0.05,
        degree: int = 1,
        robust: bool = False,
        **kwargs,
    ) -> None:
        """Initialize LocalPolynomial object."""
        self.kernel_name = kernel_name
        self.bandwidth = bandwidth
        self.degree = degree
        self.robust = robust

    @property
    def kernel_name(self) -> str:
        """Getter for `kernel_name`."""
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, new_kernel_name: str) -> None:
        self._kernel_name = new_kernel_name
        self._kernel = _kernel(new_kernel_name)

    @property
    def bandwidth(self) -> float:
        """Getter for `bandwidth`."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, new_bandwidth: float) -> None:
        if new_bandwidth <= 0:
            raise ValueError("Bandwidth parameter must be strictly positive.")
        self._bandwidth = new_bandwidth

    @property
    def degree(self) -> float:
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree: int) -> None:
        if new_degree < 0:
            raise ValueError("Degree parameter must be positive.")
        self._degree = new_degree
        self._poly_features = PolynomialFeatures(degree=new_degree)

    @property
    def robust(self) -> bool:
        """Getter for `robust`."""
        return self._robust

    @robust.setter
    def robust(self, new_robust: bool) -> None:
        self._robust = new_robust

    @property
    def kernel(self) -> Callable:
        """Getter for `kernel`."""
        return self._kernel

    @property
    def poly_features(self) -> PolynomialFeatures:
        """Getter for `poly_features`."""
        return self._poly_features

    def predict(
        self,
        y: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        x_new: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Predict using local polynomial regression.

        Parameters
        ----------
        y
            Target values.
        x
            Training data.
        x_new
            Query points at which estimates the function. If ``None``, the
            (unique) training data are used as query points. The shape of the
            array must be (n_points, n_dim).

        Returns
        -------
        npt.NDArray[np.float64], shape = (n_samples,)
            Return predicted values.

        Notes
        -----
        Be careful that, for two-dimensional and higher-dimensional data, not
        passing a ``x_new`` argument may result to something unexpected as for
        now, the function ``np.unique`` re-order the columns of the data. To be
        sure of the results, please provide a ``x_new`` argument.

        Examples
        --------
        For one-dimensional data.

        >>> n_points = 101
        >>> x = np.linspace(0, 1, n_points)
        >>> y = np.sin(x) + np.random.normal(0, 0.05, n_points)
        >>> x_new = np.linspace(0, 1, 11)

        >>> lp = LocalPolynomial(
        ...     kernel_name='epanechnikov', bandwidth=0.3, degree=1
        ... )
        >>> lp.predict(y=y, x=x, x_new=x_new)

        For two-dimensional data.

        >>> n_points = 51
        >>> pts = np.linspace(0, 1, n_points)
        >>> xx, yy = np.meshgrid(pts, pts, indexing='ij')
        >>> x = np.column_stack([xx.flatten(), yy.flatten()])
        >>> eps = np.random.normal(0, 0.1, len(x))
        >>> y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + eps

        >>> lp = LocalPolynomial(
        ...     kernel_name='epanechnikov', bandwidth=0.3, degree=2
        ... )
        >>> lp.predict(y=y, x=x, x_new=x_new)

        """
        if x_new is None:
            x_new = np.unique(x, axis=0)

        if x.ndim == 1:  # because PolynomialFeatures wants 2d arrays.
            x = x.reshape(-1, 1)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)

        dmat_sampling = self.poly_features.fit_transform(x)
        dmat_query = self.poly_features.fit_transform(x_new)

        y_pred = np.zeros(x_new.shape[0])
        for idx, (pts, dmat) in enumerate(zip(x_new, dmat_query)):
            y_pred[idx] = _local_regression(
                y, x, pts, dmat_sampling, dmat, self.bandwidth, self.kernel
            )
        return y_pred


##############################################################################
# From FDApy/preprocessing/smoothing/psplines.py

########################################################################################
# Utils
def _row_tensor(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    """
    Compute the row-wise tensor product of two 2D arrays.

    The row-wise tensor product of two 2D arrays `x` and `y` is a 2D array `z` such that
    each row of `z` is the Kronecker product of the corresponding row of `x` and `y`.
    If `y` is not provided, it defaults to `x`. Note that `x` and `y` must have the
    same number of rows.

    Parameters
    ----------
    x
        A 2D array of shape `(m, n)`.
    y
        A 2D array of shape `(m, q)`. If not provided, it defaults to `x`.

    Returns
    -------
    npt.NDArray[np.float64]
        A 2D array of shape `(m, n*q)` or `(m, n*n)` if `y` is not provided.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[5, 6, 7], [7, 8, 9]])
    >>> _row_tensor(x, y)
    array([
        [ 5.,  6.,  7., 10., 12., 14.],
        [21., 24., 27., 28., 32., 36.]
    ])
    >>> _row_tensor(x)
    array([
        [ 1.,  2.,  2.,  4.],
        [ 9., 12., 12., 16.]
    ])

    Notes
    -----
    This function is adapted from [1]_.

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    if y is None:
        y = x
    if x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must have the same number of rows.")
    onex = np.ones((1, x.shape[1]))
    oney = np.ones((1, y.shape[1]))
    return np.kron(x, oney) * np.kron(onex, y)


def _h_transform(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the H-transform of a nD array `y` with respect to a 2D array `x`.

    The H-transform of a nD array `y` with respect to a 2D array `x` is a nD array `z`.
    The H-transform generalizes the pre-multiplication of vectors and matrices by a
    matrix.

    Parameters
    ----------
    x
        A 2D array of shape `(n, m)`.
    y
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float64]
        A nD array of shape `(n, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _h_transform(x, y)
    array([[22, 28]])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    if x.shape[1] != y.shape[0]:
        raise ValueError(
            "The second dimension of `x` must be equal to the first dimension of `y`."
        )
    y_dim = y.shape
    y_reshape = y.reshape(y_dim[0], np.prod(y_dim[1:]))
    xy_product = x @ y_reshape
    return xy_product.reshape((xy_product.shape[0], *y_dim[1:]))


def _rotate(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Rotate the axes of a multi-dimensional array to the right.

    The rotation of a nD array moves the first axis of a multi-dimensional array to the
    last position. This is equivalent to rotating the axes of the array to the right.
    It generalizes the transpose operation to nD arrays.

    Parameters
    ----------
    x
        A multi-dimensional array of shape `(n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float64]
        A multi-dimensional array of shape `(n2, ..., nk, n1)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8], [9, 0]]])
    >>> _rotate(x)
    array([
        [[1, 5],[2, 6]],
        [[3, 7],[4, 8]],
        [[5, 9],[6, 0]]
    ])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    return np.moveaxis(x, 0, -1)


def _rotated_h_transform(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the rotated H-transform of a nD array `y` with respect to a 2D array `x`.

    The rotated H-transform of a nD array `y` with respect to a 2D array `x` is a nD
    array `z`. This function performed a H-transform of `x` and `y`, and then rotate
    the results.

    Parameters
    ----------
    x
        A 2D array of shape `(n, m)`.
    y
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Returns
    -------
    npt.NDArray[np.float64]
        A nD array of shape `(n1, n2, ..., nk, m)`.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _rotated_h_transform(x, y)
    array([
        [22],
        [28]
    ])

    References
    ----------
    .. [1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    return _rotate(_h_transform(x, y))


def _create_permutation(p: int, k: int) -> npt.NDArray[np.float64]:
    """
    Create a permutation array for a given number of factors and levels.

    This function creates a permutation array for a given number of factors `p` and
    levels `k`. The resulting array is a 1D array of shape `(k*p,)` that contains the
    indices of all possible combinations of `p` factors with `k` levels each.

    Parameters
    ----------
    p
        The number of factors.
    k
        The number of levels.

    Returns
    -------
    npt.NDArray[np.float64]
        A 1D array of shape `(k*p,)` that contains the indices of all possible
        combinations of `p` factors with `k` levels each.

    Examples
    --------
    >>> np.tile(np.arange(3), 2)
    array([0, 1, 2, 0, 1, 2])
    >>> _create_permutation(3, 2)
    array([0, 3, 1, 4, 2, 5])
    >>> np.repeat(np.arange(3), 2)
    array([0, 0, 1, 1, 2, 2])
    >>> _create_permutation(2, 3)
    array([0, 2, 4, 1, 3, 5])

    """
    a = np.arange(0, k)
    b = np.arange(0, p)
    m = np.add.outer(a * p, b)
    return m.flatten("F")


def _tensor_product_penalties(
    penalties: list[npt.NDArray[np.float64]],
) -> list[npt.NDArray[np.float64]]:
    """
    Compute the tensor product of a list of penalty matrices.

    The `_tensor_product_penalties` function computes the tensor product of a list of
    penalty matrices. The resulting list contains the tensor product of each penalty
    matrix with itself and with the identity matrices of the same size as the other
    penalty matrices. If a penalty matrix is square, its tensor product with itself is
    symmetrized by taking the mean of the matrix and its transpose.

    Parameters
    ----------
    penalties
        A list of penalty matrices.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        A list of tensor product matrices.

    Notes
    -----
    This function is adapted from the function `tensor.prod.penalties` in [1]_.

    Examples
    --------
    >>> penalties = [
        np.array([
            [ 1., -1.,  0.],
            [-1.,  2., -1.],
            [ 0., -1.,  2.]
        ]),
        np.array([
            [ 1., -1.],
            [-1.,  2.]
        ])
    ]
    >>> _tensor_product_penalties(penalties)
    [
         array([
             [ 1.,  0., -1., -0.,  0.,  0.],
             [ 0.,  1., -0., -1.,  0.,  0.],
             [-1., -0.,  2.,  0., -1., -0.],
             [-0., -1.,  0.,  2., -0., -1.],
             [ 0.,  0., -1., -0.,  2.,  0.],
             [ 0.,  0., -0., -1.,  0.,  2.]
         ]),
         array([
             [ 1., -1.,  0., -0.,  0., -0.],
             [-1.,  2., -0.,  0., -0.,  0.],
             [ 0., -0.,  1., -1.,  0., -0.],
             [-0.,  0., -1.,  2., -0.,  0.],
             [ 0., -0.,  0., -0.,  1., -1.],
             [-0.,  0., -0.,  0., -1.,  2.]
         ]
     )]

    References
    ----------
    .. [1] Wood, S. (2023). mgcv: Mixed GAM Computation Vehicle with Automatic
        Smoothness Estimation.

    """
    n_penalties = len(penalties)
    eyes = [np.eye(penalty.shape[1]) for penalty in penalties]

    if n_penalties == 1:
        return penalties[0]
    else:
        tensors_list = []
        for idx in range(n_penalties):
            left = penalties[0] if idx == 0 else eyes[0]
            for j in range(1, n_penalties):
                right = penalties[j] if idx == j else eyes[j]
                left = np.kron(left, right)
            # Make sure the matrix is symmetric
            if left.shape[0] == left.shape[1]:
                left = (left + left.T) / 2
            tensors_list.append(left)
        return tensors_list


def _format_data(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Format input data for multidimensional P-splines smoothing.

    Parameters
    ----------
    X
        An array containing the predictor variable values.
    y
        An array containing the response variable values.

    """
    new_X = [np.unique(column) for column in X.T]
    X_matrices = np.meshgrid(*new_X, indexing="ij")

    new_y = np.zeros_like(X_matrices[0])
    for x, obs in zip(X, y):
        indices = tuple(
            np.flatnonzero(points == point)[0] for point, points in zip(x, new_X)
        )
        new_y[indices] = obs

    weights = np.ones_like(new_y)
    weights[new_y == 0] = 0
    return new_X, new_y, weights


########################################################################################
# Inner functions for the PSplines class.
def _fit_one_dimensional(
    data: npt.NDArray[np.float64],
    basis: npt.NDArray[np.float64],
    sample_weights: npt.NDArray[np.float64] | None = None,
    penalty: float = 1.0,
    order_penalty: int = 2,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Fit a one-dimensional P-splines model to the given data.

    The function fits a one-dimensional P-splines model to the given data using a basis
    matrix and an optional weight matrix. The function returns a dictionary containing
    the fitted values, the estimated coefficients, and the hat matrix.

    Parameters
    ----------
    data
        A one-dimensional array of shape `(n_obs,)` containing the response variable
        values.
    basis
        A two-dimensional array of shape `(n_basis, n_obs)` containing the basis matrix.
    sample_weights
        A one-dimensional array of shape `(n_obs,)` containing the weights for each
        observation. If not provided, all observations are assumed to have equal weight.
    penalty
        The penalty parameter for the P-splines model.
    order_penalty
        The order of the penalty difference matrix.

    Returns
    -------
    Dict[str, npt.NDArray[np.float64]]
        A dictionary containing the following keys:
        - `y_hat`: A one-dimensional array of shape `(n_obs,)` containing the fitted
        values.
        - `beta_hat`: A one-dimensional array of shape `(n_basis,)` containing the
        estimated coefficients.
        - `hat_matrix`: A one-dimensional array of shape `(n_obs,)` containing the
        diagonal of the hat matrix.

    Notes
    -----
    The implementation of adapted from [2]_. See [1]_ for more details.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> basis = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]])
    >>> _fit_one_dimensional(data, basis)
    {
        'y_hat': array([1.25143869, 1.95484728, 2.83532537, 3.89287295, 5.12749004]),
        'beta_hat': array([0.7250996 , 0.43780434, 0.08853475]),
        'hat_matrix': array([0.3756529, 0.3549800, 0.2669322, 0.2788401, 0.7545816])
    }

    References
    ----------
    .. [1] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X. (2023).
        JOPS: Practical Smoothing with P-Splines.

    """
    # Get parameters.
    n_basis, n_obs = basis.shape

    # Construct the penalty.
    pen_mat = np.diff(np.eye(n_basis), n=order_penalty, axis=0)

    # Build the different part of the model.
    if sample_weights is None:
        sample_weights = np.ones(n_obs)
    weight_mat = np.diag(sample_weights)

    bwb_mat = basis @ weight_mat @ basis.T
    pen_mat = penalty * pen_mat.T @ pen_mat
    bwy_mat = basis @ weight_mat @ data

    # Fit the model
    inv_mat = np.linalg.pinv(bwb_mat + pen_mat)
    beta_hat = inv_mat @ bwy_mat
    y_hat = basis.T @ beta_hat

    # Compute the hat matrix
    hat_matrix = np.diag(basis.T @ inv_mat @ basis @ weight_mat)

    return {"y_hat": y_hat, "beta_hat": beta_hat, "hat_matrix": hat_matrix}


def _fit_n_dimensional(
    data: npt.NDArray[np.float64],
    basis_list: List[npt.NDArray[np.float64]],
    sample_weights: npt.NDArray[np.float64] | None = None,
    penalties: tuple[float, ...] | None = None,
    order_penalty: int = 2,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Fit an nD P-splines model to the given data.

    The function fits an nD P-splines model to the given data using a list of basis
    matrices and an optional weights matrix. The function returns a dictionary
    containing the fitted values, the estimated coefficients, and the hat matrix.

    Parameters
    ----------
    data
        An nD array of shape `(n1, n2, ..., nk)` containing the response variable
        values.
    basis_list
        A list of two-dimensional arrays of shape `(m1, n1), (m2, n2), ..., (mk, nk)`
        containing the basis matrices for each dimension.
    sample_weights
        An nD array of shape `(n1, n2, ..., nk)` containing the weights for each
        observation. If not provided, all observations are assumed to have equal weight.
    penalties
        A tuple of penalty parameters for each dimension. If not provided, the penalty
        is assumed to be the same for each dimension and equal to 1.
    order_penalty
        The order of the penalty difference matrix.

    Returns
    -------
    Dict[str, npt.NDArray[np.float64]]
        A dictionary containing the following keys:
        - `y_hat`: An nD array of shape `(n1, n2, ..., nk)` containing the fitted
        values.
        - `beta_hat`: An nD array of shape `(m1, m2, ..., mk)` containing the
        estimated coefficients.
        - `hat_matrix`: A nD array of shape `(n1, n2, ..., nk)` containing the hat
        matrix.

    Notes
    -----
    The implementation of adapted from [2]_. See [1]_ for more details.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> basis_list = [np.array([[1, 1], [1, 2]]), np.array([[1, 1], [2, 3]])]
    >>> _fit_n_dimensional(data, basis_list)
    {
        'y_hat': array([
            [1., 2.],
            [3., 4.]
        ]),
        'beta_hat': array([
            [-3.,  1.],
            [ 2.,  0.]
        ]),
        'hat_matrix': array([
            [1., 1.],
            [1., 1.]
        ])
    }

    References
    ----------
    .. [1] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X. (2023).
        JOPS: Practical Smoothing with P-Splines.

    """
    if sample_weights is None:
        sample_weights = np.ones_like(data)
    if penalties is None:
        penalties = np.ones(len(data.shape))

    n_basis = tuple(basis.shape[0] for basis in basis_list)
    tensor_list = [_row_tensor(basis.T) for basis in basis_list]

    bwb_mat = _rotated_h_transform(tensor_list[0].T, sample_weights)
    for idx in np.arange(1, len(tensor_list)):
        bwb_mat = _rotated_h_transform(tensor_list[idx].T, bwb_mat)
    bwb_mat = (
        bwb_mat.reshape(np.repeat(n_basis, 2))
        .transpose(_create_permutation(2, len(n_basis)))
        .reshape((np.prod(n_basis), np.prod(n_basis)))
    )

    # Penalty
    eyes_mats = [np.eye(n) for n in n_basis]
    diff_mats = [np.diff(eyes_mat, n=order_penalty, axis=0) for eyes_mat in eyes_mats]
    prod_diff_mats = [diff_mat.T @ diff_mat for diff_mat in diff_mats]
    pen_mats = _tensor_product_penalties(prod_diff_mats)

    penalty_mat = np.sum(
        [penalty * pen_mat for (penalty, pen_mat) in zip(penalties, pen_mats)], axis=0
    )

    # Last part of the equation
    bwy_mat = _rotated_h_transform(basis_list[0], data * sample_weights)
    for idx in np.arange(1, len(basis_list)):
        bwy_mat = _rotated_h_transform(basis_list[idx], bwy_mat)
    bwy_mat = bwy_mat.reshape(np.prod(n_basis))

    # Fit
    fit = np.linalg.lstsq(bwb_mat + penalty_mat, bwy_mat, rcond=None)
    y_hat = _rotated_h_transform(basis_list[0].T, fit[0].reshape(n_basis))
    for idx in np.arange(1, len(basis_list)):
        y_hat = _rotated_h_transform(basis_list[idx].T, y_hat)

    # Compute the H matrix
    rot_hat_mat = np.linalg.pinv(bwb_mat + penalty_mat)
    rot_hat_mat = (
        rot_hat_mat.reshape(np.repeat(n_basis, 2))
        .transpose(_create_permutation(2, len(n_basis)))
        .reshape(tuple(n**2 for n in n_basis))
    )

    hat_matrix = _rotated_h_transform(tensor_list[0], rot_hat_mat)
    for idx in np.arange(1, len(tensor_list)):
        hat_matrix = _rotated_h_transform(tensor_list[idx], hat_matrix)
    hat_matrix = sample_weights * hat_matrix

    return {
        "y_hat": y_hat,
        "beta_hat": fit[0].reshape(n_basis),
        "hat_matrix": hat_matrix,
    }


########################################################################################
# class PSplines


class PSplines:
    r"""P-Splines smoothing.

    The class fits a P-splines model to the given data using a B-splines basis and an
    optional weights matrix.

    Parameters
    ----------
    n_segments
        The number of evenly spaced segments.
    degree
        The number of the degree of the basis.
    order_penalty
        The number of the order of the difference penalty.
    order_derivative
        The order of the derivative to compute.

    Attributes
    ----------
    y_hat: npt.NDArray[np.float64]
        The fitted response variable values.
    beta_hat: npt.NDArray[np.float64]
        The estimated coefficients for the basis functions.
    diagnostics: dict
        A dictionary containing diagnostic information about the fit.

    Notes
    -----
    This code is adapted from [2]_. See [1]_ for more details.

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X., (2023)
        JOPS: Practical Smoothing with P-Splines.

    """

    def __init__(
        self,
        n_segments: int | npt.NDArray[np.int64] = 10,
        degree: int | npt.NDArray[np.int64] = 3,
        order_penalty: int = 2,
        order_derivative: int = 0,
    ) -> None:
        """Initialize PSplines object."""
        self._n_segments = n_segments
        self._degree = degree
        self._order_penalty = order_penalty
        self._order_derivative = order_derivative

    @property
    def n_segments(self) -> Union[int, npt.NDArray[np.int64]]:
        """Getter for `n_segments`."""
        return self._n_segments

    @n_segments.setter
    def n_segments(self, new_n_segments: Union[int, npt.NDArray[np.int64]]) -> None:
        self._n_segments = new_n_segments

    @property
    def degree(self) -> Union[int, npt.NDArray[np.int64]]:
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree: Union[int, npt.NDArray[np.int64]]) -> None:
        self._degree = new_degree

    @property
    def order_penalty(self) -> int:
        """Getter for `order_penalty`."""
        return self._order_penalty

    @order_penalty.setter
    def order_penalty(self, new_order_penalty: int) -> None:
        if new_order_penalty < 0:
            raise ValueError("The order of the penalty must be positive.")
        self._order_penalty = new_order_penalty

    @property
    def order_derivative(self) -> int:
        """Getter for `order_derivatives`."""
        return self._order_derivative

    @order_derivative.setter
    def order_derivative(self, new_order_derivative: int) -> None:
        if new_order_derivative < 0:
            raise ValueError("The order of the derivative must be positive.")
        self._order_derivative = new_order_derivative

    def fit(
        self,
        y: npt.NDArray[np.float64],
        x: List[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
        sample_weights: npt.NDArray[np.float64] | None = None,
        penalty: tuple[float, ...] | None = None,
        **kwargs,
    ) -> None:
        """Fit a P-splines model to the given data.

        The method fits a P-splines model to the given data using a B-splines basis and
        an optional weights matrix.

        Parameters
        ----------
        y
            An nD array of shape `(n1, n2, ..., nk)` containing the response variable
            values.
        x
            A 1D or a list of 1D arrays of shape `(n1,), (n2,), ..., (nk,)` containing
            the predictor variable values.
        sample_weights
            An N-dimensional array of shape `(n1, n2, ..., nk)` containing the weights
            for each observation. If not provided, all observations are assumed to have
            equal weight.
        penalty
            A tuple of penalty parameters for each dimension.

        Returns
        -------
        self

        Examples
        --------
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> ps = P_splines(n_segments=3, degree=2)
        >>> ps.fit(y, x)
        >>> ps.y_hat
        array([1., 2., 3., 4., 5.])

        """
        # Check parameters
        self.dimension = len(y.shape)
        if isinstance(self.n_segments, int):
            self.n_segments = np.repeat(self.n_segments, self.dimension)
        if isinstance(self.degree, int):
            self.degree = np.repeat(self.degree, self.dimension)
        if penalty is None:
            penalty = tuple(self.dimension * [1])

        domain_min = kwargs.get("domain_min", self.dimension * [None])
        domain_max = kwargs.get("domain_max", self.dimension * [None])

        # Build the B-splines basis
        if isinstance(x, np.ndarray):
            x = [x]
        basis_list = [
            _basis_bsplines(
                argvals=argvals,
                n_functions=n_segments + degree,
                degree=degree,
                domain_min=do_min,
                domain_max=do_max,
            )
            for argvals, n_segments, degree, do_min, do_max in zip(
                x, self.n_segments, self.degree, domain_min, domain_max
            )
        ]

        if self.dimension == 1:
            res = _fit_one_dimensional(
                data=y,
                basis=basis_list[0],
                sample_weights=sample_weights,
                penalty=penalty,
                order_penalty=self.order_penalty,
            )
        else:
            res = _fit_n_dimensional(
                data=y,
                basis_list=basis_list,
                sample_weights=sample_weights,
                penalties=penalty,
                order_penalty=self.order_penalty,
            )

        # Export results
        self.basis = basis_list
        self.y_hat = res["y_hat"]
        self.beta_hat = res["beta_hat"]
        self.diagnostics = {"hat_matrix": res["hat_matrix"]}
        return self

    def predict(self, x: npt.NDArray[np.float64] | None = None, **kwargs) -> None:
        """Predict the response variable values for the given predictor variable values.

        The method predicts the response variable values for the given predictor
        variable values using the fitted P-splines model. If `x` is not provided, the
        method returns the fitted values.

        Parameters
        ----------
        x
            A 1D or a list of one-dimensional arrays of shape `(n1,), (n2,), ..., (nk,)`
            containing the predictor variable values. If not provided, the method
            returns the fitted values.

        Returns
        -------
        npt.NDArray[np.float64]
            An nD array of shape `(n1, n2, ..., nk)` containing the predicted response
            variable values.

        Examples
        --------
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> ps = P_splines(n_segments=3, degree=2)
        >>> ps.fit(y, x)
        >>> ps.predict(x)
        array([1., 2., 3., 4., 5.])

        """
        if x is None:
            return self.y_hat

        # Build the B-splines basis
        if isinstance(x, np.ndarray):
            x = [x]
        basis_list = [
            _basis_bsplines(
                argvals=argvals,
                n_functions=n_segments + degree,
                degree=degree,
                domain_min=kwargs.get("domain_min", np.min(argvals)),
                domain_max=kwargs.get("domain_max", np.max(argvals)),
            )
            for argvals, n_segments, degree in zip(x, self.n_segments, self.degree)
        ]

        if self.dimension == 1:
            y_pred = self.beta_hat @ basis_list[0]
        else:
            y_pred = _rotated_h_transform(basis_list[0].T, self.beta_hat)
            for idx in np.arange(1, len(basis_list)):
                y_pred = _rotated_h_transform(basis_list[idx].T, y_pred)

        return y_pred
