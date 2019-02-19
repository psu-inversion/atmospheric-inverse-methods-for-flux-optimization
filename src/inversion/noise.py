"""Generate noise.

Generated values have zero mean unless specified.

Mostly for use in testing.
"""
import numpy as np
from numpy.random import standard_normal as _standard_normal

from inversion.linalg import matrix_sqrt


def gaussian_noise(cov, size=None):
    """Generate gaussian noise with the given covariance.

    Parameters
    ----------
    cov: LinearOperator[N,N]
    size: int or tuple of int, optional

    Returns
    -------
    array_like[..., N]
        Samples of multivariate Gaussian noise

    Raises
    ------
    ValueError
        If cov not a square matrix

    Notes
    -----
    implementation largely copied from
    :func:`numpy.random.multivariate_normal`
    """
    if size is None:
        shape = []
    elif isinstance(size, (int, np.integer)):
        shape = [size]
    else:
        shape = size

    if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
        raise ValueError("cov must be 2 dimensional and square")

    sample_shape = cov.shape[0]
    final_shape = list(shape[:])
    final_shape.append(sample_shape)
    transposed_shape = final_shape[::-1]

    noise = _standard_normal(
        size=transposed_shape,
    ).reshape(sample_shape, -1)

    chol_upper = matrix_sqrt(cov)

    # x = x @ chol_upper
    noise = chol_upper.T.dot(noise).T
    return noise.reshape(final_shape)
