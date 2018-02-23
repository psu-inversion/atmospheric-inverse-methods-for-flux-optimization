"""Generate noise.

Generated values have zero mean unless specified.

Mostly for use in testing.
"""
import numpy as np
from dask.array.random import standard_normal as _standard_normal

from inversion.util import chunk_sizes, matrix_sqrt


def gaussian_noise(cov, size=None):
    """Generate gaussian noise with the given covariance.

    Parameters
    ----------
    cov: np.ndarray[N,N]
    size: int or tuple of int, optional

    Returns
    -------
    noise: np.ndarray[N]

    Note
    ----
    implementation largely copied from
    :func:`numpy.random.multivariate_normal`
    """
    if size is None:
        shape = []
    elif isinstance(size, (int, np.integer)):
        shape = [size]
    else:
        shape = size

    sample_shape = cov.shape[0]
    final_shape = list(shape[:])
    final_shape.append(sample_shape)

    x = _standard_normal(
        size=final_shape, chunks=chunk_sizes(final_shape, matrix_side=False)
    ).reshape(-1, sample_shape)

    chol_upper = matrix_sqrt(cov)

    x = x.dot(chol_upper)
    return x.reshape(final_shape)
