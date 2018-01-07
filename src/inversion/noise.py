"""Generate noise.

Generated values have zero mean unless specified.

Mostly for use in testing.
"""
import numpy as np
from numpy.random import standard_normal as _standard_normal
from numpy import asarray

# import from scipy.linalg if not using dask
from scipy.linalg import cholesky

from inversion.util import chunk_sizes


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
        size=final_shape
    ).reshape(-1, sample_shape)

    chol_upper = cholesky(asarray(cov))

    x = x.dot(chol_upper)
    return x.reshape(final_shape)
