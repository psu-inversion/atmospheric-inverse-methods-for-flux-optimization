"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc
import functools

from numpy import fromfunction, arange, atleast_1d, asanyarray
from numpy import exp, dot, diag, prod, square, where, sqrt, newaxis
from numpy import sum as np_sum
from numpy.fft import rfft, rfft2, rfftn, irfft, irfft2, irfftn
from numpy.linalg import eigh, norm
from scipy.sparse.linalg import LinearOperator
import six

ROUNDOFF = 1e-13
"""Approximate size of roundoff error for correlation matrices.

Eigenvalues less than this value will be reset to this.

Gaussian correlations with a correlation length between five and ten
cells need `ROUNDOFF` greater than 1e-15 to be numerically positive
definite.

Gaussian(15) needs 1e-13 > ROUNDOFF > 1e-14
"""
NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

"""
FOURIER_NEAR_ZERO = 1e-20
"""Where fourier coefficients are treated as zero."""


class HomogeneousIsotropicCorrelation(LinearOperator):
    """Homogeneous isotropic correlations using FFTs.

    Assumes periodic domain.  Use padding or a larger domain to avoid
    this causing problems.

    See Also
    --------
    scipy.linalg.solve_circulant
        I stole the idea from here.
    """

    def __init__(self, corr_func, shape):
        """Set up the instance.

        Assumes correlations are real.

        Parameters
        ----------
        corr_func: callable(float) -> float[-1, 1]
        shape: tuple of int
            The state is formally input as a vector, but the correlations
            depend on the layout in some other shape, usually related to the
            physical layout. This is that shape.
        """
        shape = atleast_1d(shape)
        state_size = prod(shape)

        super(HomogeneousIsotropicCorrelation, self).__init__(
            dtype=float, shape=(state_size, state_size))

        ndims = len(shape)
        if ndims == 1:
            self._fft = rfft
            self._ifft = irfft
        elif ndims == 2:
            self._fft = rfft2
            self._ifft = irfft2
        else:
            self._fft = functools.partial(
                rfftn, axes=arange(-ndims, 0, dtype=int))
            self._ifft = functools.partial(
                irfftn, axes=arange(-ndims, 0, dtype=int))

        broadcastable_shape = shape[:, newaxis]
        while broadcastable_shape.ndim < len(shape) + 1:
            broadcastable_shape = broadcastable_shape[..., newaxis]

        def corr_from_index(*index):
            """Correlation of index with zero.

            Turns a correlation function in terms of index distance
            into one in terms of indices on a periodic domain.

            Parameters
            ----------
            index: tuple of int

            Returns
            -------
            float[-1, 1]

            See Also
            --------
            DistanceCorrelationFunction.correlation_from_index
            """
            comp2_1 = square(index)
            # Components of distance to shifted origin
            comp2_2 = square(broadcastable_shape - index)
            # use the smaller components to get the distance to the
            # closest of the shifted origins
            comp2 = where(comp2_1 < comp2_2, comp2_1, comp2_2)
            return corr_func(sqrt(np_sum(comp2, axis=0)))

        corr_struct = fromfunction(corr_from_index, shape)
        self._corr_fourier = self._fft(corr_struct)
        # This is also affected by roundoff
        self._fourier_near_zero = self._corr_fourier < FOURIER_NEAR_ZERO
        self._underlying_shape = shape

    @classmethod
    def from_function(cls, corr_func, shape):
        """Create an instance to apply the correlation function.

        Parameters
        ----------
        corr_func: callable(dist) -> float
        shape: tuple of int
            The state is formally a vector, but the correlations are
            assumed to depend on the layout in some other shape,
            usually related to the physical layout. This is the other
            shape.

        Returns
        -------
        HomogeneousIsotropicCorrelation
        """
        return cls(corr_func, shape)

    @classmethod
    def from_array(cls, corr_array):
        """Create an instance with the given correlations.

        Parameters
        ----------
        corr_array: array_like

        Returns
        -------
        HomogeneousIsotropicCorrelation
        """
        return cls(corr_array.__getitem__, corr_array.shape)

    def _matvec(self, vec):
        """The matrix product of this matrix and `vec`.

        Parameters
        ----------
        vec: array_like[N]

        Returns
        -------
        array_like[N]
        """
        field = vec.reshape(self._underlying_shape)

        spectral_field = self._fft(field)
        spectral_field *= self._corr_fourier
        result = self._ifft(spectral_field)

        return result.reshape(self.shape[-1])

    def solve(self, vec):
        """Solve A x = vec.

        Parameters
        ----------
        vec: array_like[N]

        Returns
        -------
        array_like[N]
            Solution of `self @ x = vec`
        """
        field = vec.reshape(self._underlying_shape)

        spectral_field = self._fft(field)
        spectral_field /= self._corr_fourier
        # Dividing by a small number is numerically unstable. This is
        # nearly an SVD solve already, so borrow that solution.
        spectral_field[self._fourier_near_zero] = 0
        result = self._ifft(spectral_field)

        return result.reshape(self.shape[-1])


def make_matrix(corr_func, shape):
    """Make a correlation matrix for a domain with shape `shape`.

    Parameters
    ----------
    corr_func: callable(float) -> float[-1, 1]
        Function giving correlation between two indices a distance d
        from each other.
    shape: tuple of int
        The underlying shape of the domain. It is viewed as a vector
        here, but may be more naturally seen as an N-D array. This is
        the shape of that array.
        `N = prod(shape)`

    Returns
    -------
    corr: np.ndarray[N, N]

    """
    shape = tuple(atleast_1d(shape))
    n_points = prod(shape)

    tmp_res = fromfunction(corr_func.correlation_from_index,
                           2 * shape).reshape(
        (n_points, n_points))
    where_small = tmp_res < NEAR_ZERO
    where_small &= tmp_res > -NEAR_ZERO

    # This isn't always positive definite.  I reset the values on
    # the negative side of roundoff to the positive side
    vals, vecs = eigh(tmp_res)
    del tmp_res
    vals[vals < ROUNDOFF] = ROUNDOFF

    result = dot(vecs, diag(vals).dot(vecs.T))

    # Now, there's more roundoff
    # make the values that were originally small zero
    result[where_small] = 0
    return result


class DistanceCorrelationFunction(six.with_metaclass(abc.ABCMeta)):
    """A correlation function that depends only on distance."""

    def __init__(self, length):
        """Set up instance.

        Parameters
        ----------
        length: float
            The correlation length in index space. Unitless.
        """
        self._length = float(length)

    @abc.abstractmethod
    def __call__(self, dist):
        """The correlation between points whose indices differ by dist.

        Parameters
        ----------
        dist: float

        Returns
        -------
        correlation: float[-1, 1]
        """
        pass

    def correlation_from_index(self, *indices):
        """Find the correlation between the indices.

        Should be independent of the length of the underlying shape,
        but `indices` must still be even.

        Parameters
        ----------
        indices: tuple of int

        Returns
        -------
        float

        """
        half = len(indices) // 2
        point1 = asanyarray(indices[:half])
        point2 = asanyarray(indices[half:])
        dist = norm(point1 - point2, axis=0)
        return self(dist)


class GaussianCorrelation(DistanceCorrelationFunction):
    """A gaussian correlation structure.

    Note
    ----
    Correlation given by exp(-dist**2 / (2 * length**2)) where dist is the
    distance between the points.
    """

    def __call__(self, dist):
        """The correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        scaled_dist2 = square(dist / self._length)
        return exp(-.5 * scaled_dist2)


class ExponentialCorrelation(DistanceCorrelationFunction):
    """A exponential correlation structure.

    Note
    ----
    Correlation given by exp(-dist/length)
    where dist is the distance between the points.
    """

    def __call__(self, dist):
        """The correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        return exp(-dist/self._length)
