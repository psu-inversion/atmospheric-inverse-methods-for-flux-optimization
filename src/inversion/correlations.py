"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc
import functools
from itertools import islice

# Need to specify numpy versions in some instances.
import numpy as np
# Not in dask.array
from numpy.linalg import eigh, norm
# arange changes signature
from numpy import arange, newaxis, asanyarray
from scipy.special import gamma, kv as K_nu

from numpy import fromfunction, asarray, hstack
from numpy import exp, square, fmin, sqrt, zeros
from numpy import logical_or, concatenate, isnan
from numpy import sum as da_sum
from numpy import where
from numpy.fft import rfft, rfft2, rfftn, irfft, irfft2, irfftn
import six

from inversion.util import schmidt_decomposition, is_odd
from inversion.util import tolinearoperator, kron
from inversion.util import DaskLinearOperator

ROUNDOFF = 1e-13
"""Approximate size of roundoff error for correlation matrices.

Eigenvalues less than this value will be reset to this.

Gaussian correlations with a correlation length between five and ten
cells need `ROUNDOFF` greater than 1e-15 to be numerically positive
definite.

Gaussian(15) needs 1e-13 > ROUNDOFF > 1e-14

Also used in SchmidtKroneckerProduct to determine how many terms in the
Schmidt decomposition should be used.
"""
NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.
"""
FOURIER_NEAR_ZERO = 1e-15
"""Where fourier coefficients are treated as zero.

1e-20 produces overflow with the dask tests
"""
DTYPE = np.float64


class HomogeneousIsotropicCorrelation(DaskLinearOperator):
    """Homogeneous isotropic correlations using FFTs.

    Assumes periodic domain.  Use padding or a larger domain to avoid
    this causing problems.

    See Also
    --------
    scipy.linalg.solve_circulant
        I stole the idea from here.
    """

    def __init__(self, shape):
        """Set up the instance.

        .. note::

            Do not invoke this directly. Use
            :func:`HomogeneousIsotropicCorrelation.from_function` or
            :func:`HomogeneousIsotropicCorrelaiton.from_array` instead.

        Parameters
        ----------
        corr_func: callable(float) -> float[-1, 1]
        shape: tuple of int
            The state is formally input as a vector, but the correlations
            depend on the layout in some other shape, usually related to the
            physical layout. This is that shape.
        """
        state_size = np.prod(shape)

        super(HomogeneousIsotropicCorrelation, self).__init__(
            dtype=DTYPE, shape=(state_size, state_size))

        self._fft, self._ifft = self._rfft_irfft(shape)
        self._underlying_shape = tuple(shape)

    # noqa W0212
    @staticmethod
    def _rfft_irfft(shape):
        """Get the forward and inverse rffts for the given dimensions.

        Parameters
        ----------
        ndim: int

        Returns
        -------
        fft, ifft: callable
        """
        ndims = len(shape)
        if ndims == 1:
            fft = functools.partial(rfft, axis=0)
            ifft = functools.partial(irfft, n=shape[0], axis=0)
        elif ndims == 2:
            fft = functools.partial(rfft2, axes=(0, 1))
            ifft = functools.partial(irfft2, s=shape, axes=(0, 1))
        else:
            fft = functools.partial(
                rfftn, axes=arange(0, ndims, dtype=int))
            ifft = functools.partial(
                irfftn, axes=arange(0, ndims, dtype=int), s=shape)
        return fft, ifft

    # noqa W0212
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
        shape = np.atleast_1d(shape)
        ndims = len(shape)
        self = cls(shape)

        broadcastable_shape = shape[:, newaxis]
        while broadcastable_shape.ndim < ndims + 1:
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
            comp2 = fmin(comp2_1, comp2_2)
            return corr_func(sqrt(da_sum(comp2, axis=0)))

        corr_struct = fromfunction(
            corr_from_index, shape=tuple(shape),
            dtype=DTYPE)

        # The dask fft functions require all relevant axes to be in
        # memory already, so keeping the array in memory won't hurt.
        # TODO delegate to from_array
        corr_fourier = (self._fft(corr_struct))
        self._corr_fourier = (corr_fourier)
        # This is also affected by roundoff
        self._fourier_near_zero = abs(corr_fourier) < FOURIER_NEAR_ZERO
        return self

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
        corr_array = asarray(corr_array)
        self = cls(corr_array.shape)
        # The fft axes need to be a single chunk for the dask ffts
        # It's in memory already anyway
        # TODO: create a from_spectrum to delegate to
        corr_fourier = (self._fft(corr_array))
        self._corr_fourier = (corr_fourier)
        self._fourier_near_zero = (corr_fourier < FOURIER_NEAR_ZERO)
        return self

    def sqrt(self):
        """Compute an S such that S.T @ S == self.

        Returns
        -------
        S: HomogeneousLinearOperator
        """
        result = HomogeneousIsotropicCorrelation(self._underlying_shape)
        result._corr_fourier = sqrt(self._corr_fourier)
        # I still don't much trust these.
        result._fourier_near_zero = self._fourier_near_zero
        return result

    def _matvec(self, vec):
        """Evaluate the matrix product of this matrix and `vec`.

        Parameters
        ----------
        vec: array_like[N]

        Returns
        -------
        array_like[N]
        """
        _shape = self._underlying_shape
        field = asarray(vec).reshape(_shape)
        # TODO: Test this

        spectral_field = self._fft(field)
        spectral_field *= self._corr_fourier
        result = self._ifft(spectral_field)

        return result.reshape(vec.shape)

    _rmatvec = _matvec
    # Matrix is symmetric, so self.T @ x = self @ x

    def _matmat(self, mat):
        """Evaluate the matrix product of self and `mat`.

        Parameters
        ----------
        mat: array_like[N, K]

        Returns
        -------
        array_like[N, K]
        """
        _shape = self._underlying_shape
        fields = asarray(mat).reshape(_shape + (-1,))
        # TODO: Test this

        spectral_fields = self._fft(fields)
        spectral_fields *= self._corr_fourier[..., np.newaxis]
        results = self._ifft(spectral_fields)

        return results.reshape(mat.shape)

    def _transpose(self):
        """Evaluate the transpose of this operator."""
        # Correlation matrices are symmetric.
        return self

    _adjoint = _transpose
    # Correlation matrices are also real

    def inv(self):
        """Construct the matrix inverse of this operator."""
        # TODO: Test this
        # TODO: Return a HomogeneousIsotropicLinearOperator
        return DaskLinearOperator(
            shape=self.shape, dtype=self.dtype,
            matvec=self.solve, rmatvec=self.solve)

    def solve(self, vec):
        """Solve A @ x = vec.

        Parameters
        ----------
        vec: array_like[N]

        Returns
        -------
        array_like[N]
            Solution of `self @ x = vec`
        """
        field = asarray(vec).reshape(self._underlying_shape)

        spectral_field = self._fft(field)
        spectral_field /= self._corr_fourier
        # Dividing by a small number is numerically unstable. This is
        # nearly an SVD solve already, so borrow that solution.
        spectral_field[self._fourier_near_zero] = 0
        result = self._ifft(spectral_field)

        return result.reshape(self.shape[-1])

    def kron(self, other):
        """Construct the Kronecker product of this operator and other.

        Parameters
        ----------
        other: HomogeneousIsotropicCorrelation
            The other operator for the Kronecker product.
            This implementation will accept other objects,
            passing them along to :class:`SchmidtKroneckerProduct`.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
        """
        if not isinstance(other, HomogeneousIsotropicCorrelation):
            return SchmidtKroneckerProduct(self, other)
        shape = self._underlying_shape + other._underlying_shape
        shift = len(self._underlying_shape)

        self_index = tuple(slice(None) if i < shift else np.newaxis
                           for i in range(len(shape)))
        other_index = tuple(np.newaxis if i < shift else slice(None)
                            for i in range(len(shape)))

        # rfft makes the first axis half the size
        # When combining things like this, I need to re-double that size again
        if is_odd(self._underlying_shape[-1]):
            reverse_start = -1
        else:
            reverse_start = -2
        expanded_fft = hstack(
            (self._corr_fourier,
             self._corr_fourier[..., reverse_start:0:-1].conj()))

        expanded_near_zero = concatenate(
            (self._fourier_near_zero,
             self._fourier_near_zero[..., reverse_start:0:-1]), axis=-1)

        newinst = HomogeneousIsotropicCorrelation(shape)
        newinst._corr_fourier = (expanded_fft[self_index] *
                                 other._corr_fourier[other_index])
        newinst._fourier_near_zero = logical_or(
            expanded_near_zero[self_index],
            other._fourier_near_zero[other_index])
        return newinst


class SchmidtKroneckerProduct(DaskLinearOperator):
    """Kronecker product of two operators using Schmidt decomposition.

    This works best when the input vectors are nearly Kronecker
    products as well, dominated by some underlying structure with
    small variations.  One example would be average net flux + trend
    in net flux + average daily cycle + daily cycle timing variations
    across domain + localized events + ...

    Multiplications are roughly the same time complexity class as with
    an explicit Kronecker Product, perhaps a factor of two or three
    slower in the best case, but the memory requirements are
    :math:`N_1^2 + N_2^2` rather than :math:`(N_1 * N_2)^2`, plus this
    approach works with sparse matrices and other LinearOperators
    which can further reduce the memory requirements and may decrease
    the time complexity.

    Forming the Kronecker product from the component vectors currently
    requires the whole thing to be in memory, so a new implementation
    of kron would be needed to take advantage of this. There may be
    some difficulties with the dask cache getting flushed and causing
    repeat work in this case. I don't know how to get around this.
    """

    def __init__(self, operator1, operator2):
        """Set up the instance.

        Parameters
        ----------
        operator1, operator2: scipy.sparse.linalg.LinearOperator
            The operators input to the Kronecker product.
        """
        operator1 = tolinearoperator(operator1)
        operator2 = tolinearoperator(operator2)
        total_shape = np.multiply(operator1.shape, operator2.shape)

        super(SchmidtKroneckerProduct, self).__init__(
            shape=tuple(total_shape),
            dtype=np.result_type(operator1.dtype, operator2.dtype))

        self._inshape1 = operator1.shape[1]
        self._inshape2 = operator2.shape[1]
        self._operator1 = operator1
        self._operator2 = operator2

    def _matvec(self, vector):
        """Evaluate the indicated matrix-vector product.

        Parameters
        ----------
        vector: array_like[N]

        Returns
        -------
        array_like[M]
        """

        if vector.ndim == 1:
            result_shape = self.shape[0]
        else:
            result_shape = (self.shape[0], 1)

        lambdas, vecs1, vecs2 = schmidt_decomposition(
            asarray(vector), self._inshape1, self._inshape2)

        # The vector should fit in memory, and I need specific
        # elements of lambdas
        lambdas = np.asarray(lambdas)
        vecs1 = np.asarray(vecs1)
        vecs2 = np.asarray(vecs2)

        small_lambdas = np.nonzero(lambdas < lambdas[0] * ROUNDOFF)[0]
        if small_lambdas.any():
            last_lambda = int(small_lambdas[0])
        else:
            last_lambda = len(lambdas)

        result = zeros(shape=result_shape,
                       dtype=np.result_type(self.dtype, vector.dtype))
        for lambd, vec1, vec2 in islice(zip(lambdas, vecs1, vecs2),
                                        0, last_lambda):
            result += kron(
                asarray(lambd * self._operator1.dot(vec1).reshape(-1, 1)),
                asarray(self._operator2.dot(vec2).reshape(-1, 1))
            ).reshape(result_shape)

        return asarray(result)

    def _matmat(self, matrix):
        """Evaluate the indicated matrix-matrix product.

        Parameters
        ----------
        matrix: array_like

        Returns
        -------
        array_like
        """
        # TODO: look into Kronecker decomposition of matrix
        # as indicated in references below.
        # Mathematica code here:
        # https://mathematica.stackexchange.com/
        # questions/91651/nearest-kronecker-product,
        # drawn from Pitsianis-Van Loan algorithm from here:
        # https://link.springer.com/chapter/10.1007%2F978-94-015-8196-7_17
        return hstack([self.matvec(column.reshape(-1, 1))
                       for column in matrix.T])
    #     result = zeros(shape=(self.shape[0], matrix.shape[1]),
    #                    dtype=np.result_type(self.dtype, matrix.dtype))
    #     for i, column in enumerate(matrix.T):
    #         result[:, i] = self._matvec(column)
    #     return result


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

    See Also
    --------
    :func:`statsmodels.stats.correlation_tools.corr_clipped`, which
    does something similar, and refers to other functions that may
    give more accurate results.

    Returns
    -------
    corr: np.ndarray[N, N]
        Positive definite dense array, entirely in memory
    """
    shape = tuple(np.atleast_1d(shape))
    n_points = np.prod(shape)

    # Since dask doesn't have eigh, using dask in this section slows
    # the test suite by about 25%.  Since it all ends up in memory,
    # may as well start with it there instead of converting back and
    # forth a few times.
    tmp_res = np.fromfunction(corr_func.correlation_from_index,
                              shape=2 * shape,
                              dtype=DTYPE).reshape(
        (n_points, n_points))
    where_small = tmp_res < NEAR_ZERO
    where_small &= tmp_res > -NEAR_ZERO

    # This isn't always positive definite.  I reset the values on
    # the negative side of roundoff to the positive side
    vals, vecs = eigh(tmp_res)
    del tmp_res
    vals[vals < ROUNDOFF] = ROUNDOFF

    result = np.dot(vecs, np.diag(vals).dot(vecs.T))

    # Now, there's more roundoff
    # make the values that were originally small zero
    result[where_small] = 0
    return asarray(result)


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
        """Get the correlation between points whose indices differ by dist.

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
        """Get the correlation between the points.

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
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        return exp(-dist / self._length)


class BalgovindCorrelation(DistanceCorrelationFunction):
    """A Balgovind correlation structure.

    Follows Balgovind et al. 1983, modified so the correlation length
    better matches that used by other correlation functions.

    Note
    ----
    Correlation given by :math:`(1 + 2*dist/length) exp(-2*dist/length)`

    Note
    ----
    This implementation has problems for length == 10.
    I have no idea why.  3 and 30 are fine.
    """

    def __init__(self, length):
        """Set up instance with proper length.

        This folds a constant into the length.
        """
        super(BalgovindCorrelation, self).__init__(.5 * length)

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        scaled_dist = dist / self._length
        return (1 + scaled_dist) * exp(-scaled_dist)


class MaternCorrelation(DistanceCorrelationFunction):
    r"""A Matern correlation structure.

    Follows Matern (1986) *Spatial Variation*

    Note
    ----
    Correlation given by
    :math:`[2^{\kappa-1}\Gamma(\kappa)]^{-1} (d/L)^{\kappa} K_{\kappa}(d/L)`
    where :math:`\kappa` is a smoothness parameter and
    :math:`K_{\kappa}` is a modified Bessel function of the third kind.
    """

    def __init__(self, length, kappa=1):
        r"""Set up instance.

        Parameters
        ----------
        length: float
            The correlation length in index space. Unitless.
        kappa: float
            The smoothness parameter
            :math:`kappa=\infty` is equivalent to Gaussian correlations
            :math:`kappa=\frac{1}{2}` is equivalent to exponential
            Default value is only for full equivalence with other classes.
            The default value is entirely arbitrary and may change without
            notice.
        """
        super(MaternCorrelation, self).__init__(length)
        self._kappa = kappa
        # Make sure correlation at zero is one
        self._scale_const = .5 * gamma(kappa)

    def __call__(self, dist):
        """Get the correlation between the points.

        Parameters
        ----------
        dist: float

        Returns
        -------
        corr: float
        """
        kappa = self._kappa
        scaled_dist = dist / self._length
        result = ((.5 * scaled_dist) ** kappa *
                  K_nu(kappa, scaled_dist) / self._scale_const)
        # K_nu returns nan at zero
        return where(isnan(result), 1, result)
