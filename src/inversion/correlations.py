"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc
import functools

from numpy import ones_like, fromfunction, arange
from numpy import sqrt, exp, dot, diag, prod
from numpy.fft import rfft, rfft2, rfftn, irfft, irfft2, irfftn
from numpy.linalg import eigh
from scipy.sparse.linalg import LinearOperator
import six

ROUNDOFF = 1e-13
"""Approximate size of roundoff error for correlation matrices.

Eigenvalues less than this value will be reset to this.

Gaussian correlations with a correlation length between five and ten
cells need `ROUNDOFF` greater than 1e-15 to be positive definite.

Gaussian(15) needs 1e-13 > ROUNDOFF > 1e-14
"""
NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

"""


class HomogeneousIsotropicCorrelation(LinearOperator):
    """Homogeneous isotropic correlations using FFTs.

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
        corr_func: callable
        shape: tuple of int
            Shape of input domain expected by `corr_func`.
            The state is formally input as a vector, but the correlations
            depend on the layout in some other shape, usually related to the
            physical layout. This is that shape.
        """
        state_size = prod(shape)

        super(HomogeneousIsotropicCorrelation, self).__init__(
            self, dtype=float, shape=(state_size, state_size),
            matvec=self._matvec, rmatvec=self._matvec, matmat=self._matmat)

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

        corr_struct = fromfunction(corr_func, tuple(ones_like(shape)) + shape)
        self._corr_fourier = self._fft(corr_struct[0, 0])
        # This is also affected by roundoff
        self._fourier_near_zero = self._corr_fourier < ROUNDOFF
        self._underlying_shape = shape

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


class CorrelationFunction2D(six.with_metaclass(abc.ABCMeta)):
    """Base class for 2D correlations."""

    def __init__(self, length):
        """Set up the instance.

        Parameters
        ----------
        length: float
            Unitless: physical correlation length / grid spacing
        """
        self._length = float(length)

    @abc.abstractmethod
    def __call__(self, y1, x1, y2, x2):
        """The correlation between the points.

        Argument order for use with :func:`np.fromfunction`.  Should
        be similar to results from :func:`scipy.linalg.kron` when
        reshaped to a 2D array for separable correlation functions

        Parameters
        ----------
        x1, x2: float
        y1, y2: float

        Returns
        -------
        corr: float

        """
        pass

    def make_matrix(self, ny, nx):
        """Make a correlation matrix for an `nx` by `ny` domain.

        Parameters
        ----------
        ny: int
        nx: int

        Returns
        -------
        corr: np.ndarray[ny*nx, ny*nx]
        """
        n_points = ny * nx
        tmp_res = fromfunction(self, (ny, nx, ny, nx)).reshape(
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


class Gaussian2DCorrelation(CorrelationFunction2D):
    """A 2D gaussian correlation structure.

    Note
    ----
    Correlation given by exp(-dist**2 / (2 * length**2)) where dist is the
    distance between the points.
    """

    def __call__(self, y1, x1, y2, x2):
        """The correlation between the points.

        Argument order mirrors :func:`np.fromfunction`

        Parameters
        ----------
        x1, x2: float
        y1, y2: float

        Returns
        -------
        corr: float
        """
        dist2 = ((x1 - x2)**2 + (y1 - y2)**2)
        return exp(-dist2 / (2 * self._length ** 2))


class Exponential2DCorrelation(CorrelationFunction2D):
    """A 2D exponential correlation structure.

    Note
    ----
    Correlation given by exp(-dist/length)
    where dist is the distance between the points.
    """

    def __call__(self, y1, x1, y2, x2):
        """The correlation between the points.

        Argument order mirrors :func:`np.fromfunction`

        Parameters
        ----------
        x1, x2: float
        y1, y2: float

        Returns
        -------
        corr: float
        """
        dist = sqrt((x1 - x2)**2 + (y1-y2)**2)
        return exp(-dist/self._length)


class CorrelationFunction1D(six.with_metaclass(abc.ABCMeta)):
    """Base class for 1D correlations."""

    def __init__(self, length):
        """Set up the instance.

        Parameters
        ----------
        length: float
            Unitless: physical correlation time / bin dt
        """
        self._length = float(length)

    @abc.abstractmethod
    def __call__(self, t1, t2):
        """The correlation between the points.

        Argument order for use with :func:`np.fromfunction`.

        Parameters
        ----------
        t1, t2: float

        Returns
        -------
        corr: float

        """
        pass

    def make_matrix(self, nt):
        """Make a correlation matrix for `nt` times.

        Parameters
        ----------
        nt: int

        Returns
        -------
        corr: np.ndarray[nt, nt]
        """
        tmp_res = fromfunction(self, (nt, nt))
        where_small = tmp_res < NEAR_ZERO
        where_small &= tmp_res > -NEAR_ZERO

        # This isn't always positive definite.  I reset the values on
        # the negative side of roundoff to the positive side
        vals, vecs = eigh(tmp_res)
        del tmp_res
        vals[vals < ROUNDOFF] = ROUNDOFF

        result = dot(vecs, diag(vals).dot(vecs.T))

        # Now there's more roundoff
        # make the values that were originally small zero
        result[where_small] = 0
        return result


class Exponential1DCorrelation(CorrelationFunction1D):
    """A 1D exponential correlation structure.

    Should model AR(1) processes fairly well

    Note
    ----
    Correlation given by exp(-dist/length)
    where dist is the distance between the points.
    """

    def __call__(self, t1, t2):
        """The correlation between the points.

        Argument order mirrors :func:`np.fromfunction`

        Parameters
        ----------
        t1, t2: float

        Returns
        -------
        corr: float
        """
        dist = abs(t1 - t2)
        return exp(-dist/self._length)


class Gaussian1DCorrelation(CorrelationFunction1D):
    """A 1D gaussian correlation structure.

    Note
    ----
    Correlation given by exp(-dist**2 / (2 * length**2)) where dist is the
    distance between the points.
    """

    def __call__(self, t1, t2):
        """The correlation between the points.

        Argument order mirrors :func:`np.fromfunction`

        Parameters
        ----------
        x1, x2: float
        y1, y2: float

        Returns
        -------
        corr: float
        """
        dist2 = (t1 - t2)**2
        return exp(-dist2 / (2 * self._length ** 2))
