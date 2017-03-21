"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc

from numpy import sqrt, exp, fromfunction, dot, diag
from numpy.linalg import eigh
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


class SpatialCorrelationFunction(six.with_metaclass(abc.ABCMeta)):
    """Base class for the common spatial correlations."""

    def __init__(self, length):
        """Set up the instance.

        Parameters
        ----------
        length: float
            Unitless.
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


class GaussianSpatialCorrelation(SpatialCorrelationFunction):
    """A gaussian spatial correlation structure.

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


class ExponentialSpatialCorrelation(SpatialCorrelationFunction):
    """An exponential spatial correlation structure.

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


class TemporalCorrelationFunction(six.with_metaclass(abc.ABCMeta)):
    """Base class for the common temporal correlations."""

    def __init__(self, length):
        """Set up the instance.

        Parameters
        ----------
        length: float
            Unitless.
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


class ExponentialTemporalCorrelation(TemporalCorrelationFunction):
    """An exponential Temporal correlation structure.

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
