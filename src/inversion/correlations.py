"""Correlations; used for assumed background covariances.

Multiply by diagonal matrices of the assumed standard deviations on
the right and left to obtain covariance matrices.

"""
import abc

from numpy import sqrt, exp
import six


class SpatialCorrelationFunction(six.with_metaclass(abc.ABCMeta)):
    """Base class for the common spatial correlations."""

    def __init__(self, length):
        """Set up the instance.

        Parameters
        ----------
        length: float
        """
        self._length = float(length)

    @abc.abstractmethod
    def __call__(self, x1, y1, x2, y2):
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
        pass


class GaussianSpatialCorrelation(SpatialCorrelationFunction):
    """A gaussian spatial correlation structure.

    Note
    ----
    Correlation given by exp(-dist**2 / (2 * length**2)) where dist is the
    distance between the points.
    """

    def __call__(self, x1, y1, x2, y2):
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

    def __call__(self, x1, y1, x2, y2):
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
