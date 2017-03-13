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
