"""Support classes for covariances.

See Also
--------
atmos_flux_inversion.correlations
"""
from __future__ import absolute_import

import numpy as np
from numpy import newaxis, sqrt

from .linalg import (
    ProductLinearOperator, SelfAdjointLinearOperator,
    DiagonalOperator, matrix_sqrt)

OBSERVATION_INTERVAL = np.array(1, dtype="m8[h]")
"""The time between individual observations at one site.

More precisely, these are the units used with the correlation function
used in :func:`observation_covariance_matrix`.
"""


class CorrelationStandardDeviation(ProductLinearOperator,
                                   SelfAdjointLinearOperator):
    """Represent correlation-std product."""

    def __init__(self, correlation, std):
        """Set up instance to use given parameters.

        Parameters
        ----------
        correlation: LinearOperator[N, N]
            Correlations
        std: array_like[N]
            Standard deviations
        """
        std_matrix = DiagonalOperator(std)
        super(CorrelationStandardDeviation, self).__init__(
            std_matrix, correlation, std_matrix)

    def _transpose(self):
        """Return transpose of self."""
        return self  # pragma: no cover

    def _adjoint(self):
        """Return adjoint of self."""
        return self

    def sqrt(self):
        """Find S such that S.T @ S == self."""
        std_matrix, correlation, _ = self._operators

        return ProductLinearOperator(
            matrix_sqrt(correlation), std_matrix)

    _sqrt = sqrt


def observation_covariance_matrix(variance_series, correlation_function):
    """Create the observation covariance matrix given parammeters.

    Assumes the observations are far enough apart to be uncorrelated.

    Parameters
    ----------
    variance_series : pandas.Series[n_observations]
    correlation_function : DistanceCorrelationFunction

    Returns
    -------
    np.ndarray
    """
    obs_time_name = [name for name in variance_series.index.names
                     if "time" in name][0]
    obs_site_name = [name for name in variance_series.index.names
                     if "site" in name or "name" in name][0]
    obs_times = variance_series.index.get_level_values(obs_time_name)
    obs_sites = variance_series.index.get_level_values(obs_site_name)

    # Only correlations for now
    observation_covariance = correlation_function(
        abs(obs_times[:, newaxis] - obs_times[newaxis, :]) /
        OBSERVATION_INTERVAL
    )
    observation_covariance[
        obs_sites.values[:, newaxis] != obs_sites.values[newaxis, :]
    ] = 0

    # now turn it into covariance
    obs_stds = sqrt(variance_series.values)
    observation_covariance *= obs_stds[:, newaxis]
    observation_covariance *= obs_stds[newaxis, :]
    return observation_covariance
