"""Implementations of the perturbed obs filter.

Calculate a localized :math:`P^B`, generate observations perturbed in
accordance with the observation covariances for each ensemble member,
and pass these off to one of the
:mod:`inversion.optimal_interpolation`, :mod:`inversion.variational`,
or :mod:`inversion.psas` implementations.
"""

from __future__ import absolute_import, print_function, division

from numpy import empty_like

import inversion.optimal_interpolation
import inversion.correlations
import inversion.ensemble
import inversion.noise
from inversion.noise import gaussian_noise


def simple(ensemble, localization_matrix,
           observations, observation_covariance,
           observation_operator, assimilator):
    """Straightforward implementation of a  perturbed obs filter.

    Assumes a full covariance matrix can fit in memory twice with room
    to spare.

    Parameters
    ----------
    ensemble: np.ndarray[N, K]
    localization_matrix: np.ndarray[N, N]
        Classes in :mod:`inversion.correlations` are an easy way to
        get this.  Unfortunately, building this here for the 2D
        correlations requires the domain size, so this takes the matrix.
    observations: np.ndarray[M]
    observation_covariance: np.ndarray[M,M]
    observation_operator: np.ndarray[M,N]
    assimilator: callable
        The data assimilation function that will perform the individual
        assimilations.

    Returns
    -------
    new_ensemble: np.ndarray[N, K]
    """
    ensemble_size = ensemble.shape[-1]
    perturbations = inversion.ensemble.perturbations(ensemble)

    # If I assume the ROI for observations << localization length and
    # can somehow get (L * (X @ X.T)) @ H.T in terms of X @ (H @ X).T, I can
    # reduce the memory footprint.  Until then, I do things this way.
    background_covariance = (
        1 / (ensemble_size - 1) *
        perturbations.dot(perturbations.T) *
        localization_matrix
    )

    observation_noise = gaussian_noise(observation_covariance, ensemble_size)

    new_ensemble = empty_like(ensemble, order="F")

    for i in range(ensemble_size):
        new_ensemble[:, i], _ = assimilator(
            ensemble[:, i], background_covariance,
            observations + observation_noise[i, :],
            observation_covariance, observation_operator)

    return new_ensemble
