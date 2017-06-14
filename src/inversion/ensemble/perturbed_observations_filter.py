"""Implementations of the perturbed obs filter.

Calculate a localized :math:`P^B`, generate observations perturbed in
accordance with the observation covariances for each ensemble member,
and pass these off to one of the
:mod:`inversion.optimal_interpolation`, :mod:`inversion.variational`,
or :mod:`inversion.psas` implementations.

This is a stochastic filter, using randomized observations to account
for the filter updating perturbations rather than covariances.
"""

from __future__ import absolute_import, print_function, division

from numpy import empty_like

import inversion.optimal_interpolation
import inversion.correlations
import inversion.ensemble
import inversion.noise
from inversion.noise import gaussian_noise


class SimplePerturbedObsFilter:
    """Perturbed obs filter.

    Performs separate assimilations on each member. Alone, this
    reduces the posterior covariance to :math:`(I-KH)P_B(I-KH)` rather
    than :math:`(I-KH)P_B`. Adding random errors to the observations
    in accordance with their covariance (seperately for each
    ensemble's assimilation) brings this back where it should be.

    """

    def __init__(self, assimilator):
        """Set up using given per-member assimilator.

        Parameters
        ----------
        assimilator: callable
            The data assimilation function that will perform the individual
            assimilations.
        """
        self._assimilator = assimilator

    def __call__(self, ensemble, localization_matrix,
                 observations, observation_error_covariance,
                 observation_operator):
        """Straightforward implementation of a  perturbed obs filter.

        Assumes a full covariance matrix can fit in memory twice with room
        to spare.

        Parameters
        ----------
        ensemble: np.ndarray[K, N]
        localization_matrix: np.ndarray[N, N]
            Classes in :mod:`inversion.correlations` are an easy way to
            get this.  Unfortunately, building this here for the 2D
            correlations requires the domain shape, so this takes the matrix.
        observations: np.ndarray[M]
        observation_error_covariance: np.ndarray[M,M]
        observation_operator: np.ndarray[M,N]

        Returns
        -------
        new_ensemble: np.ndarray[K, N]
        """
        ensemble_size = ensemble.shape[0]
        perturbations = inversion.ensemble.perturbations(ensemble)

        # If I assume the ROI for observations << localization length or
        # can somehow get (L * (X @ X.T)) @ H.T in terms of X @ (H @ X).T,
        # I can reduce the memory footprint.  Until then, I do things this way.
        # Using dask arrays may also help with this.
        background_covariance = (
            1 / (ensemble_size - 1) *
            perturbations.T.dot(perturbations) *
            localization_matrix
        )

        observation_noise = gaussian_noise(observation_error_covariance,
                                           ensemble_size)

        new_ensemble = empty_like(ensemble)

        assimilator = self._assimilator
        for i in range(ensemble_size):
            new_ensemble[i, :], _ = assimilator(
                ensemble[i, :], background_covariance,
                # The posterior spread should be (I - KH)P^B
                # Without the observation perturbations, it is instead
                # (I - KH)P^B(I - KH)
                observations + observation_noise[i, :],
                observation_error_covariance, observation_operator)

        return new_ensemble
