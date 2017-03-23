"""Hybrid ensemble-static assimilation schemes."""

import numpy as np

import inversion.ensemble

class SimpleEns3DVar:

    def __init__(self, control_assimilator, ensemble_assimilator):
        """Set assimilators.

        Parameters
        ----------
        control_assimilator: callable
        ensemble_assimilator: callable
        """
        self._control_assim = control_assimilator
        self._ens_assim = ensemble_assimilator

    def __call__(self, control, ensemble,
                 background_error_covariance, localization_matrix,
                 observations, observation_error_covariance,
                 observation_operator, ens_weight=.5,
                 control_to_ensemble_function=None):
        """Assimilate the observations into the control and ensemble forecasts.

        Control analysis is used as posterior ensemble mean.
        Ensemble provides flow-dependence for control assimilation.

        Assumes full background error covariance matrix can fit in
        memory many times over.

        Currently assumes `control` and rows of `ensemble` have the
        same dtype and are the same length.

        Parameters
        ----------
        control: np.ndarray[N]
        ensemble: np.ndarray[K, N]
        background_error_covariance: np.ndarray[N, N]
        localization_matrix: np.ndarray[N, N]
        observations: np.ndarray[M]
        observation_error_covariance: np.ndarray[M, M]
        observation_operator: np.ndarray[M, N]
        ens_weight: float
        control_to_ensemble_function: callable
            Function to 

        Returns
        -------
        control_analysis: np.ndarray[N]
        analysis_ensemble: np.ndarray[K, N]

        """
        bg_cov = ((1 - ens_weight) * background_error_covariance +
                  ens_weight * ensemble.T.dot(ensemble) * localization_matrix)

        control_analysis, _ = self._control_assim(
            control, bg_cov, observations, observation_error_covariance,
            observation_operator)

        new_ensemble = self._ens_assim(
            ensemble, localization_matrix, observations,
            observation_error_covariance, observation_operator)

        analysis_perturbations = inversion.ensemble.perturbations(new_ensemble)
        analysis_ensemble = inversion.ensemble.states_from_perturbations(
            control_analysis, analysis_perturbations)

        return control_analysis, analysis_ensemble
