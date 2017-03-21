"""Facilities for working with ensembles."""

import numpy as np


def mean(ensemble_states):
    """Ensemble mean.

    Parameters
    ----------
    ensemble_states: np.ndarray[N, K]

    Returns
    -------
    ensemble_mean: np.ndarray[N]
    """
    return np.mean(ensemble_states, axis=-1)


def spread(ensemble_states):
    """Ensemble spread.

    Assume this uses the definition of variance used in derivation of
    OI as the BLUE.

    Parameters
    ----------
    ensemble_states: np.ndarray[N, K]

    Returns
    -------
    ensemble_spread: float

    """
    return np.sum(np.var(ensemble_states, axis=-1))


def perturbations(ensemble_states):
    """Ensemble_perturbations.

    Parameters
    ----------
    ensemble_states: np.ndarray[N, K]

    Returns
    -------
    ensemble_perturbations: np.ndarray[N, K]
    """
    ensemble_mean = mean(ensemble_states)
    # Ensure broadcasting occurs over the correct dimension
    return ensemble_states - ensemble_mean[:, np.newaxis]
