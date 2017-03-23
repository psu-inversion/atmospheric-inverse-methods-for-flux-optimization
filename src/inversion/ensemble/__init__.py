"""Facilities for working with ensembles."""

import numpy as np


def mean(ensemble_states):
    """Ensemble mean.

    Parameters
    ----------
    ensemble_states: np.ndarray[K, N]

    Returns
    -------
    ensemble_mean: np.ndarray[N]
    """
    return np.mean(ensemble_states, axis=0)


def spread(ensemble_states):
    """Ensemble spread.

    Assume this uses the definition of variance used in derivation of
    OI as the BLUE.

    Parameters
    ----------
    ensemble_states: np.ndarray[K, N]

    Returns
    -------
    ensemble_spread: float

    """
    return np.sum(np.var(ensemble_states, axis=0))


def mean_and_perturbations(ensemble_states):
    """Ensemble mean and perturbations.

    Parameters
    ----------
    ensemble_states: np.ndarray[K, N]

    Returns
    -------
    ensemble_mean: np.ndarray[N]
    ensemble_perturbations: np.ndarray[K, N]

    See Also
    --------
    states_from_perturbations
    """
    ensemble_mean = mean(ensemble_states)
    # Ensure broadcasting occurs over the correct dimension
    return ensemble_mean, ensemble_states - ensemble_mean[np.newaxis, :]


def perturbations(ensemble_states):
    """Ensemble perturbations.

    Parameters
    ----------
    ensemble_states: np.ndarray[K, N]

    Returns
    -------
    ensemble_perturbations: np.ndarray[K, N]
    """
    return mean_and_perturbations(ensemble_states)[-1]


def states_from_perturbations(ensemble_mean, ensemble_perturbations):
    """Recover ensemble states from mean and perturbations.

    Parameters
    ----------
    ensemble_mean: np.ndarray[N]
    ensemble_perturbations: np.ndarray[K, N]

    Returns
    -------
    ensemble_states: np.ndarray[K, N]

    See Also
    --------
    mean_and_perturbations
    """
    return ensemble_mean[np.newaxis, :] + ensemble_perturbations
