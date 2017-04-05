"""Classes for ensemble inflation.

Parameters set at instantion; thereafter act as function calls.

"""
import random

import inversion.ensemble


class MultiplicativeInflation:
    """Multiplicative inflation of ensemble perturbations."""

    def __init__(self, factor):
        """Set up factor.

        Parameter
        ---------
        factor: float
        """
        self._factor = factor

    def __call__(self, ensemble):
        """Apply multiplicative inflation to the ensemble.

        Parameters
        ----------
        ensemble: array_like[K, N]

        Returns
        -------
        new_ensemble: array_like[K, N]
        """
        mean, perturbations = inversion.ensemble.mean_and_perturbations(
            ensemble)

        perturbations *= self._factor

        return inversion.ensemble.states_from_perturbations(
            mean, perturbations)


class AdditiveInflation:
    """Additive inflation of ensemble perturbations."""

    def __init__(self, climatology, factor):
        """Set climatology to be used for inflation.

        Parameters
        ----------
        climatology: array_like[T_climo, N]
            The climatology from which to draw inflating perturbations
        factor: float
            How much of a perturbation to add
        """
        self._climatology = climatology
        self._factor = factor

    def __call__(self, ensemble):
        """Apply additive inflation to the ensemble.

        Parameters
        ----------
        ensemble: array_like[K, N]

        Returns
        -------
        new_ensemble: array_like[K, N]
        """
        samp_ind = random.sample(
            range(self._climatology.shape[0]), ensemble.shape[0])
        sample_states = self._climatology[samp_ind, :]

        samp_ind = random.sample(
            range(self._climatology.shape[0]), ensemble.shape[0])
        sample_states -= self._climatology[samp_ind, :]

        sample_states *= self._factor
        return ensemble + sample_states
