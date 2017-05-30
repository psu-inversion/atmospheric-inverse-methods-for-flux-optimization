"""Estimate background error covariances from model runs."""

from numpy.fft import rfft, irfft
from numpy import cov, var, diff

from scipy.linalg import toeplitz

ASSIMILATION_SPIN_UP = 10
"""How many DA cycles it takes to converge to steady state."""


def nmc_covariances(forecasts, difference_lag, assume_homogeneous=False):
    """Estimated background error covariances from forecasts.

    Uses the method popularized by the National Meteorological Center
    (NMC), now the National Centers for Environmental Prediction.

    Assumes the bias is zero and removes any indication of nonzero
    means.

    Parameters
    ----------
    forecasts: array_like[N_climo, 2, N]
        Array of forecasts. The first axis goes over forecast number,
        the second over lead time, and the third over the state space.
    difference_lag: int
        The number of forecast cycles between the lead times. This is
        required to ensure the errors are generated from forecasts for
        the same valid time. This is traditionally one day, to reduce
        the effects of the diurnal cycle on the covariances.
    assume_homogeneous: bool

    Returns
    -------
    estimated_covariances: array_like[N, N]

    References
    ----------
    Parrish and Derber 1992
    """
    state_size = forecasts.shape[-1]
    differences = (forecasts[:-difference_lag, 1] -
                   forecasts[difference_lag:, 0])

    if not assume_homogeneous:
        result = cov(differences[ASSIMILATION_SPIN_UP:, :].T)
    else:
        difference_ffts = rfft(differences, axis=1)

        real_variances = var(
            difference_ffts[ASSIMILATION_SPIN_UP:].real, axis=0)
        imag_variances = var(
            difference_ffts[ASSIMILATION_SPIN_UP:].imag, axis=0)

        # This forces homogeneity.
        # Not sure if multiplying imag_variances by 1j would force isotropy
        lag_covariances = (irfft(real_variances + imag_variances,
                                 n=state_size) /
                           state_size)
        result = toeplitz(lag_covariances)

    # There are two realizations of the forecast error in
    # the differences: this corrects for that.
    result /= 2

    return result


def canadian_quick_covariances(free_run, assume_homogeneous=False):
    """Estimate background error covariances from forecast tendencies.

    Does poorly with phenomena that have a frequency that is a
    multiple of the reporting frequency.

    Parameters
    ----------
    free_run: array_like[N_climo, N]
        A free run of the model, starting once the model has spun up.
    assume_homogeneous: bool

    Returns
    -------
    estimated_covariances: array_like[N, N]
    """
    state_size = free_run.shape[-1]
    forecast_tendencies = diff(free_run)

    if not assume_homogeneous:
        result = cov(forecast_tendencies.T)
    else:
        tendency_ffts = rfft(forecast_tendencies, axis=1)

        real_variances = var(tendency_ffts.real, axis=0)
        imag_variances = var(tendency_ffts.imag, axis=0)

        lag_covariances = (irfft(real_variances + imag_variances,
                                 n=state_size) /
                           state_size)
        result = toeplitz(lag_covariances)

    result /= 2

    return result
