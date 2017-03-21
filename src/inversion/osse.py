"""Facilities for running DA experiments."""
import collections

import numpy as np
from numpy.fft import rfft
import scipy.linalg

from inversion.noise import gaussian_noise

ASSIMILATION_SPIN_UP = 10
"""How many DA cycles it takes to converge to steady state."""


def _make_tuple(*args):
    """Return args as a tuple.

    This is an identity function to provide an assignment context for
    *lst magics

    Returns
    -------
    tuple
    """
    return args


def root_mean_squared_difference(arry1, arry2):
    """Root mean square difference of arrays.

    Parameters
    ----------
    arry1, arry2: np.ndarray[N]

    Returns
    -------
    float
    """
    return np.sqrt(np.mean(np.square(arry1 - arry2)))


def identical_twin(model, integrator, initial_state, dt,
                   assimilation_cycle_time, experiment_length,
                   observation_operator, observation_error_covariance,
                   background_error_covariance,
                   assimilation_function,
                   nmc_comparison=None,
                   nmc_homogeneous=False,
                   observation_err=False):
    """Run an identical twin experiment.

    Parameters
    ----------
    model: callable
        Follows signature of :func:`inversion.models.Lorenz96`
    integrator: callable
        Follows signature of :func:`inversion.integrators.forward_euler`
    initial_state: np.ndarray[N]
    dt: float
        Model integration time step
    assimilation_cycle_time: float
    experiment_length: float
    observation_operator: np.ndarray[M, N]
    observation_error_covariance: np.ndarray[M, M]
    background_error_covariance: np.ndarray[N, N]
    assimilation_function: callable
        Follows signature of :func:`inversion.optimal_interpolation.simple`
    nmc_comparison: tuple, optional
        Two floats in increasing order giving the times to use for
        estimating the background covariances using the NMC method.
    nmc_homogeneous: bool, optional
        Whether to assume the covariances from the NMC method are spatially
        homogeneous.
    observation_err: bool, optional
        whether to return observation diagnostics

    Returns
    -------
    background_rmse: list
    analysis_rmse: list
    nmc_covariances: np.ndarray[N, N]
        if nmc_comparison is provided
    observation_background_rmsd: list
        if observation_err is True
    observation_analysis_rmsd: list
        if observation_err is True

    References
    ----------
    Parrish and Derber 1992
        doi: 10.1175/1520-0493(1992)120<1747:TNMCSS>2.0.CO;2
    """
    total_cycles = int(np.rint(experiment_length / assimilation_cycle_time))
    dtype = initial_state.dtype
    if nmc_comparison is not None:
        difference_delay = int(np.rint(
            np.diff(nmc_comparison) / assimilation_cycle_time))
        past_long_forecasts = collections.deque(maxlen=difference_delay)
        if not nmc_homogeneous:
            forecast_differences = np.empty(
                (len(initial_state),
                 total_cycles - difference_delay),
                dtype=dtype,
                order="F")
        else:
            # Using the increments in spectral space assumes homogeneity
            # and isotropy of the errors, which is justified if the
            # observing network is homogeneous
            complex_dtype = rfft(initial_state).dtype
            forecast_diff_ffts = np.empty(
                (len(initial_state) // 2 + 1,
                 total_cycles - difference_delay),
                dtype=complex_dtype,
                order="F")

    background_rmse = np.empty(total_cycles, dtype=dtype)
    analysis_rmse = np.empty(total_cycles, dtype=dtype)

    if observation_err:
        observation_background_rmsd = np.empty(total_cycles, dtype=dtype)
        observation_analysis_rmsd = np.empty_like(observation_background_rmsd)

    curr_truth = initial_state.copy()
    curr_model = np.zeros_like(initial_state)

    truth_forecast_times = (0, assimilation_cycle_time)
    # Ensure this still works if the NMC times start at 0 or 6 hours
    # (original paper and UK Met Office)
    forecast_times = sorted(set(_make_tuple(
        0, assimilation_cycle_time,
        *(nmc_comparison if nmc_comparison is not None else ()))))
    next_background_index = forecast_times.index(assimilation_cycle_time)

    if nmc_comparison is not None:
        nmc_indices = [forecast_times.index(time)
                       for time in nmc_comparison]

    for i, init_time in enumerate(np.arange(
            0, experiment_length, assimilation_cycle_time)):
        background_rmse[i] = root_mean_squared_difference(
            curr_truth, curr_model)

        true_obs = observation_operator.dot(curr_truth)
        obs_err = gaussian_noise(observation_error_covariance)
        observations = true_obs + obs_err
        del true_obs, obs_err

        analysis, _ = assimilation_function(
            curr_model, background_error_covariance,
            observations, observation_error_covariance,
            observation_operator)

        analysis_rmse[i] = root_mean_squared_difference(
            curr_truth, analysis)

        if observation_err:
            background_obs = observation_operator.dot(curr_model)
            analysis_obs = observation_operator.dot(analysis)

            observation_background_rmsd[i] = root_mean_squared_difference(
                observations, background_obs)
            observation_analysis_rmsd[i] = root_mean_squared_difference(
                observations, analysis_obs)

        truth_forecasts = integrator(
            lambda y, t: model(y),
            curr_truth,
            truth_forecast_times,
            dt)
        curr_truth = truth_forecasts[1]

        forecasts = integrator(
            lambda y, t: model(y),
            analysis,
            forecast_times,
            dt)

        curr_model = forecasts[next_background_index, :]

        if nmc_comparison is not None:
            if len(past_long_forecasts) == difference_delay:
                long_forecast = past_long_forecasts.popleft()
                forecast_difference = long_forecast - forecasts[nmc_indices[0]]

                if nmc_homogeneous:
                    forecast_diff_ffts[:, i - difference_delay] = (
                        rfft(forecast_difference))
                else:
                    forecast_differences[:, i - difference_delay] = (
                        forecast_difference)
            past_long_forecasts.append(forecasts[nmc_indices[1]])

    if nmc_comparison is None:
        result = background_rmse, analysis_rmse

    # some portion will be DA spin-up
    elif not nmc_homogeneous:
        result = (background_rmse, analysis_rmse,
                  np.cov(forecast_differences[:, ASSIMILATION_SPIN_UP:]))
    else:
        real_variances = np.var(np.real(
            forecast_diff_ffts[:, ASSIMILATION_SPIN_UP:]), 1)
        imag_variances = np.var(np.imag(
            forecast_diff_ffts[:, ASSIMILATION_SPIN_UP:]), 1)
        # This seems to add the variances for each variable
        # Averaging should match the statistics better
        # The imag and real parts are the sines and cosines
        # doing them together may cause problems with the mean
        lag_covariances = (np.fft.irfft(real_variances + imag_variances) /
                           len(curr_model))

        result = (background_rmse, analysis_rmse,
                  scipy.linalg.toeplitz(lag_covariances))

    if observation_err:
        return result + (observation_background_rmsd,
                         observation_analysis_rmsd)
    return result
