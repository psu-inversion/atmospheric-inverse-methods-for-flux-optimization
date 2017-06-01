"""Facilities for running DA experiments."""
import collections

import numpy as np
from dask.array.fft import rfft, irfft
import scipy.linalg

import xarray

import inversion.models
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


def root_mean_squared_difference(arry1, arry2, axis=None):
    """Root mean square difference of arrays.

    Parameters
    ----------
    arry1, arry2: np.ndarray[N]
    axis: int, optional

    Returns
    -------
    float
    """
    return np.sqrt(np.mean(np.square(arry1 - arry2), axis=axis))


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
        lag_covariances = (irfft(real_variances + imag_variances) /
                           len(curr_model))

        result = (background_rmse, analysis_rmse,
                  scipy.linalg.toeplitz(lag_covariances))

    if observation_err:
        return result + (observation_background_rmsd,
                         observation_analysis_rmsd)
    return result


def ensemble_osse(
        model, integrator,
        dt, assimilation_cycle_time, experiment_length,
        observations, observation_operator, observation_error_covariance,
        localization_matrix,
        initial_ensemble, assimilation_function,
        additional_forecast_times=(),
        multiplicative_inflation_factor=1):
    """Run an OSSE using the given observations and assumed errors.

    Assumes ensemble trajectory and full background covariance
    matrices can fit in memory.

    Parameters
    ----------
    model: callable
        Follows signature of :class:`inversion.models.Lorenz96`
        instances.
    integrator: callable
        Follows signature of functions in :mod:`inversion.ensemble.integrators`
    dt: float
        Model integration time step
    assimilation_cycle_time: float
    experiment_length: float
    observations: np.ndarray[N_CYCLES, M]
        Observations generated from the truth run.
    observation_operator: np.ndarray[M, N]
        The assumed observation operator.
        Currently assumes that :math:`h(x) = H x`
    observation_error_covariance: np.ndarray[M, M]
        The assumed observation error covariance matrix.
    localization_matrix: np.ndarray[N, N]
        np.ones((N, N)) gives no localization.
        Classes in :mod:`inversion.correlations` offer other options.
    initial_ensemble: np.ndarray[K, N]
        The initial ensemble to start the experiment.
    assimilation_function: callable
        Follows signature of
        :func:`inversion.ensemble.perturbed_observations_filter.SimplePerturbedObsFilter`.
    additional_forecast_times: sequence of float, optional
        Additional forecast times to include in the output.
    multiplicative_inflation_factor: float, optional
        Number to multiply the perturbations by after the assimilation.

    Returns
    -------
    ensemble_forecast_trajectory: xarray.Dataset[N_CYCLES, LEAD_TIME, K, N]
    """
    n_cycles = int(np.rint(experiment_length / assimilation_cycle_time))
    ensemble_size, state_size = initial_ensemble.shape

    curr_ensemble = initial_ensemble.copy()

    forecast_times = sorted(set(_make_tuple(
        0, assimilation_cycle_time, *additional_forecast_times)))
    next_background_index = forecast_times.index(assimilation_cycle_time)
    forecast_times = np.array(forecast_times)

    ensemble_forecast_trajectory = np.empty(
        (n_cycles, len(forecast_times), ensemble_size, state_size),
        dtype=initial_ensemble.dtype)

    forecast_init_times = np.arange(
        0, experiment_length, assimilation_cycle_time)

    for i, init_time in enumerate(forecast_init_times):
        curr_obs = observations[i, :]

        curr_ensemble = assimilation_function(
            curr_ensemble, localization_matrix, curr_obs,
            observation_error_covariance, observation_operator)

        if multiplicative_inflation_factor != 1:
            ens_mean, ens_perts = inversion.ensemble.mean_and_perturbations(
                curr_ensemble)
            ens_perts *= multiplicative_inflation_factor
            curr_ensemble = inversion.ensemble.states_from_perturbations(
                ens_mean, ens_perts)

        forecasts = integrator(
            inversion.models.ArgsYTWrapper(model),
            curr_ensemble,
            forecast_times,
            dt)

        ensemble_forecast_trajectory[i, :, :, :] = forecasts[:, :, :]
        curr_ensemble = forecasts[next_background_index, :, :]

    result = xarray.Dataset(
        dict(
            ensemble_forecasts=(
                ("forecast_reference_time", "forecast_lead_time",
                 "ensemble_member", "state_vec_index"),
                ensemble_forecast_trajectory,
                dict(long_name="OSSE_results"))),
        dict(
            forecast_reference_time=(
                ("forecast_reference_time",),
                forecast_init_times,
                dict(standard_name="forecast_reference_time")),
            forecast_lead_time=(
                ("forecast_lead_time",),
                forecast_times,
                dict(long_name="forecast_lead_time")),
            ensemble_member=(
                ("ensemble_member",),
                range(ensemble_size),
                dict(long_name="ensemble_member",
                     standard_name="realization")),
            state_vec_index=(
                ("state_vec_index",),
                range(state_size),
                dict(long_name="state_vector_index")),
            time=(
                ("forecast_reference_time", "forecast_lead_time"),
                forecast_init_times[:, np.newaxis] +
                forecast_times[np.newaxis, :],
                dict(standard_name="time")),
        ),
        attrs=dict(Conventions="CF-1.6"),
    )
    return result


def hybrid_osse(
        model, control_integrator, ensemble_integrator,
        dt, assimilation_cycle_time, experiment_length,
        observations, observation_operator, observation_error_covariance,
        background_error_covariance, localization_matrix,
        initial_control, initial_ensemble, assimilation_function,
        additional_forecast_times=(),
        multiplicative_inflation_factor=1,
        ensemble_covariance_weight=.5):
    """Run an OSSE using the given observations and assumed errors.

    Assumes ensemble trajectory and full background covariance
    matrices can fit in memory. Assumes `control` and rows of
    `ensemble` are the same length and have the same dtype.

    Parameters
    ----------
    model: callable
        Follows signature of :class:`inversion.models.Lorenz96`
        instances.
    control_integrator: callable
        Follows signature of functions in :mod:`inversion.integrators`
    ensemble_integrator: callable
        Follows signature of functions in :mod:`inversion.ensemble.integrator`
    dt: float
        Model integration time step
    assimilation_cycle_time: float
    experiment_length: float
    observations: np.ndarray[N_CYCLES, M]
        Observations generated from the truth run.
    observation_operator: np.ndarray[M, N]
        The assumed observation operator.
        Currently assumes that :math:`h(x) = H x`
    observation_error_covariance: np.ndarray[M, M]
        The assumed observation error covariance matrix.
    background_error_covariance: np.ndarray[N, N]
    localization_matrix: np.ndarray[N, N]
        np.ones((N, N)) gives no localization.
        Classes in :mod:`inversion.correlations` offer other options.
    initial_control: np.ndarray[N]
    initial_ensemble: np.ndarray[K, N]
        The initial ensemble to start the experiment.
    assimilation_function: callable
        Follows signature of
        :func:`inversion.ensemble.hybrid.SimpleEns3DVar`.
    additional_forecast_times: sequence of float, optional
        Additional forecast times to include in the output.
    multiplicative_inflation_factor: float, optional
        Number to multiply the perturbations by after the assimilation.
    ensemble_covariance_weight: float

    Returns
    -------
    xarray.Dataset
        Contains control and ensemble trajectories.
    """
    n_cycles = int(np.rint(experiment_length / assimilation_cycle_time))
    ensemble_size, state_size = initial_ensemble.shape

    curr_control = initial_control.copy()
    curr_ensemble = initial_ensemble.copy()

    forecast_times = sorted(set(_make_tuple(
        0, assimilation_cycle_time, *additional_forecast_times)))
    next_background_index = forecast_times.index(assimilation_cycle_time)
    forecast_times = np.array(forecast_times)

    control_forecast_trajectory = np.empty(
        (n_cycles, len(forecast_times), state_size),
        dtype=initial_control.dtype)
    ensemble_forecast_trajectory = np.empty(
        (n_cycles, len(forecast_times), ensemble_size, state_size),
        dtype=initial_ensemble.dtype)

    forecast_init_times = np.arange(
        0, experiment_length, assimilation_cycle_time)

    for i, init_time in enumerate(forecast_init_times):
        curr_obs = observations[i, :]

        curr_control, curr_ensemble = assimilation_function(
            curr_control, curr_ensemble,
            background_error_covariance, localization_matrix, curr_obs,
            observation_error_covariance, observation_operator,
            ensemble_covariance_weight)

        if multiplicative_inflation_factor != 1:
            ens_mean, ens_perts = inversion.ensemble.mean_and_perturbations(
                curr_ensemble)
            ens_perts *= multiplicative_inflation_factor
            curr_ensemble = inversion.ensemble.states_from_perturbations(
                ens_mean, ens_perts)

        ens_forecasts = ensemble_integrator(
            inversion.models.ArgsYTWrapper(model),
            curr_ensemble,
            forecast_times,
            dt)
        control_forecasts = control_integrator(
            inversion.models.ArgsYTWrapper(model),
            curr_control, forecast_times, dt)

        control_forecast_trajectory[i, :, :] = control_forecasts
        ensemble_forecast_trajectory[i, :, :, :] = ens_forecasts[:, :, :]

        curr_control = control_forecasts[next_background_index, :]
        curr_ensemble = ens_forecasts[next_background_index, :, :]

    result = xarray.Dataset(
        dict(
            ensemble_forecasts=(
                ("forecast_reference_time", "forecast_lead_time",
                 "ensemble_member", "state_vec_index"),
                ensemble_forecast_trajectory,
                dict(long_name="OSSE_ensemble_trajectories")),
            control_forecast=(
                ("forecast_reference_time", "forecast_lead_time",
                 "state_vec_index"),
                control_forecast_trajectory,
                dict(long_name="OSSE_control_trajectory")),
        ),
        dict(
            forecast_reference_time=(
                ("forecast_reference_time",),
                forecast_init_times,
                dict(standard_name="forecast_reference_time")),
            forecast_lead_time=(
                ("forecast_lead_time",),
                forecast_times,
                dict(long_name="forecast_lead_time")),
            ensemble_member=(
                ("ensemble_member",),
                range(ensemble_size),
                dict(long_name="ensemble_member",
                     standard_name="realization")),
            state_vec_index=(
                ("state_vec_index",),
                range(state_size),
                dict(long_name="state_vector_index")),
            time=(
                ("forecast_reference_time", "forecast_lead_time"),
                forecast_init_times[:, np.newaxis] +
                forecast_times[np.newaxis, :],
                dict(standard_name="time", long_name="forecast_valid_time")),
        ),
        attrs=dict(Conventions="CF-1.6"),
    )
    return result
