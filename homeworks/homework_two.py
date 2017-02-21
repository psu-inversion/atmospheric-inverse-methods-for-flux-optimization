#!/usr/bin/env python
from __future__ import division, print_function
import collections
import fractions
import time
import sys
import os

import numpy as np
import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import inversion.optimal_interpolation
import inversion.integrators
import inversion.variational
import inversion.models
import inversion.osse

MODEL_SIZE = 40
FORCING = 8
DAY_LEN = .2

model = inversion.models.Lorenz96(FORCING, MODEL_SIZE)
dt = .005
plots_per_day = 4
PAUSE = .002
LONG_PAUSE = 1

# Start from all zeros
state_start = np.zeros((MODEL_SIZE,), dtype=np.float64)
# Make one point different from the next by a bit more than roundoff error
# 64-bit floating point has just under 16 digits of precision
state_start[0] = 1e-15
integrator = inversion.integrators.scipy_odeint

fig = plt.figure()
ax = fig.add_subplot(111)

start_printing = False
xlocs = np.arange(MODEL_SIZE)

############################################################
# Problem 1
for day, daytime in enumerate(np.arange(40) * DAY_LEN):
    states = integrator(lambda y, t: model(y), state_start,
                        np.linspace(0, DAY_LEN, plots_per_day + 1),
                        dt)

    for state in states[1:]:
        ax.clear()
        ax.plot(xlocs, state)
        ax.set_xlim((0, MODEL_SIZE))
        ax.set_ylim((-16, 16))
        fig.suptitle("Lorenz '96 model: Day {day:d}".format(day=day))
        plt.show()
        plt.pause(PAUSE)

    std = state.std()
    if std > 1e-5:
        start_printing = True
    if start_printing:
        print(day, state.std())
    state_start = states[-1,:]

spun_up_state = state_start.copy()

# Run the model for 1000 days
# save every six hours
saved_days = np.arange(0, 1000.0001, .25)
saved_times = saved_days * DAY_LEN

original_states = integrator(lambda y, t: model(y),
                             spun_up_state,
                             saved_times,
                             dt)
del states, fig, ax

############################################################
# Problem 2
# Generate perturbation
perturbation = np.random.normal(0, .1, MODEL_SIZE)
# generate perturbed trajectory
perturbed_states = integrator(lambda y, t: model(y),
                              spun_up_state + perturbation,
                              saved_times,
                              dt)
del perturbation

# calculate root mean square differences
differences = perturbed_states - original_states
rms_diffs = np.sqrt(
    np.mean(
        np.square(differences),
        axis=1))

err_fig = plt.figure()
err_ax = err_fig.add_subplot(111)

err_ax.plot(saved_days, rms_diffs)
err_ax.set_xlabel("Day")
err_ax.set_ylabel("Root mean square error")
err_fig.suptitle("Root Mean Square Error in Perturbed Trajectory")
plt.show()
#plt.pause(LONG_PAUSE)

# The error saturates by about day five
# The linear growth phase lasts roughly a day and a half to two days
err_ax.clear()
err_ax.semilogy(saved_days[:8], rms_diffs[:8])
# The error doubling time looks to be about two days

linear_data = pd.DataFrame(dict(
    RMSE=rms_diffs[:7],
    days=saved_days[:7]))

lin_growth_fit = smf.ols("np.log2(RMSE) ~ days", data=linear_data).fit()
print("Error doubling period:", 1/lin_growth_fit.params.days)
err_ax.semilogy(
    saved_days[:8],
    np.exp2(lin_growth_fit.params.Intercept +
            lin_growth_fit.params.days * saved_days[:8]))
del lin_growth_fit

# climatological variance
# use only times after variance settles
# five days with four times saved per day
variances = np.var(differences, axis=1)
print("Climatological variance:", np.mean(variances[20:]))
del variances, rms_diffs

# Bred vectors
# TODO: get bred vectors
print("TODO: get bred vectors")

del perturbed_states, original_states

############################################################
# Problem 3

EXP_LEN = 100
DA_CYCLE = .25
NMC_COMPARISON = (1, 2)
TRUTH_DTYPE = np.float64
MODEL_DTYPE = np.float64
TRUTH_INTEGRATOR = inversion.integrators.forward_euler
MODEL_INTEGRATOR = inversion.integrators.forward_euler
saved_days = np.arange(0, EXP_LEN, DA_CYCLE)
saved_times = saved_days * DAY_LEN

# produce a truth for the identical twin experiment
model_truth = TRUTH_INTEGRATOR(
    lambda y, t: model(y),
    np.asanyarray(spun_up_state, dtype=TRUTH_DTYPE),
    saved_times,
    dt)

# set up the observations
obs_op = np.eye(MODEL_SIZE)
obs_cov = np.eye(MODEL_SIZE)

# Ensure the observations are reproducible
random_state = np.random.get_state()

true_obs = model_truth.dot(obs_op.T)
obs_err = np.random.multivariate_normal(np.zeros(MODEL_SIZE), obs_cov)
observations = true_obs + obs_err

############################################################
# Problem 4

def root_mean_squared_difference(arry1, arry2):
    """Root mean squared difference between arrays.

    Parameters
    ----------
    arry1, arry2: np.ndarray[N]

    Returns
    -------
    float
    """
    return np.sqrt(np.mean(np.square(arry1 - arry2)))

def make_tuple(*args):
    """Make the arguments a tuple.

    Returns
    -------
    tuple
    """
    return args

bg_cov = np.eye(MODEL_SIZE)
da_fun = inversion.optimal_interpolation.simple

model_init = np.zeros(MODEL_SIZE, dtype=MODEL_DTYPE)

# Include 1 and 2 day forecasts for NMC background covariances
forecast_days = np.array(make_tuple(0, DA_CYCLE, *NMC_COMPARISON))
forecast_times = forecast_days * DAY_LEN

background_rmse = []
analysis_rmse = []

curr_state = model_init.copy()

# set up infrastructure for NMC method
# only need past four cycles of 48hr forecasts
# to compare to most recent 24hr forecast
difference_delay = int(np.rint(np.diff(NMC_COMPARISON) / DA_CYCLE))
past_f48 = collections.deque(maxlen=difference_delay)
f48_f24_diffs = np.empty((MODEL_SIZE,
                          int(np.rint(EXP_LEN / DA_CYCLE)) - difference_delay),
                         order="F", dtype=MODEL_DTYPE)
print(f48_f24_diffs.shape)

for i, init_day in enumerate(np.arange(0, EXP_LEN, DA_CYCLE)):
    init_time = init_day * DAY_LEN

    background_rmse.append(root_mean_squared_difference(
        curr_state, model_truth[i,:]))

    curr_obs = observations[i,:]

    analysis, _ = da_fun(curr_state, bg_cov,
                         curr_obs, obs_cov,
                         obs_op)

    analysis_rmse.append(root_mean_squared_difference(
        analysis, model_truth[i,:]))

    forecasts = MODEL_INTEGRATOR(
        lambda y, t: model(y),
        analysis,
        forecast_times,
        dt)

    curr_state = forecasts[1,:]

    if len(past_f48) == past_f48.maxlen:
        if i < 10 or i > 390:
            print(i, difference_delay)
        f48 = past_f48.popleft()
        f48_f24_diffs[:, i - difference_delay] = (
            f48 - forecasts[2])

    past_f48.append(forecasts[3])

rmse_fig = plt.figure()
rmse_ax = rmse_fig.add_subplot(211)

rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
rmse_ax.plot(saved_days, background_rmse, label="Background")
rmse_ax.legend()
rmse_ax.set_xlabel("Day")
rmse_ax.set_ylabel("RMSE")
rmse_fig.suptitle("DA Errors for a Lorenz '96\n"
                  "Fully Observed Identical Twin Setup")
rmse_ax.set_title("Prescribed Covariances")
# plt.pause(LONG_PAUSE)
# DA system spins up in about a day,
# asymptoting to an error of roughly one
print(f48_f24_diffs.shape)
nmc_cov = np.cov(f48_f24_diffs)
print(nmc_cov.shape)
print(nmc_cov)
print(np.diag(nmc_cov))
np.linalg.cholesky(nmc_cov)

background_rmse = []
analysis_rmse = []

curr_state = model_init.copy()

# set up infrastructure for NMC method
# only need past four cycles of 48hr forecasts
# to compare to most recent 24hr forecast
difference_delay = int(np.rint(np.diff(NMC_COMPARISON) / DA_CYCLE))
past_f48 = collections.deque(maxlen=difference_delay)
f48_f24_diffs = np.empty((MODEL_SIZE,
                          int(np.rint(EXP_LEN / DA_CYCLE)) - difference_delay),
                         order="F", dtype=MODEL_DTYPE)

for i, init_day in enumerate(np.arange(0, EXP_LEN, DA_CYCLE)):
    init_time = init_day * DAY_LEN

    background_rmse.append(root_mean_squared_difference(
        curr_state, model_truth[i,:]))

    curr_obs = observations[i,:]

    analysis, _ = da_fun(curr_state, nmc_cov,
                         curr_obs, obs_cov,
                         obs_op)

    analysis_rmse.append(root_mean_squared_difference(
        analysis, model_truth[i,:]))

    forecasts = MODEL_INTEGRATOR(
        lambda y, t: model(y),
        analysis,
        forecast_times,
        dt)

    curr_state = forecasts[1,:]

    if len(past_f48) == past_f48.maxlen:
        if i < 10:
            print(i, difference_delay)

        f48 = past_f48.popleft()
        f48_f24_diffs[:, i - difference_delay] = (
            f48 - forecasts[2])

    past_f48.append(forecasts[3])

rmse_ax = rmse_fig.add_subplot(212)

rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
rmse_ax.plot(saved_days, background_rmse, label="Background")
rmse_ax.legend()
rmse_ax.set_xlabel("Day")
rmse_ax.set_ylabel("RMSE")
rmse_ax.set_title("NMC Covariances")
# plt.pause(LONG_PAUSE)
nmc_cov = np.cov(f48_f24_diffs)
print(nmc_cov.shape)
print(nmc_cov)
print(np.diag(nmc_cov))

# It is not evident that the NMC covariances decrease the RMSE;
# however; they do smooth it out, decreasing the maximum and
# increasing the minimum. It also appears that they decrease the
# forecast errors.

plt.show(block=True)
