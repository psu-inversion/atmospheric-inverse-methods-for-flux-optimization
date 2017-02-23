#!/usr/bin/env python
from __future__ import division, print_function
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
import inversion.noise
import inversion.osse
from inversion.noise import gaussian_noise

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
# use the value given in email
state_start[0] = .25
integrator = inversion.integrators.scipy_odeint

fig = plt.figure()
ax = fig.add_subplot(111)

start_printing = False
xlocs = np.arange(MODEL_SIZE)
print("Day, Std. Dev.")

############################################################
# Problem 1
for day, daytime in enumerate(np.arange(15) * DAY_LEN):
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
    if std > 1e-4:
        start_printing = True
    if start_printing:
        print("{:3d}  {:7.4f}".format(day, state.std()))
    if day % 2 == 1:
        fig.savefig("Lorenz96_spinup_day{day:02d}.pdf".format(day=day+1))
    state_start = states[-1, :]

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
err_ax.set_ylabel("Root mean square difference")
err_ax.set_xlim(0, 1000)
err_ax.set_ylim(0)
err_fig.suptitle("Root Mean Square Error in Perturbed Trajectory")
plt.pause(LONG_PAUSE)
err_fig.savefig("perturbed_rmse.pdf")

# The error saturates by about day five
# The linear growth phase lasts roughly a day and a half to two days
err_ax.clear()
err_ax.semilogy(saved_days[:9], rms_diffs[:9])
# The error doubling time looks to be about two days

linear_data = pd.DataFrame(dict(
    RMSE=rms_diffs[1:9],
    days=saved_days[1:9]))

lin_growth_fit = smf.ols("np.log2(RMSE) ~ days", data=linear_data).fit()
print("Error doubling period:", 1/lin_growth_fit.params.days)
err_ax.semilogy(
    saved_days[:9],
    np.exp2(lin_growth_fit.params.Intercept +
            lin_growth_fit.params.days * saved_days[:9]))
del lin_growth_fit

# climatological variance
# use only times after variance settles
# five days with four times saved per day
variances = np.var(differences, axis=1)
print("Climatological variance of differences:", np.mean(variances[20:]))
del variances, rms_diffs

# Bred vectors
N_BRED_VECTORS = 20
RESCALING_PERIOD = .25
BREEDING_CYCLES = 40
forecast_times = (0, RESCALING_PERIOD * DAY_LEN)

bred_fig, bred_axes = plt.subplots(
    nrows=4, ncols=1, sharex=True, sharey=False, squeeze=True,
    gridspec_kw=dict(top=.85, hspace=.35, wspace=.1))
bred_fig.suptitle("Bred Vectors for a Lorenz '96 System")

for i, initial_size in enumerate((3, 1, .3, .1)):
    bred_ax = bred_axes[i]
    bred_ax.set_title("Perturbations Scaled to {size:.1f}".format(
        size=initial_size))
    bred_ax.set_xlim(0, 40)

    for _ in range(N_BRED_VECTORS):
        # The diagonal elements are variances, not standard deviations.
        perturbation = gaussian_noise(np.eye(MODEL_SIZE) * initial_size ** 2)
        state = original_states[0] + perturbation

        # Even going out only 100 iterations, the largest singular
        # mode dominates all but the first. It seems to be able to get
        # halfway to saturation in six hours consistently.
        for j, time in enumerate(saved_times[:BREEDING_CYCLES]):
            perturbation = state - original_states[j]
            perturbation *= initial_size / perturbation.std()
            states = integrator(
                lambda y, t: model(y),
                original_states[j] + perturbation,
                forecast_times,
                dt)
            state = states[1]
        perturbation = state - original_states[BREEDING_CYCLES - 1]

        bred_ax.plot(perturbation)
    plt.pause(PAUSE)

bred_fig.savefig("Lorenz96_bred_vectors.pdf")

del perturbed_states, original_states

############################################################
# Problem 3

EXP_LEN = 100
DA_CYCLE = .25
# The interval here is traditionally a day to avoid problems with the
# daily cycle of observations and temperatures.
# The original paper used 0 and 24 hour forecasts, 24 and 48 is standard,
# UK Met Office used 6 and 30 at one point
# Given that a perturbation of .1 stops growing linearly at two days,
# probably shouldn't go too far for this.
NMC_COMPARISON = np.array((.5, 1))
DTYPE = np.float64
MODEL_INTEGRATOR = inversion.integrators.forward_euler
saved_days = np.arange(0, EXP_LEN, DA_CYCLE)
saved_times = saved_days * DAY_LEN

# produce a truth for the identical twin experiment
model_truth = MODEL_INTEGRATOR(
    lambda y, t: model(y),
    np.asanyarray(spun_up_state, dtype=DTYPE),
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
# Problems 4 and 5
# Fully observed

def getname(method):
    """Get a name for the function.

    Parameters
    ----------
    method: callable

    Returns
    -------
    str
    """
    module = method.__module__
    group = module.split(".")[-1]
    variant = method.__name__

    return "{group:s} ({variant:s})".format(group=group,
                                            variant=variant)


rmse_fig, rmse_axes = plt.subplots(
    nrows=3, ncols=2, sharex=True, sharey=True,
    gridspec_kw=dict(top=.85, hspace=.25, wspace=.1))
rmse_fig.suptitle("DA Errors for a Lorenz '96\n"
                  "Fully Observed Identical Twin Setup")

for rmse_ax in rmse_axes.flat:
    rmse_ax.set_xlim(0, EXP_LEN)
    rmse_ax.set_ylim(.25, 1)
for rmse_ax in rmse_axes[:, 0]:
    rmse_ax.set_ylabel("RMSE")
for rmse_ax in rmse_axes[-1, :]:
    rmse_ax.set_xlabel("Day")

for col, da_fun in enumerate(
        (inversion.optimal_interpolation.simple,
         inversion.variational.simple)):
    print(getname(da_fun))
    bg_cov = np.eye(MODEL_SIZE)

    # Ensure the observations are those obtained above
    np.random.set_state(random_state)

    # Basic covariances. different for each combination
    background_rmse, analysis_rmse, nmc_cov = inversion.osse.identical_twin(
        model, MODEL_INTEGRATOR, spun_up_state, dt,
        DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
        obs_op, obs_cov, bg_cov,
        da_fun, NMC_COMPARISON * DAY_LEN, True)

    rmse_ax = rmse_axes[0, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Prescribed")
    plt.pause(PAUSE)
    # DA system spins up in about a day,
    # asymptoting to an error of roughly one
    # print(nmc_cov.shape)
    # print(nmc_cov)
    # print(np.diag(nmc_cov))
    np.linalg.cholesky(nmc_cov)
    print("Prescribed")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

    np.random.set_state(random_state)

    # See if assuming homogeneous covariances provides a better fit
    background_rmse, analysis_rmse, nmc_cov = inversion.osse.identical_twin(
        model, MODEL_INTEGRATOR, spun_up_state, dt,
        DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
        obs_op, obs_cov, nmc_cov, da_fun,
        NMC_COMPARISON * DAY_LEN, False)

    rmse_ax = rmse_axes[1, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Homogeneous NMC")
    plt.pause(PAUSE)
    # print(nmc_cov.shape)
    # print(nmc_cov)
    # print(np.diag(nmc_cov))
    np.linalg.cholesky(nmc_cov)
    print("NMC Homogeneous")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

    np.random.set_state(random_state)
    # See if inhomogenizing the covariances provides a better fit
    background_rmse, analysis_rmse = inversion.osse.identical_twin(
        model, MODEL_INTEGRATOR, spun_up_state, dt,
        DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
        obs_op, obs_cov, nmc_cov, da_fun)

    rmse_ax = rmse_axes[2, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Inhomogeneous NMC")
    plt.pause(PAUSE)

    print("Inhomogeneous NMC")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

# The current implementation improves the RMSE each time through

rmse_fig.savefig("lorenz96_full.pdf")

############################################################
# Lorenz-Emanuel Land-ocean

# set up the observations
# The variational assimilation does not converge with unit observation variances
# the first variational run converges with obs_std == .5
# this works if the homogeneous NMC covariances are estimated and used
# to get the inhomogeneous NMC covariances rather than the other way around

# note that because of the different number of observations per cycle,
# these runs will not be using the same observations as the last set.
# They will, however, be using the same observations as each other.
obs_op = np.eye(MODEL_SIZE // 2, MODEL_SIZE)
obs_cov = np.eye(MODEL_SIZE // 2) / 20

rmse_fig, rmse_axes = plt.subplots(
    nrows=3, ncols=2, sharex=True, sharey=True,
    gridspec_kw=dict(top=.85, hspace=.25, wspace=.1))
rmse_fig.suptitle("DA Errors for a Lorenz '96\n"
                  "Land-Ocean Identical Twin Setup")

for rmse_ax in rmse_axes.flat:
    rmse_ax.set_xlim(0, EXP_LEN)
    rmse_ax.set_ylim(0, 6)
for rmse_ax in rmse_axes[:, 0]:
    rmse_ax.set_ylabel("RMSE")
for rmse_ax in rmse_axes[-1, :]:
    rmse_ax.set_xlabel("Day")

obs_rmsd_fig, obs_rmsd_axes = plt.subplots(
    nrows=3, ncols=2, sharex=True, sharey=True,
    gridspec_kw=dict(top=.85, hspace=.25, wspace=.1))
obs_rmsd_fig.suptitle("DA Observation Mismatch for a Lorenz '96\n"
                  "Land-Ocean Identical Twin Setup")

for obs_rmsd_ax in obs_rmsd_axes.flat:
    obs_rmsd_ax.set_xlim(0, EXP_LEN)
    obs_rmsd_ax.set_ylim(0, 6)
for obs_rmsd_ax in obs_rmsd_axes[:, 0]:
    obs_rmsd_ax.set_ylabel("RMSE")
for obs_rmsd_ax in obs_rmsd_axes[-1, :]:
    obs_rmsd_ax.set_xlabel("Day")

for col, da_fun in enumerate(
        (inversion.optimal_interpolation.fold_common,
         inversion.variational.incremental)):
    print(getname(da_fun))
    bg_cov = np.eye(MODEL_SIZE)

    # Ensure the observations are those obtained above
    np.random.set_state(random_state)

    # Basic covariances. different for each combination
    (background_rmse, analysis_rmse, nmc_cov,
     obs_bg_rmsd, obs_analysis_rmsd) = inversion.osse.identical_twin(
         model, MODEL_INTEGRATOR, spun_up_state, dt,
         DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
         obs_op, obs_cov, bg_cov,
         da_fun, NMC_COMPARISON * DAY_LEN, True,
         observation_err=True)

    rmse_ax = rmse_axes[0, col]
    obs_rmsd_ax = obs_rmsd_axes[0, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Prescribed")

    # DA system spins up in about a day,
    # asymptoting to an error of roughly one
    # print(nmc_cov.shape)
    # print(nmc_cov)
    # print(np.diag(nmc_cov))
    np.linalg.cholesky(nmc_cov)
    print("Prescribed")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

    obs_rmsd_ax.plot(saved_days, obs_analysis_rmsd, label="Analysis")
    obs_rmsd_ax.plot(saved_days, obs_bg_rmsd, label="Background")
    obs_rmsd_ax.legend()
    obs_rmsd_ax.set_title("Prescribed")
    print("Observation Background RMSD:", np.mean(obs_bg_rmsd))
    print("Observation Analysis RMSD:", np.mean(obs_analysis_rmsd))
    plt.pause(PAUSE)

    np.random.set_state(random_state)

    # See if assuming homogeneous covariances provides a better fit
    (background_rmse, analysis_rmse, nmc_cov,
     obs_bg_rmsd, obs_analysis_rmsd) = inversion.osse.identical_twin(
         model, MODEL_INTEGRATOR, spun_up_state, dt,
         DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
         obs_op, obs_cov, nmc_cov, da_fun,
         NMC_COMPARISON * DAY_LEN, False,
         observation_err=True)

    rmse_ax = rmse_axes[1, col]
    obs_rmsd_ax = obs_rmsd_axes[1, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Homogeneous NMC")
    np.linalg.cholesky(nmc_cov)
    print("Homogeneous NMC")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

    obs_rmsd_ax.plot(saved_days, obs_analysis_rmsd, label="Analysis")
    obs_rmsd_ax.plot(saved_days, obs_bg_rmsd, label="Background")
    obs_rmsd_ax.legend()
    obs_rmsd_ax.set_title("Homogeneous NMC")
    print("Observation Background RMSD:", np.mean(obs_bg_rmsd))
    print("Observation Analysis RMSD:", np.mean(obs_analysis_rmsd))
    plt.pause(PAUSE)

    np.random.set_state(random_state)
    # See if inhomogenizing the covariances provides a better fit
    (background_rmse, analysis_rmse,
     obs_bg_rmsd, obs_analysis_rmsd) = inversion.osse.identical_twin(
         model, MODEL_INTEGRATOR, spun_up_state, dt,
         DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
         obs_op, obs_cov, nmc_cov, da_fun,
         observation_err=True)

    rmse_ax = rmse_axes[2, col]
    obs_rmsd_ax = obs_rmsd_axes[2, col]

    rmse_ax.plot(saved_days, analysis_rmse, label="Analysis")
    rmse_ax.plot(saved_days, background_rmse, label="Background")
    rmse_ax.legend()
    rmse_ax.set_title("Inhomogeneous NMC")

    print("Inhomogeneous NMC")
    print("Average Background RMSE:", np.mean(background_rmse))
    print("Average Analysis RMSE:", np.mean(analysis_rmse))

    obs_rmsd_ax.plot(saved_days, obs_analysis_rmsd, label="Analysis")
    obs_rmsd_ax.plot(saved_days, obs_bg_rmsd, label="Background")
    obs_rmsd_ax.legend()
    obs_rmsd_ax.set_title("NMC Inhomogeneous")
    print("Observation Background RMSD:", np.mean(obs_bg_rmsd))
    print("Observation Analysis RMSD:", np.mean(obs_analysis_rmsd))
    plt.pause(PAUSE)

# The current implementation improves the RMSE each time through

rmse_fig.savefig("lorenz96_landocean.pdf")
obs_rmsd_fig.savefig("lorenz96_landocean_obsdiag.pdf")
plt.show(block=True)
