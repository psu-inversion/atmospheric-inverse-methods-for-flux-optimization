#!/usr/bin/env python
from __future__ import division, print_function
import random
import sys
import os

import numpy as np
import cycler
import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import seaborn as sns
import scipy.io

import iris.quickplot as qplt
import iris.analysis
import iris.coords
import iris.cube

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'src'))

import inversion.ensemble.perturbed_observations_filter
import inversion.ensemble.integrators
import inversion.ensemble.hybrid
import inversion.integrators
import inversion.models
import inversion.noise
import inversion.osse
from inversion.noise import gaussian_noise

MODEL_SIZE = 40
FORCING = 8
DAY_LEN = .2

MODEL = inversion.models.Lorenz96(FORCING, MODEL_SIZE)
SINGLE_INTEGRATOR = inversion.integrators.scipy_odeint
ENSEMBLE_INTEGRATOR = (inversion.ensemble.integrators.
                       SerialEnsembleIntegrator(
                           SINGLE_INTEGRATOR))
dt = .005
plots_per_day = 4
PAUSE = .002
LONG_PAUSE = 1
# RMSE near spread for K=100 (borderline divergent)
# Filter diverges for K=60
ENSEMBLE_SIZE = 40

TRUE_DTYPE = np.float64
MODEL_DTYPE = np.float64

sns.set_context("paper")
# Set1 and CMRmap should both work
sns.set_palette("Set1")
# No colors
# plt.style.use("grayscale")
plt.rc("axes",
       prop_cycle=(
           cycler.cycler("linestyle", ("-", "--", "-.", ":", "-", "--")) +
           plt.rcParams["axes.prop_cycle"]))

# Start from all zeros
state_start = np.zeros((MODEL_SIZE,), dtype=TRUE_DTYPE)
# Make one point different from the next by a bit more than roundoff error
# 64-bit floating point has just under 16 digits of precision
state_start[0] = 1e-15
# use the value given in email
state_start[0] = .25
integrator = inversion.integrators.scipy_odeint

start_printing = False
xlocs = np.arange(MODEL_SIZE)
print("Day, Std. Dev.")

############################################################
# Problem 1
for day, daytime in enumerate(np.arange(15) * DAY_LEN):
    states = integrator(lambda y, t: MODEL(y), state_start,
                        np.linspace(0, DAY_LEN, plots_per_day + 1),
                        dt)

    std = states[-1].std()
    if std > 1e-4:
        start_printing = True
    if start_printing:
        print("{:3d}  {:7.4f}".format(day, std))
    state_start = states[-1, :]

spun_up_state = state_start.copy()

# Run the model for 1000 days
# save every six hours
CLIMATOLOGY_LEN = 1000
saved_days = np.arange(0, CLIMATOLOGY_LEN + .0001, .25)
saved_times = saved_days * DAY_LEN

climatology = SINGLE_INTEGRATOR(lambda y, t: MODEL(y),
                               spun_up_state,
                               saved_times,
                               dt)
fig = plt.figure(figsize=(8, 6))
plt.contourf(saved_days, range(40), climatology.T)
plt.xlim((0, 300))
plt.xlabel("Time / days")
plt.title("Lorenz '96 Model Evolution")
plt.savefig("climatology_start.png")
plt.pause(PAUSE)

sample_index = [random.randrange(climatology.shape[0])
                for _ in range(ENSEMBLE_SIZE)]
initial_ensemble = climatology[sample_index, :]

############################################################
# Problem 2
EXP_LEN = 100
DA_CYCLE = .25

OSSE_N_TIMES = int(np.rint(EXP_LEN / DA_CYCLE))

osse_start = random.randrange(climatology.shape[0] - OSSE_N_TIMES)
osse_stop = osse_start + OSSE_N_TIMES
osse_truth = climatology[osse_start:osse_stop]
truth_cube = iris.cube.Cube(
    osse_truth, long_name="osse_truth",
    dim_coords_and_dims=(
        (iris.coords.DimCoord(
            np.arange(0, EXP_LEN, DA_CYCLE),
            standard_name="time",
            units="days"), 0),
        (iris.coords.DimCoord(
            range(MODEL_SIZE),
            long_name="state_vec_index"), 1)
    ),
)

obs_op = np.eye(MODEL_SIZE)
obs_cov = np.eye(MODEL_SIZE)


observations = (osse_truth.dot(obs_op.T) +
                gaussian_noise(obs_cov, OSSE_N_TIMES))

ensemble_vals = inversion.osse.ensemble_osse(
    MODEL, ENSEMBLE_INTEGRATOR, dt,
    DA_CYCLE * DAY_LEN, EXP_LEN * DAY_LEN,
    observations, obs_op, obs_cov,
    np.ones((MODEL_SIZE, MODEL_SIZE), dtype=MODEL_DTYPE),
    initial_ensemble,
    inversion.ensemble.perturbed_observations_filter.SimplePerturbedObsFilter(
        inversion.optimal_interpolation.scipy_chol))

# The OSSE can't tell what the time units are, so set those here
osse_units = "{:.2f} days".format(1 / DAY_LEN)
for coord in ("time", "forecast_reference_time", "forecast_lead_time"):
    time_coord = ensemble_vals.coord(coord)
    time_coord.units = osse_units
    time_coord.convert_units("days")

ensemble_mean = ensemble_vals.collapsed(
    "ensemble_member", iris.analysis.MEAN)

# analysis = ensemble_mean[:, 0]
# iris.util.promote_aux_coord_to_dim_coord(analysis, "time")
# analysis_errors = analysis - truth_cube

# background = ensemble_mean[:, 1]
# iris.util.promote_aux_coord_to_dim_coord(background, "time")
# background_errors = background - truth_cube
perturbations = ensemble_vals - ensemble_mean

# analysis_rmse = analysis_errors.collapsed(
#     "state_vec_index", iris.analysis.RMS)
# background_rmse = background_errors.collapsed(
#     "state_vec_index", iris.analysis.RMS)
analysis_rmse = inversion.osse.root_mean_squared_difference(
    ensemble_mean.data[:, 0, :], truth_cube.data[:, :],
    axis=-1)
background_rmse = inversion.osse.root_mean_squared_difference(
    ensemble_mean.data[:-1, 1], truth_cube.data[1:, :],
    axis=-1)
rmsd = perturbations.collapsed(
    "ensemble_member", iris.analysis.RMS).collapsed(
        "state_vec_index", iris.analysis.RMS)
rmsd.long_name = "ensemble_spread"

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=True,
                       figsize=(8, 5))
times = truth_cube.coord("time").points
ax.plot(times, analysis_rmse, label="Analysis Error")
ax.plot(times[1:], background_rmse, label="Background Error")
qplt.plot(rmsd[:, 0],
          axes=ax, label="Analysis Spread")
qplt.plot(rmsd[:, 1],
          axes=ax, label="Background Spread")
plt.legend()
plt.title("Fully Observed Lorenz '96 Non-localized: {n_mem:d} members".format(
    n_mem=ENSEMBLE_SIZE))
plt.pause(PAUSE)
plt.savefig("PO_EnKF_{n_mem:d}_spread.pdf".format(
    n_mem=ENSEMBLE_SIZE))

LOC_TEST_LEN = 30
LOC_TEST_N_TIMES = int(np.rint(LOC_TEST_LEN / DA_CYCLE))

# A full run takes a while
# The correlation lengths closest are one, as are the inflation factors
# 40-member Gaussian(1) * 1 works best, it seems
LOC_CLASSES = (inversion.correlations.Exponential1DCorrelation,
               inversion.correlations.Gaussian1DCorrelation)
CORR_LENS = (1, 1.1, 1.2, 1.3, 1.5) #, 2, 5, 10, 15, 30)
ENSEMBLE_SIZES = (40, 30, 20, 10)
MULT_INFL_FACTS = (1, 1.05, 1.1)

ALLOWABLE_MISMATCH = 1.1

analysis_rmse_spread_ratio = np.ones(
    (len(LOC_CLASSES), len(CORR_LENS),
     len(ENSEMBLE_SIZES), len(MULT_INFL_FACTS)))
analysis_rmse_values = np.empty_like(analysis_rmse_spread_ratio)
background_rmse_spread_ratio = np.ones(
    (len(LOC_CLASSES), len(CORR_LENS),
     len(ENSEMBLE_SIZES), len(MULT_INFL_FACTS)))

for i, loc_class in enumerate(LOC_CLASSES):
    for j, corr_len in enumerate(CORR_LENS):
        loc_fun = loc_class(corr_len)
        loc_mat = loc_fun.make_matrix(MODEL_SIZE)

        for k, ensemble_size in enumerate(ENSEMBLE_SIZES):

            for l, mult_infl_fact in enumerate(MULT_INFL_FACTS):
                ensemble_vals = inversion.osse.ensemble_osse(
                    MODEL, ENSEMBLE_INTEGRATOR, dt,
                    DA_CYCLE * DAY_LEN, LOC_TEST_LEN * DAY_LEN,
                    observations, obs_op, obs_cov,
                    # actually passing this will probably help a lot
                    loc_mat,
                    initial_ensemble[:ensemble_size, :],
                    inversion.ensemble.perturbed_observations_filter.
                    SimplePerturbedObsFilter(
                        inversion.optimal_interpolation.scipy_chol),
                    multiplicative_inflation_factor=mult_infl_fact)

                # The OSSE can't tell what the time units are,
                # so set those here
                osse_units = "{:.2f} days".format(1 / DAY_LEN)
                for coord in ("time", "forecast_reference_time",
                              "forecast_lead_time"):
                    time_coord = ensemble_vals.coord(coord)
                    time_coord.units = osse_units
                    time_coord.convert_units("days")

                ensemble_mean = ensemble_vals.collapsed(
                    "ensemble_member", iris.analysis.MEAN)
                perturbations = ensemble_vals - ensemble_mean

                analysis_rmse = inversion.osse.root_mean_squared_difference(
                    ensemble_mean.data[:,0,:],
                    truth_cube.data[:LOC_TEST_N_TIMES, :],
                    axis=-1)
                background_rmse = inversion.osse.root_mean_squared_difference(
                    ensemble_mean.data[:-1,1,:],
                    truth_cube.data[1:LOC_TEST_N_TIMES, :],
                    axis=-1)
                rmsd = perturbations.collapsed(
                    "ensemble_member", iris.analysis.RMS).collapsed(
                        "state_vec_index", iris.analysis.RMS)
                rmsd.long_name = "ensemble_spread"

                mean_background_error = background_rmse.mean()
                mean_background_rmsd = rmsd[:, 1].collapsed(
                    "forecast_reference_time", iris.analysis.MEAN)
                mean_background_spread = mean_background_rmsd.data

                background_rmse_spread_ratio[i, j, k, l] = (
                    mean_background_error / mean_background_spread)
                analysis_rmse_spread_ratio[i, j, k, l] = (
                    analysis_rmse.mean() / rmsd[:, 0].collapsed(
                        "forecast_reference_time", iris.analysis.MEAN).data)
                analysis_rmse_values[i, j, k, l] = (
                    analysis_rmse.mean())

                if ((mean_background_error >
                     ALLOWABLE_MISMATCH * mean_background_spread) or
                    (mean_background_spread >
                     ALLOWABLE_MISMATCH * mean_background_error)):
                    continue
                print(ensemble_size, loc_class.__name__, corr_len,
                      mult_infl_fact)
                print("RMSE:  ", mean_background_error)
                print("Spread:", mean_background_spread)

                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                                       squeeze=True, figsize=(8, 5))
                times = truth_cube.coord("time").points
                ax.plot(times[:LOC_TEST_N_TIMES], analysis_rmse,
                        label="Analysis Error")
                ax.plot(times[1:LOC_TEST_N_TIMES], background_rmse,
                        label="Background Error")
                qplt.plot(rmsd[:, 0],
                          axes=ax, label="Analysis Spread")
                qplt.plot(rmsd[:, 1],
                          axes=ax, label="Background Spread")
                plt.title("Fully Observed Lorenz '96 {n_mem:d}-member {name:s}\n"
                          "Localization: {form:s} {dist:.1f} "
                          "Inflation: {fact:.1f}".format(
                              name="Perturbed Obs Filter",
                              form=loc_class.__name__,
                              dist=corr_len, n_mem=ensemble_size,
                              fact=mult_infl_fact))
                plt.legend()
                plt.pause(PAUSE)
                plt.savefig("PO_EnKF_{n_mem:d}_mem_{form:s}_"
                            "loc_to_{dist:.1f}_infl_{fact:.1f}_full.pdf".format(
                                n_mem=ensemble_size, fact=mult_infl_fact,
                                form=loc_class.__name__, dist=corr_len))

best_every_index = np.unravel_index(
    np.argmin(np.abs(analysis_rmse_spread_ratio - 1)),
    analysis_rmse_spread_ratio.shape)
print("Best Ratio was at index", best_every_index)
print(LOC_CLASSES[best_every_index[0]], CORR_LENS[best_every_index[1]])
print(ENSEMBLE_SIZES[best_every_index[2]], MULT_INFL_FACTS[best_every_index[3]])
best_every_index = np.unravel_index(
    np.argmin(analysis_rmse_values),
    analysis_rmse_values.shape)
print("Best RMSE was at index", best_every_index)
print(LOC_CLASSES[best_every_index[0]], CORR_LENS[best_every_index[1]])
print(ENSEMBLE_SIZES[best_every_index[2]], MULT_INFL_FACTS[best_every_index[3]])
observed_every = iris.cube.Cube(
    analysis_rmse_values,
    long_name="analysis_rmse_fully_observed",
    dim_coords_and_dims=(
        (iris.coords.DimCoord(
            CORR_LENS, long_name="correlation_length"), 1),
        (iris.coords.DimCoord(
            ENSEMBLE_SIZES, long_name="ensemble_sizes"), 2),
        (iris.coords.DimCoord(
            MULT_INFL_FACTS, long_name="multiplicative_inflation_factors"), 3),
        ),
    )
iris.save(observed_every, "Lorenz96_full_ens_rmse.nc")
plt.pause(LONG_PAUSE)
plt.show()

new_obs_op = obs_op[::2, :]
new_obs_cov = obs_cov[::2, ::2]
new_observations = observations[:, ::2]

analysis_rmse_spread_ratio = np.ones((len(LOC_CLASSES), len(CORR_LENS),
                                       len(ENSEMBLE_SIZES), len(MULT_INFL_FACTS)))
background_rmse_spread_ratio = np.ones((len(LOC_CLASSES), len(CORR_LENS),
                                        len(ENSEMBLE_SIZES), len(MULT_INFL_FACTS)))
analysis_rmse_values = np.empty_like(analysis_rmse_spread_ratio)

for i, loc_class in enumerate(LOC_CLASSES):
    for j, corr_len in enumerate(CORR_LENS):
        loc_fun = loc_class(corr_len)
        loc_mat = loc_fun.make_matrix(MODEL_SIZE)

        for k, ensemble_size in enumerate(ENSEMBLE_SIZES):

            for l, mult_infl_fact in enumerate(MULT_INFL_FACTS):
                ensemble_vals = inversion.osse.ensemble_osse(
                    MODEL, ENSEMBLE_INTEGRATOR, dt,
                    DA_CYCLE * DAY_LEN, LOC_TEST_LEN * DAY_LEN,
                    new_observations, new_obs_op, new_obs_cov,
                    # actually passing this will probably help a lot
                    loc_mat,
                    initial_ensemble[:ensemble_size, :],
                    inversion.ensemble.perturbed_observations_filter.
                    SimplePerturbedObsFilter(
                        inversion.optimal_interpolation.scipy_chol),
                    multiplicative_inflation_factor=mult_infl_fact)

                # The OSSE can't tell what the time units are,
                # so set those here
                osse_units = "{:.2f} days".format(1 / DAY_LEN)
                for coord in ("time", "forecast_reference_time",
                              "forecast_lead_time"):
                    time_coord = ensemble_vals.coord(coord)
                    time_coord.units = osse_units
                    time_coord.convert_units("days")

                ensemble_mean = ensemble_vals.collapsed(
                    "ensemble_member", iris.analysis.MEAN)
                perturbations = ensemble_vals - ensemble_mean

                analysis_rmse = inversion.osse.root_mean_squared_difference(
                    ensemble_mean.data[:,0,:],
                    truth_cube.data[:LOC_TEST_N_TIMES, :],
                    axis=-1)
                background_rmse = inversion.osse.root_mean_squared_difference(
                    ensemble_mean.data[:-1,1,:],
                    truth_cube.data[1:LOC_TEST_N_TIMES, :],
                    axis=-1)
                rmsd = perturbations.collapsed(
                    "ensemble_member", iris.analysis.RMS).collapsed(
                        "state_vec_index", iris.analysis.RMS)
                rmsd.long_name = "ensemble_spread"

                mean_background_error = background_rmse.mean()
                mean_background_rmsd = rmsd[:, 1].collapsed(
                    "forecast_reference_time", iris.analysis.MEAN)
                mean_background_spread = mean_background_rmsd.data

                background_rmse_spread_ratio[i, j, k, l] = (
                    mean_background_error / mean_background_spread)
                analysis_rmse_spread_ratio[i, j, k, l] = (
                    analysis_rmse.mean() / rmsd[:, 0].collapsed(
                        "forecast_reference_time", iris.analysis.MEAN).data)
                analysis_rmse_values[i, j, k, l] = (
                    analysis_rmse.mean())

                if ((mean_background_error >
                     ALLOWABLE_MISMATCH * mean_background_spread) or
                    (mean_background_spread >
                     ALLOWABLE_MISMATCH * mean_background_error)):
                    continue
                print(ensemble_size, loc_class.__name__, corr_len,
                      mult_infl_fact)
                print("RMSE:  ", mean_background_error)
                print("Spread:", mean_background_spread)

                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                                       squeeze=True, figsize=(8, 5))
                times = truth_cube.coord("time").points
                ax.plot(times[:LOC_TEST_N_TIMES], analysis_rmse,
                        label="Analysis Error")
                ax.plot(times[1:LOC_TEST_N_TIMES], background_rmse,
                        label="Background Error")
                qplt.plot(rmsd[:, 0],
                          axes=ax, label="Analysis Spread")
                qplt.plot(rmsd[:, 1],
                          axes=ax, label="Background Spread")
                plt.title("Half-Observed Lorenz '96 {n_mem:d}-member {name:s}\n"
                          "Localization: {form:s} {dist:.1f} "
                          "Inflation: {fact:.1f}".format(
                              name="Perturbed Obs Filter",
                              form=loc_class.__name__,
                              dist=corr_len, n_mem=ensemble_size,
                              fact=mult_infl_fact))
                plt.legend()
                plt.pause(PAUSE)
                plt.savefig("PO_EnKF_{n_mem:d}_mem_{form:s}_"
                            "loc_to_{dist:.1f}_infl_{fact:.1f}_every_other.pdf".format(
                                n_mem=ensemble_size, fact=mult_infl_fact,
                                form=loc_class.__name__, dist=corr_len))

best_every_other_index = np.unravel_index(
    np.argmin(np.abs(analysis_rmse_spread_ratio - 1)),
    analysis_rmse_spread_ratio.shape)
print("Best ratio was at index", best_every_other_index)
print(LOC_CLASSES[best_every_other_index[0]],
      CORR_LENS[best_every_other_index[1]])
print(ENSEMBLE_SIZES[best_every_other_index[2]],
      MULT_INFL_FACTS[best_every_other_index[3]])
best_every_other_index = np.unravel_index(
    np.argmin(analysis_rmse_values),
    analysis_rmse_spread_ratio.shape)
print("Best RMSE was at index", best_every_other_index)
print(LOC_CLASSES[best_every_other_index[0]],
      CORR_LENS[best_every_other_index[1]])
print(ENSEMBLE_SIZES[best_every_other_index[2]],
      MULT_INFL_FACTS[best_every_other_index[3]])

nmc_cov = scipy.io.loadmat("3dVarInhomogeneousNMCCovariances.mat")["nmc_cov"]

loc_fun = LOC_CLASSES[best_every_index[0]](CORR_LENS[best_every_index[1]])
loc_mat = loc_fun.make_matrix(MODEL_SIZE)

ENS_WEIGHTS = np.arange(.1, 1, .1)

analysis_rmse_spread_ratio = np.ones((len(ENS_WEIGHTS)))
background_rmse_spread_ratio = np.ones((len(ENS_WEIGHTS)))
analysis_rmse_values = np.empty_like(analysis_rmse_spread_ratio)

for i, ens_weight in enumerate(ENS_WEIGHTS):
    control_vals, ensemble_vals = inversion.osse.hybrid_osse(
        MODEL, SINGLE_INTEGRATOR, ENSEMBLE_INTEGRATOR, dt,
        DA_CYCLE * DAY_LEN, LOC_TEST_LEN * DAY_LEN,
        observations, obs_op, obs_cov,
        # actually passing this will probably help a lot
        nmc_cov, loc_mat,
        inversion.ensemble.mean(initial_ensemble),
        initial_ensemble[:ensemble_size, :],
        inversion.ensemble.hybrid.SimpleEns3DVar(
            inversion.optimal_interpolation.scipy_chol,
            inversion.ensemble.perturbed_observations_filter.
            SimplePerturbedObsFilter(inversion.optimal_interpolation.scipy_chol)),
        multiplicative_inflation_factor=mult_infl_fact)

    # The OSSE can't tell what the time units are,
    # so set those here
    osse_units = "{:.2f} days".format(1 / DAY_LEN)
    for coord in ("time", "forecast_reference_time",
                  "forecast_lead_time"):
        time_coord = ensemble_vals.coord(coord)
        time_coord.units = osse_units
        time_coord.convert_units("days")

    ensemble_mean = ensemble_vals.collapsed(
        "ensemble_member", iris.analysis.MEAN)
    perturbations = ensemble_vals - ensemble_mean

    analysis_rmse = inversion.osse.root_mean_squared_difference(
        control_vals.data[:,0,:],
        truth_cube.data[:LOC_TEST_N_TIMES, :],
        axis=-1)
    background_rmse = inversion.osse.root_mean_squared_difference(
        control_vals.data[:-1,1,:],
        truth_cube.data[1:LOC_TEST_N_TIMES, :],
        axis=-1)
    rmsd = perturbations.collapsed(
        "ensemble_member", iris.analysis.RMS).collapsed(
            "state_vec_index", iris.analysis.RMS)
    rmsd.long_name = "ensemble_spread"

    mean_background_error = background_rmse.mean()
    mean_background_rmsd = rmsd[:, 1].collapsed(
        "forecast_reference_time", iris.analysis.MEAN)
    mean_background_spread = mean_background_rmsd.data

    background_rmse_spread_ratio[i] = (
        mean_background_error / mean_background_spread)
    analysis_rmse_spread_ratio[i] = (
        analysis_rmse.mean() / rmsd[:, 0].collapsed(
            "forecast_reference_time", iris.analysis.MEAN).data)
    analysis_rmse_values[i] = analysis_rmse.mean()

    if ((mean_background_error >
         ALLOWABLE_MISMATCH * mean_background_spread) or
        (mean_background_spread >
         ALLOWABLE_MISMATCH * mean_background_error)):
        continue
    print(ensemble_size, loc_class.__name__, corr_len,
          mult_infl_fact)
    print("RMSE:  ", mean_background_error)
    print("Spread:", mean_background_spread)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                           squeeze=True, figsize=(8, 5))
    times = truth_cube.coord("time").points
    ax.plot(times[:LOC_TEST_N_TIMES], analysis_rmse,
            label="Analysis Error")
    ax.plot(times[1:LOC_TEST_N_TIMES], background_rmse,
            label="Background Error")
    qplt.plot(rmsd[:, 0],
              axes=ax, label="Analysis Spread")
    qplt.plot(rmsd[:, 1],
              axes=ax, label="Background Spread")
    plt.title("Fully Observed Lorenz '96 {n_mem:d}-member {name:s}\n"
              "Localization: {form:s} {dist:.1f} "
              "Inflation: {fact:.1f} Hybrid Assimilation".format(
                  name="Perturbed Obs Filter",
                  form=loc_fun.__class__.__name__,
                  dist=loc_fun._corr_len, n_mem=perturbations.shape[-2],
                  fact=mult_infl_fact))
    plt.legend()
    plt.pause(PAUSE)
    plt.savefig("Hybrid_3DVar_POEnKF_{n_mem:d}_mem_{form:s}_"
                "loc_to_{dist:.1f}_infl_{fact:.1f}_full.pdf".format(
                    n_mem=ensemble_size, fact=mult_infl_fact,
                    form=loc_class.__name__, dist=corr_len))

best_hybrid_every_index = np.unravel_index(
    np.argmin(np.abs(analysis_rmse_spread_ratio - 1)),
    analysis_rmse_spread_ratio.shape)
print("Best ratio was at index", best_every_index)
print(ENS_WEIGHTS[best_hybrid_every_index[0]])
best_hybrid_every_index = np.unravel_index(
    np.argmin(analysis_rmse_values),
    analysis_rmse_values.shape)
print("Best ratio was at index", best_every_index)
print(ENS_WEIGHTS[best_hybrid_every_index[0]])

plt.show()
