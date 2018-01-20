#!/usr/bin/env python
r"""Run a flux inversion with a simple H and B.

Welcome to the problem that combines all the best features of GARCH,
VAR, SARIMA, and HMM models, all in one place. Bad estimates of the
HMM observation operator! Bad initial estimates of the VAR noise
process covariance matrix! Seasonal variations in variance from the
annual cycle! Systematic problems in the prior estimates! And much,
much more!

For the fraternal twin setup, we use the different covariance matrices
in the noise generation and the inversion.
"""

import itertools
import datetime
import os.path
import sys

import numpy as np
import numpy.linalg as la
import dateutil.tz
import scipy.linalg
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import cf_units
import cartopy.crs as ccrs
import iris.cube
import iris.util
import iris.coords
import iris.analysis.maths
import iris.plot as iplt
import iris.quickplot as qplt
import xarray

try:
    sys.path.append(os.path.join(os.path.dirname(__file__),
                                 "..", "src"))
except NameError:
    sys.path.append(os.path.join(os.getcwd(), "..", "src"))

import inversion.optimal_interpolation
import inversion.correlations
import inversion.covariances
import inversion.noise
import inversion.tests

isqrt = iris.analysis.maths.IFunc(
    np.sqrt, lambda cube: cube.units.root(2))

NX = 60
NY = 40
N_FLUX_TIMES = 24 * 7

N_TIMES_BACK = 24 * 5
N_SITES = 4

N_GRID_POINTS = NX * NY
N_OBS_TIMES = N_FLUX_TIMES - N_TIMES_BACK + 1

TRUE_CORR_LEN = 5
ASSUMED_CORR_LEN = 5
TRUE_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(TRUE_CORR_LEN))
ASSUMED_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(ASSUMED_CORR_LEN))
DAY_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(14))
HOUR_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(3))
STDS = np.ones((N_FLUX_TIMES, NY, NX))

COORD_ADJOINT_STR = "adjoint_"
CUBE_ADJOINT_STR = "adjoint_of_"

INVERSION_FUNCTIONS = inversion.tests.ALL_METHODS
N_FUNCTIONS = len(INVERSION_FUNCTIONS)
FUNCTION_COORD = iris.coords.AuxCoord(
    [inversion.tests.getname(func)
     for func in INVERSION_FUNCTIONS],
    long_name="inversion_function_name")

DIVERGING_CMAP = plt.get_cmap("RdBu_r")
FLUX_FILE = "/mc1s2/s4/dfw5129/data/Marthas_2010_wrfouts/wrf_fluxes_all.nc"


DX = iris.cube.Cube(27, units="km")
X_COORD = iris.coords.DimCoord(DX.data * np.arange(0, NX), units="km",
                               standard_name="projection_x_coordinate")
Y_COORD = iris.coords.DimCoord(DX.data * np.arange(0, NY), units="km",
                               standard_name="projection_y_coordinate")
FLUX_TIME_COORD = iris.coords.DimCoord(
    np.arange(N_FLUX_TIMES),
    units="days since 2010-01-01 00:00:00+0000",
    standard_name="time",
    long_name="flux_time")
OBS_TIME_COORD = iris.coords.DimCoord(
    FLUX_TIME_COORD.points[-N_OBS_TIMES:],
    units=FLUX_TIME_COORD.units,
    standard_name="forecast_reference_time",
    long_name="observation_time")
TIME_BACK_COORD = iris.coords.DimCoord(
    np.arange(N_TIMES_BACK),
    units="days",
    long_name="time_before_observation")
TOWER_COORD = iris.coords.DimCoord(
    np.arange(N_SITES),
    long_name="tower_number",
)

FLUX_INTERVAL = 3
"""The interval at which fluxes become available in hours.

Fluxes are usually integrated forward from hourly input, but I can't
solve for that in a reasonable timeframe.

This determines both the input and the output time resolution as well
as what the inversion solves for.

Note
----
Must divide twenty-four.
"""
FLUX_NAME = "E_TRA2"
TRACER_NAME = "tracer_2_subset"
FLUX_CHUNKS = 64
"""How many flux times to treat at once.

Must be a multiple of day length.
"""
OBS_CHUNKS = 62
"""How many observations to treat at once.

Must allow a few chunks of the influence function to be in memory at
once.  Will need to extend this if (N_OBS_TIMES * N_SITES) ** 2 array
stops fitting in memory.
"""

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
FLUX_WINDOW = HOURS_PER_DAY * DAYS_PER_WEEK * 2
"""How long fluxes considered to have an influence.

Measured in hours.
Implemented by slicing out this much of the stored influence functions
for use as the linearized observation operator in the inversion.
"""
OBS_HOURS = (datetime.time(14), datetime.time(18))
"""Which observation times will be used in the inversion.

Assumed to be UTC. Should give afternoon hours for the domain.  Not
currently set up for domains spanning many time zones. This shouldn't
matter for OSSEs, but will for real-data cases (PBL schemes don't
treat night well)

I really hope I can assume this doesn't depend on latitude. That would
make this much more complicated.
"""
OBS_TIMES_PER_DAY = OBS_HOURS[1].hour - OBS_HOURS[0].hour
"""Observations used per site per day."""
OBS_WINDOW = 1
CO2_MOLAR_MASS = 16 * 2 + 12.01
"""Molar mass of CO2 (g/mol).

Used to convert WRF fluxes to units expected by observation operator.
"""
CO2_MOLAR_MASS_UNITS = cf_units.Unit("g/mol")
FLUX_UNITS = cf_units.Unit("g/m^2/hr")


def sort_key_to_consecutive(sequence):
    """Turn a list of sort keys into a list of consecutive numbers.

    Parameters
    ----------
    sequence: collections.Sequence
        The list to sort.

    Returns
    -------
    tuple
        ``argsort(sequence)``, I think
    """
    items = list(enumerate(sorted(sequence)))

    items.sort(key=lambda item: sequence.index(item[1]))
    return tuple(item[0] for item in items)


UTC = dateutil.tz.tzutc()
SECONDS_PER_HOUR = 3600

# I'm pretty sure this is the proper memory order for products to work
# Unfortunately, this conflicts with the intuition I get from the
# $E(\ce \ce^T)$ definition
TRUE_SP_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        TRUE_SP_ERROR_CORRELATION_FUN, (NY, NX)))
ASSUMED_SP_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        ASSUMED_SP_ERROR_CORRELATION_FUN, (NY, NX)))
HOUR_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        HOUR_ERROR_CORRELATION_FUN, (8,)))
DAY_ERROR_CORRELATION = (
    inversion.correlations.make_matrix(
        DAY_ERROR_CORRELATION_FUN, (N_FLUX_TIMES // 8,)))
TRUE_ERROR_CORRELATION = inversion.util.kronecker_product(
    DAY_ERROR_CORRELATION,
    inversion.util.kronecker_product(
        HOUR_ERROR_CORRELATION,
        TRUE_SP_ERROR_CORRELATION))
ASSUMED_ERROR_CORRELATION = inversion.util.kronecker_product(
    DAY_ERROR_CORRELATION,
    inversion.util.kronecker_product(
        HOUR_ERROR_CORRELATION,
        ASSUMED_SP_ERROR_CORRELATION))

# Do this part before or after the kronecker product?
STDS = np.ones(N_FLUX_TIMES * N_GRID_POINTS)
DIAG_STDS = inversion.covariances.DiagonalOperator(STDS)
TRUE_ERROR_COVARIANCE = DIAG_STDS.dot(TRUE_ERROR_CORRELATION.dot(DIAG_STDS))
ASSUMED_ERROR_COVARIANCE = DIAG_STDS.dot(
    ASSUMED_ERROR_CORRELATION.dot(DIAG_STDS))
print(TRUE_ERROR_COVARIANCE.shape)
OBS_STDS = np.ones((N_OBS_TIMES, N_SITES)) / 5
OBSERVATION_COVARIANCE = np.diag(np.square(OBS_STDS.reshape(-1)))

# This one goes forward in time
INFLUENCE_DATASET = xarray.open_mfdataset(
    "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/ACT-America_trial4/2010/01/GROUP1/LPDM_2010_01_03hrly_footprints.nc4",
    # Total runtime by chunks
    # 6m10s for 62, 4, 8*7*2, full
    # 6m to segfault with 62, 4, 64, full
    chunks=dict(observation_time=OBS_CHUNKS, site=N_SITES,
                time_before_observation=FLUX_WINDOW // FLUX_INTERVAL,
                dim_y=NY, dim_x=NX)).isel(
    observation_time=slice(0, N_OBS_TIMES),
    time_before_observation=slice(0, N_TIMES_BACK))
INFLUENCE_FUNCTIONS = INFLUENCE_DATASET.H
# Use site names as index/dim coord for site dim
INFLUENCE_FUNCTIONS["site"] = np.char.decode(INFLUENCE_FUNCTIONS["site_names"].values, "ascii")

OBS_TIME_INDEX = INFLUENCE_DATASET.indexes["observation_time"]
TIME_BACK_INDEX = INFLUENCE_DATASET.indexes["time_before_observation"]

# NB: Remember to change frequency and time zone as necessary.
FLUX_START = (OBS_TIME_INDEX[-1] - TIME_BACK_INDEX[-1]).replace(
    hour=0, microsecond=0, nanosecond=0)
if OBS_TIME_INDEX[0].hour != 0:
    FLUX_END = OBS_TIME_INDEX[0].replace(hour=0) + datetime.timedelta(days=1)
else:
    FLUX_END = OBS_TIME_INDEX[0]
FLUX_TIMES_INDEX = pd.date_range(FLUX_START,
                                 FLUX_END,
                                 freq="{flux_interval:d}H".format(flux_interval=FLUX_INTERVAL),
                                 tz="UTC",
                                 closed="right",
                                 name="flux_times")
N_FLUX_TIMES = len(FLUX_TIMES_INDEX)

FLUX_DATASET = xarray.open_mfdataset(
    FLUX_FILE,
    chunks=dict(projection_x_coordinate=NX, projection_y_coordinate=NY, XTIME=1),
    concat_dim="Time",
)
# Many of the times are off by about four milliseconds.
# This difference is irrelevant here.
wrf_orig_times = FLUX_DATASET["XTIME"].to_index()
timestamps = [timestamp.replace(microsecond=0,
                                nanosecond=0)
              for timestamp in wrf_orig_times]
timestamps[-1] += datetime.timedelta(hours=FLUX_INTERVAL/2-1)
timestamps[0] -= datetime.timedelta(hours=1)
wrf_new_times = pd.DatetimeIndex(timestamps,
                                 name="XTIME")
FLUX_DATASET.coords["Time"] = wrf_new_times
FLUX_DATASET.set_index(XTIME="Time", inplace=True)
FLUX_DATASET = FLUX_DATASET.rename(
    dict(
        XTIME="Time", projection_y_coordinate="south_north",
        projection_x_coordinate="west_east"),
    inplace=True)
# Assign a few more coords and pull out only the fluxes we need.
FLUX_DATASET = FLUX_DATASET.sel(Time=FLUX_TIMES_INDEX)
TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i+1)
                                for i in range(10)]).isel(emissions_zdim=0)
TRUE_FLUXES_MATCHED = TRUE_FLUXES.rename(dict(
    south_north="dim_y", west_east="dim_x", Time="flux_time")) * CO2_MOLAR_MASS
for flux_part, flux_orig in zip(TRUE_FLUXES_MATCHED.data_vars.values(), TRUE_FLUXES.data_vars.values()):
    unit = (cf_units.Unit(flux_orig.attrs["units"]) *
            CO2_MOLAR_MASS_UNITS)
    # For whatever reason this is backwards from the conversion
    # factors used elsewhere.
    flux_part *= (unit / FLUX_UNITS).convert(1, 1)
    flux_part.attrs["units"] = str(FLUX_UNITS)


print(datetime.datetime.now(UTC).strftime("%c"), "Getting solar times")
############################################################
# Get time zone representing local solar time for each site
LOCAL_TIME_ZONES = list(map(
    dateutil.tz.tzoffset,
    ("Local time for {name!s}".format(name=name)
     for name in INFLUENCE_FUNCTIONS.coords["site"].values),
    [round(offset_hr * SECONDS_PER_HOUR)
     for offset_hr in INFLUENCE_FUNCTIONS.coords["site_lons"].values / 15]))

print(datetime.datetime.now(UTC).strftime("%c"),
      "Converting influence function to have alignment necessary for H")
############################################################
# Do the inversion
############################################################
# The Kalman filter version subtracts an hour from these. Why?
obs_times = (INFLUENCE_FUNCTIONS.indexes["observation_time"][::-1])

############################################################
# Align observation flux times
# Take care of missing obs
# Also subsetting for late afternoon steady convective boundary layer
# takes roughly 11 min.
# Takes 6 min with everything in memory
# Two minutes with big chunks?
site_obs_index = []
print(datetime.datetime.now(UTC).strftime("%c"), "Selecting observations")
for i, site in enumerate(INFLUENCE_FUNCTIONS.indexes["site"]):
    local_times = pd.Index([
        obs_time.tz_localize(UTC)
        .tz_convert(LOCAL_TIME_ZONES[i])
        for obs_time in obs_times], name="local_times")

    # np.where is designed for multi-dimensional arrays
    site_obs_index.extend(zip(
        np.where((OBS_HOURS[0] < local_times.time) &
                 (local_times.time < OBS_HOURS[1]))[0],
        itertools.repeat(site)))
obs_index, site_index = zip(*site_obs_index)
site_index = np.array(list(site_index))
pd_obs_index = obs_times[list(obs_index)]
site_obs_pd_index = pd.MultiIndex.from_tuples(
    list(zip(site_index, pd_obs_index)), names=("site", "observation_time"))

print(datetime.datetime.now(UTC).strftime("%c"),
      "Aligning flux times in influence function")
sys.stdout.flush()
dimension_order = [item for item in INFLUENCE_FUNCTIONS.dims]
dimension_order.insert(dimension_order.index("time_before_observation"), "flux_time")
dimension_order.insert(dimension_order.index("observation_time"), "observation")
dimension_order = tuple(dimension_order)
aligned_influences = xarray.concat(
    [here_infl.set_index(
        time_before_observation="flux_time").rename(
            dict(time_before_observation="flux_time"))
     for here_infl in INFLUENCE_FUNCTIONS.sel_points(
             site=site_index, observation_time=pd_obs_index)],
    "observation").set_index(
    observation=("observation_time", "site"))
print(datetime.datetime.now(UTC).strftime("%c"), "Aligned flux times in influence function, aligning fluxes with influence function")
sys.stdout.flush()
aligned_influences, aligned_fluxes = xarray.align(aligned_influences, TRUE_FLUXES_MATCHED[FLUX_NAME],
                                                  exclude=("dim_x", "dim_y"),
                                                  join="outer", copy=False)
# aligned_influences.reindex(flux_time=FLUX_TIMES_INDEX)
print(datetime.datetime.now(UTC).strftime("%c"), "Aligned fluxes and influence function")
aligned_fluxes = aligned_fluxes.chunk(dict(
        dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS))
aligned_influences = aligned_influences.chunk(dict(
        dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS,
        # One chunk per site.  Should be fast with the zeros in R.
        observation=OBS_CHUNKS))
print(datetime.datetime.now(UTC).strftime("%c"), "Rechunked to square")
print("Aligned fluxes:", aligned_fluxes.dims, aligned_fluxes.shape)
print(aligned_fluxes.chunks)
print("Aligned influences:", aligned_influences.dims, aligned_influences.shape)
print(aligned_influences.chunks)
aligned_influences = aligned_influences.fillna(0)
transpose_arg = sort_key_to_consecutive([dimension_order.index(dim)
                                         for dim in aligned_influences.dims])

OBSERVATION_OPERATOR = iris.cube.Cube(
    INFLUENCE_FUNCTIONS[:N_OBS_TIMES, :N_SITES, :N_TIMES_BACK].data,
    long_name="influence_function",
    units="ppmv/(g/km^2/hr)",
    dim_coords_and_dims=(
        (OBS_TIME_COORD, 0),
        (TOWER_COORD, 1),
        (TIME_BACK_COORD, 2),
        (Y_COORD, 3),
        (X_COORD, 4),
    ),
    aux_coords_and_dims=(
        (iris.coords.AuxCoord(
            np.random.uniform(X_COORD.points[0], X_COORD.points[-1], N_SITES),
            units=X_COORD.units,
            standard_name="projection_x_coordinate",
            long_name="tower_x"), (1,)),
        (iris.coords.AuxCoord(
            np.random.uniform(Y_COORD.points[0], Y_COORD.points[-1], N_SITES),
            units=Y_COORD.units,
            standard_name="projection_y_coordinate",
            long_name="tower_y"), (1,)),
        (iris.coords.AuxCoord(
            scipy.linalg.hankel(FLUX_TIME_COORD[:N_OBS_TIMES].points,
                                FLUX_TIME_COORD[-N_TIMES_BACK:].points),
            units=FLUX_TIME_COORD.units,
            standard_name=FLUX_TIME_COORD.standard_name,
            long_name=FLUX_TIME_COORD.long_name), (0, 2)),
    )
)

tower_x = OBSERVATION_OPERATOR.coord(long_name="tower_x")
tower_y = OBSERVATION_OPERATOR.coord(long_name="tower_y")

fig = plt.figure()
# Currently goes 0 to .05
qplt.contourf(OBSERVATION_OPERATOR.collapsed(
    (OBS_TIME_COORD, TOWER_COORD, TIME_BACK_COORD),
    iris.analysis.SUM),
              vmin=0, vmax=.06)
iplt.plot(tower_x, tower_y, "*")
fig.savefig("crosscheck_obs_op.png")
plt.close(fig)
print(TRUE_CORR_LEN, ASSUMED_CORR_LEN)

if __name__ == "__main__":
    truth_shaped = iris.cube.Cube(
        np.zeros((N_FLUX_TIMES, NY, NX)),
        long_name="true_fluxes",
        units="g/km^2/hr",
        dim_coords_and_dims=(
            (FLUX_TIME_COORD, 0),
            (Y_COORD, 1),
            (X_COORD, 2),
        ),
    )
    truth_vec = truth_shaped.data.reshape(-1)
    fig = plt.figure()
    qplt.contourf(truth_shaped[0])
    fig.savefig("crosscheck_truth.png")
    plt.close(fig)

    # This will need to be refined for N_TIMES_BACK != N_FLUX_TIMES
    true_obs_shaped = (OBSERVATION_OPERATOR * truth_shaped).collapsed(
        (X_COORD, Y_COORD, FLUX_TIME_COORD),
        iris.analysis.SUM)
    true_obs_vec = true_obs_shaped.data.reshape(-1)

    chi2s = iris.cube.Cube(
        np.empty((N_FUNCTIONS)),
        long_name="chi_squared_statistics",
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    flux_totals = iris.cube.Cube(
        np.empty(N_FUNCTIONS),
        long_name="flux_posterior_totals",
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    innovations = iris.cube.Cube(
        np.empty((N_FUNCTIONS, N_OBS_TIMES, N_SITES)),
        long_name="innovations",
        dim_coords_and_dims=(
            (OBS_TIME_COORD, 1),
        ),
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    increments = iris.cube.Cube(
        np.empty((N_FUNCTIONS, N_TIMES_BACK, NY, NX)),
        long_name="increments",
        dim_coords_and_dims=(
            (TIME_BACK_COORD, 1),
            (Y_COORD, 2),
            (X_COORD, 3),
        ),
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )

    noise = inversion.noise.gaussian_noise(TRUE_ERROR_COVARIANCE)

    prior_shaped = truth_shaped + noise.reshape(truth_shaped.shape)
    prior_shaped.long_name = "Prior flux"
    prior_vec = prior_shaped.data.reshape(-1)
    prior_cov = ASSUMED_ERROR_COVARIANCE[:N_TIMES_BACK * N_GRID_POINTS,
                                         :N_TIMES_BACK * N_GRID_POINTS]
    fig = plt.figure()
    qplt.contourf(prior_shaped[0], cmap=DIVERGING_CMAP)
    fig.savefig("crosscheck_prior.png")
    plt.close(fig)
    prior_tot = prior_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                       iris.analysis.SUM)
    print("Prior total:")
    print(prior_tot)
    print(prior_tot.data)


    obs_noise = inversion.noise.gaussian_noise(OBSERVATION_COVARIANCE)
    observations = true_obs_vec + obs_noise
    observations_shaped = true_obs_shaped.copy()
    observations_shaped.data = observations.reshape(true_obs_shaped.shape)
    observations_shaped.long_name = "pseudo-observations"

    for i, inversion_func in enumerate(INVERSION_FUNCTIONS):
        # For N_TIMES_BACK != N_FLUX_TIMES, need to figure out how to loop.
        # With OI, iterating mean and covariance should work,
        # 3D-Var will work with relaxation to prior errors
        # PSAS: only iterate mean. Keep static covariances and evaluate after.
        # Will need to get :math:`M/M^T` figured out.
        # Note: Will need to store part of posterior as we go
        obs_op = OBSERVATION_OPERATOR.data.reshape(
            (N_OBS_TIMES * N_SITES, N_TIMES_BACK * N_GRID_POINTS))

        try:
            posterior, posterior_cov = inversion_func(
                prior_vec, prior_cov, observations, OBSERVATION_COVARIANCE,
                obs_op)
        except inversion.ConvergenceError as err:
            print("Convergence Not Achieved:", inversion_func)
            posterior = err.guess
            posterior_cov = err.hess_inv

        prior_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                                prior_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)
        #print(prior_mismatch)
        print(prior_mismatch.data)

        post_shaped = iris.cube.Cube(
            posterior.reshape((N_TIMES_BACK, NY, NX)),
            long_name="posterior_flux",
            units=prior_shaped.units,
            dim_coords_and_dims=(
                (FLUX_TIME_COORD, 0),
                (Y_COORD, 1),
                (X_COORD, 2),
            ),
        )
        fig = plt.figure()
        qplt.contourf(post_shaped[0], cmap=DIVERGING_CMAP)
        fig.savefig("crosscheck_posterior_{name:s}.png".format(
            name=FUNCTION_COORD.points[i].replace(" ", "_").replace("(", "").replace(")", "")))
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False,
                                 gridspec_kw=dict(wspace=.1, hspace=.1),
                                 figsize=(8, 4), )
        contours = axes[0, 0].contourf(
            X_COORD.points, Y_COORD.points,
            np.diag(posterior_cov).reshape(
                N_TIMES_BACK, NY, NX)[0],
            vmin=.5, vmax=1)
        axes[0, 0].set_title("Posterior Variances")
        fig.colorbar(contours, ax=axes[0, 0], orientation="horizontal", pad=.1)
        fig.suptitle("Results for method {name:s}".format(
            name=FUNCTION_COORD.points[i]))

        post_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                               post_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)

        gain = (1 -
                (iris.analysis.maths.abs(prior_shaped) /
                 iris.analysis.maths.abs(post_shaped)).data)
        plt.sca(axes[0, 1])
        contours = plt.contourf(
            X_COORD.points, Y_COORD.points,
            gain[0], np.linspace(-1, 1, 7), vmin=-1, vmax=1,
            extend="both",
            norm=mpl.colors.Normalize(-1, 1, clip=True))
        axes[0, 1].set_title("Gain")
        fig.colorbar(contours, ax=axes[0, 1], orientation="horizontal", pad=.1)
        fig.savefig("crosscheck_posterior_var_gain_{name:s}.png".format(
            name=FUNCTION_COORD.points[i].replace(" ", "_").replace("(", "").replace(")", "")))
        plt.close(fig)
        # error_reduction = (1 -
        #                    np.sqrt(np.diag(prior_cov) / np.diag(posterior_cov)))
        

        error_proj = obs_op.dot(prior_cov.dot(obs_op.T))
        total_err_cov = error_proj + OBSERVATION_COVARIANCE

        chisq = prior_mismatch.data.dot(
            la.solve(total_err_cov, prior_mismatch.data))
        df_expected = np.prod(prior_mismatch.shape)
        # print("Chi squared statistic:", chisq)
        # print("Expected value:       ", df_expected)

        # print("Chi squared reduced:", chisq / df_expected)
        chi2s.data[i] = chisq
        innovations.data[i, :, :] = prior_mismatch.data[np.newaxis, :]
        increments.data[i] = (post_shaped - prior_shaped).data
        flux_tot = post_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                    iris.analysis.SUM)
        flux_totals.data[i] = flux_tot.data

        print(FUNCTION_COORD.points[i])
        #print(flux_tot)
        print(flux_tot.data)

    # print("To increase this statistic, decrease the flux variances\n"
    #       "To decrease this statistic, increase the flux variances\n"
    #       "If this is not close to one for this perfect-model setup,\n"
    #       "we have big problems.")
    iris.save([post_shaped, innovations, increments, chi2s, flux_totals,
               truth_shaped, OBSERVATION_OPERATOR, prior_shaped, observations_shaped],
              "crosscheck_exponential_actual_{true:d}_assumed_{assumed:d}.nc".format(
                  true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN),
              zlib=True)
