#!/usr/bin/env python
"""Run an identical twin flux inversion OSSE with real data.

Use xarray/dask to grab influence functions and priors from netCDF
files.
"""
from __future__ import print_function, division, unicode_literals

import itertools
import datetime
import os.path
import glob
import sys

import dask.array as da
import pandas as pd
import dateutil.tz
import numpy as np
import cf_units
import netCDF4
import xarray
import wrf

try:
    THIS_DIR = os.path.dirname(__file__)
except NameError:
    THIS_DIR = os.getcwd()

sys.path.insert(0, os.path.join(
    THIS_DIR, "..", "src"))
sys.path.append(THIS_DIR)

import inversion.optimal_interpolation
import inversion.variational
import inversion.correlations
import inversion.covariances
from inversion.util import kronecker_product, asarray
import cf_acdd

INFLUENCE_PATH = "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/ACT-America_trial5/2010/01/GROUP1"
PRIOR_PATH = "/mc1s2/s4/dfw5129/data/Marthas_2010_wrfouts"
OBS_PATH = "/mc1s2/s4/dfw5129/inversion"

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
OBS_FILES = glob.glob(os.path.join(OBS_PATH,
                                   "2010_01_4tower_WRF*concentrations*.nc"))
FLUX_FILES = glob.glob(os.path.join(PRIOR_PATH, "wrf_fluxes_all.nc"))
FLUX_FILES.sort()
OBS_FILES.sort()
INFLUENCE_FILES = glob.glob(os.path.join(
    INFLUENCE_PATH,
    "LPDM_2010_01_{flux_interval:02d}hrly_027km_footprints.nc4"
    .format(flux_interval=FLUX_INTERVAL)))

print("Flux files", FLUX_FILES)
print("Influence Files", INFLUENCE_FILES)

FLUX_NAME = "E_TRA2"
TRACER_NAME = "tracer_2_subset"
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

FLUX_CHUNKS = HOURS_PER_DAY * 1 // FLUX_INTERVAL
"""How many flux times to treat at once.

Must be a multiple of day length.
"""
OBS_CHUNKS = 1
"""How many observations to treat at once.

Must allow a few chunks of the influence function to be in memory at
once.  Will need to extend this if (N_OBS_TIMES * N_SITES) ** 2 arrays
stop fitting nicely in memory.
"""


############################################################
# Utility functions.
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


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

############################################################
# Read sizes from influence function file
TEST_DS = netCDF4.Dataset(INFLUENCE_FILES[0])

NX = len(TEST_DS.dimensions["dim_x"])
NY = len(TEST_DS.dimensions["dim_y"])
N_TIMES_BACK = len(TEST_DS.dimensions["time_before_observation"])

N_SITES = len(TEST_DS.dimensions["site"])
N_OBS_TIMES = len(TEST_DS.dimensions["observation_time"])

TEST_DS.close()
del TEST_DS

if N_TIMES_BACK < FLUX_WINDOW / FLUX_INTERVAL:
    raise ValueError("FLUX_WINDOW too long for file")

N_GRID_POINTS = NY * NX
STATE_SIZE = N_GRID_POINTS * FLUX_WINDOW

OBS_VEC_SIZE = N_SITES * OBS_WINDOW
OBS_VEC_TOTAL_SIZE = N_SITES * N_OBS_TIMES

############################################################
# Read influence functions

# Laptop timings
# save_sum, obs chunk 24, flux_chunk 6 days
# Flux window 14 days
# 1 day obs:   4m11s
# 2 days obs:  5m48s,  8m
# 4 days obs: 10m44s, 16m (high load)
# 8 days obs: Crashes with smallest chunks possible, needs *128 chunks
INFLUENCE_DATASET = xarray.open_mfdataset(
    INFLUENCE_FILES,
    chunks=dict(observation_time=OBS_CHUNKS, site=1,
                time_before_observation=FLUX_WINDOW // FLUX_INTERVAL,
                dim_y=NY, dim_x=NX)).isel(
    observation_time=slice(0, 8 * HOURS_PER_DAY),
    time_before_observation=slice(0, FLUX_WINDOW // FLUX_INTERVAL))
INFLUENCE_FUNCTIONS = INFLUENCE_DATASET.H
# Use site names as index/dim coord for site dim
INFLUENCE_FUNCTIONS.coords["site"] = np.char.decode(
    INFLUENCE_FUNCTIONS["site_names"].values, "ascii")

OBS_TIME_INDEX = INFLUENCE_DATASET.indexes["observation_time"].round("S")
TIME_BACK_INDEX = INFLUENCE_DATASET.indexes["time_before_observation"]

INFLUENCE_FUNCTIONS.coords["observation_time"] = OBS_TIME_INDEX

# NB: Remember to change frequency and time zone as necessary.
FLUX_START = (OBS_TIME_INDEX[-1] - TIME_BACK_INDEX[-1]).replace(hour=0)
if OBS_TIME_INDEX[0].hour != 0:
    FLUX_END = OBS_TIME_INDEX[0].replace(hour=0) + datetime.timedelta(days=1)
else:
    FLUX_END = OBS_TIME_INDEX[0]
FLUX_TIMES_INDEX = pd.date_range(
    FLUX_START, FLUX_END,
    freq="{flux_interval:d}H".format(flux_interval=FLUX_INTERVAL),
    tz="UTC", closed="right",
    name="flux_times")
N_FLUX_TIMES = len(FLUX_TIMES_INDEX)

############################################################
# Set some constants based on the WRF file
with netCDF4.Dataset(FLUX_FILES[0]) as ds:
    WRF_PROJECTION = wrf.util.getproj(**wrf.util.get_proj_params(ds))
    WRF_CRS = WRF_PROJECTION.cartopy()

    # Alternate method
    # Currently broken for some reason.
    # WRF_CRS = wrf.get_cartopy(ds)

WRF_TOWER_COORDS = WRF_CRS.transform_points(
    WRF_CRS.as_geodetic(),
    INFLUENCE_FUNCTIONS.coords["site_lons"].values,
    INFLUENCE_FUNCTIONS.coords["site_lats"].values,
    INFLUENCE_FUNCTIONS.coords["site_heights"].values)
OBS_ROUGH_LEVEL = 11
"""Because I don't feel like destaggering the geopotential."""
OBS_ROUGH_SIGMA = 0.9486
"""Also eyeballed from a single time."""

print(datetime.datetime.now(UTC).strftime("%c"),
      "Have constants, getting priors")
############################################################
# Read prior fluxes
FLUX_DATASET = xarray.open_mfdataset(
    FLUX_FILES,
    chunks=dict(projection_x_coordinate=NX, projection_y_coordinate=NY,
                XTIME=1),
    concat_dim="Time",
)
OBS_DATASET = xarray.open_mfdataset(
    OBS_FILES,
    chunks=dict(time=OBS_CHUNKS),
    concat_dim="Time",
)
print(datetime.datetime.now(UTC).strftime("%c"), "Have obs, normalizing")
sys.stdout.flush()

# Many of the times are off by about four milliseconds.
# This difference is irrelevant here.
wrf_orig_times = FLUX_DATASET["XTIME"].to_index().round("S")
timestamps = list(wrf_orig_times)
timestamps[-1] += datetime.timedelta(hours=FLUX_INTERVAL / 2 - 1)
timestamps[0] -= datetime.timedelta(hours=1)
wrf_new_times = pd.DatetimeIndex(timestamps,
                                 name="XTIME")
FLUX_DATASET.coords["Time"] = wrf_new_times

print(OBS_DATASET.dims, OBS_DATASET.coords)
OBS_DATASET.set_index(dim3="projection_x_coordinate",
                      dim2="projection_y_coordinate",
                      inplace=True)
OBS_DATASET.rename(dict(dim3="projection_x_coordinate",
                        dim2="projection_y_coordinate"),
                   inplace=True)
print(OBS_DATASET.dims, OBS_DATASET.coords)
FLUX_DATASET.set_index(XTIME="Time", inplace=True)
FLUX_DATASET = FLUX_DATASET.rename(
    dict(
        XTIME="Time", projection_y_coordinate="south_north",
        projection_x_coordinate="west_east"),
    inplace=True)
# Assign a few more coords and pull out only the fluxes we need.
FLUX_DATASET = FLUX_DATASET.sel(Time=FLUX_TIMES_INDEX)

WRF_DX = FLUX_DATASET.attrs["DX"]

TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i + 1)
                                for i in range(10)]).isel(emissions_zdim=0)
TRUE_FLUXES_MATCHED = TRUE_FLUXES.rename(dict(
    south_north="dim_y", west_east="dim_x", Time="flux_time")) * CO2_MOLAR_MASS
for flux_part, flux_orig in zip(TRUE_FLUXES_MATCHED.data_vars.values(),
                                TRUE_FLUXES.data_vars.values()):
    unit = (cf_units.Unit(flux_orig.attrs["units"]) *
            CO2_MOLAR_MASS_UNITS)
    # For whatever reason this is backwards from the conversion
    # factors used elsewhere.
    flux_part *= (unit / FLUX_UNITS).convert(1, 1)
    flux_part.attrs["units"] = str(FLUX_UNITS)

WRF_OBS = OBS_DATASET.get(
    ["tracer_{:d}_subset".format(i + 1)
     for i in (1, 6, 8)]).isel(
         atmosphere_sigma_coordinate=slice(8, 15))
WRF_OBS_MATCHED = WRF_OBS.rename(dict(
    projection_y_coordinate="dim_y", projection_x_coordinate="dim_x",
    time="observation_time"))
WRF_OBS_SITE = (
    WRF_OBS_MATCHED.sel(atmosphere_sigma_coordinate=OBS_ROUGH_SIGMA,
                        method="nearest")
    .isel_points(dim_x=range(4), dim_y=range(4),
                 dim=INFLUENCE_FUNCTIONS.coords["site"])).transpose()
print(WRF_OBS_MATCHED.dims, WRF_OBS_MATCHED.coords)

WRF_OBS_START = WRF_OBS_MATCHED.indexes["observation_time"][0]
WRF_OBS_INTERVAL = (WRF_OBS_START -
                    WRF_OBS_MATCHED.indexes["observation_time"][1])

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
dimension_order.insert(dimension_order.index("time_before_observation"),
                       "flux_time")
dimension_order.insert(dimension_order.index("observation_time"),
                       "observation")
dimension_order = tuple(dimension_order)
aligned_influences = xarray.concat(
    [here_infl.set_index(
        time_before_observation="flux_time").rename(
            dict(time_before_observation="flux_time"))
     for here_infl in INFLUENCE_FUNCTIONS.sel_points(
         site=site_index, observation_time=pd_obs_index)],
    "observation").set_index(
    observation=("observation_time", "site"))
print(datetime.datetime.now(UTC).strftime("%c"),
      "Aligned flux times in influence function, "
      "aligning fluxes with influence function")
sys.stdout.flush()
aligned_influences, aligned_fluxes = xarray.align(
    aligned_influences, TRUE_FLUXES_MATCHED[FLUX_NAME],
    exclude=("dim_x", "dim_y"),
    join="outer", copy=False)
print(datetime.datetime.now(UTC).strftime("%c"),
      "Aligned fluxes and influence function")
aligned_fluxes = aligned_fluxes.chunk(dict(
    dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS))
aligned_influences = aligned_influences.chunk(dict(
    dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS,
    # One chunk per site.  Should be fast with the zeros in R.
    observation=OBS_CHUNKS))
print(datetime.datetime.now(UTC).strftime("%c"), "Rechunked to square")
aligned_influences = aligned_influences.fillna(0)
transpose_arg = sort_key_to_consecutive([dimension_order.index(dim)
                                         for dim in aligned_influences.dims])

posterior_var_atts = aligned_fluxes.attrs.copy()
posterior_var_atts.update(dict(
    long_name="posterior_fluxes",
    units=TRUE_FLUXES_MATCHED[FLUX_NAME].attrs["units"],
    description="posterior fluxes using dask for a month",
    origin="OI using dask for a month",
    prior_flux_name=FLUX_NAME,
    flux_window=FLUX_WINDOW,
    observation_window=OBS_WINDOW))
increment_var_atts = aligned_fluxes.attrs.copy()
increment_var_atts.update(dict(
    long_name="flux_increment",
    units=TRUE_FLUXES_MATCHED[FLUX_NAME].attrs["units"],
    description="Change from prior to posterior using dask",
    origin="OI and dask",
    flux_window=FLUX_WINDOW,
    observation_window=OBS_WINDOW))
posterior_global_atts = cf_acdd.global_attributes_dict()
posterior_global_atts.update(dict(
    title="Posterior fluxes",
    summary="Posterior fluxes",
    creator_institution="PSU Department of Meteorology",
    product_version="v0.0.0.dev0",
    cdm_data_type="grid",
    institution="PSU Department of Meteorology",
    source="Test inversion using OI for a monthlong window",
))

############################################################
# Define correlation constants and get covariances
print(datetime.datetime.now(UTC).strftime("%c"), "Getting covariances")
sys.stdout.flush()
CORRELATION_LENGTH = 84
GRID_RESOLUTION = 27
spatial_correlations = (
    inversion.correlations.HomogeneousIsotropicCorrelation.
    # First guess at correlation length on the order of previous studies
    from_function(
        inversion.correlations.ExponentialCorrelation(
            CORRELATION_LENGTH / GRID_RESOLUTION),
        (len(TRUE_FLUXES_MATCHED.coords["dim_y"]),
         len(TRUE_FLUXES_MATCHED.coords["dim_x"]))))
print(datetime.datetime.now(UTC).strftime("%c"), "Have spatial correlations")
sys.stdout.flush()
HOURLY_FLUX_TIMESCALE = 3
hour_correlations = (
    inversion.correlations.HomogeneousIsotropicCorrelation.
    from_function(
        inversion.correlations.ExponentialCorrelation(HOURLY_FLUX_TIMESCALE),
        (HOURS_PER_DAY // FLUX_INTERVAL,)))
print(datetime.datetime.now(UTC).strftime("%c"), "Have hourly correlations")
sys.stdout.flush()
DAILY_FLUX_TIMESCALE = 14
day_correlations = (
    inversion.correlations.make_matrix(
        inversion.correlations.ExponentialCorrelation(DAILY_FLUX_TIMESCALE),
        (len(TRUE_FLUXES_MATCHED.coords["flux_time"]) *
         FLUX_INTERVAL // HOURS_PER_DAY,)))
print(datetime.datetime.now(UTC).strftime("%c"), "Have daily correlations")
sys.stdout.flush()

full_correlations = kronecker_product(
    day_correlations,
    kronecker_product(hour_correlations, spatial_correlations))
print(datetime.datetime.now(UTC).strftime("%c"), "Have combined correlations")
sys.stdout.flush()

# I would like to add a fixed minimum at some point.
# full stds would then be sqrt(fixed^2 + varying^2)
# average seasonal variation (or some fraction thereof) might work.
FLUX_VARIANCE_VARYING_FRACTION = .3
flux_stds = FLUX_VARIANCE_VARYING_FRACTION * da.fabs(aligned_fluxes.data)
flux_stds_matrix = inversion.covariances.DiagonalOperator(
    flux_stds.reshape(-1))

# TODO: use actual heights
here_obs = WRF_OBS_SITE[TRACER_NAME].sel_points(
    observation_time=pd_obs_index, site=site_index
)

OBSERVATION_VARIANCE = .1
OBS_CORR_FUN = inversion.correlations.ExponentialCorrelation(3)
"""Temporal correlations in observation error.

Mostly reflects transport error.  Given in units of obs_time.
Since the input is hourly this is a three-hour decay time.
"""
OBS_INTERVAL = np.array(1, dtype='m8[h]')
observation_covariance = OBS_CORR_FUN(
    abs(pd_obs_index[:, np.newaxis] - pd_obs_index[np.newaxis, :]) /
    OBS_INTERVAL)
# Assumes no correlations between observations.
observation_covariance[
    site_index[:, np.newaxis] != site_index[np.newaxis, :]] = 0
observation_covariance = asarray(observation_covariance)

print(datetime.datetime.now(UTC).strftime("%c"),
      "Got covariance parts, getting posterior")
sys.stdout.flush()
posterior, correlations = inversion.optimal_interpolation.save_sum(
    aligned_fluxes.data.reshape(N_GRID_POINTS * N_FLUX_TIMES),
    inversion.util.ProductLinearOperator(
        flux_stds_matrix, full_correlations, flux_stds_matrix),
    here_obs.data,
    observation_covariance,
    (aligned_influences.data
     .transpose(transpose_arg)
     .reshape(aligned_influences.shape[0],
              np.prod(aligned_influences.shape[-3:]))))
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have posterior values, making dataset")
sys.stdout.flush()
posterior = posterior.reshape(aligned_fluxes.shape)
posterior_ds = xarray.Dataset(
    dict(posterior=(aligned_fluxes.dims, posterior,
                    posterior_var_atts),
         prior=aligned_fluxes,
         increment=(aligned_fluxes.dims, posterior - aligned_fluxes,
                    increment_var_atts),
         ),
    TRUE_FLUXES_MATCHED.coords,
    posterior_global_atts)
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have posterior structure, evaluating and writing")
sys.stdout.flush()
posterior_ds.to_netcdf("monthly_inversion_{flux_interval:02d}h_027km_output.nc4"
                       .format(flux_interval=FLUX_INTERVAL))
print(datetime.datetime.now(UTC).strftime("%c"), "Wrote posterior")
sys.stdout.flush()
