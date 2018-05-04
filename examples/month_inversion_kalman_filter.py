#!/usr/bin/env python
"""Run an identical twin flux inversion OSSE with simulated obs.

Use xarray/dask to grab influence functions, priors, and obs from
netCDF files.

Extend month_inversion_magic_dask to cover whole month
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
import cartopy
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
from inversion.util import kronecker_product, tolinearoperator, asarray
from inversion.noise import gaussian_noise
import cf_acdd

UTC = dateutil.tz.tzutc()
SECONDS_PER_HOUR = 3600

print(datetime.datetime.now(UTC).strftime("%c"), "Finished imports, setting constants")
sys.stdout.flush()
INFLUENCE_PATH = ("/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                  "ACT-America_trial5/2010/01/GROUP1")
PRIOR_PATH = "/mc1s2/s4/dfw5129/inversion_code/data_files"
OBS_PATH = "/mc1s2/s4/dfw5129/inversion"

FLUX_INTERVAL = 6
"""The interval at which fluxes become available in hours.

Fluxes are usually integrated forward from hourly input, but I can't
solve for that in a reasonable timeframe.

This determines both the input and the output time resolution as well
as what the inversion solves for.

Note
----
Must divide twenty-four.
"""
FLUX_INTERVAL_DT = datetime.timedelta(hours=FLUX_INTERVAL)
FLUX_RESOLUTION = 27
"""FLux resolution in km.

Resolution for the inversion.
"""
CORR_FUN = "balg"
CORR_LEN = 200
"""Prior noise realization to use."""

# OBS_FILES = glob.glob(os.path.join(PRIOR_PATH, "wrfout_d01_*.nc"))
OBS_FILES = glob.glob(os.path.join(
    OBS_PATH, "2010_01_4tower_LPDM_concentrations?.nc"))
FLUX_FILES = glob.glob(os.path.join(
    PRIOR_PATH,
    ("osse_priors_{interval:1d}h_{res:02d}km_noise_{corr_fun:s}{corr_len:d}km"
     "_exp14d_exp3h.nc").format(
        interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        corr_fun=CORR_FUN, corr_len=CORR_LEN)))
FLUX_FILES.sort()
OBS_FILES.sort()
INFLUENCE_FILES = glob.glob(os.path.join(
    INFLUENCE_PATH,
    "LPDM_2010_01*{flux_interval:02d}hrly_{res:03d}km_footprints.nc4".format(
        flux_interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))

print(datetime.datetime.now(UTC).strftime("%c"))
print("Flux files", FLUX_FILES)
print("Influence Files", INFLUENCE_FILES)
sys.stdout.flush()

# tracers are:
#  diurnal bio, fossil, ocean, biomass burn, biofuel,
#  ship, posterior bio, empty, prior bio, empty
#    CMS posterior mole fractions are added to the first empty tracer,
#    and note the modified CMS posterior mole fraction file used for the
#     last tracer (prefix 'tsq')
TRUE_FLUX_NAME = "E_TRA7"
TRACER_NAME = "tracer_7_LPDM"
PRIOR_FLUX_NAME = TRUE_FLUX_NAME + "_noisy"

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
FLUX_WINDOW = HOURS_PER_DAY * DAYS_PER_WEEK * 2 - FLUX_INTERVAL
"""How long fluxes considered to have an influence.

Measured in hours.
Implemented by slicing out this much of the stored influence functions
for use as the linearized observation operator in the inversion.
"""
OBS_HOURS = (datetime.time(12), datetime.time(16))
"""Which observation times will be used in the inversion.

Assumed to be UTC. Should give afternoon hours for the domain.  Not
currently set up for domains spanning many time zones. This shouldn't
matter for OSSEs, but will for real-data cases (PBL schemes don't
treat night well)

I really hope I can assume this doesn't depend on latitude. That would
make this much more complicated.
"""
OBS_WINDOW = 16
"""Days of obs to process at once.
"""
OBS_PER_DAY = (OBS_HOURS[1].hour - OBS_HOURS[0].hour)
CO2_MOLAR_MASS = 16 * 2 + 12.01
"""Molar mass of CO2 (g/mol).

Used to convert WRF fluxes to units expected by observation operator.
"""
CO2_MOLAR_MASS_UNITS = cf_units.Unit("g/mol")
FLUX_UNITS = cf_units.Unit("g/m^2/hr")

FLUX_CHUNKS = 3 * HOURS_PER_DAY // FLUX_INTERVAL
OBS_CHUNKS_ALL = 24
OBS_CHUNKS_USED = 96
REALIZATION_CHUNK = 20

# Inverting a single day of observations
# Four stations; afternoon is four hours
# FLUX_WINDOW is one day: 25s, 19s
# FLUX_WINDOW is two days: 17s
# FLUX_WINDOW is four days: 25s
# seven days: 29s
# ten days: 57s
# fourteen days: 78s
# twenty-one days: >100s, 9.5 min.

# Inverting two days of observations
# FLUX_WINDOW is fourteen days: 86s -- OPTIMUM
# Three days of four hours at four towers
# fourteen days back: 168s

# Two days of observations, FLUX_WINDOW=14 days
# numpy OI: 108s
# numpy var: >607s
# dask var: >1228s
# dask OI: 99s
# Var may need differnt tuning; explore later

############################################################
# Utility functions.
# def grouper(iterable, n, fillvalue=None):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
#     args = [iter(iterable)] * n
#     return itertools.zip_longest(*args, fillvalue=fillvalue)


def grouper(lst, blocksize, total_len=None):
    """Return blocks of `blocksize` from `lst`.

    Parameters
    ----------
    lst: sequence
    blocksize: int
    total_len: int, optional
        Total length where this cannot be determined from lst

    Yields
    ------
    sequence
        slices from lst
    """
    for start in range(0, len(lst), blocksize):
        yield lst[start:start + blocksize]


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


############################################################
# Read sizes from influence function file
TEST_DS = netCDF4.Dataset(INFLUENCE_FILES[0])

NX = TEST_DS.dimensions["dim_x"].size
NY = TEST_DS.dimensions["dim_y"].size
N_TIMES_BACK = TEST_DS.dimensions["time_before_observation"].size

N_SITES = TEST_DS.dimensions["site"].size
N_OBS_TIMES = TEST_DS.dimensions["observation_time"].size

TEST_DS.close()
del TEST_DS

if N_TIMES_BACK < FLUX_WINDOW / FLUX_INTERVAL:
    raise ValueError("FLUX_WINDOW too long for file")

N_GRID_POINTS = NY * NX
STATE_SIZE = N_GRID_POINTS * FLUX_WINDOW

OBS_VEC_SIZE = N_SITES * OBS_WINDOW * OBS_PER_DAY
OBS_VEC_TOTAL_SIZE = N_SITES * N_OBS_TIMES

############################################################
# Read influence functions

# obs time chunk size works best as one.  Need to iterate over single
# hyperslabs along this dimension to have single flux time to line up
# with the time coordinate in the fluxes
# seven days: > 288012
# One day:
INFLUENCE_DATASET = xarray.open_mfdataset(
    INFLUENCE_FILES,
    # Kind of ad-hoc obs time chunk to match above
    # These may be too slow. I don't know how to check.
    chunks=dict(observation_time=OBS_CHUNKS_ALL, site=1,
                time_before_observation=FLUX_CHUNKS,
                dim_y=NY, dim_x=NX)).isel(
    # observation_time=slice(0, 6 * HOURS_PER_DAY),
    time_before_observation=slice(0, FLUX_WINDOW // FLUX_INTERVAL))
INFLUENCE_FUNCTIONS = INFLUENCE_DATASET.H
# Use site names as index/dim coord for site dim
INFLUENCE_FUNCTIONS["site"] = np.char.decode(
    INFLUENCE_FUNCTIONS["site_names"].values, "ascii")

# Not entirely sure why this is one too many
# N_FLUX_TIMES = INFLUENCE_DATASET.dims["observation_time"] + FLUX_WINDOW - 1
OBS_TIME_INDEX = INFLUENCE_DATASET.indexes["observation_time"].round("S")
TIME_BACK_INDEX = INFLUENCE_DATASET.indexes["time_before_observation"].round("S")

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

    # TOWER_WRF_INDICES = wrf.ll_to_xy(
    #     ds, INFLUENCE_FUNCTIONS.coords["site_lats"],
    #     INFLUENCE_FUNCTIONS.coords["site_lons"])

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

print(datetime.datetime.now(UTC).strftime("%c"), "Have constants, getting priors")
sys.stdout.flush()


############################################################
# Read prior fluxes
def fix_wrf_times(ds):
    wrf_times = ds["XTIME"].to_index().round("S")
    xtime_dim = "XTIME" in ds.dims
    timestamps = pd.DatetimeIndex(
        wrf_times,
        name="XTIME" if xtime_dim else "Time")
    ds.coords["Time"] = timestamps
    if xtime_dim:
        ds.set_index(XTIME="Time",
                     inplace=True)
    else:
        ds.set_index(Time="Time",
                     inplace=True)
    return ds


FLUX_DATASET = xarray.open_mfdataset(
    FLUX_FILES,
    chunks=dict(dim_x=NX, dim_y=NY,
                flux_time=FLUX_CHUNKS),
    concat_dim="flux_time",
    # preprocess=fix_wrf_times,
)
print(datetime.datetime.now(UTC).strftime("%c"), "Have fluxes, getting obs")
sys.stdout.flush()
OBS_DATASET = xarray.open_mfdataset(
    OBS_FILES,
    chunks=dict(forecast_reference_time=24),
    concat_dim="forecast_reference_time",
    # preprocess=fix_wrf_times,
    # drop_variables=("HGT", "PH", "PHB", "ZS"),
)
print(datetime.datetime.now(UTC).strftime("%c"), "Have obs, normalizing")
sys.stdout.flush()

# Many of the times are of by about four milliseconds.
# This difference is irrelevant here.
wrf_times = OBS_DATASET["forecast_reference_time"].to_index().round("S")
OBS_DATASET.coords["forecast_reference_time"] = wrf_times

print(OBS_DATASET.dims, OBS_DATASET.coords)
OBS_DATASET.coords["site"] = list(
    map(lambda x: x.decode("ascii"),
        OBS_DATASET["name_of_observation_site"].values))
OBS_DATASET.set_index(dim1="site",
                      inplace=True)
OBS_DATASET.rename(dict(dim1="site"),
                   inplace=True)
del OBS_DATASET.coords["name_of_observation_site"]
print(OBS_DATASET.dims, OBS_DATASET.coords)


wrf_times = FLUX_DATASET["flux_time"].to_index().round("S")
timestamps = list(wrf_times)
timestamps[-1] += datetime.timedelta(hours=FLUX_INTERVAL/2-1)
timestamps[0] -= datetime.timedelta(hours=1)
wrf_new_times = pd.DatetimeIndex(timestamps,
                                 name="flux_time")
FLUX_DATASET["flux_time"] = wrf_new_times

# Assign a few more coords and pull out only the fluxes we need.
FLUX_DATASET = FLUX_DATASET.sel(flux_time=FLUX_TIMES_INDEX)
N_REALIZATIONS = len(FLUX_DATASET.indexes["realization"])

WRF_DX = FLUX_DATASET.attrs["DX"]
# OBS_DATASET = OBS_DATASET.set_index(
#     dim3="projection_x_coordinate",
#     dim2="projection_y_coordinate",
#     inplace=True)
# OBS_DATASET.rename(dict(
#     dim3="projection_x_coordinate",
#     dim2="projection_y_coordinate"),
#                    inplace=True)

TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i+1)
                                for i in range(10)])
TRUE_FLUXES_MATCHED = TRUE_FLUXES # * CO2_MOLAR_MASS
# for flux_part, flux_orig in zip(TRUE_FLUXES_MATCHED.data_vars.values(), TRUE_FLUXES.data_vars.values()):
#     unit = (cf_units.Unit(flux_orig.attrs["units"]) *
#             CO2_MOLAR_MASS_UNITS)
#     # For whatever reason this is backwards from the conversion
#     # factors used elsewhere.
#     flux_part *= (unit / FLUX_UNITS).convert(1, 1)
#     flux_part.attrs["units"] = str(FLUX_UNITS)

PRIOR_FLUXES = FLUX_DATASET.get(["E_TRA{:d}_noisy".format(i + 1)
                                 for i in (6,)])
PRIOR_FLUXES_MATCHED = PRIOR_FLUXES # * CO2_MOLAR_MASS
# for flux_part, flux_orig in zip(PRIOR_FLUXES_MATCHED.data_vars.values(), PRIOR_FLUXES.data_vars.values()):
#     unit = (cf_units.Unit(flux_orig.attrs["units"]) *
#             CO2_MOLAR_MASS_UNITS)
#     # For whatever reason this is backwards from the conversion
#     # factors used elsewhere.
#     flux_part *= (unit / FLUX_UNITS).convert(1, 1)
#     flux_part.attrs["units"] = str(FLUX_UNITS)

WRF_OBS = OBS_DATASET.get(
    ["tracer_{:d}_LPDM".format(i+1)
     for i in (1, 6, 8)])
WRF_OBS_MATCHED = WRF_OBS.rename(dict(
    forecast_reference_time="observation_time"))
WRF_OBS_SITE = WRF_OBS_MATCHED

WRF_OBS_START = WRF_OBS_MATCHED.indexes["observation_time"].round("S")[0]
WRF_OBS_INTERVAL = (
    WRF_OBS_START - WRF_OBS_MATCHED.indexes["observation_time"].round("S")[1])

print(datetime.datetime.now(UTC).strftime("%c"), "Getting solar times")
sys.stdout.flush()
############################################################
# Get time zone representing local solar time for each site
LOCAL_TIME_ZONES = list(map(
    dateutil.tz.tzoffset,
    ("Local time for {name!s}".format(name=name)
     for name in INFLUENCE_FUNCTIONS.coords["site"].values),
    [round(offset_hr * SECONDS_PER_HOUR)
     for offset_hr in INFLUENCE_FUNCTIONS.coords["site_lons"].values / 15]))

# Define proper dimension order for INFLUENCE_FUNCTIONS
dimension_order = [item for item in INFLUENCE_FUNCTIONS.dims]
dimension_order.insert(dimension_order.index("time_before_observation"),
                       "flux_time")
dimension_order.insert(dimension_order.index("observation_time"), "observation")
dimension_order = tuple(dimension_order)

############################################################
# Define metadata for output variables
############################################################
posterior_var_atts = PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].attrs.copy()
posterior_var_atts.update(dict(
    long_name="posterior_fluxes",
    # units=TRUE_FLUXES_MATCHED[FLUX_NAME].attrs["units"],
    description="posterior fluxes using dask for a week at a time, overlapped for a month",
    origin="OI using dask for a month",
    prior_flux_name=PRIOR_FLUX_NAME,
    flux_window=FLUX_WINDOW,
    observation_window=OBS_WINDOW))
increment_var_atts = PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].attrs.copy()
increment_var_atts.update(dict(
    long_name="flux_increment",
    # units=TRUE_FLUXES_MATCHED[FLUX_NAME].attrs["units"],
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
    source="Test inversion using OI for a 8-day windows strung together for a month",
))


############################################################
# Define correlation constants and get correlations
# Covariances have time-varying component
print(datetime.datetime.now(UTC).strftime("%c"), "Getting correlations")
sys.stdout.flush()
CORRELATION_LENGTH = 200
GRID_RESOLUTION = FLUX_RESOLUTION
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
    from_function(inversion.correlations.ExponentialCorrelation(HOURLY_FLUX_TIMESCALE),
                  (HOURS_PER_DAY // FLUX_INTERVAL,)))
hour_correlations_matrix = hour_correlations.dot(
    np.eye(hour_correlations.shape[0]))
print(datetime.datetime.now(UTC).strftime("%c"), "Have hourly correlations")
sys.stdout.flush()
DAILY_FLUX_TIMESCALE = 14
day_correlations = (
    inversion.correlations.make_matrix(
        inversion.correlations.ExponentialCorrelation(DAILY_FLUX_TIMESCALE),
        ((FLUX_WINDOW) // HOURS_PER_DAY + OBS_WINDOW,)))
print(datetime.datetime.now(UTC).strftime("%c"), "Have daily correlations")
sys.stdout.flush()
temporal_correlations = kronecker_product(day_correlations, hour_correlations_matrix)

print(datetime.datetime.now(UTC).strftime("%c"), "Have combined correlations")
sys.stdout.flush()

OBSERVATION_VARIANCE = .2
OBS_CORR_FUN = inversion.correlations.ExponentialCorrelation(3)
"""Temporal correlations in observation error.

Mostly reflects transport error.  Given in units of obs_time.
Since the input is hourly this is a three-hour decay time.
"""
OBS_INTERVAL = np.array(1, dtype='m8[h]')

############################################################
# Do the inversion
############################################################
obs_times = (INFLUENCE_FUNCTIONS.indexes["observation_time"][::-1] -
             datetime.timedelta(hours=1))

# list of the parts of the posterior, collected to merge at once
have_posterior_part = False
for i, inversion_period in enumerate(grouper(obs_times, OBS_WINDOW * HOURS_PER_DAY)):
    ############################################################
    # Align observation flux times
    # Take care of missing obs
    # Also subsetting for late afternoon steady convective boundary layer
    # takes roughly 11 min.
    site_obs_index = []
    print(datetime.datetime.now(UTC).strftime("%c"), "Selecting observations")
    for j, site in enumerate(INFLUENCE_FUNCTIONS.indexes["site"]):
        local_times = pd.Index([
            obs_time.tz_localize(UTC)
            .tz_convert(LOCAL_TIME_ZONES[j])
            for obs_time in inversion_period], name="local_times")

        # np.where is designed for multi-dimensional arrays
        site_obs_index.extend(zip(
            np.where((OBS_HOURS[0] < local_times.time) &
                     (local_times.time < OBS_HOURS[1]))[0],
            itertools.repeat(site)))
    obs_index, site_index = zip(*site_obs_index)
    site_index = np.array(list(site_index))
    pd_obs_index = inversion_period[list(obs_index)]
    site_obs_pd_index = pd.MultiIndex.from_tuples(
        list(zip(site_index, pd_obs_index)),
        names=("site", "observation_time"))

    print(datetime.datetime.now(UTC).strftime("%c"),
          "Aligning flux times in influence function")
    sys.stdout.flush()
    print(datetime.datetime.now(UTC).strftime("%c"),
          "Converting influence function to have alignment necessary for H")
    matched_influences = xarray.concat(
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
    start_date, end_date = [
        dt.replace(hour=0)
        for dt in matched_influences.indexes["flux_time"][[0, -1]]]
    if matched_influences.indexes["flux_time"][-1].hour != 0:
        end_date += datetime.timedelta(days=1)
    start_date += FLUX_INTERVAL_DT

    # xarray appears to have changed to include poth endpoints at some
    # point.  To counter this and get back to proper half-open
    # indexing, I move the start one flux-interval later, since the
    # end could be physically relevant
    if not have_posterior_part:
        unaligned_fluxes = PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].sel(
            flux_time=slice(start_date, end_date)).persist()
    else:
        print(posterior_ds.coords["flux_time"][OBS_WINDOW * HOURS_PER_DAY // FLUX_INTERVAL:])
        print(PRIOR_FLUXES_MATCHED.coords["flux_time"][
            (FLUX_WINDOW + (i*OBS_WINDOW) * HOURS_PER_DAY) // FLUX_INTERVAL:])

        old_posterior_fluxes = posterior_ds["posterior"].isel(
            flux_time=slice(OBS_WINDOW * HOURS_PER_DAY // FLUX_INTERVAL, None))
        new_fluxes = PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].sel(
            flux_time=slice(
                old_posterior_fluxes.indexes["flux_time"][-1] + FLUX_INTERVAL_DT,
                end_date))
        unaligned_fluxes = xarray.concat(
            [old_posterior_fluxes,
             # TODO: fix this for the last time window
             new_fluxes.rename("posterior")],
            dim="flux_time").persist()
        print(unaligned_fluxes)
    print("Unaligned flux coords")
    print(unaligned_fluxes.coords)
    aligned_influences, aligned_fluxes = xarray.align(
        matched_influences.isel(flux_time=slice(1, None)),
                                unaligned_fluxes,
        exclude=("dim_x", "dim_y", "observation"),
        join="outer", copy=False)
    print("Aligned flux coords")
    print(aligned_fluxes.coords)

    print("Aligned incluences")
    print(aligned_influences)
    print("Aligned fluxes")
    print(aligned_fluxes)
    print(N_GRID_POINTS * N_FLUX_TIMES)
    # aligned_influences.reindex(flux_time=FLUX_TIMES_INDEX)
    print(datetime.datetime.now(UTC).strftime("%c"), "Aligned fluxes and influence function")
    aligned_influences = aligned_influences.fillna(0).chunk(dict(
            dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS,
            observation=OBS_CHUNKS_USED)).transpose(
        "observation", "flux_time", "dim_y", "dim_x")
    aligned_fluxes = aligned_fluxes.chunk(dict(
            dim_y=NY, dim_x=NX, flux_time=FLUX_CHUNKS))
    print(datetime.datetime.now(UTC).strftime("%c"), "Influence functions now H")
    transpose_arg = sort_key_to_consecutive([dimension_order.index(dim)
                                             for dim in aligned_influences.dims])

    # I would like to add a fixed minimum at some point.
    # full stds would then be sqrt(fixed^2 + varying^2)
    # average seasonal variation (or some fraction thereof) might work.
    FLUX_VARIANCE_VARYING_FRACTION = 30.
    # TODO: Check times using numpy for inversion.* and no dask here.
    # Using dask prompts xarray to load the data. I'd like to do that all at once later.
    flux_std_pattern = xarray.open_dataset("../data_files/wrf_flux_rms.nc").get(
        ["E_TRA{:d}".format(i + 1) for i in range(10)]).isel(emissions_zdim=0)
    # Ensure units work out
    for flux_part in flux_std_pattern.data_vars.values():
        unit = (cf_units.Unit(flux_part.attrs["units"]))
        if unit is not FLUX_UNITS:
            flux_part *= (
                unit * CO2_MOLAR_MASS_UNITS / FLUX_UNITS
            ).convert(1, 1) * CO2_MOLAR_MASS
            flux_part.attrs["units"] = str(FLUX_UNITS)
    flux_stds = (
        FLUX_VARIANCE_VARYING_FRACTION * flux_std_pattern[TRUE_FLUX_NAME].data)

    # Calculate temporal covariances having same number of days as the fluxes
    day_correlations = (
        inversion.correlations.make_matrix(
            inversion.correlations.ExponentialCorrelation(DAILY_FLUX_TIMESCALE),
            (len(aligned_fluxes.indexes["flux_time"]) * FLUX_INTERVAL // HOURS_PER_DAY,)))
    print(datetime.datetime.now(UTC).strftime("%c"), "Have daily correlations")
    sys.stdout.flush()
    temporal_correlations = kronecker_product(day_correlations, hour_correlations_matrix)

    print(datetime.datetime.now(UTC).strftime("%c"), "Have combined correlations")
    sys.stdout.flush()

    prior_covariance = kronecker_product(
        temporal_correlations,
        inversion.util.CorrelationStandardDeviation(
            spatial_correlations, flux_stds))

    print(datetime.datetime.now(UTC).strftime("%c"), "Have covariances")
    sys.stdout.flush(); sys.stderr.flush()

    # TODO: use actual heights
    here_obs = WRF_OBS_SITE[TRACER_NAME].sel_points(
        observation_time=pd_obs_index, site=site_index
        )

    observation_covariance = OBS_CORR_FUN(
        abs(pd_obs_index[:, np.newaxis] - pd_obs_index[np.newaxis, :]) /
        OBS_INTERVAL)
    # Assumes no correlations between observations.
    observation_covariance[
        site_index[:, np.newaxis] != site_index[np.newaxis, :]] = 0
    observation_covariance = asarray(observation_covariance)

    used_observation_vals = (
        here_obs.data[:, np.newaxis] +
        gaussian_noise(observation_covariance, N_REALIZATIONS).T.reshape(
            here_obs.shape + (N_REALIZATIONS,))).persist()
    used_observations = xarray.DataArray(
        used_observation_vals,
        here_obs.coords,
        here_obs.dims + ("realization",),
        "pseudo_observations",
        # obs_atts
    ).rename(dict(longitude_0="tower_lon", latitude_0="tower_lat")).chunk(
        dict(realization=REALIZATION_CHUNK))
    used_observations.coords["realization"] = range(N_REALIZATIONS)
    used_observations.coords["realization"].attrs.update(dict(
        standard_name="realization"))
    print(datetime.datetime.now(UTC).strftime("%c"), "Have observation noise")
    sys.stdout.flush(); sys.stderr.flush()

    print(datetime.datetime.now(UTC).strftime("%c"), "Got covariance parts, getting posterior")
    sys.stdout.flush()
    posterior, post_cov = inversion.optimal_interpolation.save_sum(
        aligned_fluxes.data.reshape(
            N_GRID_POINTS * len(aligned_influences.indexes["flux_time"]),
            N_REALIZATIONS).compute(),
        prior_covariance,
        used_observations.data.compute(),
        observation_covariance,
        (aligned_influences.data
         .transpose(transpose_arg)
         .reshape(aligned_influences.shape[0],
                  np.prod(aligned_influences.shape[-3:]))).compute(),
        np.ones((1, 1)),
        np.ones((used_observations.shape[0], 1)))

    print(datetime.datetime.now(UTC).strftime("%c"), "Have posterior values, making dataset")
    sys.stdout.flush()
    posterior = posterior.reshape(aligned_fluxes.shape)
    posterior_ds = xarray.Dataset(
        dict(posterior=(aligned_fluxes.dims, posterior,
                        posterior_var_atts),
             prior=(aligned_fluxes.dims, aligned_fluxes,
                    aligned_fluxes.attrs),
             increment=(aligned_fluxes.dims, posterior - aligned_fluxes,
                        increment_var_atts),
             ),
        aligned_fluxes.coords,
        posterior_global_atts
    )
    print(posterior_ds)
    used_observations.to_netcdf("observation_realizations_for_{flux_interval:02d}h_{step:02d}.nc4"
                                .format(flux_interval=FLUX_INTERVAL, step=i))
    posterior_part = posterior_ds.isel(flux_time=slice(None, OBS_WINDOW * HOURS_PER_DAY//FLUX_INTERVAL))
    posterior_part.to_netcdf("monthly_inversion_{flux_interval:02d}h_output_{step:02d}.nc4".format(
            flux_interval=FLUX_INTERVAL, step=i))
    have_posterior_part = True
    print(datetime.datetime.now(UTC).strftime("%c"), "Have posterior dataset, looping for next obs")
    sys.stdout.flush()


print(datetime.datetime.now(UTC).strftime("%c"), "Have posterior structure, evaluating and writing")
sys.stdout.flush()
print("Parts of posterior already written, catenate parts with ncrcat.")
print("Not all of posterior written; writing rest")
posterior_ds.to_netcdf("monthly_inversion_{flux_interval:02d}h_output_zz.nc4".format(
            flux_interval=FLUX_INTERVAL))
print(datetime.datetime.now(UTC).strftime("%c"), "Wrote posterior")
sys.stdout.flush()
