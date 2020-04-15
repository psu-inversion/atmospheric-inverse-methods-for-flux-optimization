#!/usr/bin/env python
"""Run an identical twin flux inversion OSSE with real data.

Use xarray/dask to grab influence functions and priors from netCDF
files.
"""
from __future__ import print_function, division, unicode_literals

import itertools
import datetime
import textwrap
import os.path
import glob
import sys

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

import atmos_flux_inversion.optimal_interpolation
import atmos_flux_inversion.variational
import atmos_flux_inversion.correlations
import atmos_flux_inversion.covariances
from atmos_flux_inversion.util import kronecker_product
from atmos_flux_inversion.linalg import asarray, kron
from atmos_flux_inversion.noise import gaussian_noise
import cf_acdd

INFLUENCE_PATHS = ["/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "ACT-America_trial5/2010/01/GROUP1",
                   "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "candidacy_more_towers/2010/01/GROUP1"]
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
FLUX_RESOLUTION = 27
"""Resolution of fluxes and influence functions in kilometers."""
UNCERTAINTY_RESOLUTION_REDUCTION_FACTOR = 16
"""How much coarser uncertainty is than mean estimate in the x direction.

If we compute the uncertainty at full resolution, the resulting file
is huge, starting in the hundreds of terabytes for a month.
Coarsening the resolution for the uncertainties allows us to still
report uncertainties within current computing constraints.
"""
# 4: 2h49 wall 132GiB mem 3h57 cpu
# 3: 5h11 wall 140GiB mem 6h27 cpu
UNCERTAINTY_FLUX_RESOLUTION = (UNCERTAINTY_RESOLUTION_REDUCTION_FACTOR *
                               FLUX_RESOLUTION * 1e3)
"""Resolution of posterior uncertainties in meters."""
UNCERTAINTY_TEMPORAL_RESOLUTION = "7D"
"""The resolution at which the uncertainty is calculated and saved.

Higher resolution means the uncertainties will be more accurate.
"""

# Linear interpolation in space
OBS_FILES = glob.glob(os.path.join(
    OBS_PATH,
    "2010_07_[45]tower_{inter:02d}hr_{res:03d}km_"
    "LPDM_concentrations?.nc".format(
        inter=FLUX_INTERVAL, res=FLUX_RESOLUTION)))
CORR_FUN = "exp"
CORR_LEN = 200
TIME_CORR_FUN = "exp"
TIME_CORR_LEN = 21
FLUX_FILES = glob.glob(os.path.join(
    PRIOR_PATH,
    ("2010-07_osse_bio_priors_{interval:1d}h_{res:02d}km_noise_"
     "{corr_fun:s}{corr_len:d}km_"
     "{corr_fun_time:s}{corr_len_time:d}d_exp3h.nc").format(
        interval=FLUX_INTERVAL, corr_fun=CORR_FUN, corr_len=CORR_LEN,
        corr_fun_time=TIME_CORR_FUN, corr_len_time=TIME_CORR_LEN,
        res=FLUX_RESOLUTION)))
FLUX_FILES.sort()
OBS_FILES.sort()
INFLUENCE_FILES = [
    name
    for path in INFLUENCE_PATHS
    for name in glob.glob(os.path.join(
        path,
        "LPDM_2010_01_{flux_interval:02d}hrly_{res:03d}km_molar_footprints.nc4"
        .format(flux_interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))]

print("Flux files", FLUX_FILES)
print("Influence Files", INFLUENCE_FILES)

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
FLUX_WINDOW = HOURS_PER_DAY * DAYS_PER_WEEK * 2
"""How long fluxes considered to have an influence.

Measured in hours.
Implemented by slicing out this much of the stored influence functions
for use as the linearized observation operator in the inversion.
"""
OBS_HOURS = (datetime.time(12), datetime.time(16))
"""Which observation times will be used in the inversion.

Assumed to be local solar. Should give afternoon hours for the domain.

I really hope I can assume this doesn't depend on latitude. That would
make this much more complicated.
"""
OBS_TIMES_PER_DAY = OBS_HOURS[1].hour - OBS_HOURS[0].hour
"""Observations used per site per day."""
CO2_MOLAR_MASS = 16 * 2 + 12.01
"""Molar mass of CO2 (g/mol).

Used to convert WRF fluxes to units expected by observation operator.
"""
DAYS_DROPPED_FROM_END = 1
"""Currently 1 to avoid problems with lack of fluxes in August."""
OBS_DAYS = 30
N_REALIZATIONS = 80
#  1    9m55 (80 realizations)    9m46
# 30 2h48m40 (80 realizations) 3h56m41
"""Number of days of obs to use."""
OBS_WINDOW = OBS_DAYS * OBS_TIMES_PER_DAY
"""Number of observation times."""
CO2_MOLAR_MASS_UNITS = cf_units.Unit("g/mol")
FLUX_UNITS = cf_units.Unit("mol/m^2/s")
BAD_SITES = ("WGC", "OSI")

FLUX_CHUNKS = HOURS_PER_DAY * 8 // FLUX_INTERVAL
# 48 51s
#  4 54s
#  2 4m23
"""How many flux times to treat at once.

Must be a multiple of day length.
"""
OBS_CHUNKS_ALL = 24
"""How many observations to load at once.

Must allow a few chunks of the influence function to be in memory at
once.  Will need to extend this if (N_OBS_TIMES * N_SITES) ** 2 arrays
stop fitting nicely in memory.
"""
OBS_CHUNKS_USED = 96
"""Obs chunks treated at once in inversion code.

Must also allow a few chunks to be loaded at once.
"""
REALIZATION_CHUNK = 20
NC_ENGINE = "netcdf4"


############################################################
# Utility functions.
def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
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


def flush_output_streams():
    """Flush stdout and stderr."""
    sys.stdout.flush()
    sys.stderr.flush()


def write_progress_message(msg):
    """Write the message to stdout with time.

    Parameters
    ----------
    msg: str
    """
    flush_output_streams()
    print(datetime.datetime.now(UTC).strftime("%c"), msg)
    flush_output_streams()


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
print("Days of obs used:", OBS_DAYS)

# Quadratic form in save_sum
# OPTIMAL_ELEMENTS 1e4, not changed in save_sum
# obs_chunk: 24: 96 flux_chunk 3 days
# 2 week flux window, 40 realizations in loaded noise
# 1* OPTIMAL_ELEMENTS**2
# OPTIMAL_ELEMENTS 5e4
# flux_chunk 3 days, obs_chunk 24: 96
# run under debugger
#  2 days obs:  2.5 min, 2.3min
#  4 days obs:  4.3 min
#  8 days obs:  9.5 min
# 16 days obs: 24.5 min
# 31 days obs: 83.3 min
#   8m for Kd, 75 to find HBH^T
INFLUENCE_DATASET = xarray.open_mfdataset(
    INFLUENCE_FILES,
    chunks=dict(observation_time=OBS_CHUNKS_ALL, site=1,
                time_before_observation=FLUX_CHUNKS,
                dim_y=NY, dim_x=NX),
    engine=NC_ENGINE,
).isel(
    observation_time=slice(DAYS_DROPPED_FROM_END * HOURS_PER_DAY,
                           (OBS_DAYS + DAYS_DROPPED_FROM_END) * HOURS_PER_DAY),
    time_before_observation=slice(0, FLUX_WINDOW // FLUX_INTERVAL))
OBS_TIME_INDEX = (INFLUENCE_DATASET.indexes["observation_time"].round("S") +
                  datetime.timedelta(days=181))
TIME_BACK_INDEX = (
    INFLUENCE_DATASET.indexes["time_before_observation"].round("S"))
INFLUENCE_DATASET.coords["observation_time"] = OBS_TIME_INDEX
INFLUENCE_DATASET.coords["time_before_observation"] = TIME_BACK_INDEX

FLUX_TIMES = (INFLUENCE_DATASET.coords["flux_time"] +
              np.array(datetime.timedelta(days=181),
                       dtype='m8[ns]'))
INFLUENCE_DATASET.coords["flux_time"] = FLUX_TIMES

INFLUENCE_FUNCTIONS = INFLUENCE_DATASET.H
# Use site names as index/dim coord for site dim
INFLUENCE_FUNCTIONS.coords["site"] = np.char.decode(
    INFLUENCE_FUNCTIONS["site_names"].values, "ascii")

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
assert N_FLUX_TIMES % 4 == 0

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
    chunks=dict(dim_x=NX, dim_y=NY,
                flux_time=FLUX_CHUNKS,
                realization=REALIZATION_CHUNK),
    concat_dim="flux_time",
    engine=NC_ENGINE,
).isel(realization=slice(0, N_REALIZATIONS))

OBS_FILES_4 = [name for name in OBS_FILES if "4tower" in name]
OBS_FILES_5 = [name for name in OBS_FILES if "5tower" in name]

OBS_DATASET_4 = xarray.open_mfdataset(
    OBS_FILES_4,
    chunks=dict(forecast_reference_time=OBS_CHUNKS_USED),
    engine=NC_ENGINE,
)
OBS_DATASET_5 = xarray.open_mfdataset(
    OBS_FILES_5,
    chunks=dict(forecast_reference_time=OBS_CHUNKS_USED),
    engine=NC_ENGINE,
)
OBS_DATASET = xarray.concat([OBS_DATASET_4, OBS_DATASET_5],
                            dim="dim1")

write_progress_message("Have obs, normalizing")

wrf_times = OBS_DATASET.indexes["forecast_reference_time"].round("S")
OBS_DATASET.coords["forecast_reference_time"] = wrf_times

print(OBS_DATASET.dims, OBS_DATASET.coords)
OBS_DATASET.coords["site"] = list(
    map(lambda x: x.decode("ascii"),
        OBS_DATASET["name_of_observation_site"].values))
OBS_DATASET = (
    OBS_DATASET
    .set_index(dim1="site")
    .rename(dict(dim1="site"))
)
del OBS_DATASET.coords["name_of_observation_site"]
print(OBS_DATASET.dims, OBS_DATASET.coords)
# Assign a few more coords and pull out only the fluxes we need.
FLUX_DATASET = FLUX_DATASET.sel(flux_time=FLUX_TIMES_INDEX.tz_convert(None))
N_REALIZATIONS = len(FLUX_DATASET.indexes["realization"])

WRF_DX = FLUX_DATASET.attrs["DX"]

TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i + 1)
                                for i in range(10)])
TRUE_FLUXES_MATCHED = TRUE_FLUXES
# for flux_part, flux_orig in zip(
#         TRUE_FLUXES_MATCHED.data_vars.values(),
#         TRUE_FLUXES.data_vars.values()):
#     unit = (cf_units.Unit(flux_orig.attrs["units"]) *
#             CO2_MOLAR_MASS_UNITS)
#     # For whatever reason this is backwards from the conversion
#     # factors used elsewhere.
#     flux_part *= (unit / FLUX_UNITS).convert(1, 1)
#     flux_part.attrs["units"] = str(FLUX_UNITS)

PRIOR_FLUXES = FLUX_DATASET.get(["E_TRA{:d}_noisy".format(i + 1)
                                 for i in (6,)])
PRIOR_FLUXES_MATCHED = PRIOR_FLUXES
# for flux_part, flux_orig in zip(
#         PRIOR_FLUXES_MATCHED.data_vars.values(),
#         PRIOR_FLUXES.data_vars.values()):
#     unit = (cf_units.Unit(flux_orig.attrs["units"]) *
#             CO2_MOLAR_MASS_UNITS)
#     # For whatever reason this is backwards from the conversion
#     # factors used elsewhere.
#     flux_part *= (unit / FLUX_UNITS).convert(1, 1)
#     flux_part.attrs["units"] = str(FLUX_UNITS)

WRF_OBS = OBS_DATASET.get(
    ["tracer_{:d}_LPDM".format(i + 1)
     for i in (1, 6, 8)])
WRF_OBS_MATCHED = WRF_OBS.rename(dict(
    forecast_reference_time="observation_time"))
WRF_OBS_SITE = WRF_OBS_MATCHED
print(WRF_OBS_MATCHED.dims, WRF_OBS_MATCHED.coords)

WRF_OBS_START = WRF_OBS_MATCHED.indexes["observation_time"][0]
WRF_OBS_INTERVAL = (WRF_OBS_START -
                    WRF_OBS_MATCHED.indexes["observation_time"][1])

write_progress_message("Getting solar times")
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
write_progress_message("Selecting observations")
for i, site in enumerate(INFLUENCE_FUNCTIONS.indexes["site"]):
    if site in BAD_SITES:
        continue
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
flush_output_streams()
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
     for here_infl in INFLUENCE_FUNCTIONS.sel(
         site=xarray.DataArray(site_index, name="observation",
                               dims=dict(observation=site_obs_pd_index)),
         observation_time=xarray.DataArray(
             pd_obs_index, name="observation",
             dims=dict(observation=site_obs_pd_index)
         )
    )],
    "observation").set_index(
    observation=("observation_time", "site"))
print(datetime.datetime.now(UTC).strftime("%c"),
      "Aligned flux times in influence function, "
      "aligning fluxes with influence function")
flush_output_streams()
aligned_influences, aligned_true_fluxes, aligned_prior_fluxes = (
    xarray.align(
        aligned_influences,
        TRUE_FLUXES_MATCHED[TRUE_FLUX_NAME],
        PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME],
        exclude=("dim_x", "dim_y"),
        join="outer", copy=False))
print(datetime.datetime.now(UTC).strftime("%c"),
      "Aligned fluxes and influence function")
flush_output_streams()
aligned_true_fluxes = aligned_true_fluxes.transpose(
    "flux_time", "dim_y", "dim_x")
aligned_prior_fluxes = aligned_prior_fluxes.transpose(
    "flux_time", "dim_y", "dim_x", "realization")
aligned_influences = aligned_influences.transpose(
    "observation", "flux_time", "dim_y", "dim_x")
write_progress_message("Rechunked to square")
aligned_influences = aligned_influences.fillna(0)
aligned_true_fluxes.load()
aligned_prior_fluxes.load()
aligned_influences.load()
write_progress_message("Loaded data")

posterior_var_atts = aligned_prior_fluxes.attrs.copy()
posterior_var_atts.update(dict(
    long_name="posterior_fluxes",
    units=PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].attrs["units"],
    description="posterior fluxes using dask for a month",
    origin="OI using dask for a month",
    prior_flux_name=PRIOR_FLUX_NAME,
    flux_window=FLUX_WINDOW,
    observation_window=OBS_WINDOW,
    ancillary_variables="reduced_posterior_covariance",
))
increment_var_atts = aligned_prior_fluxes.attrs.copy()
increment_var_atts.update(dict(
    long_name="flux_increment",
    units=PRIOR_FLUXES_MATCHED[PRIOR_FLUX_NAME].attrs["units"],
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
    external_variables="reduced_posterior_covariance reduced_prior_covariance",
))

############################################################
# Define correlation constants and get covariances
# Constants for inversion
write_progress_message("Getting covariances")
CORRELATION_LENGTH = 1000
GRID_RESOLUTION = 27
spatial_correlations = (
    atmos_flux_inversion.correlations.make_matrix(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            CORRELATION_LENGTH / GRID_RESOLUTION),
        (len(TRUE_FLUXES_MATCHED.coords["dim_y"]),
         len(TRUE_FLUXES_MATCHED.coords["dim_x"])),
    )
)
# spatial_correlation_remapper = np.full(
#     # Grid points at full resolution, grid points at reduced resolution
#     (spatial_correlations.shape[0], 1),
#     # 1/Number of gridpoints combined in above mapping
#     1. / spatial_correlations.shape[0])
# # reduced_spatial_correlations = (
# #     spatial_correlation_remapper.dot(
# #         spatial_correlations.dot(spatial_correlation_remapper)))
import atmos_flux_inversion.remapper
spatial_obs_op_remapper, spatial_correlation_remapper = (
    atmos_flux_inversion.remapper.get_remappers(
        (len(TRUE_FLUXES_MATCHED.coords["dim_y"]),
         len(TRUE_FLUXES_MATCHED.coords["dim_x"])),
        UNCERTAINTY_RESOLUTION_REDUCTION_FACTOR))
REDUCED_NY, REDUCED_NX = spatial_correlation_remapper.shape[:2]
REDUCED_N_GRID_POINTS = REDUCED_NY * REDUCED_NX
write_progress_message("Have spatial correlations")

HOURLY_FLUX_TIMESCALE = 3
INTERVALS_PER_DAY = HOURS_PER_DAY // FLUX_INTERVAL
hour_correlations = (
    atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
    from_function(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            HOURLY_FLUX_TIMESCALE / FLUX_INTERVAL),
        (INTERVALS_PER_DAY,),
        is_cyclic=True))
hour_correlations_matrix = hour_correlations.dot(np.eye(
    hour_correlations.shape[0]))
write_progress_message("Have hourly correlations")
DAILY_FLUX_TIMESCALE = 7
DAILY_FLUX_FUN = "exp"
day_correlations = (
    atmos_flux_inversion.correlations.make_matrix(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            DAILY_FLUX_TIMESCALE
        ),
        (len(aligned_prior_fluxes.indexes["flux_time"]) *
         FLUX_INTERVAL // HOURS_PER_DAY,)))
write_progress_message("Have daily correlations")
temporal_correlations = kron(day_correlations,
                             hour_correlations_matrix)
print("Temporal:", type(temporal_correlations))
temporal_correlation_ds = xarray.DataArray(
    temporal_correlations,
    dict(flux_time=aligned_prior_fluxes.indexes["flux_time"].values,
         flux_time_adjoint=aligned_prior_fluxes.indexes["flux_time"].values),
    ("flux_time", "flux_time_adjoint"),
    "temporal_correlations",
    dict(long_name="temporal_correlations",
         units="dimensionless")
)
# Mean of fluxes, so sum of variances over square of number of
# elements in the group.  The number of rows and columns for each
# group will each be the number of elements in that group.
reduced_temporal_correlation_ds = (
    temporal_correlation_ds
    .resample(flux_time=UNCERTAINTY_TEMPORAL_RESOLUTION).mean("flux_time")
    .resample(flux_time_adjoint=UNCERTAINTY_TEMPORAL_RESOLUTION).mean(
        "flux_time_adjoint"
    )
)
write_progress_message("Have temporal correlations")
print(reduced_temporal_correlation_ds.values)
flush_output_streams()

full_correlations = kronecker_product(
    temporal_correlations,
    spatial_correlations)
print("Full:", type(full_correlations))
write_progress_message("Have combined correlations")

# I would like to add a fixed minimum at some point.
# full stds would then be sqrt(fixed^2 + varying^2)
# average seasonal variation (or some fraction thereof) might work.
# x2 since MsTMIP spread does not represent full uncertainty
# x5 since MsTMIP spread only represents monthly values and this uses sub-daily
# x10 matches model-model for Raczka for 200km/21d
# x4 matches model-model for 1000km/7d
FLUX_VARIANCE_VARYING_FRACTION = 4.
flux_std_pattern = xarray.open_dataset(
    "../data_files/2010_MsTMIP_flux_std.nc4",
    engine=NC_ENGINE
).get(
    ["E_TRA{:d}".format(i + 1) for i in range(1)]
).sel(
    Time=reduced_temporal_correlation_ds.indexes["flux_time"]
).mean(
    dim="Time",
    keep_attrs=True,
)

# Ensure units work out
for flux_part in flux_std_pattern.data_vars.values():
    unit = (cf_units.Unit(flux_part.attrs["units"]))
    if unit is not FLUX_UNITS:
        flux_part *= unit.convert(1, FLUX_UNITS)
        flux_part.attrs["units"] = str(FLUX_UNITS)

# flux_stds = (
#     FLUX_VARIANCE_VARYING_FRACTION *
#     flux_std_pattern["E_TRA1"].rolling(
#         center=True, min_periods=3, Time=21 * 8
#     ).mean(dim="Time").sel(
#         Time=temporal_correlation_ds.indexes["flux_time"]
#     ).data)
reduced_flux_stds = (
    FLUX_VARIANCE_VARYING_FRACTION *
    flux_std_pattern["E_TRA1"].data)
write_progress_message("Have standard deviations")

spatial_covariance = (
    atmos_flux_inversion.covariances.CorrelationStandardDeviation(
        spatial_correlations, reduced_flux_stds
    )
)
write_progress_message("Have full spatial covariance")

reduced_spatial_covariance = spatial_correlation_remapper.reshape(
    REDUCED_N_GRID_POINTS, N_GRID_POINTS
).dot(
    spatial_covariance.dot(
        spatial_correlation_remapper.reshape(
            REDUCED_N_GRID_POINTS, N_GRID_POINTS
        ).T
    )
)
write_progress_message("Have reduced spatial covariance")

write_progress_message("Have spatial covariances")
print(reduced_spatial_covariance)
flush_output_streams()

######################################################################
# Get an observation operator for the month
#
# To treat the flux as a mean, we need to sum the influence function
# spatial_obs_op_remapper_ds = xarray.DataArray(
#     spatial_obs_op_remapper,
#     dict(
#     ("reduced_dim_y", "reduced_dim_x", "dim_y", "dim_x"),
#     "spatial_obs_op_remapper_ds",
# )

reduced_influences = (
    aligned_influences
    .groupby_bins(
        "dim_x",
        pd.interval_range(
            0.,
            (aligned_influences.indexes["dim_x"][-1] +
             UNCERTAINTY_FLUX_RESOLUTION),
            freq=UNCERTAINTY_FLUX_RESOLUTION,
            closed="left")
        # np.arange(
        #     -1,
        #     (aligned_influences.indexes["dim_x"][-1] +
        #      UNCERTAINTY_FLUX_RESOLUTION),
        #     UNCERTAINTY_FLUX_RESOLUTION),
    ).sum("dim_x")
    .groupby_bins(
        "dim_y",
        pd.interval_range(
            0,
            (aligned_influences.indexes["dim_y"][-1] +
             UNCERTAINTY_FLUX_RESOLUTION),
            freq=UNCERTAINTY_FLUX_RESOLUTION, closed="left")
        # np.arange(
        #     -1,
        #      (aligned_influences.indexes["dim_y"][-1] +
        #       UNCERTAINTY_FLUX_RESOLUTION),
        #      UNCERTAINTY_FLUX_RESOLUTION),
    ).sum("dim_y")
    .resample(flux_time=UNCERTAINTY_TEMPORAL_RESOLUTION).sum("flux_time")
).rename(dim_x_bins="reduced_dim_x", dim_y_bins="reduced_dim_y",
         flux_time="reduced_flux_time")
reduced_influences.load()
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have influence for monthly average plots")
flush_output_streams()

prior_covariance = kronecker_product(
    temporal_correlations,
    spatial_covariance)
# Yay separability!
reduced_prior_covariance = kron(
    reduced_temporal_correlation_ds.data,
    reduced_spatial_covariance)
print("Covariance:", type(prior_covariance))
write_progress_message("Have covariances")
print(reduced_prior_covariance)
flush_output_streams()

# I realize this isn't quite the intended use for OBS_CHUNK
prior_fluxes = aligned_prior_fluxes.transpose(
    "flux_time", "dim_y", "dim_x", "realization")
write_progress_message("Have prior noise")

# TODO: use actual heights
here_obs = WRF_OBS_SITE[TRACER_NAME].sel(
    observation_time=xarray.DataArray(
        pd_obs_index,
        name="observation",
        dims=dict(observation=site_obs_pd_index)),
    site=xarray.DataArray(
        site_index,
        name="observation",
        dims=dict(observation=site_obs_pd_index)),
).rename(dict(projection_x_coordinate="tower_x",
              projection_y_coordinate="tower_y"))
print(here_obs)

OBSERVATION_STD = 2.
"""Standard deviation of observation transport error

This assumes similar deviations can be expected at each site.

Representativeness error from Gerbig et al 2003 for 27 km is .2 ppm
Ken says transport error is usually given as O(2-3ppm)
"""
OBS_CORR_FUN = atmos_flux_inversion.correlations.ExponentialCorrelation(3)
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
observation_covariance *= OBSERVATION_STD ** 2
observation_covariance = asarray(observation_covariance)
obs_diag_index = np.arange(observation_covariance.shape[0])
# Add representativeness and instrument errors
observation_covariance[obs_diag_index, obs_diag_index] += 0.4 ** 2 + 0.1 ** 2

used_observation_vals = (
    here_obs.data[:, np.newaxis] +
    gaussian_noise(observation_covariance, N_REALIZATIONS).T.reshape(
        here_obs.shape + (N_REALIZATIONS,))).persist()
used_observations = xarray.DataArray(
    used_observation_vals,
    here_obs.coords,
    here_obs.dims + ("realization",),
    "pseudo_observations",
    dict(
        observation_standard_deviation=OBSERVATION_STD,
        observation_correlation_time=OBS_CORR_FUN._length)
).rename(dict(longitude_0="tower_lon", latitude_0="tower_lat"))
used_observations.coords["realization"] = range(N_REALIZATIONS)
used_observations.coords["realization"].attrs.update(dict(
    standard_name="realization"))
write_progress_message("Have observation noise")

print(datetime.datetime.now(UTC).strftime("%c"),
      "Got covariance parts, getting posterior")
flush_output_streams()
posterior, reduced_posterior_covariances = (
    atmos_flux_inversion.optimal_interpolation.save_sum(
        aligned_prior_fluxes.values.reshape(
            N_GRID_POINTS * len(aligned_prior_fluxes.indexes["flux_time"]),
            N_REALIZATIONS),
        prior_covariance,
        used_observations.values,
        observation_covariance,
        aligned_influences.stack(
            fluxes=("flux_time", "dim_y", "dim_x")
        ).values,
        # .reshape(aligned_influences.shape[0],
        #          np.prod(aligned_influences.shape[-3:]))),
        reduced_prior_covariance,
        reduced_influences.stack(
            fluxes=("reduced_flux_time", "reduced_dim_y",
                    "reduced_dim_x")).values
    )
)
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have posterior values, making dataset")
flush_output_streams()

posterior = posterior.reshape(aligned_prior_fluxes.shape)
posterior_ds = xarray.Dataset(
    dict(posterior=(aligned_prior_fluxes.dims, posterior,
                    posterior_var_atts),
         prior=(aligned_prior_fluxes.dims, aligned_prior_fluxes,
                aligned_prior_fluxes.attrs),
         truth=(aligned_true_fluxes.dims, aligned_true_fluxes,
                aligned_true_fluxes.attrs),
         ),
    aligned_prior_fluxes.coords,
    posterior_global_atts)
posterior_ds["pseudo_observations"] = used_observations
print(datetime.datetime.now(UTC).strftime("%c"),
      "Have posterior structure, evaluating and writing")
flush_output_streams()

encoding = {name: {"_FillValue": -99}
            for name in posterior_ds.data_vars}
encoding.update({name: {"_FillValue": None}
                 for name in posterior_ds.coords})

posterior_ds.to_netcdf(
    ("2010-07_monthly_inversion_{flux_interval:02d}h_027km_"
     "noise{ncorr_fun:s}{ncorr_len:d}km{ncorr_fun_time:s}{ncorr_len_time:d}d_"
     "icov{icorr_fun:s}{icorr_len:d}km{icorr_fun_time:s}{icorr_len_time:d}d_"
     "output_dense_spatial_corr.nc4")
    .format(flux_interval=FLUX_INTERVAL, ncorr_fun=CORR_FUN,
            ncorr_len=CORR_LEN, icorr_len=CORRELATION_LENGTH, icorr_fun="exp",
            icorr_len_time=DAILY_FLUX_TIMESCALE, icorr_fun_time=DAILY_FLUX_FUN,
            ncorr_fun_time=TIME_CORR_FUN, ncorr_len_time=TIME_CORR_LEN),
    encoding=encoding, engine=NC_ENGINE, mode="w")
write_progress_message("Wrote posterior")


write_progress_message(
    "Finding posterior covariance without corrections for "
    "aggregation error"
)

infl_fun_red = reduced_influences.stack(
    fluxes=("reduced_flux_time", "reduced_dim_y",
            "reduced_dim_x")
).values
B_HT_red = reduced_prior_covariance.dot(infl_fun_red.T)
red_post_cov_no_agg = reduced_prior_covariance - B_HT_red.dot(
    atmos_flux_inversion.linalg.solve(
        infl_fun_red.dot(B_HT_red) + observation_covariance,
        B_HT_red.T
    )
)
write_progress_message(
    "Found reduced posterior covariance without "
    "aggregation error approximations"
)

posterior_covariance_ds = xarray.Dataset(
    dict(
        reduced_posterior_covariance=(
            ("reduced_flux_time_adjoint",
             "reduced_dim_y_adjoint",
             "reduced_dim_x_adjoint",
             "reduced_flux_time",
             "reduced_dim_y",
             "reduced_dim_x",
             ),
            reduced_posterior_covariances.reshape(
                reduced_temporal_correlation_ds.shape[0],
                reduced_influences.shape[2],
                reduced_influences.shape[3],
                reduced_temporal_correlation_ds.shape[1],
                reduced_influences.shape[2],
                reduced_influences.shape[3]
            ),
            dict(
                standard_name=("surface_upward_mole_flux_of_carbon_dioxide "
                               "standard_error"),
                standard_error_multiplier=1.,
                long_name=(
                    "reduced_covariance_matrix_for_posterior_fluxes_full_HBHT"
                ),
                units=(FLUX_UNITS ** 2).format(),
                description=textwrap.dedent("""\
                Reduced-resolution approximation to the posterior
                covariance matrix, with no attempt to account for the
                increased aggregation error due to the increased
                resolution.
                """),
            ),
        ),
        reduced_posterior_covariance_no_aggregation=(
            ("reduced_flux_time_adjoint",
             "reduced_dim_y_adjoint",
             "reduced_dim_x_adjoint",
             "reduced_flux_time",
             "reduced_dim_y",
             "reduced_dim_x",
             ),
            red_post_cov_no_agg.reshape(
                reduced_temporal_correlation_ds.shape[0],
                reduced_influences.shape[2],
                reduced_influences.shape[3],
                reduced_temporal_correlation_ds.shape[1],
                reduced_influences.shape[2],
                reduced_influences.shape[3]
            ),
            dict(
                standard_name=("surface_upward_mole_flux_of_carbon_dioxide "
                               "standard_error"),
                standard_error_multiplier=1.,
                long_name=(
                    "reduced_covariance_matrix_for_posterior_fluxes_red_HBHT"
                ),
                units=(FLUX_UNITS ** 2).format(),
                description=textwrap.dedent("""\
                Reduced-resolution approximation to the posterior
                covariance matrix, with no attempt to account for the
                increased aggregation error due to the increased
                resolution.
                """),
            ),
        ),
        reduced_prior_covariance=(
            ("reduced_flux_time_adjoint",
             "reduced_dim_y_adjoint",
             "reduced_dim_x_adjoint",
             "reduced_flux_time",
             "reduced_dim_y",
             "reduced_dim_x",
             ),
            reduced_prior_covariance.reshape(
                reduced_temporal_correlation_ds.shape[0],
                reduced_influences.shape[2],
                reduced_influences.shape[3],
                reduced_temporal_correlation_ds.shape[1],
                reduced_influences.shape[2],
                reduced_influences.shape[3]
            ),
            dict(
                standard_name=("surface_upward_mole_flux_of_carbon_dioxide "
                               "standard_error"),
                standard_error_multiplier=1.,
                long_name="reduced_covariance_matrix_for_prior_fluxes",
                units=(FLUX_UNITS ** 2).format(),
            ),
        ),
    ),
    dict(
        reduced_flux_time=(
            reduced_temporal_correlation_ds.coords["flux_time"].values
        ),
        reduced_flux_time_adjoint=(
            reduced_temporal_correlation_ds.coords["flux_time_adjoint"].values
        ),
        wrf_proj=((), -1, WRF_PROJECTION.cf()),
    ),
    posterior_global_atts
)

RED_DIM_Y = reduced_influences.indexes["reduced_dim_y"]
posterior_covariance_ds.coords["reduced_dim_y"] = RED_DIM_Y.left
posterior_covariance_ds.coords["reduced_dim_y_bnds"] = (
    ("reduced_dim_y", "bnds2"),
    np.vstack([RED_DIM_Y.left, RED_DIM_Y.right]).T,
    dict(closed="left")
)
posterior_covariance_ds.coords["reduced_dim_y"].attrs.update(
    aligned_influences.coords["dim_y"].attrs)
posterior_covariance_ds.coords["reduced_dim_y"].attrs.update(
    dict(bounds="reduced_dim_y_bnds",
         adjoint="reduced_dim_y_adjoint"))

RED_DIM_X = reduced_influences.indexes["reduced_dim_x"]
posterior_covariance_ds.coords["reduced_dim_x"] = RED_DIM_X.left
posterior_covariance_ds.coords["reduced_dim_x_bnds"] = (
    ("reduced_dim_x", "bnds2"),
    np.vstack([RED_DIM_X.left, RED_DIM_X.right]).T,
    dict(closed="left"))
posterior_covariance_ds.coords["reduced_dim_x"].attrs.update(
    aligned_influences.coords["dim_x"].attrs)
posterior_covariance_ds.coords["reduced_dim_x"].attrs.update(
    dict(bounds="reduced_dim_x_bnds",
         adjoint="reduced_dim_x"))

posterior_covariance_ds.coords["reduced_dim_y_adjoint"] = (
    posterior_covariance_ds.coords["reduced_dim_y"].values)
posterior_covariance_ds.coords["reduced_dim_y_adjoint_bnds"] = (
    ("reduced_dim_y", "bnds2"),
    np.vstack([RED_DIM_Y.left, RED_DIM_Y.right]).T,
    dict(closed="left")
)
posterior_covariance_ds.coords["reduced_dim_y_adjoint"].attrs.update(
    aligned_influences.coords["dim_y"].attrs)
posterior_covariance_ds.coords["reduced_dim_y_adjoint"].attrs.update(
    dict(bounds="reduced_dim_y_adjoint_bnds",
         adjoint="reduced_dim_y"))

posterior_covariance_ds.coords["reduced_dim_x_adjoint"] = (
    posterior_covariance_ds.coords["reduced_dim_x"].values)
posterior_covariance_ds.coords["reduced_dim_x_adjoint_bnds"] = (
    ("reduced_dim_x", "bnds2"),
    np.vstack([RED_DIM_X.left, RED_DIM_X.right]).T,
    dict(closed="left"))
posterior_covariance_ds.attrs.update(
    dict(title="posterior_flux_uncertainty",
         summary="Covariance matrices for prior and posterior fluxes.",
         )
)

encoding = {name: {"_FillValue": -99}
            for name in posterior_covariance_ds.data_vars}
encoding.update({name: {"_FillValue": None}
                 for name in posterior_covariance_ds.coords})

posterior_covariance_ds.to_netcdf(
    ("2010-07_monthly_inversion_{flux_interval:02d}h_027km_"
     "noise{ncorr_fun:s}{ncorr_len:d}km{ncorr_fun_time:s}{ncorr_len_time:d}d_"
     "icov{icorr_fun:s}{icorr_len:d}km{icorr_fun_time:s}{icorr_len_time:d}d_"
     "{cov_res:d}km_{cov_tres:s}_covariance_output_dense_spatial_corr.nc4")
    .format(flux_interval=FLUX_INTERVAL, ncorr_fun=CORR_FUN,
            ncorr_len=CORR_LEN, icorr_len=CORRELATION_LENGTH, icorr_fun="exp",
            icorr_len_time=DAILY_FLUX_TIMESCALE, icorr_fun_time=DAILY_FLUX_FUN,
            ncorr_fun_time=TIME_CORR_FUN, ncorr_len_time=TIME_CORR_LEN,
            cov_res=int(UNCERTAINTY_FLUX_RESOLUTION // 1e3),
            cov_tres=UNCERTAINTY_TEMPORAL_RESOLUTION),
    encoding=encoding, engine=NC_ENGINE, mode="w")
write_progress_message("Wrote posterior covariance")
# UNCERTAINTY_FLUX_RESOLUTION UNCERTAINTY_TEMPORAL_RESOLUTION
