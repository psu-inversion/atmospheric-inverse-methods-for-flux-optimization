#!/usr/bin/env python
"""Run an identical twin flux inversion OSSE with real data.

Use xarray/dask to grab influence functions and priors from netCDF
files.
"""

import itertools
import datetime
import os.path
import glob
import sys

import dask.array as da
import pandas as pd
import dateutil.tz
import numpy as np
import cartopy
import netCDF4
import xarray
import wrf

try:
    THIS_DIR = os.path.dirname(__file__)
except NameError:
    THIS_DIR = os.getcwd()

sys.path.append(os.path.join(
    THIS_DIR, "..", "src"))

import inversion.optimal_interpolation
import inversion.correlations
import inversion.noise

INFLUENCE_PATH = os.path.join(THIS_DIR, "..", "data_files")
PRIOR_PATH = INFLUENCE_PATH

FLUX_FILES = glob.glob(os.path.join(PRIOR_PATH, "wrfout_d01_*"))
INFLUENCE_FILES = glob.glob(os.path.join(INFLUENCE_PATH, "*footprints.nc4"))

FLUX_WINDOW = 24 * 7 * 1
OBS_HOURS = (datetime.time(14), datetime.time(18))
"""Which observation times will be used in the inversion.

Assumed to be UTC. Should give afternoon hours for the domain.  Not
currently set up for domains spanning many time zones. This shouldn't
matter for OSSEs, but will for real-data cases (PBL schemes don't
treat night well)

I really hope I can assume this doesn't depend on latitude. That would
make this much more complicated.
"""
OBS_WINDOW = 1

############################################################
# Utility functions.
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# def grouper(lst, blocksize, total_len=None):
#     """Return blocks of `blocksize` from `lst`.

#     Parameters
#     ----------
#     lst: sequence
#     blocksize: int
#     total_len: int, optional
#         Total length where this cannot be determined from
    

#     Yields
#     ------
#     sequence
#         slices from lst
#     """
#     for start in range(0, len(lst), n):
#         yield lst[start:start + n]


UTC = dateutil.tz.tzutc()
SECONDS_PER_HOUR = 3600

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

if N_TIMES_BACK < FLUX_WINDOW:
    raise ValueError("FLUX_WINDOW too long for file")

N_GRID_POINTS = NY * NX
STATE_SIZE = N_GRID_POINTS * FLUX_WINDOW

OBS_VEC_SIZE = N_SITES * OBS_WINDOW
OBS_VEC_TOTAL_SIZE = N_SITES * N_OBS_TIMES

############################################################
# Read influence functions

# obs time chunk size works best as one.  Need to iterate over single
# hyperslabs along this dimension to have single flux time to line up
# with the time coordinate in the fluxes
INFLUENCE_DATASET = xarray.open_mfdataset(
    INFLUENCE_FILES,
    chunks=dict(observation_time=1, site=N_SITES,
                time_before_observation=FLUX_WINDOW,
                dim_y=NY, dim_x=NX))
INFLUENCE_FUNCTIONS = INFLUENCE_DATASET.H.isel(
    time_before_observation=slice(0, FLUX_WINDOW))
# Use site names as index/dim coord for site dim
INFLUENCE_FUNCTIONS["site"] = np.char.decode(INFLUENCE_FUNCTIONS["site_names"].values, "ascii")
# It does not appear to be possible to localize the indices.
# INFLUENCE_FUNCTIONS.indexes["observation_time"] = (
#     INFLUENCE_FUNCTIONS.indexes["observation_time"]
#     .tz_localize(UTC))

# Only need this to get LPDM observations
# OBSERVATION_OPERATOR = INFLUENCE_FUNCTIONS.data.reshape(OBS_VEC_TOTAL_SIZE,
#                                                         STATE_SIZE)

############################################################
# Set some constants based on the WRF file
with netCDF4.Dataset(FLUX_FILES[0]) as ds:
    WRF_PROJECTION = wrf.util.getproj(**wrf.util.get_proj_params(ds))
    WRF_CRS = WRF_PROJECTION.cartopy()

    TOWER_WRF_INDICES = wrf.ll_to_xy(
        ds, INFLUENCE_FUNCTIONS.coords["site_lats"],
        INFLUENCE_FUNCTIONS.coords["site_lons"])

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

############################################################
# Read prior fluxes
FLUX_DATASET = xarray.open_mfdataset(
    FLUX_FILES,
    chunks=dict(west_east=NX, south_north=NY, Time=1)
)
FLUX_DATASET.set_index(Time="XTIME",
                       bottom_top="ZNU", bottom_top_stag="ZNW",
                       soil_layers_stag="ZS",
                       inplace=True)
FLUX_DATASET = FLUX_DATASET.assign_coords(
    geopot_hgt=lambda ds: (ds.PH + ds.PHB) / 9.8,
    HGT=lambda ds: ds.HGT)

WRF_DX = FLUX_DATASET.attrs["DX"]
for dim in [dimname + suffix for
            dimname in ("south_north", "west_east")
            for suffix in ("", "_stag")]:
    domain_bound = (FLUX_DATASET.dims[dim] - 1) / 2 * WRF_DX
    FLUX_DATASET = FLUX_DATASET.assign_coords(**{
        dim: np.arange(-domain_bound, domain_bound + WRF_DX / 2, WRF_DX)
    })

TRUE_FLUXES = FLUX_DATASET.get(["E_TRA{:d}".format(i+1)
                                for i in range(10)]).isel(emissions_zdim=0)
TRUE_FLUXES_MATCHED = TRUE_FLUXES.rename(dict(
    south_north="dim_y", west_east="dim_x", Time="flux_time"))

WRF_OBS = FLUX_DATASET.get(
    ["tracer_{:d}".format(i+1)
     for i in range(10)]).isel(
             bottom_top=slice(8, 15))
WRF_OBS_MATCHED = WRF_OBS.rename(dict(
    south_north="dim_y", west_east="dim_x", Time="observation_time"))
WRF_OBS_SITE = (
    WRF_OBS_MATCHED.sel(bottom_top=OBS_ROUGH_SIGMA, method="nearest")
    .sel_points(dim_x=WRF_TOWER_COORDS[:, 0], dim_y=WRF_TOWER_COORDS[:, 1],
                method="nearest", dim=INFLUENCE_FUNCTIONS.coords["site"]))

WRF_OBS_START = WRF_OBS_MATCHED.indexes["observation_time"][0]
WRF_OBS_INTERVAL = WRF_OBS_START - WRF_OBS_MATCHED.indexes["observation_time"][1]

############################################################
# Get time zone representing local solar time for each site
LOCAL_TIME_ZONES = list(map(
    dateutil.tz.tzoffset,
    ("Local time for {name!s}".format(name=name)
     for name in INFLUENCE_FUNCTIONS.coords["site"].values),
    [round(offset_hr * SECONDS_PER_HOUR)
     for offset_hr in INFLUENCE_FUNCTIONS.coords["site_lons"].values / 15]))

############################################################
# Do the inversion
obs_times = reversed(INFLUENCE_FUNCTIONS.indexes["observation_time"])
# Take care of missing obs
# Also subsetting for late afternoon steady convective boundary layer
site_obs_index = []
for i, site in enumerate(INFLUENCE_FUNCTIONS.indexes["site"]):
    local_times = pd.Index([
        obs_time.tz_localize(UTC)
        .tz_convert(LOCAL_TIME_ZONES[i])
        for obs_time in obs_times], name="local_times")

    # np.where is designed for multi-dimensional arrays
    site_obs_index.extend(zip(
        np.where((OBS_HOURS[0] < local_times.time) &
                 (local_times.time < OBS_HOURS[1]))[0],
        itertools.repeat(i)))
obs_index, site_index = zip(*site_obs_index)

aligned_influences = xarray.concat(
    [here_infl.set_index(
        time_before_observation="flux_time").rename(
            dict(time_before_observation="flux_time"))
     for here_infl in INFLUENCE_FUNCTIONS.isel_points(
             site=site_index, observation_time=obs_index)],
    "observation_time").fillna(0)
# TODO: use actual heights
here_obs = WRF_OBS_MATCHED.isel_points(
    observation_time=obs_index, site=site_index
    ).sel(bottom_top=[OBS_ROUGH_SIGMA],
          method="nearest")

posterior, posterior_err = inversion.optimal_interpolation.fold_common(
    TRUE_FLUXES_MATCHED, TRUE_FLUXES_MATCHED,
    here_obs, .1, aligned_influences)
posterior_array = xarray.Dataset(
    dict(posterior=(

posterior.to_netcdf("monthly_inversion_output.nc4")
