#!/usr/bin/env python
# ~*~ coding: utf8 ~*~
"""Find the domain average flux uncertainties."""
from __future__ import division, print_function
import sys
import glob
import os.path
import datetime

import numpy as np
import dateutil.tz
import netCDF4
import cf_units
import xarray

try:
    THIS_DIR = os.path.dirname(__file__)
except NameError:
    THIS_DIR = os.getcwd()

sys.path.insert(0, os.path.join(
    THIS_DIR, "..", "src"))
sys.path.append(THIS_DIR)

import atmos_flux_inversion.correlations
import atmos_flux_inversion.covariances
from atmos_flux_inversion.util import kronecker_product
from atmos_flux_inversion.linalg import kron


############################################################
# A few utility functions
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
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

DAILY_FLUX_TIMESCALE = 21
DAILY_FLUX_FUN = "exp"
HOURLY_FLUX_TIMESCALE = 3
CORRELATION_LENGTH = 200
GRID_RESOLUTION = 27
# I would like to add a fixed minimum at some point.
# full stds would then be sqrt(fixed^2 + varying^2)
# average seasonal variation (or some fraction thereof) might work.
# x2 since MsTMIP spread does not represent full uncertainty
# x5 since MsTMIP spread only represents monthly values and this uses sub-daily
# x10 matches model-model for Raczka for 200km/21d (0.68)
# x3.5 matches model-model for 1000km/7d (0.65)
FLUX_VARIANCE_VARYING_FRACTION = 10.
NC_ENGINE = "netcdf4"
FLUX_UNITS = "umol/m2/s"
FLUX_WINDOW = HOURS_PER_DAY * DAYS_PER_WEEK * 2

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


############################################################
# Get grid parameters

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
INTERVALS_PER_DAY = HOURS_PER_DAY // FLUX_INTERVAL
FLUX_RESOLUTION = 27
"""Resolution of fluxes and influence functions in kilometers."""
UNCERTAINTY_RESOLUTION_REDUCTION_FACTOR = 6
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
UNCERTAINTY_TEMPORAL_RESOLUTION = "2D"
"""The resolution at which the uncertainty is calculated and saved.

Higher resolution means the uncertainties will be more accurate.
"""

INFLUENCE_PATHS = ["/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "ACT-America_trial5/2010/01/GROUP1",
                   "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "candidacy_more_towers/2010/01/GROUP1"]
INFLUENCE_FILES = [
    name
    for path in INFLUENCE_PATHS
    for name in glob.glob(os.path.join(
        path,
        "LPDM_2010_01_{flux_interval:02d}hrly_{res:03d}km_molar_footprints.nc4"
        .format(flux_interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))]

TEST_DS = netCDF4.Dataset(INFLUENCE_FILES[0])

NX = len(TEST_DS.dimensions["dim_x"])
NY = len(TEST_DS.dimensions["dim_y"])
N_TIMES_BACK = len(TEST_DS.dimensions["time_before_observation"])

N_SITES = len(TEST_DS.dimensions["site"])
N_OBS_TIMES = len(TEST_DS.dimensions["observation_time"])

TEST_DS.close()
del TEST_DS

N_TIMES = INTERVALS_PER_DAY * 30

if N_TIMES_BACK < FLUX_WINDOW / FLUX_INTERVAL:
    raise ValueError("FLUX_WINDOW too long for file")

N_GRID_POINTS = NY * NX
STATE_SIZE = N_GRID_POINTS * FLUX_WINDOW

OBS_VEC_SIZE = N_SITES * OBS_WINDOW
OBS_VEC_TOTAL_SIZE = N_SITES * N_OBS_TIMES


write_progress_message("Getting covariances")
spatial_correlations = (
    atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
    from_function(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            CORRELATION_LENGTH / GRID_RESOLUTION),
        (NY, NX),
        is_cyclic=False))
write_progress_message("Have spatial correlations")

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
day_correlations = (
    atmos_flux_inversion.correlations.make_matrix(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            DAILY_FLUX_TIMESCALE
        ),
        (N_TIMES // INTERVALS_PER_DAY,)))
write_progress_message("Have daily correlations")
temporal_correlations = kron(day_correlations,
                             hour_correlations_matrix)
write_progress_message("Have temporal correlations")

full_correlations = kronecker_product(
    temporal_correlations,
    spatial_correlations)
write_progress_message("Have combined correlations")
flux_std_pattern = xarray.open_dataset(
    "../data_files/2010_MsTMIP_flux_std.nc4",
    engine=NC_ENGINE
).get(
    ["E_TRA{:d}".format(i + 1) for i in range(1)]
).sel(
    Time=slice("2010-07-01", "2010-07-30")
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

prior_covariance = kronecker_product(
    temporal_correlations,
    spatial_covariance)

averager = np.full(
    prior_covariance.shape[0], 1. / prior_covariance.shape[0], dtype=np.float32
)

write_progress_message("Getting mean covariance for all")
cov_of_avg = averager.dot(prior_covariance.dot(averager))
write_progress_message("Got mean covariance for all")

write_progress_message("Covariance is {0:5.3f}\nStandard Deviation: {1:5.3f}"
                       .format(cov_of_avg, np.sqrt(cov_of_avg)))
