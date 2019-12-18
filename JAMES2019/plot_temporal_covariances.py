"""Plot the temporal correlations given configuration.

Produces plots of temporal correlations given configuration from
run_inversion_osse.py
"""
from __future__ import print_function, division, unicode_literals

import itertools
import datetime
import os.path
import glob
import sys

import pandas as pd
import dateutil.tz
import numpy as np
import cf_units
import netCDF4
import xarray

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
from atmos_flux_inversion.util import kron
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
    ("2010-07_osse_priors_{interval:1d}h_{res:02d}km_noise_"
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
#  1  3m51 (10 realizations)
#  2  6m49 (10 realizations)
#  2 20m10 (80 realizations)
# 21 3h10  (80 realizations)
# 30
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
# Define correlation constants and get covariances
print(datetime.datetime.now(UTC).strftime("%c"), "Getting covariances")
flush_output_streams()
CORRELATION_LENGTH = 200
GRID_RESOLUTION = 27
# spatial_correlations = (
#     atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
#     # First guess at correlation length on the order of previous studies
#     from_function(
#         atmos_flux_inversion.correlations.ExponentialCorrelation(
#             CORRELATION_LENGTH / GRID_RESOLUTION),
#         (len(TRUE_FLUXES_MATCHED.coords["dim_y"]),
#          len(TRUE_FLUXES_MATCHED.coords["dim_x"]))))
print(datetime.datetime.now(UTC).strftime("%c"), "Have spatial correlations")
flush_output_streams()
HOURLY_FLUX_TIMESCALE = 3
INTERVALS_PER_DAY = HOURS_PER_DAY // FLUX_INTERVAL
hour_correlations = (
    atmos_flux_inversion.correlations.HomogeneousIsotropicCorrelation.
    from_function(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            HOURLY_FLUX_TIMESCALE / FLUX_INTERVAL),
        (INTERVALS_PER_DAY,)))
hour_correlations_matrix = hour_correlations.dot(np.eye(
    hour_correlations.shape[0]))
print(datetime.datetime.now(UTC).strftime("%c"), "Have hourly correlations")
flush_output_streams()
DAILY_FLUX_TIMESCALE = 21
DAILY_FLUX_FUN = "exp"
day_correlations = (
    atmos_flux_inversion.correlations.make_matrix(
        atmos_flux_inversion.correlations.ExponentialCorrelation(
            DAILY_FLUX_TIMESCALE
        ),
        (40 * 4 *
         FLUX_INTERVAL // HOURS_PER_DAY,)))
print(datetime.datetime.now(UTC).strftime("%c"), "Have daily correlations")
flush_output_streams()
temporal_correlations = kron(day_correlations,
                             hour_correlations_matrix)
print("Temporal:", type(temporal_correlations))
print(datetime.datetime.now(UTC).strftime("%c"), "Have temporal correlations")
flush_output_streams()

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

time_index = pd.timedelta_range(start="0 day", end="40 day", freq="6H")
array = xarray.DataArray(data=temporal_correlations,
                         coords=dict(time1=time_index[:-1],
                                     time2=time_index[:-1]),
                         dims=("time1", "time2"),
                         attrs=dict(day_timescale=np.int8(21), day_fun="exp",
                                    hour_timescale=np.int8(3), hour_fun="exp",
                                    long_name="temporal_correlation_matrix"),
                         name="temporal_correlation_matrix")
array.coords["time_in_days1"] = (
    "time1", np.arange(0, 40, .25, dtype=np.float32), dict(units="days"))
array.coords["time_in_days2"] = (
    "time2", np.arange(0, 40, .25, dtype=np.float32), dict(units="days"))
array.attrs.update(cf_acdd.global_attributes_dict())

array.plot(x="time_in_days1", y="time_in_days2", yincrease=False,
           vmin=0, vmax=1, figsize=(5, 4))
plt.suptitle("Temporal correlation matrix")
plt.xlabel("Days since start")
plt.ylabel("Days since start")
plt.savefig("temporal_correlation_matrix_exp{day_corr:d}day_exp3hour.png"
            .format(day_corr=DAILY_FLUX_TIMESCALE))

fig = plt.figure(figsize=(5, 2.5))
plt.plot(array.coords["time_in_days2"], array.isel(time1=0))
plt.xlim(0, 30)
plt.ylim(0, 1)
plt.ylabel("Correlation")
plt.xlabel("Time difference in days")
plt.suptitle("Temporal correlation function")
plt.tight_layout()
plt.subplots_adjust(top=.9)
fig.savefig("temporal_correlation_function_exp{day_corr:d}days_exp3hours.pdf"
            .format(day_corr=DAILY_FLUX_TIMESCALE))

array.to_netcdf(
    "temporal_correlation_matrix_exp{day_corr:d}days_exp03hours.nc4"
    .format(day_corr=DAILY_FLUX_TIMESCALE),
    engine="h5netcdf",
    encoding=dict(time_in_days1={"_FillValue": None},
                  time_in_days2={"_FillValue": None})
)
