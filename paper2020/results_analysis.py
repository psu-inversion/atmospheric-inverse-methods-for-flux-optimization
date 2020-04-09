#!/usr/bin/env python
# ~*~ coding: utf-8 ~*~
"""Plot results from inversion.

Only a single run.  Currently set up for plots from
run_inversion_osse
"""
from __future__ import print_function, division
import datetime
import glob
import sys
import os

import numpy as np
import scipy.stats
import scipy.special
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeat
import iris
import dask.array as da
import xarray.plot
print(datetime.datetime.now(), "Imports")
sys.stdout.flush()

YEAR = 2010
MONTH = 7

NOISE_FUNCTION = "exp"
NOISE_LENGTH = 200
NOISE_TIME_FUN = "exp"
NOISE_TIME_LEN = 21
INV_FUNCTION = "exp"
INV_LENGTH = 1000
INV_TIME_FUN = "exp"
INV_TIME_LEN = 7

FLUX_INTERVAL = 6
FLUX_RESOLUTION = 27


def write_console_message(msg):
    """Write a message to stdout and flush output streams.

    Parameters
    ----------
    msg: str
    """
    sys.stderr.flush()
    print(datetime.datetime.now(), msg, flush=True)


def long_description(df, ci_width=0.95):
    """Print longer description of df.

    Parameters
    ----------
    df: pd.DataFrame
    ci_width: float
         Width of confidence intervals.
         Must between 0 and 1.

    Returns
    -------
    pd.DataFrame
    """
    df_stats = df.describe()
    df_stats_loc = df_stats.loc
    # Robust measures of scale
    df_stats_loc["IQR", :] = df_stats_loc["75%", :] - df_stats_loc["25%", :]
    df_stats_loc["mean abs. dev.", :] = df.mad()
    deviation_from_median = df - df_stats_loc["50%", :]
    df_stats_loc["med. abs. dev.", :] = deviation_from_median.abs().median()
    # Higher-order moments
    df_stats_loc["Fisher skewness", :] = df.skew()
    df_stats_loc["Y-K skewness", :] = (
        (df_stats_loc["75%", :] + df_stats_loc["25%", :] -
         2 * df_stats_loc["50%", :]) /
        (df_stats_loc["75%", :] - df_stats_loc["25%", :])
    )
    df_stats_loc["Fisher kurtosis", :] = df.kurt()
    # Confidence intervals
    for col_name in df:
        # I'm already dropping NAs for the rest of these.
        mean, var, std = scipy.stats.bayes_mvs(
            df[col_name].dropna(),
            alpha=ci_width
        )
        # Record mean
        df_stats_loc["Mean point est", col_name] = mean[0]
        df_stats_loc[
            "Mean {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = mean[1][0]
        df_stats_loc[
            "Mean {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = mean[1][1]
        # Record var
        df_stats_loc["Var. point est", col_name] = var[0]
        df_stats_loc[
            "Var. {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = var[1][0]
        df_stats_loc[
            "Var. {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = var[1][1]
        # Record Std Dev
        df_stats_loc["std point est", col_name] = std[0]
        df_stats_loc[
            "std {width:2d}%CI low".format(width=round(ci_width * 100)),
            col_name
        ] = std[1][0]
        df_stats_loc[
            "std {width:2d}%CI high".format(width=round(ci_width * 100)),
            col_name
        ] = std[1][1]
    return df_stats


PRIOR_PATH = (
    "../data_files/"
    "{year:04d}-{month:02d}_osse_bio_priors_{interval:d}h_{res:d}km_"
    "noise_{fun:s}{len:d}km_{time_fun:s}{time_len:d}d_exp3h.nc"
).format(
    year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
    fun=NOISE_FUNCTION, len=NOISE_LENGTH, time_fun=NOISE_TIME_FUN,
    time_len=NOISE_TIME_LEN)
# 2010-07_monthly_inversion_06h_027km_noiseexp100kmexp14d_icovexp100kmexp14d_output.nc4
FRAT_POSTERIOR_PATH = (
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d"
    "_icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d"
    "_output.nc4"
).format(year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
         noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
         noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
         invfun=INV_FUNCTION, invlen=INV_LENGTH,
         inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN)
IDEN_POSTERIOR_PATH = (
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d"
    "_icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d"
    "_output.nc4"
).format(year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
         noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
         noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
         invfun=NOISE_FUNCTION, invlen=NOISE_LENGTH,
         inv_time_fun=NOISE_TIME_FUN, inv_time_len=NOISE_TIME_LEN)

WRF_CRS = ccrs.LambertConformal(
    standard_parallels=(30, 60), central_latitude=40,
    central_longitude=-96, false_easting=0, false_northing=0,
    globe=ccrs.Globe(semimajor_axis=6370e3, semiminor_axis=6360e3,
                     ellipse=None))
LPDM_PROJ = ccrs.LambertConformal(
    central_longitude=-96, central_latitude=40, standard_parallels=[30, 60],
    false_easting=3347998.5116325677, false_northing=2470499.376688077,
    globe=ccrs.Globe(semimajor_axis=6370e3, semiminor_axis=6370e3,
                     ellipse=None))

BIG_LAKES = cfeat.NaturalEarthFeature(
    "physical", "lakes", "110m",
    edgecolor="gray", facecolor="none", linewidth=.5)
STATES = cfeat.NaturalEarthFeature(
    "cultural", "admin_1_states_provinces_lines", "110m",
    edgecolor="gray", facecolor="none", linewidth=.5)

TRACER_NAMES = [
    "diurnal_bio",
    "fossil",
    "ocean",
    "biomass_burn",
    "biofuel",
    "ship",
    "posterior_bio",
    "boundaries",
    "prior_bio",
    "",
]

# WEST_BOUNDARY_LPDM = 2.7e6
# WEST_BOUNDARY_WRF = WRF_CRS.transform_point(
#     WEST_BOUNDARY_LPDM, 0, LPDM_PROJ)[0]

# Estimates for West Virginia, roughly
WEST_BOUNDARY_WRF = 1.13e6
EAST_BOUNDARY_WRF = 1.52e6
SOUTH_BOUNDARY_WRF = -1.76e5
NORTH_BOUNDARY_WRF = 2.02e5

LPDM_BOUNDS = LPDM_PROJ.transform_points(
    WRF_CRS,
    np.array([WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF]),
    np.array([SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF]),
)

print(datetime.datetime.now(), "Constants")
sys.stdout.flush()


def plot_realizations(data_array):
    """Plot a `Dataarray` with realizations."""
    time_dim = [dim for dim in data_array.dims if "time" in dim][0]
    x_dim = [dim for dim in data_array.dims if "_x" in dim][0]
    y_dim = [dim for dim in data_array.dims if "_y" in dim][0]
    xarray.plot.pcolormesh(
        data_array.isel(**{time_dim: slice(3, None, 96),
                           "realization": slice(3)}),
        x_dim, y_dim,
        col=time_dim, row="realization", cmap="RdBu_r", center=0,
        aspect=1.35, size=1.8,
        subplot_kws=dict(projection=WRF_CRS),
    )

    post_fig = plt.gcf()
    axes = post_fig.axes
    try:
        axes[-1].set_ylabel(
            "{long_name:s}  (${units:s}$)".format(**data_array.attrs))
    except KeyError:
        pass
    xlim = data_array[x_dim][[0, -1]]
    ylim = data_array[y_dim][[0, -1]]

    for ax in axes[:-1]:
        ax.coastlines()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_feature(cfeat.BORDERS)

    return post_fig


def plot_fluxes(data_array):
    """Plot a single flux realization."""
    time_dim = [dim for dim in data_array.dims if "time" in dim][0]
    x_dim = [dim for dim in data_array.dims if "_x" in dim][0]
    y_dim = [dim for dim in data_array.dims if "_y" in dim][0]
    xarray.plot.pcolormesh(
        data_array.isel(**{time_dim: slice(3, None, 32)}),
        x_dim, y_dim,
        col=time_dim, col_wrap=3, cmap="RdBu_r", center=0,
        aspect=1.35, size=1.8,
        subplot_kws=dict(projection=WRF_CRS),
    )

    post_fig = plt.gcf()
    axes = post_fig.axes
    try:
        axes[-1].set_ylabel(
            "{long_name:s} (${units:s}$)".format(**data_array.attrs))
    except KeyError:
        pass
    xlim = data_array[x_dim][[0, -1]]
    ylim = data_array[y_dim][[0, -1]]

    for ax in axes[:-1]:
        ax.coastlines()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_feature(cfeat.BORDERS)

    post_fig.subplots_adjest(top=.9, bottom=.0256, left=.0216,
                             right=.82, hspace=.2, wspace=.08)
    return post_fig


print(datetime.datetime.now(), "Functions")
sys.stdout.flush()

PRIOR_CUBES = iris.load(PRIOR_PATH)
PRIOR_DS = xarray.open_dataset(
    PRIOR_PATH, chunks=dict(realization=1,
                            flux_time=8 * 7)
).set_coords(
    "lambert_conformal_conic"
)  # .rename(
#     dict(south_north="dim_y", west_east="dim_x"))

PRIOR_CUBES.sort(key=lambda cube: cube.name())
PRIOR_CUBES.sort(key=lambda cube: len(cube.name()))

for name, var in PRIOR_DS.data_vars.items():
    num_end = name.rindex("_")
    if num_end < 5:
        num_end = None
    tracer_num = int(name[5:num_end])
    long_name = TRACER_NAMES[tracer_num - 1]
    if name.endswith("_noisy"):
        long_name += "_with_added_noise"
    var.attrs["long_name"] = long_name
    try:
        var.attrs["units"] = "\N{MICRO SIGN}" + var.attrs["units"]
        var *= 1e6
    except KeyError:
        print(name)
        pass


NOISE_STD_DS = xarray.open_dataset(
    "../data_files/{year:04d}_MsTMIP_flux_std.nc4".format(
        year=YEAR, month=MONTH),
    chunks=dict(Time=21 * 8)
)[["E_TRA1"]].sel(
    Time=slice("2010-06-16", "2010-07-31")
).mean("Time", keep_attrs=True)
NOISE_STD_DS["E_TRA1"].attrs["units"] = "umol/m^2/s"

for name, var in NOISE_STD_DS.data_vars.items():
    if name.startswith("E_TRA"):
        tracer_num = int(name[5:])
    elif name.startswith("tracer_"):
        tracer_num = int(name[7:])
    else:
        continue
    long_name = TRACER_NAMES[tracer_num - 1]
    long_name += "_rms"
    var.attrs["long_name"] = long_name
    if var.attrs["units"] == "mol km^-2 hr^-1":
        var /= 3.6e3
        var.attrs["units"] = "\N{MICRO SIGN}mol/m^2/s"

# NOISE_STD_DS.coords["south_north"] = PRIOR_DS.coords["dim_y"].data
# NOISE_STD_DS.coords["west_east"] = PRIOR_DS.coords["dim_x"].data

NOISE_STD_DS["E_TRA7"] = NOISE_STD_DS["E_TRA1"]

############################################################
# Load fraternal-twin dataset
FRAT_INVERSION_DS = xarray.open_dataset(
    FRAT_POSTERIOR_PATH,
    chunks=dict(realization=1, flux_time=8 * 14)
)
FRAT_INVERSION_DS = FRAT_INVERSION_DS.set_index(
    observation=["observation_time", "site"])

FRAT_PSEUDO_OBS_DS = FRAT_INVERSION_DS["pseudo_observations"]
FRAT_POSTERIOR_DS = FRAT_INVERSION_DS[["posterior", "prior", "truth"]].rename(
    dict(dim_x="projection_x_coordinate", dim_y="projection_y_coordinate",
         flux_time="time"))
FRAT_POSTERIOR_DS.coords["projection_x_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_x_coordinate", axis="X"))
FRAT_POSTERIOR_DS.coords["projection_y_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_y_coordinate", axis="Y"))
FRAT_POSTERIOR_DS.coords["time"] = FRAT_POSTERIOR_DS.indexes["time"].round("S")

for var in FRAT_POSTERIOR_DS.data_vars.values():
    var *= 1e6
    var.attrs["units"] = "\N{MICRO SIGN}" + var.attrs["units"]
del var

###############################################################################
# Load identical-twin dataset
IDEN_INVERSION_DS = xarray.open_dataset(
    IDEN_POSTERIOR_PATH,
    chunks=dict(realization=1, flux_time=8 * 14)
)
IDEN_INVERSION_DS = IDEN_INVERSION_DS.set_index(
    observation=["observation_time", "site"]
)

IDEN_PSEUDO_OBS_DS = IDEN_INVERSION_DS["pseudo_observations"]
IDEN_POSTERIOR_DS = IDEN_INVERSION_DS[["posterior", "prior", "truth"]].rename(
    dict(dim_x="projection_x_coordinate", dim_y="projection_y_coordinate",
         flux_time="time"))
IDEN_POSTERIOR_DS.coords["projection_x_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_x_coordinate", axis="X"))
IDEN_POSTERIOR_DS.coords["projection_y_coordinate"].attrs.update(
    dict(units="m", standard_name="projection_y_coordinate", axis="Y"))
IDEN_POSTERIOR_DS.coords["time"] = FRAT_POSTERIOR_DS.indexes["time"].round("S")

for var in IDEN_POSTERIOR_DS.data_vars.values():
    var *= 1e6
    var.attrs["units"] = "\N{MICRO SIGN}" + var.attrs["units"]
del var


NOISE_STD_DS.coords["south_north"] = (
    FRAT_POSTERIOR_DS.coords["projection_y_coordinate"].data
)
NOISE_STD_DS.coords["west_east"] = (
    FRAT_POSTERIOR_DS.coords["projection_x_coordinate"].data
)

FRAT_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=INV_FUNCTION, invlen=INV_LENGTH,
        inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)

IDEN_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=NOISE_FUNCTION, invlen=NOISE_LENGTH,
        inv_time_fun=NOISE_TIME_FUN, inv_time_len=NOISE_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)

############################################################
# Read in lower-resolution posterior covariance datasets

LOWER_RES_FRAT_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "216km_7D_covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=INV_FUNCTION, invlen=INV_LENGTH,
        inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)

LOWER_RES_IDEN_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "216km_7D_covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=NOISE_FUNCTION, invlen=NOISE_LENGTH,
        inv_time_fun=NOISE_TIME_FUN, inv_time_len=NOISE_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)

LOWEST_RES_FRAT_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "432km_7D_covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=INV_FUNCTION, invlen=INV_LENGTH,
        inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)

LOWEST_RES_IDEN_COVARIANCE_DS = xarray.open_dataset(
    "{year:04d}-{month:02d}_monthly_inversion_{interval:02d}h_{res:03d}km_"
    "noise{noisefun:s}{noiselen:d}km{noise_time_fun:s}{noise_time_len:d}d_"
    "icov{invfun:s}{invlen:d}km{inv_time_fun:s}{inv_time_len:d}d_"
    "432km_7D_covariance_output.nc4".format(
        year=YEAR, month=MONTH, interval=FLUX_INTERVAL, res=FLUX_RESOLUTION,
        noisefun=NOISE_FUNCTION, noiselen=NOISE_LENGTH,
        noise_time_fun=NOISE_TIME_FUN, noise_time_len=NOISE_TIME_LEN,
        invfun=NOISE_FUNCTION, invlen=NOISE_LENGTH,
        inv_time_fun=NOISE_TIME_FUN, inv_time_len=NOISE_TIME_LEN
    ),
    chunks=dict(reduced_flux_time_adjoint=3, reduced_dim_y_adjoint=3,
                reduced_dim_x_adjoint=3),
)


############################################################
# Read in the influence functions
INFLUENCE_PATHS = ["/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "ACT-America_trial5/2010/01/GROUP1",
                   "/mc1s2/s4/dfw5129/data/LPDM_2010_fpbounds/"
                   "candidacy_more_towers/2010/01/GROUP1"]


COLLAPSED_INFLUENCE_DS = xarray.open_dataset(
    "../data_files/LPDM_2010_01_31day_027km_molar_footprints.nc4",
).set_coords(
    ["observation_time", "time_before_observation",
     "lpdm_configuration", "wrf_configuration"])
FULL_INFLUENCE_DS = xarray.open_mfdataset(
    [name
     for path in INFLUENCE_PATHS
     for name in glob.iglob(os.path.join(
         path,
         ("LPDM_2010_01*{flux_interval:02d}hrly_{res:03d}km_"
          "molar_footprints.nc4").format(
             flux_interval=FLUX_INTERVAL, res=FLUX_RESOLUTION)))],
    concat_dim="site",
    chunks=dict(time_before_observation=4 * 7, observation_time=24),
).set_coords(["lpdm_configuration", "wrf_configuration"])

print(datetime.datetime.now(), "Files", flush=True)
sys.stderr.flush()

write_console_message("Getting influence")
INFLUENCE_TEMPORAL_ONLY = FULL_INFLUENCE_DS.H.sum(
    ["dim_x", "dim_y"]
).mean("site")
ALIGNED_TEMPORAL_INFLUENCES = xarray.concat(
    [here_infl.set_index(
        time_before_observation="flux_time"
    ).rename(
        dict(time_before_observation="flux_time")
    )
     for here_infl in INFLUENCE_TEMPORAL_ONLY],
    "observation_time"
)
OBSERVATIONAL_CONSTRAINT = ALIGNED_TEMPORAL_INFLUENCES.sum("observation_time")
OBSERVATIONAL_CONSTRAINT.coords["flux_time"] = (
    OBSERVATIONAL_CONSTRAINT.coords["flux_time"] +
    (np.array("2010-07-01T00:00:00", dtype="M8[ns]") -
     np.array("2010-01-01T00:00:00", dtype="M8[ns]"))
)
OBSERVATIONAL_CONSTRAINT.load()
write_console_message("Got influence of fluxes on observations")

############################################################
# Plot standard deviations
fig, axes = plt.subplots(
    1, 1, figsize=(5.5, 3.3), subplot_kw=dict(projection=WRF_CRS))
(2. * 5. * NOISE_STD_DS.E_TRA7).plot.pcolormesh(robust=True)
axes = fig.axes
axes[0].coastlines()
axes[1].set_ylabel("standard deviation of noise (µmol/m$^2$/s)")
fig.suptitle("Standard deviation of added noise")
axes[0].set_xlim(FRAT_POSTERIOR_DS.coords["projection_x_coordinate"][[0, -1]])
axes[0].set_ylim(FRAT_POSTERIOR_DS.coords["projection_y_coordinate"][[0, -1]])
axes[0].set_title("")
axes[0].add_feature(cfeat.BORDERS)
axes[0].add_feature(STATES)
axes[0].add_feature(BIG_LAKES)

fig.savefig("{year:04d}-{month:02d}_noise_standard_deviation.png".format(
    year=YEAR, month=MONTH))
plt.close(fig)
write_console_message("Made std plot")

############################################################
# Plot pseudo-observations
fig, axes = plt.subplots(len(FRAT_PSEUDO_OBS_DS.site),
                         sharex=True,
                         figsize=(8, 1.5 * len(FRAT_PSEUDO_OBS_DS.site)))
write_console_message("Made figure")
fig.autofmt_xdate()
pseudo_obs = FRAT_PSEUDO_OBS_DS
for i, site in enumerate(set(FRAT_PSEUDO_OBS_DS.site.values)):
    site_obs = pseudo_obs.sel(site=site).transpose(
        "observation_time", "realization"
    )
    axes[i].plot(site_obs.observation_time.values, site_obs.values)
    axes[i].text(0.01, 0.98, site, transform=axes[i].transAxes,
                 horizontalalignment="left", verticalalignment="top")

fig.suptitle("Pseudo-observations used in inversion")
# fig.savefig("{year:04d}-{month:02d}_pseudo_obs_afternoon.png".format(
#     year=YEAR, month=MONTH))
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_pseudo_obs_afternoon.pdf"
    .format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN))
plt.close(fig)
write_console_message("Made pseudo-obs plot")

############################################################
# Plot "truth", prior, and posterior side-by-side
for_plotting = xarray.concat((FRAT_POSTERIOR_DS.prior.isel(realization=0),
                              IDEN_POSTERIOR_DS.posterior.isel(realization=0),
                              FRAT_POSTERIOR_DS.posterior.isel(realization=0)),
                             dim="type")
del for_plotting.coords["realization"]

# e_tra7_for_plot = PRIOR_DS["E_TRA7"].rename(
#     dict(dim_x="projection_x_coordinate", dim_y="projection_y_coordinate",
#          flux_time="time"))
for_plotting = xarray.concat((FRAT_POSTERIOR_DS.truth, for_plotting),
                             dim="type")
for_plotting.coords["type"] = ['"Truth"', "Prior", "Posterior\nIdentical-Twin",
                               "Posterior\nFraternal-Twin"]
for_plotting.persist()

xlim = for_plotting.coords["projection_x_coordinate"][[0, -1]]
ylim = for_plotting.coords["projection_y_coordinate"][[0, -1]]

plots = for_plotting.isel(time=slice(55, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    col="type", row="time", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    center=0, vmin=-40, vmax=40,
    cmap="RdBu_r", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.cbar.ax.set_ylabel("CO$_2$ Flux (\N{MICRO SIGN}mol/m$^2$/s)")
plots.axes[0, 0].set_title('"Truth"')
plots.axes[0, 1].set_title("Prior")
plots.axes[0, 2].set_title("Posterior\nIdentical-Twin")
plots.axes[0, 3].set_title("Posterior\nFraternal-Twin")

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_osse_realization.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN),
    dpi=400)
plt.close(plots.fig)
write_console_message("Done realization plot")

############################################################
# Plot tower locations
fig, ax = plt.subplots(
    1, 1, subplot_kw=dict(projection=WRF_CRS), figsize=(4, 3))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.coastlines()
ax.add_feature(BIG_LAKES)
ax.add_feature(cfeat.BORDERS)
ax.add_feature(STATES)
ax.scatter(pseudo_obs.tower_lon, pseudo_obs.tower_lat,
           transform=WRF_CRS.as_geodetic())
fig.suptitle("WRF domain and tower locations")
fig.savefig("tower_locations.pdf")
plt.close(fig)
write_console_message("Done tower loc plot")

############################################################
# Plot differences
differences = (for_plotting.isel(type=slice(1, None)) -
               for_plotting.isel(type=0))
differences.persist()
plots = differences.isel(time=slice(68 - 1, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    col="type", row="time", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    center=0, vmin=-10, vmax=10,
    cmap="RdBu_r", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.cbar.ax.set_ylabel("CO$_2$ Flux (\N{MICRO SIGN}mol/m$^2$/s)")
plots.axes[0, 0].set_title("Prior $-$ \"Truth\"")
plots.axes[0, 1].set_title("Posterior $-$ \"Truth\"\nIdentical-Twin")
plots.axes[0, 2].set_title("Posterior $-$ \"Truth\"\nFraternal-Twin")

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_osse_errors.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(plots.fig)
write_console_message("Done error plot")

############################################################
# Plot "truth", prior, and increments
for_incr_plotting = xarray.concat(
    [for_plotting.isel(type=slice(None, 2)),
     for_plotting.isel(type=slice(2, None)) -
     for_plotting.sel(type="Prior")],
    dim="type"
)
for_incr_plotting.coords["type"] = ['"Truth"', "Prior",
                                    "Posterior - Prior: Identical-Twin",
                                    "Posterior - Prior: Fraternal-Twin"]
for_incr_plotting.persist()

xlim = for_incr_plotting.coords["projection_x_coordinate"][[0, -1]]
ylim = for_incr_plotting.coords["projection_y_coordinate"][[0, -1]]

plots = for_incr_plotting.isel(time=slice(55, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    col="type", row="time", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    center=0, vmin=-40, vmax=40,
    cmap="RdBu_r", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.cbar.ax.set_ylabel("CO$_2$ Flux (\N{MICRO SIGN}mol/m$^2$/s)")
plots.axes[0, 0].set_title('"Truth"')
plots.axes[0, 1].set_title("Prior")
plots.axes[0, 2].set_title("Posterior - Prior\nIdentical-Twin")
plots.axes[0, 3].set_title("Posterior - Prior\nFraternal-Twin")

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_osse_realization_increment.png"
    .format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN),
    dpi=400)
plt.close(plots.fig)
write_console_message("Done realization increment plot")

############################################################
# Plot gain over time
gain = 1 - (
    da.fabs(differences.sel(
        type=[name for name in differences.coords["type"].values
              if "Posterior" in name])) /
    da.fabs(differences.sel(type="Prior"))
)
gain.attrs["long_name"] = "inversion_gain"
gain.attrs["units"] = "1"
# gain.load()

plots = gain.isel(time=slice(68 - 1, None, 40)).plot.pcolormesh(
    "projection_x_coordinate", "projection_y_coordinate",
    row="time", col="type", subplot_kws=dict(projection=WRF_CRS),
    aspect=1.3, size=1.8,
    vmin=0, vmax=1, cmap="viridis", levels=None)

for ax in plots.axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.coastlines()

plots.fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_pointwise_gain.png"
    .format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(plots.fig)
write_console_message("Done pointwise gain")

############################################################
# Find and plot gains for all realizations
all_differences = xarray.concat(
    (FRAT_POSTERIOR_DS.prior - for_plotting.sel(type='"Truth"'),
     IDEN_POSTERIOR_DS.posterior - for_plotting.sel(type='"Truth"'),
     FRAT_POSTERIOR_DS.posterior - for_plotting.sel(type='"Truth"')),
    dim="type")
all_differences.coords["type"] = ["prior_error", "iden_posterior_error",
                                  "frat_posterior_error"]

print(datetime.datetime.now(), "Getting January means east of line")
sys.stdout.flush()
time_mean_error = all_differences.mean("time")
time_mean_error.load()
all_mean_error = time_mean_error.mean(
    ("projection_x_coordinate", "projection_y_coordinate"))
all_mean_error.load()
small_mean_error = time_mean_error.sel(
    projection_x_coordinate=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    projection_y_coordinate=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF)
).mean(
    ("projection_x_coordinate", "projection_y_coordinate"))
small_mean_error.load()
print(datetime.datetime.now(), "Have means")
sys.stdout.flush()

total_gain = 1 - (
    da.fabs(all_mean_error.sel(type=[
        "iden_posterior_error", "frat_posterior_error"
    ])) /
    da.fabs(all_mean_error.sel(type="prior_error"))
)
total_gain.load()

fig = plt.figure()
total_gain.plot.hist(range=(-2, 1), bins=15)
fig.suptitle("Total gain")
plt.xlabel("Gain in total flux")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_total_gain_hist.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done large gain plot")

############################################################
# Describe the distribution
with open(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_gain_dist.txt".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN), "w") as result_file:
    print(datetime.datetime.now(), "Total gain", file=result_file)
    print(scipy.stats.describe(total_gain), file=result_file)
    print(total_gain.quantile((0, .2, .25, .4, .5, .6, .75, .8, 1)),
          file=result_file)

write_console_message("Done gain statistics files")

############################################################
# Plot timeseries of mean flux
spatial_avg_differences = all_differences.mean(
    ["projection_x_coordinate", "projection_y_coordinate"])

fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
spatial_avg_differences.sel(
    type="prior_error", realization=0).plot.line(
    "-", label="Prior")
spatial_avg_differences.sel(
    type="iden_posterior_error", realization=0).plot.line(
    "--", label="Posterior: Iden.")
spatial_avg_differences.sel(
    type="frat_posterior_error", realization=0).plot.line(
    "-.", label="Posterior: Frat.")

ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)
ax.axhline(0, color="black", linewidth=.75)

plt.legend()
ax.set_ylabel("Average flux error over whole domain\n"
              "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
plt.title("Spatial average flux error")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_spatial_"
    "avg_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done timeseries")


small_spatial_avg_differences = all_differences.sel(
    projection_x_coordinate=slice(WEST_BOUNDARY_WRF, EAST_BOUNDARY_WRF),
    projection_y_coordinate=slice(SOUTH_BOUNDARY_WRF, NORTH_BOUNDARY_WRF)
).mean(["projection_x_coordinate", "projection_y_coordinate"])

fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
small_spatial_avg_differences.sel(
    type="prior_error", realization=0).plot.line(
    "-", label="Prior")
small_spatial_avg_differences.sel(
    type="iden_posterior_error", realization=0).plot.line(
    "--", label="Posterior")
small_spatial_avg_differences.sel(
    type="frat_posterior_error", realization=0).plot.line(
    "--", label="Posterior")

ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)

plt.legend()
ax.set_ylabel("Average flux error over West Virginia\n"
              "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
plt.title("Spatial average flux error")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_small_spatial_"
    "avg_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done small timeseries")


############################################################
# Plot timeseries of increment
spatial_avg_increment = (
    spatial_avg_differences.sel(type=["iden_posterior_error",
                                      "frat_posterior_error"]) -
    spatial_avg_differences.sel(type="prior_error")
)
fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
spatial_avg_increment.isel(realization=0).plot.line('-', hue="type")

# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Average increment over whole domain\n"
              "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
plt.title("Spatial average increment")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_spatial_"
    "avg_increment_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)


small_spatial_avg_increment = (
    small_spatial_avg_differences.sel(type=["iden_posterior_error",
                                            "frat_posterior_error"]) -
    small_spatial_avg_differences.sel(type="prior_error")
)
fig = plt.figure(figsize=(5, 3.4))
fig.autofmt_xdate()
plt.subplots_adjust(left=.18)
ax = plt.gca()
try:
    ax.set_xlim(mpl.dates.datestr2num(
        ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
    small_spatial_avg_increment.isel(realization=0).plot.line('-', hue="type")
except ValueError:
    ax.set_xlim(mpl.dates.datestr2num(
        ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
    small_spatial_avg_increment.isel(realization=0).plot.line('-', hue="type")

# for xval, color in (zip(
#         mpl.dates.datestr2num(
#             ["2010-07-01T00:00:00Z", "2010-07-03T00:00:00Z",
#              "2010-07-17T00:00:00Z"]),
#         ["red", "gray", "red"])):
#     ax.axvline(xval, color=color)

ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Average increment over West Virginia\n"
              "(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
plt.title("Spatial average increment")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_small_spatial_"
    "avg_increment_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done increment plots")

spatial_avg_differences.load()
spatial_avg_increment.load()

############################################################
# Make big combined plot
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
fig.subplots_adjust(hspace=.27, bottom=.12, top=.85)
ax = axes[0]
prior_line = spatial_avg_differences.sel(
    type="prior_error", realization=0).plot.line(
    "-", label="Prior", ax=ax)
id_post_line = spatial_avg_differences.sel(
    type="iden_posterior_error", realization=0).plot.line(
    "--", label="Posterior: Identical", ax=ax)
fr_post_line = spatial_avg_differences.sel(
    type="frat_posterior_error", realization=0).plot.line(
    "-.", label="Posterior: Fraternal", ax=ax)

ax.set_xlim(mpl.dates.datestr2num(
    ["2010-06-18T00:00:00Z", "2010-08-01T00:00:00Z"]))
ax.axhline(0, color="black", linewidth=.75)
ax.set_ylim(-5, 5)

fig.legend([prior_line[0], id_post_line[0], fr_post_line[0]],
           ["Prior", "Posterior: Identical", "Posterior: Fraternal"],
           (.2, .9),
           ncol=3)
ax.set_ylabel("Average flux error\n(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
ax.set_title("Spatial average flux error")

ax = axes[1]
spatial_avg_increment.isel(
    realization=0
).sel(
    type="iden_posterior_error"
).plot.line('--', color="tab:orange", ax=ax)
spatial_avg_increment.isel(
    realization=0
).sel(
    type="frat_posterior_error"
).plot.line('-.', color="tab:green", ax=ax)

ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Average increment\n(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)")
ax.set_xlabel("")
ax.set_title("Spatial average increment")

ax = axes[2]
OBSERVATIONAL_CONSTRAINT.plot.line("-", ax=ax)
ax.axhline(0, color="black", linewidth=.75)
ax.set_ylabel("Influence\n"
              "(ppm/(\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s))")
ax.set_xlabel("OSSE Flux Time")
ax.set_title("Influence Of Fluxes On Observations")
ax.set_ylim(0, 5e7)

for ax in axes:
    ax.axvline(mpl.dates.datestr2num("2010-07-01T00:00:00Z"),
               color="black", linewidth=.75)

fig.suptitle("Average fluxes, increment, and constraint over whole domain")
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_realization_spatial_"
    "avg_combined_timeseries.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)
write_console_message("Done combined timeseries plot")


############################################################
# Calculate prior and posterior variances
write_console_message("Calculating variances for identical-twin OSSE")
iden_prior_theoretical_variance = (
    IDEN_COVARIANCE_DS["reduced_prior_covariance"].mean() * 1e12
).values
write_console_message("Done prior, starting posterior")
iden_posterior_theoretical_variance = (
    IDEN_COVARIANCE_DS["reduced_posterior_covariance"].mean() * 1e12
).values
write_console_message("Done posterior w/ agg, starting w/o")
iden_posterior_theoretical_variance_no_agg = (
    IDEN_COVARIANCE_DS["reduced_posterior_covariance_no_aggregation"].mean() *
    1e12
).values
write_console_message("Calculating variances for fraternal-twin OSSE")
frat_prior_theoretical_variance = (
    FRAT_COVARIANCE_DS["reduced_prior_covariance"].mean() * 1e12
).values
write_console_message("Done prior, starting posterior")
frat_posterior_theoretical_variance = (
    FRAT_COVARIANCE_DS["reduced_posterior_covariance"].mean() * 1e12
).values
write_console_message("Done posterior w/ agg, starting w/o")
frat_posterior_theoretical_variance_no_agg = (
    FRAT_COVARIANCE_DS["reduced_posterior_covariance_no_aggregation"].mean() *
    1e12
).values
write_console_message("Done highest resolution, starting lower resolution")
LOWER_RES_REDUCED_IDEN_COVARIANCE_DS = (
    LOWER_RES_IDEN_COVARIANCE_DS.mean() * 1e12
).persist()
lower_res_iden_prior_theoretical_variance = (
    LOWER_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_prior_covariance"
    ].values
)
lower_res_iden_posterior_theoretical_variance = (
    LOWER_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_posterior_covariance"
    ].values
)
lower_res_iden_posterior_theoretical_variance_no_agg = (
    LOWER_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_posterior_covariance_no_aggregation"
    ].values
)
LOWER_RES_REDUCED_FRAT_COVARIANCE_DS = (
    LOWER_RES_FRAT_COVARIANCE_DS.mean() * 1e12
).persist()
lower_res_frat_prior_theoretical_variance = (
    LOWER_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_prior_covariance"
    ].values
)
lower_res_frat_posterior_theoretical_variance = (
    LOWER_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_posterior_covariance"
    ].values
)
lower_res_frat_posterior_theoretical_variance_no_agg = (
    LOWER_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_posterior_covariance_no_aggregation"
    ].values
)
write_console_message("Done lower resolution, starting lowest_resolution")
LOWEST_RES_REDUCED_IDEN_COVARIANCE_DS = (
    LOWEST_RES_IDEN_COVARIANCE_DS.mean() * 1e12
).persist()
lowest_res_iden_prior_theoretical_variance = (
    LOWEST_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_prior_covariance"
    ].values
)
lowest_res_iden_posterior_theoretical_variance = (
    LOWEST_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_posterior_covariance"
    ].values
)
lowest_res_iden_posterior_theoretical_variance_no_agg = (
    LOWEST_RES_REDUCED_IDEN_COVARIANCE_DS[
        "reduced_posterior_covariance_no_aggregation"
    ].values
)
LOWEST_RES_REDUCED_FRAT_COVARIANCE_DS = (
    LOWEST_RES_FRAT_COVARIANCE_DS.mean() * 1e12
).persist()
lowest_res_frat_prior_theoretical_variance = (
    LOWEST_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_prior_covariance"
    ].values
)
lowest_res_frat_posterior_theoretical_variance = (
    LOWEST_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_posterior_covariance"
    ].values
)
lowest_res_frat_posterior_theoretical_variance_no_agg = (
    LOWEST_RES_REDUCED_FRAT_COVARIANCE_DS[
        "reduced_posterior_covariance_no_aggregation"
    ].values
)
write_console_message("Done calculating variances")

############################################################
# Plot histogram of average flux errors
mean_error_df = all_mean_error.to_dataframe(
    name="error"
).loc[:, "error"].unstack(0).loc[
    :, ["prior_error", "iden_posterior_error", "frat_posterior_error"]
]
error_range = da.asarray(
    [all_mean_error.min(), all_mean_error.max()]
).compute()

fig, ax = plt.subplots(1, 1)
mean_error_df.plot.hist(ax=ax, alpha=.5, xlim=(-1.8, 1.8))
ax.set_ylabel("Count")
ax.set_xlabel(
    "Error in mean estimate (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)"
)
ax.legend(["Prior means", "Posterior means: Identical",
           "Posterior means: Fraternal"])
mean_error_df["prior_error"].plot.box(
    ax=ax, vert=False, positions=(31,), color="blue", widths=1.5)
mean_error_df["iden_posterior_error"].plot.box(
    ax=ax, vert=False, positions=(33,), color="orange", widths=1.5)
mean_error_df["frat_posterior_error"].plot.box(
    ax=ax, vert=False, positions=(35,), color="tab:green", widths=1.5)
ax.set_ylim(0, 37)
ax.set_yticks(np.arange(0, 31, 5))
ax.set_yticklabels(np.arange(0, 31, 5))
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_flux_error_hist.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

############################################################
# Write file of statistics for flux errors
ZSTAR_90 = scipy.stats.norm.ppf(.95)
with open(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_flux_error_stats.txt".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN), "w") as out_file:
    print("Theoretical/analytic/deterministic standard deviations: "
          "Identical-twin OSSE", file=out_file)
    print(np.sqrt([iden_prior_theoretical_variance,
                   iden_posterior_theoretical_variance,
                   iden_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Theoretical/analytic/deterministic standard deviations: "
          "Fraternal-twin OSSE", file=out_file)
    print(np.sqrt([frat_prior_theoretical_variance,
                   frat_posterior_theoretical_variance,
                   frat_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Lower-resolution Theoretical/analytic/deterministic "
          "standard deviations: Identical-twin OSSE", file=out_file)
    print(np.sqrt([lower_res_iden_prior_theoretical_variance,
                   lower_res_iden_posterior_theoretical_variance,
                   lower_res_iden_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Lower-resolution Theoretical/analytic/deterministic "
          "standard deviations: Fraternal-twin OSSE", file=out_file)
    print(np.sqrt([lower_res_frat_prior_theoretical_variance,
                   lower_res_frat_posterior_theoretical_variance,
                   lower_res_frat_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Lowest-resolution Theoretical/analytic/deterministic "
          "standard deviations: Identical-twin OSSE", file=out_file)
    print(np.sqrt([lowest_res_iden_prior_theoretical_variance,
                   lowest_res_iden_posterior_theoretical_variance,
                   lowest_res_iden_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Lowest-resolution Theoretical/analytic/deterministic "
          "standard deviations: Fraternal-twin OSSE", file=out_file)
    print(np.sqrt([lowest_res_frat_prior_theoretical_variance,
                   lowest_res_frat_posterior_theoretical_variance,
                   lowest_res_frat_posterior_theoretical_variance_no_agg]),
          file=out_file)
    print("Description of errors", file=out_file)
    for n_realizations in (5, 10, 20, 40, 80):
        print(
            "Number of realizations considered:", n_realizations, file=out_file
        )
        ldesc = long_description(mean_error_df.iloc[:n_realizations, :])
        print(ldesc, file=out_file)
    print("Coverage for 90% confidence interval, identical prior:",
          file=out_file)
    number_in_prior_ci = (
        np.abs(mean_error_df["prior_error"] /
               np.sqrt(iden_prior_theoretical_variance))
        < ZSTAR_90
    ).sum()
    print(number_in_prior_ci / mean_error_df.shape[0], file=out_file)
    print("Coverage for 90% confidence interval, identical posterior:",
          file=out_file)
    number_in_posterior_ci = (
        np.abs(mean_error_df["iden_posterior_error"] /
               np.sqrt(iden_posterior_theoretical_variance))
        < ZSTAR_90
    ).sum()
    print(number_in_posterior_ci / mean_error_df.shape[0], file=out_file)
    print("Coverage for 90% confidence interval, posterior (no agg.):",
          file=out_file)
    number_in_posterior_ci_no_agg = (
        np.abs(mean_error_df["iden_posterior_error"] /
               np.sqrt(iden_posterior_theoretical_variance_no_agg))
        < ZSTAR_90
    ).sum()
    print(number_in_posterior_ci_no_agg / mean_error_df.shape[0],
          file=out_file)
    print("P-values are one of:", file=out_file)
    dist = scipy.stats.binom(mean_error_df.shape[0], .9)
    print(
        dist.cdf([number_in_prior_ci, number_in_posterior_ci,
                  number_in_posterior_ci_no_agg]) * 2,
        file=out_file
    )
    print(
        dist.sf([number_in_prior_ci, number_in_posterior_ci,
                 number_in_posterior_ci_no_agg]) * 2,
        file=out_file
    )
    print("Coverage for 90% confidence interval, fraternal prior:",
          file=out_file)
    number_in_prior_ci = (
        np.abs(mean_error_df["prior_error"] /
               np.sqrt(frat_prior_theoretical_variance))
        < ZSTAR_90
    ).sum()
    print(number_in_prior_ci / mean_error_df.shape[0], file=out_file)
    print("Coverage for 90% confidence interval, fraternal posterior:",
          file=out_file)
    number_in_posterior_ci = (
        np.abs(mean_error_df["frat_posterior_error"] /
               np.sqrt(frat_posterior_theoretical_variance))
        < ZSTAR_90
    ).sum()
    print(number_in_posterior_ci / mean_error_df.shape[0], file=out_file)
    print("Coverage for 90% confidence interval, posterior (no agg.):",
          file=out_file)
    number_in_posterior_ci_no_agg = (
        np.abs(mean_error_df["frat_posterior_error"] /
               np.sqrt(frat_posterior_theoretical_variance_no_agg))
        < ZSTAR_90
    ).sum()
    print(number_in_posterior_ci_no_agg / mean_error_df.shape[0],
          file=out_file)
    print("P-values are one of:", file=out_file)
    dist = scipy.stats.binom(mean_error_df.shape[0], .9)
    print(
        dist.cdf([number_in_prior_ci, number_in_posterior_ci,
                  number_in_posterior_ci_no_agg]) * 2,
        file=out_file
    )
    print(
        dist.sf([number_in_prior_ci, number_in_posterior_ci,
                 number_in_posterior_ci_no_agg]) * 2,
        file=out_file
    )
    print("K-S tests for distributions (identical-twin):", file=out_file)
    print(
        "Identical-twin Prior:    ",
        scipy.stats.kstest(
            mean_error_df["prior_error"],
            scipy.stats.norm(
                scale=np.sqrt(iden_prior_theoretical_variance)).cdf
        ),
        file=out_file
    )
    print(
        "Identical-twin Posterior:",
        scipy.stats.kstest(
            mean_error_df["iden_posterior_error"],
            scipy.stats.norm(
                scale=np.sqrt(iden_posterior_theoretical_variance)).cdf
        ),
        file=out_file
    )
    print(
        "Identical-twin Posterior (no agg.):",
        scipy.stats.kstest(
            mean_error_df["iden_posterior_error"],
            scipy.stats.norm(
                scale=np.sqrt(iden_posterior_theoretical_variance_no_agg)).cdf
        ),
        file=out_file
    )
    print("K-S tests for distributions (Fraternal-twin):", file=out_file)
    print(
        "Fraternal-twin Prior:    ",
        scipy.stats.kstest(
            mean_error_df["prior_error"],
            scipy.stats.norm(
                scale=np.sqrt(frat_prior_theoretical_variance)).cdf
        ),
        file=out_file
    )
    print(
        "Fraternal-twin Posterior:",
        scipy.stats.kstest(
            mean_error_df["frat_posterior_error"],
            scipy.stats.norm(
                scale=np.sqrt(frat_posterior_theoretical_variance)).cdf
        ),
        file=out_file
    )
    print(
        "Fraternal-twin Posterior (no agg.):",
        scipy.stats.kstest(
            mean_error_df["frat_posterior_error"],
            scipy.stats.norm(
                scale=np.sqrt(frat_posterior_theoretical_variance_no_agg)).cdf
        ),
        file=out_file
    )
    print("\N{GREEK SMALL LETTER CHI}\N{SUPERSCRIPT TWO} test for variance",
          file=out_file)
    degrees_freedom = ldesc.loc["count", :] - 1
    degrees_freedom = np.array([degrees_freedom[0],
                                degrees_freedom[1],
                                degrees_freedom[1]])
    std = ldesc.loc["std", :]
    std = np.array([[std[0], std[1], std[1]],
                    [std[0], std[2], std[2]]])
    statistic = (
        degrees_freedom * std ** 2 /
        np.asarray([[iden_prior_theoretical_variance,
                     iden_posterior_theoretical_variance,
                     iden_posterior_theoretical_variance_no_agg],
                    [frat_prior_theoretical_variance,
                     frat_posterior_theoretical_variance,
                     frat_posterior_theoretical_variance_no_agg]])
    )
    print("First line is identical-twin, second line is fraternal-twin",
          file=out_file)
    print("Statistics are\n", statistic, file=out_file, sep="")
    print("One-sided test for sample variance "
          "larger than theoretical variance:\n",
          scipy.stats.chi2.sf(statistic, df=degrees_freedom),
          file=out_file, sep="")
    print("\nPearson product-moment correlations\n",
          mean_error_df.corr(),
          "\nSpearman rank correlations\n",
          mean_error_df.corr("spearman"),
          "\nKendall Tau correlation\n",
          mean_error_df.corr("kendall"),
          file=out_file)


N_FLUXES = 200
sample_fluxes = np.linspace(-3, 3, N_FLUXES)

iden_prior_mean_flux_density = (
    np.exp(-0.5 * sample_fluxes ** 2 / iden_prior_theoretical_variance) /
    np.sqrt(2 * np.pi * iden_prior_theoretical_variance)
)
iden_posterior_mean_flux_density = (
    np.exp(-0.5 * sample_fluxes ** 2 / iden_posterior_theoretical_variance) /
    np.sqrt(2 * np.pi * iden_posterior_theoretical_variance)
)

iden_prior_flux_densities = (
    np.exp(
        -0.5 *
        (sample_fluxes[:, np.newaxis] -
         mean_error_df["prior_error"].values[np.newaxis, :]) ** 2 /
        iden_prior_theoretical_variance
    ) /
    np.sqrt(2 * np.pi * iden_prior_theoretical_variance)
)
iden_prior_flux_density_estimate = iden_prior_flux_densities.mean(axis=1)

iden_posterior_flux_densities = (
    np.exp(
        -0.5 *
        (sample_fluxes[:, np.newaxis] -
         mean_error_df["iden_posterior_error"].values[np.newaxis, :]) ** 2 /
        iden_posterior_theoretical_variance
    ) /
    np.sqrt(2 * np.pi * iden_posterior_theoretical_variance)
)
iden_posterior_flux_density_estimate = iden_posterior_flux_densities.mean(
    axis=1)

frat_prior_mean_flux_density = (
    np.exp(-0.5 * sample_fluxes ** 2 / frat_prior_theoretical_variance) /
    np.sqrt(2 * np.pi * frat_prior_theoretical_variance)
)
frat_posterior_mean_flux_density = (
    np.exp(-0.5 * sample_fluxes ** 2 / frat_posterior_theoretical_variance) /
    np.sqrt(2 * np.pi * frat_posterior_theoretical_variance)
)

frat_prior_flux_densities = (
    np.exp(
        -0.5 *
        (sample_fluxes[:, np.newaxis] -
         mean_error_df["prior_error"].values[np.newaxis, :]) ** 2 /
        frat_prior_theoretical_variance
    ) /
    np.sqrt(2 * np.pi * frat_prior_theoretical_variance)
)
frat_prior_flux_density_estimate = frat_prior_flux_densities.mean(axis=1)

frat_posterior_flux_densities = (
    np.exp(
        -0.5 *
        (sample_fluxes[:, np.newaxis] -
         mean_error_df["frat_posterior_error"].values[np.newaxis, :]) ** 2 /
        frat_posterior_theoretical_variance
    ) /
    np.sqrt(2 * np.pi * frat_posterior_theoretical_variance)
)
frat_posterior_flux_density_estimate = frat_posterior_flux_densities.mean(
    axis=1)

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
fig.subplots_adjust(top=.85, bottom=.15, left=.1, right=.9)
ax.axvline(0, linewidth=.5)
# ax.plot(sample_fluxes, prior_mean_flux_density,  # 'r--',
#         label="Analytic density for prior means")
# ax.plot(sample_fluxes, posterior_mean_flux_density,  # "m--"
#         label="Analytic density for posterior means")

# Confusing, and Dr. Keller backed off on them being useful
# ax.plot(sample_fluxes, prior_flux_density_estimate,
#         label="Combined prior density")
# ax.plot(sample_fluxes, posterior_flux_density_estimate,
#         label="Combined posterior density")

for prior_density in iden_prior_flux_densities.T:
    prior_lines = ax.plot(
        sample_fluxes, prior_density,
        alpha=.2, color="tab:blue",
        linewidth=.5, label="Prior densities",
    )

for posterior_density in iden_posterior_flux_densities.T:
    posterior_lines = ax.plot(
        sample_fluxes, posterior_density,
        alpha=.2, color="tab:orange",
        linewidth=.5, label="Posterior densities",
    )

fig.legend(
    handles=prior_lines + posterior_lines,
    labels=("Prior densities", "Posterior densities"),
    ncol=2, loc="upper center"
)
ax.set_ylabel("Density")
ax.set_xlabel(
    "Error in identical-twin mean estimate (\N{MICRO SIGN}mol/m$^2$/s)"
)
ax.set_ylim(0, 2)
ax.set_xlim(-3, 3)
yticks = np.arange(0, 1.61, .2)
ax.set_yticks(yticks)
ax.set_yticklabels(["{:.1f}".format(tick) for tick in yticks])

color_blue = dict(color="tab:blue", marker=None)
ax.boxplot(mean_error_df["prior_error"], vert=False, positions=(1.64,),
           widths=.19, manage_xticks=False, showmeans=True,
           boxprops=color_blue, whiskerprops=color_blue, capprops=color_blue,
           flierprops=dict(color="tab:blue", marker="o"),
           meanprops=dict(alpha=.7), medianprops=color_blue)
color_orange = dict(color="tab:orange", marker=None)
ax.boxplot(mean_error_df["iden_posterior_error"], vert=False,
           positions=(1.88,),
           widths=.19, manage_xticks=False, showmeans=True,
           boxprops=color_orange, whiskerprops=color_orange,
           capprops=color_orange,
           flierprops=dict(color="tab:orange", marker="o"),
           meanprops=dict(alpha=.7), medianprops=color_orange)

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_flux_error_density.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

############################################################
small_error_range = da.asarray(
    [small_mean_error.min(), small_mean_error.max()]).compute()
fig, ax = plt.subplots(1, 1)
# ax.hist([small_mean_error.sel(type="prior_error"),
#          small_mean_error.sel(type="posterior_error")],
#         range=[-4, 4],
#         label=["Prior", "Posterior"])
ax.set_xlabel("West Virginia mean flux error (\N{MICRO SIGN}mol/m$^2$/s)")
small_mean_error_df = small_mean_error.to_dataframe(
    name="error"
).loc[:, "error"].unstack(0).loc[
    :, ["prior_error", "iden_posterior_error", "frat_posterior_error"]
]
small_mean_error_df.plot.kde(
    ax=ax, subplots=False, xlim=(-30, 30), ylim=(0, None),
)

sns.rugplot(small_mean_error_df["prior_error"], axis=ax, color="b")
sns.rugplot(small_mean_error_df["iden_posterior_error"], axis=ax,
            color="tab:orange")
sns.rugplot(small_mean_error_df["frat_posterior_error"], axis=ax,
            color="tab:green")

# sample_fluxes = np.linspace(-20, 20, N_FLUXES)
# prior_flux_density = (
#     np.exp(-0.5 * sample_fluxes ** 2 / prior_small_theoretical_variance) /
#     np.sqrt(2 * np.pi * prior_small_theoretical_variance)
# )
# ax.plot(sample_fluxes, prior_flux_density, 'r--')
# posterior_flux_density = (
#     np.exp(-0.5 * sample_fluxes ** 2 /
#            posterior_small_theoretical_variance) /
#     np.sqrt(2 * np.pi * posterior_small_theoretical_variance)
# )
# ax.plot(sample_fluxes, posterior_flux_density, "m--")

# ax.legend(["Prior Means", "Posterior Means", "PDF for Prior Means",
#            "PDF for Posterior Means"])
# ax.set_ylabel("Density")

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_flux_error_density.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

fig, ax = plt.subplots(1, 1)
small_mean_error_df.plot.hist(ax=ax, alpha=.5, xlim=(-30, 30))
ax.set_ylabel("Count")
ax.set_xlabel(
    "Error in mean estimate (\N{MICRO SIGN}mol/m\N{SUPERSCRIPT TWO}/s)"
)
ax.legend(["Prior means", "Posterior means"])
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_small_flux_error_hist.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

write_console_message("Wrote error densities")

############################################################
# Gain vs. magnitude of prior error
fig = plt.figure(figsize=(4, 2.5))
plt.subplots_adjust(left=.24, bottom=.2)
plt.plot(total_gain.T, all_mean_error.sel(type="prior_error"), ".")
plt.xlabel("Gain")
plt.ylabel("Prior error (\N{MICRO SIGN}mol/m$^2$/s)")
plt.xlim(-1, 1)
plt.legend(["Identical-twin", "Fraternal-twin"])

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_gain_prior_mag.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)

############################################################
# Error reduction
time_mean_error_var = time_mean_error.var("realization")

time_mean_error_reduction = 1 - da.sqrt(
    time_mean_error_var.sel(type=["iden_posterior_error",
                                  "frat_posterior_error"]) /
    time_mean_error_var.sel(type="prior_error"))

grid = time_mean_error_reduction.plot.pcolormesh(
    vmin=-.5, vmax=.5, center=0, cmap="RdBu_r", col="type",
    subplot_kws=dict(projection=WRF_CRS), aspect=1.2,
    size=2.2)
fig = grid.fig
axes = grid.axes

fig.suptitle("Error reduction")
axes.flat[0].set_title("Identical-Twin")
axes.flat[1].set_title("Fraternal-Twin")
for ax in axes.flat:
    ax.coastlines()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_error_reduction.png".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH, inv_time_fun=INV_TIME_FUN,
        inv_time_len=INV_TIME_LEN))
plt.close(fig)


############################################################
# Plot tower locations and influence functions

fig, axes = plt.subplots(
    1, 1, figsize=(5, 3), subplot_kw=dict(projection=LPDM_PROJ))

COLLAPSED_INFLUENCE_DS.H.plot(robust=True)
plt.scatter(FULL_INFLUENCE_DS.coords["site_lons"],
            FULL_INFLUENCE_DS.coords["site_lats"],
            transform=LPDM_PROJ.as_geodetic(), color="red")

plt.title("Integrated influence functions")
fig.axes[-1].set_ylabel("H (ppmv/(mol/m$^2$/s))")
axes.set_xlim(COLLAPSED_INFLUENCE_DS.coords["dim_x"][[0, -1]])
axes.set_ylim(COLLAPSED_INFLUENCE_DS.coords["dim_y"][[0, -1]])

axes.coastlines()
axes.add_feature(cfeat.BORDERS)
axes.add_feature(BIG_LAKES, facecolor="none")
axes.add_feature(STATES)
# axes.axvline(WEST_BOUNDARY_LPDM, color="white")

fig.savefig("{year:04d}-{month:02d}_integrated_influence_functions.png".format(
    year=YEAR, month=MONTH), dpi=400)
plt.close(fig)

############################################################
# Scatter plot of prior and posterior
fig, ax = plt.subplots(1, 1, figsize=(3.3, 3))
mean_error_df.plot("prior_error", "iden_posterior_error", style='.',
                   ax=ax, legend=False)
mean_error_df.plot("prior_error", "frat_posterior_error", style='.',
                   ax=ax, legend=False)
ax.set_xlabel("Prior error (\N{MICRO SIGN}mol/m$^2$/s)")
ax.set_ylabel("Posterior error (\N{MICRO SIGN}mol/m$^2$/s)")
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
fig.legend(["Identical-twin", "Fraternal-twin"], loc="upper center",
           ncol=2)

fig.tight_layout()
fig.subplots_adjust(top=.87)
fig.savefig(
    "{year:04d}-{month:02d}_noise_{noise_fun:s}{noise_len:03d}km_"
    "{noise_time_fun:s}{noise_time_len:02d}d_inv_{inv_fun:s}{inv_len:03d}km_"
    "{inv_time_fun:s}{inv_time_len:02d}d_flux_error_scatter.pdf".format(
        year=YEAR, month=MONTH, noise_fun=NOISE_FUNCTION,
        noise_len=NOISE_LENGTH, noise_time_fun=NOISE_TIME_FUN,
        noise_time_len=NOISE_TIME_LEN, inv_fun=INV_FUNCTION,
        inv_len=INV_LENGTH,
        inv_time_fun=INV_TIME_FUN, inv_time_len=INV_TIME_LEN))
plt.close(fig)

write_console_message("Done all figures")
