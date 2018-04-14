"""Wrap the details of an inversion.

Hide most of the implementation details.
Take prior and parameters, return posterior.
"""
import subprocess
import datetime
import sys
import os
from shlex import quote
from pwd import getpwuid

import numpy as np
import dateutil.tz

import xarray

import inversion.optimal_interpolation
import inversion.correlations
import inversion.covariances
import inversion.util

HOURS_PER_DAY = 24
OBSERVATION_TEMPORAL_CORRELATION_FUNCTION = (
    inversion.correlations.ExponentialCorrelation)
UTC = dateutil.tz.tzutc()
UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)


def invert_uniform(prior_fluxes, observations,
                   observation_operator,
                   prior_correlation_length, prior_correlation_structure,
                   prior_correlation_time_days, prior_correlation_time_hours,
                   prior_flux_stds,
                   observation_correlation_time,
                   method=inversion.optimal_interpolation.fold_common,
                   output_uncertainty_frequency="MS"):
    """Perform an inversion.

    Assumes error correlations are invariant in time and space and
    that the error variances are constant in time.  Also assumes the
    observation correlations between towers is zero.

    Parameters
    ----------
    prior_fluxes: xarray.Dataarray[flux_time, y, x]
    observations: xarray.Dataarray[obs_time, site]
    observation_operator: xarray.Dataarray[obs_time, site, flux_time, y, x]
        The linearized operator mapping a flux distribution to
        observations.
    prior_correlation_length: float
        The lengthscale for the spatial correlations in the prior.
    prior_correlation_structure: inversion.correlations.DistanceCorrelationFunction
        The structure of the spatial correlations in the prior.
    prior_correlation_time_days: float
        The correlation timescale for fluxes across different days.
    prior_correlation_time_hours: float
        The correlation timescale for fluxes within the same day.
    prior_flux_stds: xarray.Dataarray[y, x]
        The standard deviations of the prior flux distribution at each
        point in space.
    observation_correlation_time: float
        The correlation timescale for consecutive observations at a
        single site.
    method: function
        The method to use.  Must match the signature of
        :func:`inversion.optimal_interpolation.simple`
    output_uncertainty_frequency: str
        One of the frequencies from
        `http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases`_
        or "season".  Often "M", output_uncertainty_frequency, "Y",
        "YS", or "season" make the most sense.  Used to provide
        lower-temporal-resolution posterior uncertainties.

    Returns
    -------
    xarray.Dataset[
        prior[flux_time, y, x],
        increment[flux_time, y, x],
        posterior[flux_time, y, x],
    ]
    """
    x_index_name = [name for name in prior_fluxes.coords
                    if "x" in name.lower()][0]
    y_index_name = x_index_name.replace("x", "y")
    obs_time_index_name = [name for name in observations.coords
                           if "time" in name.lower()][0]
    flux_time_index_name = [name for name in prior_fluxes.coords
                            if "time" in name.lower()][0]
    site_index_names = [name for name in observations.coords
                        if "site" in name.lower()]

    if len(site_index_names) == 1:
        site_index_name = site_index_names[0]
    elif "site" in site_index_names:
        site_index_name = "site"
    elif any("name" in name.lower() for name in site_index_names):
        site_index_name = [name for name in site_index_names
                           if "name" in name.lower()][0]

    x_index = prior_fluxes[x_index_name]
    y_index = prior_fluxes[y_index_name]
    obs_time_index = observations.indexes[obs_time_index_name]
    flux_time_index = prior_fluxes.indexes[flux_time_index_name]
    site_index = observations.indexes[site_index_name]

    flux_time_adjoint_index = flux_time_index.copy()
    flux_time_adjoint_index.name += "_adjoint"

    dx = abs(x_index[1] - x_index[0])
    obs_interval = abs(obs_time_index[1] - obs_time_index[0])
    flux_interval = abs(flux_time_index[1] - flux_time_index[0])

    spatial_correlations = (
        inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
            prior_correlation_structure(prior_correlation_length / dx),
            prior_fluxes.shape[1:]))
    sqrt_spatial_variances = inversion.covariances.DiagonalOperator(
        prior_flux_stds.reshape(-1))
    spatial_covariances = inversion.util.ProductLinearOperator(
        sqrt_spatial_variances, spatial_correlations, sqrt_spatial_variances)

    hour_correlations = (
        inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
            inversion.correlations.ExponentialCorrelation(
                prior_correlation_time_hours / flux_interval),
            HOURS_PER_DAY // flux_interval))
    day_correlations = (
        inversion.correlations.make_matrix(
            inversion.correlations.ExponentialCorrelation(
                prior_correlation_time_days),
            prior_fluxes.shape[0]))
    temporal_covariances = (
        inversion.util.kronecker_product(day_correlations, hour_correlations))

    temporal_covariance_dataset = xarray.DataArray(
        ((flux_time_index_name, flux_time_adjoint_index.name),
         temporal_covariances,
         dict(long_name="temporal_covariances")),
        {flux_time_index_name: flux_time_index,
         flux_time_adjoint_index.name: flux_time_adjoint_index})

    if output_uncertainty_frequency is not None:
        reduced_temp_cov_ds = temporal_covariance_dataset.resample(**{
            flux_time_adjoint_index.name: output_uncertainty_frequency
        }).sum(axis=1).resample(**{
            flux_time_index_name: output_uncertainty_frequency
        }).sum(axis=0)
        reduced_obs_op = observation_operator.resample(**{
            flux_time_index_name: output_uncertainty_frequency
        }).sum(axis=2)
    else:
        reduced_temp_cov_ds = None
        reduced_obs_op = None

    prior_covariances = (
        inversion.util.kronecker_product(temporal_covariances,
                                         spatial_covariances))

    observation_temporal_correlations = (
        OBSERVATION_TEMPORAL_CORRELATION_FUNCTION(
            abs(obs_time_index[:, np.newaxis] -
                obs_time_index[np.newaxis, :]) /
            observation_correlation_time))
    observation_site_correlations = (
        site_index[:, np.newaxis] == site_index[np.newaxis, :])
    observation_correlations = (
        observation_temporal_correlations * observation_site_correlations)
    observation_covariance = observation_correlations

    posterior_fluxes, posterior_covariance = method(
        prior_fluxes.data.reshape(prior_covariances.shape[0], -1),
        prior_covariances,
        observations.data.reshape(observation_covariance.shape[0], -1),
        observation_covariance,
        # Question: should this transpose obs_op to ensure the proper
        # dimension order?
        observation_operator.reshape(observation_covariance.shape[0],
                                     prior_covariances.shape[0]),
        reduced_temp_cov_ds.data, reduced_obs_op.data
    ).reshape(prior_fluxes.shape)

    posterior_var_atts = prior_fluxes.attrs.copy()
    posterior_var_atts.update(dict(
        long_name="posterior_fluxes",
        description="posterior fluxes obtained using inversion package",
        origin="inversion package",
        units=prior_fluxes.attrs["units"],
    ))
    increment_var_atts = prior_fluxes.attrs.copy()
    increment_var_atts.update(dict(
        long_name="flux_increment",
        units=prior_fluxes.attrs["units"],
        description="Change from prior to posterior using dask",
        origin="OI and dask",
    ))
    posterior_global_atts = global_attributes_dict()
    posterior_global_atts.update(dict(
        title="Bayesian flux inversion results",
        summary="Results of a Bayesian flux inversion",
        product_version="v0.0.0.dev0",
        cdm_data_type="grid",
        source="Inversion package",
        # Descriptions of the inversion that may be hard to get later
        prior_flux_long_name=prior_fluxes.attrs["long_name"],
        correlation_length=prior_correlation_length,
        correlation_time_day=prior_correlation_time_days,
        correlation_time_hour=prior_correlation_time_hours,
        spatial_correlation_structure=prior_correlation_structure.__name__
    ))

    posterior_ds = xarray.Dataset(
        dict(posterior=(prior_fluxes.dims, posterior_fluxes,
                        posterior_var_atts),
             prior=prior_fluxes,
             increment=(prior_fluxes.dims, posterior_fluxes - prior_fluxes,
                        increment_var_atts),
             # posterior_covariance=(
             #     prior_fluxes.dims * 2, posterior_covariance,
             #     posterior_covariance_attributes),
             ),
        prior_fluxes.coords,
        posterior_global_atts)
    return posterior_ds


def get_installed_modules():
    """Get the list of installed modules.

    Returns
    -------
    list of list of str
        List of currently installed packages, using conda if that is
        available or pip if it is not.  List of two-element strings
        giving name-version pairs.
    """
    try:
        output = subprocess.check_output(
            ["conda", "list", "--export" "--no-show-channel-urls"],
            universal_newlines=True)
        package_info = [line.split("=")[:2] for line in output.split("\n")
                        if not line.startswith("#")]
        return package_info
    except OSError:
        # file not found
        pip_args = []
        if sys.executable is not None:
            pip_args = [sys.executable, "-m"]
        pip_args.extend(["pip", "freeze"])
        try:
            output = subprocess.check_output(
                pip_args,
                universal_newlines=True)
            package_info = [line.split("==") for line in output.split("\n")]
            return package_info
        except OSError:
            return []


def global_attributes_dict():
    """Set global attributes required by conventions.

    Currently CF-1.6 and ACDD-1.3.

    Still needs title, summary, source, creator_institution,
    product_version, references, cdm_data_type, institution,
    geospatial_vertical_{min,max,positive,units}, ...

    Users may want to overwrite the history and creator_name
    attributes, as these are fancy guesses.

    Returns
    -------
    global_atts: dict
        attribute_name: attribute_value mapping.
    """
    return dict(
        Conventions="CF-1.6 ACDD-1.3",
        standard_name_vocabulary="CF Standard Name Table v32",
        history=("{now:{date_fmt:s}}: Created by {progname:s}"
                 "with command line: {cmd_line:s}").format(
            now=RUN_DATE, date_fmt=UDUNITS_DATE, progname=sys.argv[0],
            cmd_line=" ".join(quote(arg) for arg in sys.argv)),
        date_created="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        date_metadata_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE),
        creator_name=getpwuid(os.getuid())[0],
        installed_modules=[
            "{name:s}={version:s}".format(name=name, version=version)
            for name, version in get_installed_modules()],
    )
