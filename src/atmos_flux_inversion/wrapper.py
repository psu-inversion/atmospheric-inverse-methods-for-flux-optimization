"""Wrap the details of an inversion.

Hide most of the implementation details.
Take prior and parameters, return posterior.

The interface is still experimental at this point.  I am likely to
remove functions and parameters as I find better ways to organize
things.  The code still works fine as an example.
"""
from distutils.version import LooseVersion
import subprocess
import datetime
import sys
import os
try:
    from shlex import quote
except ImportError:
    from pipes import quote
from pwd import getpwuid

import numpy as np
import dateutil.tz

import pandas as pd
import xarray

import atmos_flux_inversion.optimal_interpolation
import atmos_flux_inversion.correlations as inv_corr
import atmos_flux_inversion.covariances
import atmos_flux_inversion.util

HOURS_PER_DAY = 24
OBSERVATION_TEMPORAL_CORRELATION_FUNCTION = (  # noqa: C0103
    inv_corr.ExponentialCorrelation)
UTC = dateutil.tz.tzutc()
UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)
_PACKAGE_INFO = None


def invert_uniform(prior_fluxes, observations,
                   observation_operator,
                   prior_correlation_length, prior_correlation_structure,
                   prior_correlation_time_days, prior_correlation_time_hours,
                   prior_flux_stds,
                   observation_correlation_time,
                   method=(
                       atmos_flux_inversion.optimal_interpolation.fold_common
                   ),
                   output_uncertainty_frequency="MS"):
    """Perform an inversion.

    Assumes error correlations are invariant in time and space and
    that the error variances are constant in time.  Also assumes the
    observation correlations between towers is zero.

    Parameters
    ----------
    prior_fluxes: xarray.Dataarray[flux_time, y, x]
        The prior or background estimate of the fluxes
    observations: xarray.Dataarray[obs_time, site]
        The observed mole fractions
    observation_operator: xarray.Dataarray[obs_time, site, flux_time, y, x]
        The linearized operator mapping a flux distribution to
        observations.
    prior_correlation_length: float
        The lengthscale for the spatial correlations in the prior.
    prior_correlation_structure: inv_corr.DistanceCorrelationFunction
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
        :func:`atmos_flux_inversion.optimal_interpolation.simple`
    output_uncertainty_frequency: str
        One of the frequencies from
        `<http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
        or "season".  Often "M", output_uncertainty_frequency, "Y",
        "YS", or "season" make the most sense.  Used to provide
        lower-temporal-resolution posterior uncertainties.

    Returns
    -------
    xarray.Dataset
        The results of the inversion.
        Contents:

        prior[flux_time, y, x]
            The prior mean for the inversion.
        increment[flux_time, y, x]
            The change from the prior to the posterior mean estimates.
        posterior[flux_time, y, x]
            The posterior mean for the inversion.
        post_cov[red_flux_time_adj, y_adj, x_adj, red_flux_time, y, x]
            The analytic uncertainty for the posterior, expressed as a
            covariance matrix on a reduced-resolution domain.  If the
            prior and observation likelihood are Gaussian, this will
            be exact.
    """
    y_index_name = [name for name in prior_fluxes.coords
                    if "y" in name.lower()][0]
    x_index_name = y_index_name.replace("y", "x")
    flux_time_index_name = [name for name in prior_fluxes.coords
                            if "time" in name.lower()][0]
    site_index_names = [name for name in observations.coords
                        if "site" in name.lower()]
    obs_time_index_names = [name for name in observations.coords
                            if "time" in name.lower()]

    obs_is_multi = False
    for index_name, index in observations.indexes.items():
        if isinstance(index, pd.MultiIndex):
            obs_is_multi = True
            obs_time_index_names.extend([name for name in index.names
                                         if "time" in name])
            site_index_names.extend([name for name in index.names
                                     if "site" in name])
            observations = observations.rename(
                {index_name: "observations"})
            observation_operator = observation_operator.rename(
                {index_name: "observations"})

    if len(site_index_names) == 1:
        site_index_name = site_index_names[0]
    elif "site" in site_index_names:
        site_index_name = "site"

    obs_time_index_name = obs_time_index_names[0]

    x_index = prior_fluxes[x_index_name]
    y_index = prior_fluxes[y_index_name]
    obs_time_index = observations.coords[obs_time_index_name]
    flux_time_index = prior_fluxes.indexes[flux_time_index_name]

    site_names = list(set(observations.coords[site_index_name].values))

    if not obs_is_multi:
        observations_for_inversion = observations.stack(dict(
            observations=[obs_time_index_name, site_index_name]
        ))
    else:
        observations_for_inversion = observations
    long_obs_times = observations_for_inversion.coords[obs_time_index_name]
    long_site = observations_for_inversion.coords[site_index_name]

    flux_time_adjoint_index = flux_time_index.copy()
    flux_time_adjoint_index.name += "_adjoint"

    dx = abs(x_index[1] - x_index[0])
    obs_interval = abs(obs_time_index[1] - obs_time_index[0])
    flux_interval = abs(flux_time_index[1] - flux_time_index[0])

    spatial_correlations = (
        atmos_flux_inversion.correlations.
        HomogeneousIsotropicCorrelation.from_function(
            prior_correlation_structure(prior_correlation_length / dx),
            prior_fluxes.shape[1:]))
    sqrt_spatial_variances = prior_flux_stds.transpose(
        y_index_name, x_index_name)
    spatial_covariances = (
        atmos_flux_inversion.covariances.CorrelationStandardDeviation(
            spatial_correlations, sqrt_spatial_variances.data
        )
    )

    hour_correlations = (
        atmos_flux_inversion.correlations.
        HomogeneousIsotropicCorrelation.from_function(
            atmos_flux_inversion.correlations.ExponentialCorrelation(
                pd.Timedelta(hours=prior_correlation_time_hours) /
                flux_interval),
            pd.Timedelta(hours=HOURS_PER_DAY) // flux_interval))
    day_correlations = (
        atmos_flux_inversion.correlations.make_matrix(
            atmos_flux_inversion.correlations.ExponentialCorrelation(
                prior_correlation_time_days),
            prior_fluxes.shape[0]))
    temporal_covariances = atmos_flux_inversion.util.kron(
        day_correlations.dot(np.eye(*day_correlations.shape)),
        hour_correlations.dot(np.eye(*hour_correlations.shape))
    )

    temporal_covariance_dataset = xarray.DataArray(
        dims=(flux_time_index_name, flux_time_adjoint_index.name),
        data=temporal_covariances,
        attrs=dict(long_name="temporal_covariances"),
        coords={flux_time_index_name: flux_time_index,
                flux_time_adjoint_index.name: flux_time_adjoint_index}
    )

    reduced_temp_cov_ds = temporal_covariance_dataset.resample(**{
        flux_time_adjoint_index.name: output_uncertainty_frequency
    }).sum(flux_time_adjoint_index.name).resample(**{
        flux_time_index_name: output_uncertainty_frequency
    }).sum(flux_time_index_name)

    reduced_obs_op = observation_operator.resample(**{
        flux_time_index_name: output_uncertainty_frequency
    }).sum(flux_time_index_name)

    prior_covariances = (
        atmos_flux_inversion.util.kronecker_product(temporal_covariances,
                                                    spatial_covariances)
    )
    reduced_prior_covariances = (
        atmos_flux_inversion.util.kron(
            reduced_temp_cov_ds.data,
            spatial_covariances.dot(np.eye(*spatial_covariances.shape))
        )
    )

    observation_temporal_correlations = (
        OBSERVATION_TEMPORAL_CORRELATION_FUNCTION(
            observation_correlation_time)
    )(abs(
        long_obs_times.values[:, np.newaxis] -
        long_obs_times.values[np.newaxis, :]
    ).astype("m8[h]").astype(int))
    observation_site_correlations = (
        long_site.values[:, np.newaxis] == long_site.values[np.newaxis, :])
    observation_correlations = (
        observation_temporal_correlations * observation_site_correlations)
    observation_covariance = observation_correlations

    if not obs_is_multi:
        obs_stack_dict = dict(
            observations=[obs_time_index_name, site_index_name],
            fluxes=[flux_time_index_name, y_index_name, x_index_name]
        )
    else:
        obs_stack_dict = dict(
            fluxes=[flux_time_index_name, y_index_name, x_index_name]
        )

    posterior_fluxes, posterior_covariance = method(
        prior_fluxes.data.reshape(prior_covariances.shape[0]),
        prior_covariances,
        observations.data.reshape(observation_covariance.shape[0]),
        observation_covariance,
        # Question: should this transpose obs_op to ensure the proper
        # dimension order?
        observation_operator.stack(obs_stack_dict).transpose(
            "observations", "fluxes"),
        reduced_prior_covariances,
        reduced_obs_op.stack(obs_stack_dict).transpose(
            "observations", "fluxes"),
    )
    posterior_fluxes = posterior_fluxes.reshape(prior_fluxes.shape)

    posterior_var_atts = prior_fluxes.attrs.copy()
    posterior_var_atts.update(dict(
        long_name="posterior_fluxes",
        description=(
            "posterior mean fluxes obtained using atmos_flux_inversion package"
        ),
        origin="atmos_flux_inversion package",
        units=prior_fluxes.attrs["units"],
    ))
    increment_var_atts = prior_fluxes.attrs.copy()
    increment_var_atts.update(dict(
        long_name="flux_increment",
        units=prior_fluxes.attrs["units"],
        description="Change from prior to posterior using dask",
        origin="OI and dask",
    ))
    posterior_covariance_attributes = prior_fluxes.attrs.copy()
    posterior_covariance_attributes.update(dict(
        long_name="inversion_posterior_covariance",
        units="({units:s})^2".format(units=prior_fluxes.attrs["units"]),
        description=("Posterior flux covariance"
                     "obtained using atmos_flux_inversion package"),
        origin="atmos_flux_inversion package",
    ))
    posterior_global_atts = global_attributes_dict()
    posterior_global_atts.update(dict(
        title="Bayesian flux inversion results",
        summary="Results of a Bayesian flux inversion",
        product_version="v0.0.0.dev0",
        cdm_data_type="grid",
        source="Inversion package",
        # Descriptions of the inversion that may be hard to get later
        prior_flux_long_name=prior_fluxes.attrs.get("long_name", ""),
        correlation_length=prior_correlation_length,
        correlation_time_day=prior_correlation_time_days,
        correlation_time_hour=prior_correlation_time_hours,
        spatial_correlation_structure=prior_correlation_structure.__name__,
        observation_interval=obs_interval,
        observation_sites_used=site_names,
    ))

    posterior_ds = xarray.Dataset(
        dict(
            posterior=(
                prior_fluxes.dims,
                posterior_fluxes.reshape(prior_fluxes.shape),
                posterior_var_atts
            ),
            prior=prior_fluxes,
            increment=(
                prior_fluxes.dims,
                posterior_fluxes - prior_fluxes,
                increment_var_atts
            ),
            posterior_covariance=(
                ["{dim:s}_adjoint".format(dim=dim)
                 if "time" not in dim else
                 "reduced_{dim:s}_adjoint".format(dim=dim)
                 for dim in prior_fluxes.dims] +
                [dim if "time" not in dim else
                 "reduced_{dim:s}".format(dim=dim)
                 for dim in prior_fluxes.dims],
                posterior_covariance.reshape(
                    (reduced_temp_cov_ds.shape[0],
                     len(y_index),
                     len(x_index)) * 2
                ),
                posterior_covariance_attributes
            ),
        ),
        prior_fluxes.coords,
        posterior_global_atts)

    coord_list = set(posterior_ds.coords)
    for coord in coord_list:
        if "time" in coord:
            continue
        posterior_ds.coords["{coord:s}_adjoint".format(coord=coord)] = (
            posterior_ds.coords[coord]
        )

    for coord in reduced_temp_cov_ds.coords:
        posterior_ds.coords["reduced_{coord:s}".format(coord=coord)] = (
            reduced_temp_cov_ds.coords[coord]
        )

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
    global _PACKAGE_INFO
    if _PACKAGE_INFO is not None:
        return _PACKAGE_INFO

    try:  # pragma: no branch
        with open(os.devnull, "wb") as stderr:
            output = subprocess.check_output(
                ["conda", "list", "--export", "--no-show-channel-urls"],
                universal_newlines=True, stderr=stderr)
        package_info = [line.split("=")[:2] for line in output.split("\n")
                        if line and not line.startswith("#")]
        python_versions = [pair[1] for pair in package_info
                           if pair[0] == "python"]

        if ((LooseVersion(python_versions[0]) ==
             LooseVersion(sys.version.split()[0]))):
            _PACKAGE_INFO = package_info
            return package_info
    except (subprocess.CalledProcessError, OSError):  # pragma: no cover
        # Testing with python3 from conda and python2 from the system
        # will not enter this branch.
        pass  # pragma: no cover

    # Conda not present or not providing current executable
    pip_args = []
    if sys.executable is not None:  # pragma: no branch
        pip_args = [sys.executable, "-m"]
    pip_args.extend(["pip", "freeze"])
    try:  # pragma: no branch
        output = subprocess.check_output(
            pip_args,
            universal_newlines=True
        )
        package_info = [line.split("==") for line in output.split("\n")
                        if line and not line.startswith("-")]

        _PACKAGE_INFO = package_info
        return package_info
    except OSError:  # pragma: no cover
        # Avoid crashes on rare platforms lacking pip
        _PACKAGE_INFO = []    # pragma: no cover
        return _PACKAGE_INFO  # pragma: no cover


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
