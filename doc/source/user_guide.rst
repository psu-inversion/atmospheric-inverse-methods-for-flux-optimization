.. _user_guide:

====================
Inversion User Guide
====================

We are interested in using a set of :math:`M` atmospheric measurements
to obtain an estimate for a set of :math:`N` surface fluxes.

To start off, we need some previous estimate of the fluxes of interest,
the set of atmospheric measurements, and the relationship between the
two.  I will assume these are stored in variables
`previous_surface_flux_estimate`, `observations`, and `influence_functions`
[#infl_fun_deriv]_, respectively.

In an ideal world, we would have `observations ==
influence_functions.dot(previous_surface_flux_estimate)`
[#matmul_op]_, but unfortunately this is probably not the case.  We
would like to refine `previous_surface_flux_estimate` so this is
closer to true.  To do this, we need some estimate of the
uncertainties in the flux estimate and in the observations.  This
method does not explicitly represent the uncertainty in the
relationship between the fluxes and measurements, so that uncertainty
is instead included in the uncertainty for the measurements
[#transport_uncert]_.

:ref:`It turns out to be convenient <theory>` to represent these
uncertainties as covariance matrices.  We could assume that these
matrices are diagonal: that the correction needed to bring any given
flux into line with the atmospheric observations is completely
unrelated to the corrections needed for the fluxes around it in space
and time, and that the difference between an actual measurement and
what we would predict given perfect information about the fluxes is
independent of the difference at the previous and subsequent
observation times, as well as at different measurement locations.
This is equivalent to a Weighted Least Squares method and has the
advantages of being fast and memory-efficient.

Unfortunately, if we are solving for high-resolution fluxes, using a
spatially dense set of measurement locations, or using measurements
with high temporal resolution, these assumptions are unlikely to
correspond to reality.  If we discover that the previous flux estimate
at a given point was incorrect, it was likely also incorrect on the
previous and subsequent days for roughly the same reasons.  Similarly,
while the individual measurements themselves are almost certainly
independent, predicting which fluxes influenced them is still a
challenging prospect.  When that prediction goes wrong, it will likely
stay wrong in a similar manner for at least a few hours.

Given that diagonal covariances will probably not work, we must
specify the full covariance matrix.  The next-simplest assumption
about the correlations in space and time are separate from each other:
the temporal correlations in problems with our previous estimate are
independent of the spatial location of those problems, and similarly
the spatial correlations of the problems with the previous estimate
with their neighbors are independent of the time those problems
occured.  Both :class:`~inversion.linalg.DaskKroneckerProductOperator`
and :class:`~inversion.correlations.SchmidtKroneckerProduct` are
designed for this type of matrix.

I tend to assume that the pointwise variances, which are related to
how big we expect the problems in our previous estimate to be at any
given point, are a function only of space, and are independent of
time.  We can then use
:class:`~inversion.covariances.CorrelationStandardDeviation` to
represent the spatial part of the covariance, and are left with only
the spatial and temporal correlations to completely specify the
uncertainty estimate we need here.

:class:`~inversion.linalg.DaskKroneckerProductOperator` requires an
explicit array for its first argument.  The `Climate and Forecast
conventions <http://cfconventions.org>`_ suggest ordering the dimensions for
a gridded flux estimate as time by y by x, which means this would be
the temporal correlations.

The fluxes are assumed to be less correlated noon to midnight than
they are from noon to noon, so we create separate correlation
structures to reflect this.  Kronecker products work for combining
correlations into a valid larger correlation structure, so we use that
again.

.. code-block:: python

   hour_correlation_time = 3  # hours
   day_correlation_time = 14  # days

   hour_correlation_function = inversion.correlations.ExponentialCorrelation(
       hour_correlation_time / flux_dt)
   hour_correlations = inversion.correlations.make_matrix(
       hour_correlation_function, 4)

   day_correlation_function = inversion.correlations.ExponentialCorrelation(
       day_correlation_time)
   day_correlations = inversion.correlations.make_matrix(
       day_correlation_function, n_flux_days)

   temporal_correlations = inversion.util.kron(
       day_correlations, hour_correlations)

We are left with the spatial correlations.  I have seen no evidence
that the spatial correlations are related to Plant Functional Type, so
I use correlations that are a function only of distance.
:class:`~inversion.correlations.HomogeneousIsotropicCorrelation.from_function`
makes this somewhat easier, and is designed to work with instances of
:class:`~inversion.correlations.DistanceCorrelationFunction`.

.. code-block:: python

   correlation_length = 200  # km
   spatial_correlation_function = inversion.correlations.ExponentialCorrelation(
       correlation_length / dx)
   spatial_correlations = inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
       spatial_correlation_function, previous_surface_flux_estimate.shape[-2:],
       False)
   spatial_covariance = inversion.covariances.CorrelationStandardDeviation(
       spatial_correlations, standard_deviations)

   full_covariance = inversion.linalg.DaskKroneckerProductOperator(
       temporal_correlations, spatial_covariance)

The measurements tend to be somewhat less regular than the previous
flux estimate, so that covariance matrix is best constructed by
different means.  We assume the towers are far enough apart that the
uncertainties in the relationship between the fluxes and measurements
are independent at each location, though they are probably still
correlated in time.

.. code-block:: python

   observation_correlation_time = 3  # hours
   observation_standard_deviation = 2  # ppm
   observation_correlation_function = (
       inversion.correlations.ExponentialCorrelation(
       observation_correlation_time /
       observation_dt)
   )
   observation_time_units = np.array(1, dtype="m8[h]")
   observation_covariance = observation_correlation_function(
       abs(observation_time_index[:, np.newaxis] -
           observation_time_index[np.newaxis, :]) /
       observation_time_units
   )
   observation_covariance[
       observation_site_index[:, np.newaxis] !=
       observation_site_index[np.newaxis, :]
   ] = 0
   observation_covariance *= observation_standard_deviation ** 2

At this point, we are nearly ready to pass everything to
:func:`~inversion.optimal_interpolation.save_sum`; however, just
passing everything now would result in that function calculating a
full-resolution covariance matrix for its estimate.  If we want many
fluxes, this is too big to fit in memory and will crash the system.
Fortunately, :func:`~inversion.remapper.get_remappers` will give us
matrices to aggregate our flux uncertainty and influence functions to
a coarser resolution.  If we pass these to
:func:`~inversion.optimal_interpolation.save_sum`, it will calculate
the uncertainty for its estimate at this reduced resolution, as:

.. math::

   A_{red} = B_{red} -
   B_{red} H_{red}^T
   (H_{full} B_{full} H_{full}^T + R)^{-1}
   H_{red} B_{red},

taking advantage of calculations already done for the flux estimate
[#theory]_.

.. code-block:: python

   resolution_reduction_factor = 4
   uncertainty_spatial_resolution = dx * resolution_reduction_factor
   uncertainty_temporal_resolution = "7D"
   influence_function_remapper, covariance_remapper = (
       inversion.remapper.get_remappers(
           previous_surface_flux_estimate.shape[-2:],
	   resolution_reduction_factor
       )
   )
   reduced_n_grid_points = np.prod(covariance_remapper.shape[:2])

   covariance_remap_matrix = covariance_remapper.reshape(
       (reduced_n_grid_points, n_grid_points)
   )
   reduced_spatial_covariance = (
       covariance_remap_matrix.dot(
           spatial_covariance.dot(
	       covariance_remap_matrix.T
	   )
       )
   )

   reduced_influence_functions = (
       influence_functions
       .groupby_bins(
           "x_dimension",
	   pd.interval_range(
	       0,
	       x_index[-1] + uncertainty_spatial_resolution,
	       freq=uncertainty_spatial_resolution,
	       closed="left")
       ).sum("x_dimension")
       .groupby_bins(
           "y_dimension",
	   pd.interval_range(
	       0,
	       y_index[-1] + uncertainty_spatial_resolution,
	       freq=uncertainty_spatial_resolution,
	       closed="left")
       ).sum("y_dimension")
       .resample(flux_time=uncertainty_temporal_resolution)
       .sum("flux_time")
   ).rename(
       x_dimension_bins="reduced_x_dimension",
       y_dimension_bins="reduced_y_dimension",
       flux_time="reduced_flux_time",
   )

   temporal_correlation_ds = xarray.DataArray(
       dims=("flux_time_adjoint", "flux_time"),
       values=temporal_correlations,
       coords=dict(
           flux_time_adjoint=flux_time_index.values,
	   flux_time=flux_time_index.values,
       ),
       name="temporal_correlations",
   )
   reduced_temporal_correlation_ds = (
       temporal_correlation_ds
       .resample(flux_time=uncertainty_temporal_resolution)
       .mean("flux_time")
       .resample(flux_time_adjoint=uncertainty_temporal_resolution)
       .mean("flux_time_adjoint")
   )
   reduced_covariance = inversion.linalg.DaskKroneckerProductOperator(
       reduced_temporal_correlation_ds.values,
       reduced_spatial_covariance,
   )

The results from :func:`~inversion.optimal_interpolation.save_sum` are
the combined flux estimate, which uses both the previous estimate and
the atmospheric measurements, and the uncertainty of that estimate
represented as a covariance matrix, at reduced resolution if
applicable.

.. code-block:: python

   combined_flux_estimate, reduced_uncertainty = (
       inversion.optimal_interpolation.save_sum(
           previous_surface_flux_estimate.stack(
	       state_space=(
	           "flux_time",
		   "y_dimension",
		   "x_dimension",
	       ),
	   ).values,
	   full_covariance,
	   observations.values,
	   observation_covariance,
	   influence_functions.stack(
	       state_space=(
	           "flux_time",
		   "y_dimension",
		   "x_dimension",
	       ),
	   ).transpose(
	       "observation",
	       "state_space",
	   ).values,
	   reduced_covariance,
	   reduced_influence_functions,
       )
   )

If there are multiple previous estimates for which the same
uncertainty estimate applies, including those estimates as the columns
of the first argument to
:func:`~inversion.optimal_interpolation.save_sum` will produce a
collection of combined estimates, with the columns again corresponding
to the different previous estimates.  This has only been tested with
corresponding columns in the observations; a simple way to obtain
these columns would be duplicating the values of the measurements in
each column of `observations` to match the columns in the previous
flux estimate.


.. rubric:: Footnotes

.. [#infl_fun_deriv] We're assuming here that the relationship between
                     a single flux and a single observation is linear.

.. [#matmul_op] :pep:`465` introduces `@` as the matrix multiplication
                operator.

.. [#transport_uncert] The assumption is that if we are unsure about
                       which fluxes impact which measurements, we
                       cannot use as much information from the
                       measurements to inform our estimates of the
                       fluxes.

.. [#theory] The various derivations in :ref:`the section on theory
             <theory>` derive similar forms for the covariance of the
             new flux estimate.
