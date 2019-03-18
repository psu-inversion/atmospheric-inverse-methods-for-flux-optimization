====================
Inversion User Guide
====================

We are interested in using a set of :math:`M` atmospheric measurements
to obtain an estimate for a set of :math:`N` surface fluxes.

To start off, we need some previous estimate of the fluxes of interest,
the set of atmospheric measurements, and the relationship between the
two.  I will assume these are stored in variables
`prior_surface_flux`, `observations`, and `influence_functions`
[#infl_fun_deriv]_, respectively.

In an ideal world, we would have `observations == influence_functions
@ prior_surface_flux` [#matmul_op]_, but unfortunately this is probably
not the case.  We would like to refine `prior_surface_flux` so this is
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

..
    TODO: describe how I chose temporal correlations

We are left with the spatial correlations.  I have seen no evidence
that the spatial correlations are related to Plant Functional Type, so
I use correlations that are a function only of distance.
:class:`~inversion.correlations.HomogeneousIsotropicCorrelation.from_function`
makes this somewhat easier, and is designed to work with instances of
:class:`~inversion.correlations.DistanceCorrelationFunction`.

The measurements tend to be somewhat less regular than the previous
flux estimate, so that covariance matrix is best constructed manually.

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
