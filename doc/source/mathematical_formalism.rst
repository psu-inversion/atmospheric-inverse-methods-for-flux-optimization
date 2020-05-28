.. _theory:

======================================================
Mathematical Formalism for Atmospheric Flux Inversions
======================================================

Atmospheric flux inversions attempt to find what fluxes
:math:`\vec{x}_a` are consistent with a set of atmospheric
observations :math:`\vec{y}` that are related by the observation
operator :math:`H` via :math:`\vec{y} \approx H \vec{x}_a`.

Since there tend to be fewer observations than there are fluxes of
interest, we need a previous estimate of the fluxes,
:math:`\vec{x}_b`, to regularize the problem so it is solvable.

Terminology
===========

:math:`\vec{x}_a` is also called the mean of the
posterior distribution or the analysis.

The previous esitmate :math:`\vec{x}_b` is also called the background,
the mean of the prior distribution, or, in other fields, the forecast.

The rows of the observation operator :math:`H` are sometimes called
influence functions, since they indicate how much influence each flux
has on a specific observation.  :math:`H` is also referred to as the
transport matrix in some contexts, since it is based on how air is
transported from the fluxes to where it forms the observations.

The estimate of the uncertainty in the previous estimate :math:`B` is
also called the background error covariance matrix or the covariance
of the prior distribution.

The matrix :math:`K = B H^T (H B H^T + R)^{-1} = (B^{-1} + H^T R^{-1}
H)^{-1} H^T R^{-1}` is called the Kalman gain, and is called
:math:`A_2` in :ref:`the derivation for the best linear unbiased
estimator below <blue-derivation>`

:math:`A`, the estimate of the uncertainty in :math:`\vec{x}_a`, is
called the analysis error covariance matrix or the covariance of the
posterior distribution.

Methodology
===========
There are a few different ways to derive the equations used to
optimize the fluxes.

.. toctree::
   :maxdepth: 1

   best_linear_unbiased_estimator
   generalized_least_squares
   maximum_likelihood
   bayes
