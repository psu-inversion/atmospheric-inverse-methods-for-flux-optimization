==================================
Flux representation for inversions
==================================

The relationship between the state vector and the flux in each
transport model grid cell can vary greatly between inversions.

Influence functions :math:`H` for direct and scaling factor inversions
are most easily found by some variety of adjoint transport model.
Those for larger basis functions can often be found more readily by
repeated forward integrations of the transport model, as in TRANSCOM.

Direct
======

Each entry in the state vector :math:`\vec{x} = \vec{f}` is the flux
in a different grid cell.  The influence function :math:`H = H_{flux}`
determines the change in each measurement due to a one-unit change in
flux for each grid cell.

Also called "additive".

Scaling Factor
==============

Each entry in the state factor is a multiplier for the fluxes in each
grid cell.  One obtaines this by making the prior state vector a
vector of ones (i.e. :math:`\vec{x} = \vec{1}`) and moving the prior
fluxes to the influence functions (:math:`H = H_{flux}
diag(\vec{f}_{prior})`).  The fluxes can be found as :math:`\vec{f} =
\vec{x} \otimes \vec{f}_{prior}`

Larger Basis Functions
======================

The basis functions are often chosen not to nonoverlap in flux space,
with spatial boundaries corresponding to TRANSCOM regions or
ecoregions.  Another method for choosing basis functions is to find
Empirical Orthogonal Functions for the fluxes.

A matrix :math:`F` is chosen to transform the state vector into the
fluxes: :math:`\vec{f} = F \vec{x}`.  The usual choice is to make
:math:`\vec{x}` much smaller than :math:`\vec{f}`, and to make
:math:`F^T F = I`.  The influence functions can then be found as
:math:`H = H_{flux} F`.
