===========================
What is an Inverse Problem?
===========================

Forward Problem
===============

Given an initial mole fraction field :math:`\chi` for some atmospheric
trace gas, its surface flux field :math:`f`, and the wind field
:math:`\vec{u}`, it is relatively straightforward to find the mixing
ratio field at all future points in time if the trace gas is almost
inert over the timescales of interest:

.. :math: \frac{\partial\chi}{\partial t} + \vec{u} \cdot \vec{\nabla} \chi = 0,
   :name: advection-equation

plus boundary conditions, at the surface relating :math:`f` to the
gradient of :math:`\chi`, and at the top of the domain of interest
:math:`\frac{\partial\chi}{\partial z} = 0` to ensure no flow through
the top of the domain.  If the domain of interest is not global, there
will be additional boundary conditions at the lateral boundaries.

Since the PDE above is linear, one can prove the existence and
uniqueness of the solutions.

Inverse problem
===============

The problem this package tries to address is related: given the same
equation for the evolution of :math:`\chi` and a finite number of
measurements at different points in time, find :math:`f`.  Since this
problem is nearly the reverse of the problem above, it is called an
"Inverse Problem".

Usually, for practical purposes, the flux field is projected onto a
finite-dimensional subspace, allowing the flux to be described as a
vector (:math:`\vec{f}`).  This vector is usually much larger than the
vector of measurements, so that the inverse problem is
underdetermined.

One way to deal with this is to introduce an estimate for the fluxes
obtained before looking at the measurements, :math:`\vec{f}_b`, and
minimize the difference between the measurements and the prediction
using the flux estimate.  If :math:`\vec{f}_b = \vec{0}`, this is
equivalent to using the pseudo-inverse of the transport to find the
fluxes.

This allows the problem to be solved, but the problem is often
ill-posed and the solution unstable: small changes in the measurements
can lead to large changes in the fluxes.  Introducing uncertainties
for the individual measurements and elements of the flux vector
addresses some of this problem [1]_, but can still lead to
discontinuities in the final flux field, which may not be realistic
given the underlying physical processes.  Introducing correlations
between the fluxes can alliviate that problem.  Correlations also make
the problem larger and longer to solve.

Initial and Boundary Conditions
-------------------------------

The forward problem requires specifying initial conditions, and if the
domain is not global, lateral boundary conditions also need to be
specified.  These can also be included in the inverse problem as
unknowns to be solved for, that is, as an additional part of the state
vector.  [Peylinetal2005]_ indicates that, for limited-area models,
the influence of the initial conditions largely vanishes after a month
or so, having been advected out of the domain, while the lateral
boundary conditions become correspondingly more important.

As with the fluxes, the concentration field at the lateral boundaries
of the spatial domain is usually discritized into a finite-dimensional
vector, which is appended to the flux vector to form the full state
vector.

The choice of basis functions is again arbitrary.  One possibility for
the lateral boundary conditions is to model the inflow as a constant
times the concentration field of a global model.  Another, which works
better for small domains, is to model the inflow as a constant.  A
third possibility is to again use the concentration field of a global
model but treat the edges of the domain (often north, south, east, and
west) separatly.  A fourth extends this idea, and splits the inflow
farther.  One might split the inflow into boundary-layer and
free-troposphere values in any of these schemes, or choose a finer
vertical resolution.



.. [1] Incidenally, this moves the procedure from Ordinary Least Squares to
       Weighted Least Squares.  The next step is Generalized Least Squares.

References
==========

.. [Peylinetal2005] Peylin, P.; P.J. Rayner; P. Bousquet; C.
                    Carouge; F.  Hourdin; P. Hinrich; P.  Ciais; and
                    AEROCARB contributors (2005).  "Daily CO2 flux
                    estimates over Europe from continuous atmospheric
                    measurements: 1, inverse methodology" *Atmospheric
                    Chemistry and Physics* vol. 5, no. 12
                    pp. 3173-3186 :doi:`10.5194/acp-5-3173-2005`
