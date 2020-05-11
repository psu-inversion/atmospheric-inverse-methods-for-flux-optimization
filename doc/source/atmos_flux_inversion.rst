atmos_flux_inversion package
============================

Submodules
----------

Classes for defining covariances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

   atmos_flux_inversion.correlations
   atmos_flux_inversion.covariances

The classes provided by :mod:`pylops` may also be useful.

Inversion functions
~~~~~~~~~~~~~~~~~~~

All inversion functions have the same signature and give similar
answers: the difference is how they get there.  PSAS and Variational
methods use iterative solvers. Optimal Interpolation uses a
Gauss-Jordan solver.  Variational methods use a different but
equivalent formulation of the problem.

.. toctree::
   atmos_flux_inversion.optimal_interpolation
   atmos_flux_inversion.variational
   atmos_flux_inversion.psas

High-level wrappers
~~~~~~~~~~~~~~~~~~~

.. toctree::
   atmos_flux_inversion.wrapper

Other utilities
~~~~~~~~~~~~~~~

.. toctree::
   atmos_flux_inversion.linalg
   atmos_flux_inversion.noise
   atmos_flux_inversion.util

Module contents
---------------

.. automodule:: atmos_flux_inversion
    :members:
    :undoc-members:
    :show-inheritance:
