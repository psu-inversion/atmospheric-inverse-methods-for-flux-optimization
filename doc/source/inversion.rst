inversion package
=================

Submodules
----------

Classes for defining covariances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

   inversion.correlations
   inversion.covariances

Inversion functions
~~~~~~~~~~~~~~~~~~~

All inversion functions have the same signature and give similar
answers: the difference is how they get there.  PSAS and Variational
methods use iterative solvers. Optimal Interpolation uses a
Gauss-Jordan solver.  Variational methods use a different but
equivalent formulation of the problem.

.. toctree::
   inversion.optimal_interpolation
   inversion.variational
   inversion.psas

Other utility functions
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   inversion.covariance_estimation
   inversion.linalg
   inversion.noise
   inversion.util

Module contents
---------------

.. automodule:: inversion
    :members:
    :undoc-members:
    :show-inheritance:
