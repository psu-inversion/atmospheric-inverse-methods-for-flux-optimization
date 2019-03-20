inversion package
=================

Submodules
----------

Classes for defining covariances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

   inversion.correlations
   inversion.covariances

The classes provided by :mod:`pylops` may also be useful.

Inversion functions
~~~~~~~~~~~~~~~~~~~

All inversion functions have the same signature and give similar
answers: the difference is how they get there.  PSAS and Variational
methods use iterative solvers for linear systems, where Optimal
Interpolation uses Gauss-Jordan elimination.  Variational methods use
a different but equivalent formulation of the problem.

.. toctree::
   inversion.optimal_interpolation
   inversion.variational
   inversion.psas

High-level wrappers
~~~~~~~~~~~~~~~~~~~

.. toctree::
   inversion.wrapper

Helpers for real-data inversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helper functions to calculate the prior error covariance and
observation operator at reduced resolution so the inversion routines
can calculate the posterior error covariance in reasonable time and in
a reasonable amount of storage.

.. toctree::
   inversion.observation_operator
   inversion.remapper

Other utilities
~~~~~~~~~~~~~~~

.. toctree::
   inversion.linalg
   inversion.noise
   inversion.util

Module contents
---------------

.. automodule:: inversion
    :members:
    :undoc-members:
    :show-inheritance:
