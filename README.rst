Atmospheric Inverse Methods for Flux Optimization
=================================================

Python package containing functions for the application of inverse
methods to the optimization of surface fluxes to be consistent with
atmospheric observations.

My use-case is primarily continental-scale biological carbon dioxide
flux optimization using atmospheric carbon dioxide mole fraction
observations.  A paper with more details is in preparation.

Similar work is being done, using similar methods with a different
approach, by the `NOAA/GMD CarbonTracker-Lagrange Inversion code
<https://www.esrl.noaa.gov/gmd/ccgg/carbontracker-lagrange/doc/intro.html>`_.
This code is designed to be run from within Python, where theirs is
designed as a series of scripts to be run from the command line.  I
feel the flexibility from the data structures I chose to use,
specifically inheriting from classes based on `scipy's LinearOperators
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
allows greater flexibility in what this code can do.

Other software packages in Python that tackle similar problems include
`Data Assimilation with Python: a Package for Experimental Research
(DAPPER) <https://github.com/nansencenter/DAPPER>`_ and `Python
Observing System Simulation Experiments (PyOSSE)
<https://www.geos.ed.ac.uk/~lfeng/>`_, both of which have more focus
on identical-twin OSSEs and Ensemble Kalman Filters.
These packages do not use standard Python packaging frameworks to
specify dependencies, and my reasons for prefering my package to the
CT-Lagrange inversion code also apply here.

Installation
------------

.. code::

    pip install "git+https://github.com/psu-inversion/atmospheric-inverse-methods-for-flux-optimization.git"

Dependencies
------------

To install all dependencies required for development of the code, use

.. code::

   pip install -r requirements.txt

Testing
-------

To run all tests:

.. code::

    tox
    
To run only code correctness tests for the current interpreter:

.. code::

    python setup.py test

License
-------

By downloading this software you agree to the terms of use of the
`3-clause BSD license <LICENSE.txt>`_.
