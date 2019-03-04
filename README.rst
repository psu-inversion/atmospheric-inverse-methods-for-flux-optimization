Atmospheric Inverse Methods for Flux Optimization
=================================================

Python package containing functions for the application of inverse
methods to the optimization of surface fluxes to be consistent with
atmospheric observations.

My use-case is primarily continental-scale biological carbon dioxide
flux optimization using atmospheric carbon dioxide mole fraction
observations.  A paper with more details is planned.

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
