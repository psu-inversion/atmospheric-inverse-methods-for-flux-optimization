"""Utility functions for compatibility.

These functions mirror :mod:`numpy` functions but produce dask output.
"""
from __future__ import absolute_import
import functools

import numpy as np
from scipy.sparse.linalg import LinearOperator

import dask.array as da
from numpy import atleast_1d, atleast_2d

from .linalg import DaskKroneckerProductOperator, kron, solve
from .linalg_interface import ProductLinearOperator, tolinearoperator

OPTIMAL_ELEMENTS = int(2e5)
"""Optimal elements per chunk in a dask array.

Magic number, arbitrarily chosen.  Dask documentation mentions many
chunks should fit easily in memory, but each should contain at least a
million elements, recommending 10-100MiB per chunk.  This size matrix
is fast to allocate and fill, but :math:`10^5` gives a memory error.
A square matrix of float64 with ten thousand elements on a side is 762
megabytes.

A single level of our domain is 4.6e4 elements.  The calculation
proceeds much more naturally when this fits in a chunk, since it needs
to for the FFTs.  This would be for OPTIMAL_ELEMENTS**2.

Leaving this as 1e4 causes memory errors and deadlocks over an hour
and a half.  5e4 can do the same program twice in ten minutes.
I don't entirely understand how this works.

I'm going to say these problems have little use for previous results,
so this can be larger than the dask advice.  This greatly reduces the
requirements for setting up the graph.

4e4 works for both, I think.  BE VERY CAREFUL CHANGING THIS!!
"""
ARRAY_TYPES = (np.ndarray, da.Array)
"""Array types for determining Kronecker product type.

These are combined for a direct product.
"""
REAL_DTYPE_KINDS = "fiu"
"""The kinds used by dtypes to represent real numbers.

Includes subsets.
"""
MAX_EXPLICIT_ARRAY = 1 << 25
"""Maximum size for an array represented explicitly.

:func:`kronecker_product` will form products smaller than this as an
explicit matrix using :func:`kron`.  Arrays larger than this will use
:class:`DaskKroneckerProduct`.

Currently completely arbitrary.
`2 ** 16` works fine in memory, `2**17` gives a MemoryError.
Hopefully Dask knows not to try this.
"""


def kronecker_product(operator1, operator2):
    """Form the Kronecker product of the given operators.

    Delegates to ``operator1.kron()`` if possible,
    :func:`kron` if both are :const:`ARRAY_TYPES`, or
    :class:`inversion.correlations.SchmidtKroneckerProduct` otherwise.

    Parameters
    ----------
    operator1, operator2: scipy.sparse.linalg.LinearOperator
        The component operators of the Kronecker product.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
    """
    if hasattr(operator1, "kron"):
        return operator1.kron(operator2)
    elif isinstance(operator1, ARRAY_TYPES):
        if ((isinstance(operator2, ARRAY_TYPES) and
             # TODO: test this
             operator1.size * operator2.size < MAX_EXPLICIT_ARRAY)):
            return kron(operator1, operator2)
        return DaskKroneckerProductOperator(operator1, operator2)
    from inversion.correlations import SchmidtKroneckerProduct
    return SchmidtKroneckerProduct(operator1, operator2)


def method_common(inversion_method):
    """Wrap method to validate args.

    Parameters
    ----------
    inversion_method: function

    Returns
    -------
    wrapped_method: function
    """
    @functools.wraps(inversion_method)
    def wrapper(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                reduced_background_covariance=None,
                reduced_observation_operator=None):
        """Solve the inversion problem.

        Assumes everything follows a multivariate normal distribution
        with the specified covariance matrices.  Under this assumption
        `analysis_covariance` is exact, and `analysis` is the Maximum
        Likelihood Estimator and the Best Linear Unbiased Estimator
        for the underlying state in the frequentist framework, and
        specify the posterior distribution for the state in the
        Bayesian framework.  If these are not satisfied, these still
        form the Generalized Least Squares estimates for the state and
        an estimated uncertainty.

        Parameters
        ----------
        background: array_like[N]
            The background state estimate.
        background_covariance:  array_like[N, N]
            Covariance of background state estimate across
            realizations/ensemble members.  "Ensemble" is here
            interpreted in the sense used in statistical mechanics or
            frequentist statistics, and may not be derived from a
            sample as in meteorological ensemble Kalman filters
        observations: array_like[M]
            The observations constraining the background estimate.
        observation_covariance: array_like[M, M]
            Covariance of observations across realizations/ensemble
            members.  "Ensemble" again has the statistical meaning.
        observation_operator: array_like[M, N]
            The relationship between the state and the observations.
        reduced_background_covariance: array_like[Nred, Nred], optional
        reduced_observation_operator: array_like[M, Nred], optional

        Returns
        -------
        analysis: array_like[N]
            Analysis state estimate
        analysis_covariance: array_like[Nred, Nred] or array_like[N, N]
            Estimated uncertainty of analysis across
            realizations/ensemble members.  Calculated using
            reduced_background_covariance and
            reduced_observation_operator if possible
        """
        _LinearOperator = LinearOperator
        background = atleast_1d(background)
        if not isinstance(background_covariance, _LinearOperator):
            background_covariance = atleast_2d(background_covariance)

        observations = atleast_1d(observations)
        if not isinstance(observation_covariance, _LinearOperator):
            observation_covariance = atleast_2d(observation_covariance)

        if not isinstance(observation_operator, _LinearOperator):
            observation_operator = atleast_2d(observation_operator)

        if reduced_background_covariance is not None:
            if not isinstance(reduced_background_covariance, LinearOperator):
                reduced_background_covariance = atleast_2d(
                    reduced_background_covariance)

            if reduced_observation_operator is None:
                raise ValueError("Need reduced versions of both B and H")
            if not isinstance(reduced_observation_operator, LinearOperator):
                reduced_observation_operator = atleast_2d(
                    reduced_observation_operator)
        elif reduced_observation_operator is not None:
            raise ValueError("Need reduced versions of both B and H")

        analysis_estimate, analysis_covariance = (
            inversion_method(background, background_covariance,
                             observations, observation_covariance,
                             observation_operator,
                             reduced_background_covariance,
                             reduced_observation_operator))

        if analysis_covariance is None:
            if isinstance(observation_operator, _LinearOperator):
                B_HT = ProductLinearOperator(reduced_background_covariance,
                                             reduced_observation_operator.T)
                projected_reduced_background_covariance = (
                    ProductLinearOperator(
                        reduced_observation_operator,
                        reduced_background_covariance,
                        reduced_observation_operator.T))
            else:
                # H is an array
                B_HT = reduced_background_covariance.dot(
                    reduced_observation_operator.T)
                projected_reduced_background_covariance = (
                    reduced_observation_operator.dot(
                        B_HT))
            if isinstance(observation_covariance,
                          _LinearOperator):
                projected_reduced_background_covariance = tolinearoperator(
                    projected_reduced_background_covariance)

            decrease = B_HT.dot(solve(
                projected_reduced_background_covariance +
                observation_covariance,
                B_HT.T))
            if isinstance(decrease, _LinearOperator):
                reduced_background_covariance = tolinearoperator(
                    reduced_background_covariance)

            # (I - KH) B
            analysis_covariance = (
                # May need to be a LinearOperator to work properly
                reduced_background_covariance -
                decrease
            )

        return analysis_estimate, analysis_covariance
    return wrapper
