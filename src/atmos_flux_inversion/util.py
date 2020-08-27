"""Utility functions for compatibility.

Some functions mirror :mod:`numpy` functions but produce :mod:`dask`
output.  Others map similar functionality across the various methods
to accomplish that end.
"""
from __future__ import absolute_import
import functools

import numpy as np
from numpy import atleast_1d, atleast_2d
from scipy.sparse.linalg import LinearOperator

from .linalg import DaskKroneckerProductOperator, kron, solve
from .linalg_interface import ProductLinearOperator, tolinearoperator

ARRAY_TYPES = (np.ndarray,)
"""Array types for determining Kronecker product type.

These are combined for a direct product.
"""
REAL_DTYPE_KINDS = "fiu"
"""The kinds used by dtypes to represent real numbers.

Includes subsets.
"""
MAX_EXPLICIT_ARRAY = 1 << 10
"""Maximum size for an array represented explicitly.

:func:`kronecker_product` will form products smaller than this as an
explicit matrix using :func:`.linalg.kron`.  Arrays larger than this will use
:class:`.linalg.DaskKroneckerProductOperator`.

Currently completely arbitrary.
`2 ** 16` works fine in memory, `2**17` gives a :class:`MemoryError`.
Hopefully Dask knows not to try this.
"""


def kronecker_product(operator1, operator2):
    """Form the Kronecker product of the given operators.

    Delegates to ``operator1.kron()`` if possible,
    :func:`.linalg.kron` if both are :const:`ARRAY_TYPES`, or
    :class:`~atmos_flux_inversion.linalg.SchmidtKroneckerProduct`
    otherwise.

    Parameters
    ----------
    operator1, operator2: ~scipy.sparse.linalg.LinearOperator
        The component operators of the Kronecker product.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        The kronecker product of the given operators.
    """
    if hasattr(operator1, "kron"):
        return operator1.kron(operator2)

    if isinstance(operator1, ARRAY_TYPES):
        if ((isinstance(operator2, ARRAY_TYPES) and
             operator1.size * operator2.size < MAX_EXPLICIT_ARRAY)):
            return kron(operator1, operator2)
        return DaskKroneckerProductOperator(operator1, operator2)
    from atmos_flux_inversion.linalg import SchmidtKroneckerProduct
    return SchmidtKroneckerProduct(operator1, operator2)


def method_common(inversion_method):  # noqa: C901
    """Wrap method to validate args.

    Can also deal with posterior uncertainty for a reduced-resolution
    domain, where the method opts not to provide that.

    Parameters
    ----------
    inversion_method: function
        The inversion function to wrap.

    Returns
    -------
    function
        The wrapped function.
    """
    try:
        from sparse import COO
    except ImportError:
        # Probably a terrible default, but it works.
        COO = LinearOperator

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
            The covariance for a smaller state space, usually obtained by
            reducing resolution in space and time.  Note that
            `reduced_observation_operator` must also be provided
        reduced_observation_operator: array_like[M, Nred], optional
            The relationship between the reduced state space and the
            observations.  Note that `reduced_background_covariance`
            must also be provided.

        Returns
        -------
        analysis: array_like[N]
            Analysis state estimate
        analysis_covariance: array_like[Nred, Nred] or array_like[N, N]
            Estimated uncertainty of analysis across
            realizations/ensemble members.  Calculated using
            reduced_background_covariance and
            reduced_observation_operator if possible

        Raises
        ------
        ValueError
            If only one of `reduced_background_covariance` and
            `reduced_observation_operator` is provided
        """
        _LinearOperator = LinearOperator  # noqa: C0103
        _COO = COO
        _ndarray = np.ndarray
        background = atleast_1d(background)
        if not isinstance(background_covariance, _LinearOperator):
            background_covariance = atleast_2d(background_covariance)

        observations = atleast_1d(observations)
        if not isinstance(observation_covariance, _LinearOperator):
            observation_covariance = atleast_2d(observation_covariance)

        if not isinstance(observation_operator, (_LinearOperator, _COO)):
            observation_operator = atleast_2d(observation_operator)

        if ((isinstance(background_covariance, _ndarray) and
             isinstance(observation_operator, _COO))):
            try:
                observation_operator = observation_operator.todense()
            except AttributeError:
                pass

        if (
            (reduced_background_covariance is None and
             reduced_observation_operator is not None) or
            (reduced_background_covariance is not None and
             reduced_observation_operator is None)
        ):
            raise ValueError("Need reduced versions of both B and H")

        if reduced_background_covariance is not None:
            if not isinstance(reduced_background_covariance, _LinearOperator):
                reduced_background_covariance = atleast_2d(
                    reduced_background_covariance)

            if not isinstance(reduced_observation_operator,
                              (_LinearOperator, _COO)):
                reduced_observation_operator = atleast_2d(
                    reduced_observation_operator)

            if ((isinstance(reduced_background_covariance, _ndarray) and
                 isinstance(reduced_observation_operator, _COO))):
                try:
                    reduced_observation_operator = (
                        reduced_observation_operator.todense())
                except AttributeError:
                    pass

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
            elif isinstance(reduced_background_covariance, _LinearOperator):
                decrease = tolinearoperator(decrease)

            # (I - KH) B
            analysis_covariance = (
                # May need to be a LinearOperator to work properly
                reduced_background_covariance -
                decrease
            )

        return analysis_estimate, analysis_covariance
    return wrapper
