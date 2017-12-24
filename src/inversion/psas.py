"""Inversions using Physical-Space Assimilation System.

Iterative observation-space algorithm.
"""
import numpy as np
from dask.array import asarray
from scipy.linalg import cholesky
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
# I believe scipy's minimizer requires things that give boolean true
# or false from the objective, rather than a yet-to-be-realized dask
# array.
from inversion.util import atleast_1d, atleast_2d

from inversion import ConvergenceError, MAX_ITERATIONS, GRAD_TOL
from inversion.util import tolinearoperator, ProductLinearOperator


def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator):
    """Solve the inversion problem, using the equations directly.

    Assumes all arrays fit in memory with room to spare.
    This uses the algorithm from
    :func:`inversion.optimal_interpolation.simple`, except
    the matrix inversion is done with an iterative solver.

    Parameters
    ----------
    background: np.ndarray[N]
    background_covariance:  np.ndarray[N,N]
    observations: np.ndarray[M]
    observation_covariance: np.ndarray[M,M]
    observation_operator: np.ndarray[M,N]

    Returns
    -------
    analysis: np.ndarray[N]
    analysis_covariance: np.ndarray[N,N]

    Note
    ----
    Performs the matrix inversion in the Kalman gain

    .. math::

        K = B H^T (HBH^T + R)^{-1}

    approximately, with an iterative algorithm.
    There is an approximation to the analysis covariance, but it is very bad.
    """
    background = atleast_1d(background)
    if not isinstance(background_covariance, LinearOperator):
        background_covariance = atleast_2d(background_covariance)
        bg_is_arry = True
    else:
        bg_is_arry = False

    observations = np.atleast_1d(observations)
    if not isinstance(observation_covariance, LinearOperator):
        observation_covariance = atleast_2d(observation_covariance)
        obs_is_arry = True
    else:
        obs_is_arry = False

    if not isinstance(observation_operator, LinearOperator):
        observation_operator = atleast_2d(observation_operator)
        obs_op_is_arry = True
    else:
        obs_op_is_arry = False

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    if obs_op_is_arry:
        projected_background_covariance = observation_operator.dot(
            background_covariance.dot(observation_operator.T))
    else:
        projected_background_covariance = ProductLinearOperator(
            observation_operator,
            tolinearoperator(background_covariance),
            observation_operator.T)

    if (obs_op_is_arry ^ obs_is_arry):
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    # Solving A x = 0 and minimizing x^T A x give the same answer
    # solving A x = b gives the same answer as minimizing x^T A x - b^T x
    def cost_function(test_observation_increment):
        """Mismatch between prior and obs in obs space.

        Parameters
        ----------
        test_observation_increment: np.ndarray[M]

        Returns
        -------
        cost: float
        """
        return (
            0.5 *
            test_observation_increment.dot(covariance_sum.dot(
                    test_observation_increment)) -
            observation_increment.dot(
                test_observation_increment)
            )

    def cost_jacobian(test_observation_increment):
        """Gradient of cost function at `test_observation_increment`.

        Parameters
        ----------
        test_observation_increment: np.ndarray[M]

        Returns
        -------
        jac: np.ndarray[M]
        """
        return (covariance_sum.dot(test_observation_increment) -
                observation_increment)

    result = scipy.optimize.minimize(
        cost_function, observation_increment,
        method="BFGS",
        jac=cost_jacobian,
        # hess=covariance_sum,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis_increment = result.x

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + background_covariance.dot(
        observation_operator.T.dot(analysis_increment))

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    # analysis_covariance = (background_covariance -
    #                        background_covariance.dot(
    #                            observation_operator.T.dot(
    #                                result.hess_inv.dot(
    #                                    observation_operator).dot(
    #                                        background_covariance))))

    # Try a different approach to enforce invariants (symmetric
    # positive definite)
    lower = cholesky(asarray(result.hess_inv))
    lower_decrease = background_covariance.dot(
        observation_operator.T.dot(lower))
    # this will be positive
    decrease = lower_decrease.dot(lower_decrease.T)
    # this may not
    if not bg_is_arry:
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)

    return analysis, analysis_covariance


def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator):
    """Solve the inversion problem, in a slightly optimized manner.

    Assumes all arrays fit in memory with room to spare.  Evaluates
    each sub-expression only once. Uses the algorithm from
    :func:`inversion.optimal_interpolation.fold_common` with an
    iterative solver for the matrix inversion.

    Parameters
    ----------
    background: np.ndarray[N]
    background_covariance:  np.ndarray[N,N]
    observations: np.ndarray[M]
    observation_covariance: np.ndarray[M,M]
    observation_operator: np.ndarray[M,N]

    Returns
    -------
    analysis: np.ndarray[N]
    analysis_covariance: np.ndarray[N,N]

    Note
    ----
    Performs the matrix inversion in the Kalman gain

    .. math::

        K = B H^T (HBH^T + R)^{-1}

    approximately, with an iterative algorithm.
    There is an approximation to the analysis covariance, but it is very bad.
    """
    background = atleast_1d(background)
    if not isinstance(background_covariance, LinearOperator):
        background_covariance = atleast_2d(background_covariance)
        bg_is_arry = True
    else:
        bg_is_arry = False

    observations = np.atleast_1d(observations)
    if not isinstance(observation_covariance, LinearOperator):
        observation_covariance = atleast_2d(observation_covariance)
        obs_is_arry = True
    else:
        obs_is_arry = False

    if not isinstance(observation_operator, LinearOperator):
        observation_operator = atleast_2d(observation_operator)
        obs_op_is_arry = True
    else:
        obs_op_is_arry = False

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    if obs_op_is_arry:
        B_HT = background_covariance.dot(observation_operator.T)

        projected_background_covariance = (
            observation_operator.dot(B_HT))
    else:
        B_HT = tolinearoperator(background_covariance).dot(
            observation_operator.T)

        projected_background_covariance = ProductLinearOperator(
            observation_operator, tolinearoperator(background_covariance),
            observation_operator.T)

    if obs_op_is_arry ^ obs_is_arry:
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    # Solving A x = 0 and minimizing x^T A x give the same answer
    # solving A x = b gives the same answer as minimizing x^T A x - b^T x
    def cost_function(test_observation_increment):
        """Mismatch between prior and obs in obs space.

        Parameters
        ----------
        test_observation_increment: np.ndarray[M]

        Returns
        -------
        cost: float
        """
        return .5 * test_observation_increment.dot(covariance_sum.dot(
            test_observation_increment)) - observation_increment.dot(
                test_observation_increment)

    def cost_jacobian(test_observation_increment):
        """Gradient of cost function at `test_observation_increment`.

        Parameters
        ----------
        test_observation_increment: np.ndarray[M]

        Returns
        -------
        jac: np.ndarray[M]
        """
        return (covariance_sum.dot(test_observation_increment) -
                observation_increment)

    result = scipy.optimize.minimize(
        cost_function, observation_increment,
        method="BFGS",
        jac=cost_jacobian,
        # hess=covariance_sum,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis_increment = result.x

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + B_HT.dot(analysis_increment)

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    # analysis_covariance = (background_covariance -
    #                        B_HT.dot(
    #                            result.hess_inv.dot(
    #                                B_HT.T)))

    # Try a different approach to enforce invariants (symmetric
    # positive definite)
    lower = cholesky(asarray(result.hess_inv), lower=True)
    lower_decrease = B_HT.dot(lower)
    # this will be positive
    decrease = lower_decrease.dot(lower_decrease.T)
    # this may not
    if not bg_is_arry:
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)
    return analysis, analysis_covariance
