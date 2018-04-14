"""Inversions using Physical-Space Assimilation System.

Iterative observation-space algorithm.
"""
from dask.array import asarray
from dask.array.linalg import cholesky
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
# I believe scipy's minimizer requires things that give boolean true
# or false from the objective, rather than a yet-to-be-realized dask
# array.

from inversion import ConvergenceError, MAX_ITERATIONS, GRAD_TOL
from inversion.util import tolinearoperator, ProductLinearOperator
from inversion.util import method_common


@method_common
def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator,
           reduced_background_covariance=None,
           reduced_observation_operator=None):
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
    reduced_background_covariance: array_like[Nred, Nred], optional
    reduced_observation_operator: array_like[M, Nred], optional

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
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    if not isinstance(observation_operator, LinearOperator):
        projected_background_covariance = observation_operator.dot(
            background_covariance.dot(observation_operator.T))
    else:
        projected_background_covariance = ProductLinearOperator(
            observation_operator,
            tolinearoperator(background_covariance),
            observation_operator.T)

    if ((isinstance(projected_background_covariance, LinearOperator) ^
         isinstance(observation_covariance, LinearOperator))):
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

    if reduced_background_covariance is None:
        method = "BFGS"
    else:
        # The estimate from the BFGS minimization does terribly in the
        # tests.  I feel it safer to do it analytically at low
        # resolution than trying to use the high-resolution iterative
        # approximation.
        method = "CG"

    result = scipy.optimize.minimize(
        cost_function, observation_increment,
        method=method,
        jac=cost_jacobian,
        # hess=covariance_sum,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis_increment = result.x

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + background_covariance.dot(
        observation_operator.T.dot(analysis_increment))

    if reduced_background_covariance is not None:
        if not result.sucess:
            raise ConvergenceError("Did not converge: {msg:s}".format(
                msg=result.message), result, analysis, None)
        return analysis, None

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
    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)

    return analysis, analysis_covariance


@method_common
def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                reduced_background_covariance=None,
                reduced_observation_operator=None):
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
    reduced_background_covariance: array_like[Nred, Nred], optional
    reduced_observation_operator: array_like[M, Nred], optional

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
    obs_op_is_arry = not isinstance(observation_operator, LinearOperator)
    obs_is_arry = not isinstance(observation_covariance, LinearOperator)
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

    if reduced_background_covariance is None:
        method = "BFGS"
    else:
        method = "CG"

    result = scipy.optimize.minimize(
        cost_function, observation_increment,
        method=method,
        jac=cost_jacobian,
        # hess=covariance_sum,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis_increment = result.x

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + B_HT.dot(analysis_increment)

    if reduced_background_covariance is not None:
        if not result.success:
            raise ConvergenceError("Did not converge: {msg:s}".format(
                msg=result.message), result, analysis, None)
        return analysis, None

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
    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)

    return analysis, analysis_covariance
