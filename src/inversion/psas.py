"""Inversions using Physical-Space Assimilation System.

Iterative observation-space algorithm.
"""
import numpy as np
import numpy.linalg as la
import scipy.optimize

from inversion import ConvergenceError

MAX_ITERATIONS = 40
"""Max. iterations allowed during the minimization.

I think 40 is what the operational centers use.

Note
----
Must change test tolerances if this changes.
"""
GRAD_TOL = 1e-5
"""How small the gradient norm must be to declare convergence.

From `gtol` option to the BFGS method of
:func:`scipy.optimize.minimize`

Note
----
Must change test tolerances if this changes.
"""


def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator):
    """Simple PSAS implementation.

    Assumes all arrays fit in memory with room to spare.

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
    background = np.atleast_1d(background)
    background_covariance = np.atleast_2d(background_covariance)

    observations = np.atleast_1d(observations)
    observation_covariance = np.atleast_2d(observation_covariance)

    observation_operator = np.atleast_2d(observation_operator)

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        background_covariance.dot(observation_operator.T))

    covariance_sum = projected_background_covariance + observation_covariance

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
    lower = la.cholesky(result.hess_inv)
    lower_decrease = background_covariance.dot(
        observation_operator.T.dot(lower))
    # this will be positive
    decrease = lower_decrease.dot(lower_decrease.T)
    # this may not
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)

    return analysis, analysis_covariance


def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator):
    """Simple direct matrix inversion.

    Assumes all arrays fit in memory with room to spare.

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

    """
    background = np.atleast_1d(background)
    background_covariance = np.atleast_2d(background_covariance)

    observations = np.atleast_1d(observations)
    observation_covariance = np.atleast_2d(observation_covariance)

    observation_operator = np.atleast_2d(observation_operator)

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    B_HT = background_covariance.dot(observation_operator.T)
    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        B_HT)

    covariance_sum = projected_background_covariance + observation_covariance

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
    analysis = background + background_covariance.dot(
        observation_operator.T.dot(analysis_increment))

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    # analysis_covariance = (background_covariance -
    #                        B_HT.dot(
    #                            result.hess_inv.dot(
    #                                B_HT.T)))

    # Try a different approach to enforce invariants (symmetric
    # positive definite)
    lower = la.cholesky(result.hess_inv)
    lower_decrease = B_HT.dot(lower)
    # this will be positive
    decrease = lower_decrease.dot(lower_decrease.T)
    # this may not
    analysis_covariance = background_covariance - decrease

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis, analysis_covariance)
    return analysis, analysis_covariance
