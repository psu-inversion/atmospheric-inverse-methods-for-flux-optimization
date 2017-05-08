"""Functions implementing 3D-Var.

Signatures follow the functions in :mod:`inversion.optimal_interpolation`
"""
import numpy as np
import scipy.optimize
import scipy.linalg

from inversion import ConvergenceError, MAX_ITERATIONS, GRAD_TOL


def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator):
    """Feed everything to scipy's minimizer.

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
    minimizes

    .. math::

        (x - x_0)^T P_B^{-1} (x - x_0) + (y - h(x))^T R^{-1} (y - h(x))

    which has gradient

    .. math::

        P_B^{-1} (x - x_0) + H^T R^{-1} (y - h(x))
    """
    background = np.atleast_1d(background)
    background_covariance = np.atleast_2d(background_covariance)

    observations = np.atleast_1d(observations)
    observation_covariance = np.atleast_2d(observation_covariance)
    observation_operator = np.atleast_2d(observation_operator)

    def cost_function(test_state):
        """Mismatch between state, prior, and obs.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        cost: float
        """
        prior_mismatch = test_state - background
        test_obs = observation_operator.dot(test_state)
        obs_mismatch = test_obs - observations

        prior_fit = prior_mismatch.dot(np.linalg.solve(
            background_covariance, prior_mismatch))
        obs_fit = obs_mismatch.dot(np.linalg.solve(
            observation_covariance, obs_mismatch))
        return prior_fit + obs_fit

    def cost_jacobian(test_state):
        """Gradiant of cost_function at `test_state`.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        jac: np.ndarray[N]
        """
        prior_mismatch = test_state - background
        test_obs = observation_operator.dot(test_state)
        obs_mismatch = test_obs - observations

        prior_gradient = np.linalg.solve(background_covariance,
                                         prior_mismatch)
        obs_gradient = observation_operator.T.dot(
            np.linalg.solve(observation_covariance,
                            obs_mismatch))

        return prior_gradient + obs_gradient

    # def cost_hessian_product(test_state, test_step):
    #     """Hessian of cost_function at `test_state` times `test_step`.

    #     Parameters
    #     ----------
    #     test_state: np.ndarray[N]
    #     test_step: np.ndarray[N]

    #     Results
    #     -------
    #     hess_prod: np.ndarray[N]
    #     """
    #     bg_prod = np.linalg.solve(background_covariance,
    #                               test_step)
    #     obs_prod = observation_operator.T.dot(
    #         np.linalg.solve(observation_covariance,
    #                         observation_operator.dot(test_step)))
    #     return bg_prod + obs_prod

    result = scipy.optimize.minimize(
        cost_function, background,
        method="BFGS",
        jac=cost_jacobian,
        # hessp=cost_hessian_product,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result)

    return result.x, result.hess_inv


def incremental(background, background_covariance,
                observations, observation_covariance,
                observation_operator):
    """Feed everything to scipy's minimizer.

    Use the change from the background to try to avoid precision loss.

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
    minimizes

    .. math::

        (dx)^T P_B^{-1} (dx) + (y - h(x_0) - H dx)^T R^{-1} (y - h(x_0) - H dx)

    which has gradient

    .. math::

        P_B^{-1} (dx) - H^T R^{-1} (y - h(x) - H dx)

    where :math:`x = x_0 + dx`
    """
    background = np.atleast_1d(background)
    background_covariance = np.atleast_2d(background_covariance)

    observations = np.atleast_1d(observations)
    observation_covariance = np.atleast_2d(observation_covariance)
    observation_operator = np.atleast_2d(observation_operator)

    innovations = observations - observation_operator.dot(background)

    def cost_function(test_change):
        """Mismatch between state, prior, and obs.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        cost: float
        """
        obs_change = observation_operator.dot(test_change)
        obs_mismatch = innovations - obs_change

        prior_fit = test_change.dot(np.linalg.solve(
            background_covariance, test_change))
        obs_fit = obs_mismatch.dot(np.linalg.solve(
            observation_covariance, obs_mismatch))
        return prior_fit + obs_fit

    def cost_jacobian(test_change):
        """Gradiant of cost_function at `test_change`.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        jac: np.ndarray[N]
        """
        obs_change = observation_operator.dot(test_change)
        obs_mismatch = innovations - obs_change

        prior_gradient = np.linalg.solve(background_covariance,
                                         test_change)
        obs_gradient = observation_operator.T.dot(
            np.linalg.solve(observation_covariance,
                            obs_mismatch))

        return prior_gradient - obs_gradient

    # def cost_hessian_product(test_state, test_step):
    #     """Hessian of cost_function at `test_state` times `test_step`.

    #     Parameters
    #     ----------
    #     test_state: np.ndarray[N]
    #     test_step: np.ndarray[N]

    #     Results
    #     -------
    #     hess_prod: np.ndarray[N]
    #     """
    #     bg_prod = np.linalg.solve(background_covariance,
    #                               test_step)
    #     obs_prod = observation_operator.T.dot(
    #         np.linalg.solve(observation_covariance,
    #                         observation_operator.dot(test_step)))
    #     return bg_prod + obs_prod

    result = scipy.optimize.minimize(
        cost_function, np.zeros_like(background),
        method="BFGS",
        jac=cost_jacobian,
        # hessp=cost_hessian_product,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis = background + result.x

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis)

    return analysis, result.hess_inv


def incr_chol(background, background_covariance,
              observations, observation_covariance,
              observation_operator):
    """Feed everything to scipy's minimizer.

    Use the change from the background to try to avoid precision loss.
    Also use Cholesky factorization of the covariances to speed
    solution of matrix equations.

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
    minimizes

    .. math::

        (dx)^T P_B^{-1} (dx) + (y - h(x_0) - H dx)^T R^{-1} (y - h(x_0) - H dx)

    which has gradient

    .. math::

        P_B^{-1} (dx) - H^T R^{-1} (y - h(x) - H dx)

    where :math:`x = x_0 + dx`

    """
    background = np.atleast_1d(background)
    background_covariance = np.atleast_2d(background_covariance)

    observations = np.atleast_1d(observations)
    observation_covariance = np.atleast_2d(observation_covariance)
    observation_operator = np.atleast_2d(observation_operator)

    innovations = observations - observation_operator.dot(background)

    # factor the covariances to make the matrix inversions faster
    bg_cov_chol_u = scipy.linalg.cho_factor(background_covariance)
    obs_cov_chol_u = scipy.linalg.cho_factor(observation_covariance)

    from scipy.linalg import cho_solve

    def cost_function(test_change):
        """Mismatch between state, prior, and obs.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        cost: float
        """
        obs_change = observation_operator.dot(test_change)
        obs_mismatch = innovations - obs_change

        prior_fit = test_change.dot(cho_solve(
            bg_cov_chol_u, test_change))
        obs_fit = obs_mismatch.dot(cho_solve(
            obs_cov_chol_u, obs_mismatch))
        return prior_fit + obs_fit

    def cost_jacobian(test_change):
        """Gradiant of cost_function at `test_change`.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        jac: np.ndarray[N]
        """
        obs_change = observation_operator.dot(test_change)
        obs_mismatch = innovations - obs_change

        prior_gradient = cho_solve(bg_cov_chol_u,
                                   test_change)
        obs_gradient = observation_operator.T.dot(
            cho_solve(obs_cov_chol_u,
                      obs_mismatch))

        return prior_gradient - obs_gradient

    # def cost_hessian_product(test_state, test_step):
    #     """Hessian of cost_function at `test_state` times `test_step`.

    #     Parameters
    #     ----------
    #     test_state: np.ndarray[N]
    #     test_step: np.ndarray[N]

    #     Results
    #     -------
    #     hess_prod: np.ndarray[N]
    #     """
    #     bg_prod = np.linalg.solve(background_covariance,
    #                               test_step)
    #     obs_prod = observation_operator.T.dot(
    #         np.linalg.solve(observation_covariance,
    #                         observation_operator.dot(test_step)))
    #     return bg_prod + obs_prod

    result = scipy.optimize.minimize(
        cost_function, np.zeros_like(background),
        method="BFGS",
        jac=cost_jacobian,
        # hessp=cost_hessian_product,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis = background + result.x

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis)

    return analysis, result.hess_inv
