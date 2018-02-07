"""Functions implementing 3D-Var.

Signatures follow the functions in :mod:`inversion.optimal_interpolation`

Note
----
Forces in-memory computations.  BFGS method requires this, and there
are odd shape-mismatch errors if I just change to dask arrays.
Conjugate gradient solvers may work better for dask arrays if we drop
the covariance matrix from the return values.
"""
import scipy.optimize
import scipy.linalg
from scipy.sparse.linalg import LinearOperator
# I believe scipy's minimizer requires things that give boolean true
# or false from the objective, rather than a yet-to-be-realized dask
# array.
from inversion.util import atleast_1d, atleast_2d
from dask.array import asarray
from numpy import zeros_like

from inversion import ConvergenceError, MAX_ITERATIONS, GRAD_TOL
from inversion.util import solve, chunk_sizes, validate_args


@validate_args
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
    def cost_function(test_state):
        """Mismatch between state, prior, and obs.

        Parameters
        ----------
        test_state: np.ndarray[N]

        Returns
        -------
        cost: float
        """
        prior_mismatch = asarray(test_state - background)
        test_obs = observation_operator.dot(test_state)
        obs_mismatch = asarray(test_obs - observations)

        prior_fit = prior_mismatch.dot(solve(
            background_covariance, prior_mismatch))
        obs_fit = obs_mismatch.dot(solve(
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

        prior_gradient = solve(background_covariance,
                               prior_mismatch)
        obs_gradient = observation_operator.T.dot(
            solve(observation_covariance,
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
    #     bg_prod = solve(background_covariance,
    #                        test_step)
    #     obs_prod = observation_operator.T.dot(
    #         solve(observation_covariance,
    #                  observation_operator.dot(test_step)))
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


@validate_args
def incremental(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                calculate_posterior_error_covariance=True):
    """Feed everything to scipy's minimizer.

    Use the change from the background to try to avoid precision loss.

    Parameters
    ----------
    background: np.ndarray[N]
    background_covariance:  np.ndarray[N,N]
    observations: np.ndarray[M]
    observation_covariance: np.ndarray[M,M]
    observation_operator: np.ndarray[M,N]
    calculate_posterior_error_covariance: bool, optional
        Whether to calculate and return the posterior analysis error.
        This may provide a significant memory and time savings.

    Returns
    -------
    analysis: np.ndarray[N]
    analysis_covariance: np.ndarray[N,N]
        The posterior error covariance matrix. Only returned if
        `calculate_posterior_error_covariance` is :obj:`True`

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

        prior_fit = test_change.dot(asarray(solve(
            background_covariance, test_change)))
        obs_fit = obs_mismatch.dot(asarray(solve(
            observation_covariance, obs_mismatch)))
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

        prior_gradient = solve(background_covariance,
                               test_change)
        obs_gradient = observation_operator.T.dot(
            solve(observation_covariance,
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
    #     bg_prod = solve(background_covariance,
    #                        test_step)
    #     obs_prod = observation_operator.T.dot(
    #         solve(observation_covariance,
    #                  observation_operator.dot(test_step)))
    #     return bg_prod + obs_prod

    result = scipy.optimize.minimize(
        cost_function, asarray(zeros_like(background)),
        method="BFGS" if calculate_posterior_error_covariance else "CG",
        jac=cost_jacobian,
        # hessp=cost_hessian_product,
        options=dict(maxiter=MAX_ITERATIONS,
                     gtol=GRAD_TOL),
    )

    analysis = background + result.x

    if not result.success:
        raise ConvergenceError("Did not converge: {msg:s}".format(
            msg=result.message), result, analysis)

    if calculate_posterior_error_covariance:
        return analysis, result.hess_inv
    return analysis


@validate_args
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
    innovations = observations - observation_operator.dot(background)

    from scipy.linalg import cho_factor, cho_solve

    # factor the covariances to make the matrix inversions faster
    bg_cov_chol_u = cho_factor(background_covariance)
    obs_cov_chol_u = cho_factor(observation_covariance)

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
    #     bg_prod = solve(background_covariance,
    #                        test_step)
    #     obs_prod = observation_operator.T.dot(
    #         solve(observation_covariance,
    #                  observation_operator.dot(test_step)))
    #     return bg_prod + obs_prod

    result = scipy.optimize.minimize(
        cost_function, asarray(zeros_like(background)),
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
