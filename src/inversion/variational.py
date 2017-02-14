import numpy as np
import numpy.dual
import scipy.optimize
import scipy.linalg

def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator):
    """Feed everything to scipy's minimizer

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
        # hessp=cost_hessian_product
    )

    if not result.success:
        raise ValueError("Did not converge: {msg:s}".format(
            msg=result.message))

    return result.x, result.hess_inv
