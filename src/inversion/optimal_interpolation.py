"""Inversions using Optimal Interpolaiton.

Also known as Kalman Matrix Inversion or batch inversion.
"""
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from inversion.util import atleast_1d, atleast_2d, solve, tolinearoperator
from inversion.util import ProductLinearOperator


def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator):
    """Solve the inversion problem using the equations literally.

    Assumes all arrays fit in memory with room to spare.  A direct
    translation of the matrix inversion equations to Python.

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
    background = atleast_1d(background)
    if not isinstance(background_covariance, LinearOperator):
        background_covariance = atleast_2d(background_covariance)

    observations = atleast_1d(observations)
    if not isinstance(observation_covariance, LinearOperator):
        observation_covariance = atleast_2d(observation_covariance)

    if not isinstance(observation_operator, LinearOperator):
        observation_operator = atleast_2d(observation_operator)

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        background_covariance.dot(observation_operator.T))

    if isinstance(observation_covariance, LinearOperator):
        projected_background_covariance = tolinearoperator(
            projected_background_covariance)
    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            solve(
                projected_background_covariance +
                observation_covariance,
                observation_increment)))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = background_covariance.dot(
        observation_operator.T.dot(
            solve(
                projected_background_covariance +
                observation_covariance,
                observation_operator).dot(
                background_covariance)))

    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)

    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance


def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                calculate_posterior_error_covariance=True):
    """Solve the inversion problem, evaluating sub-expressions only once.

    Assumes all arrays fit in memory with room to spare.

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
    """
    background = atleast_1d(background)
    if not isinstance(background_covariance, LinearOperator):
        background_covariance = atleast_2d(background_covariance)
        bg_is_arry = True
    else:
        bg_is_arry = False

    observations = atleast_1d(observations)
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

        projected_background_covariance = observation_operator.dot(
            B_HT)
    else:
        B_HT = tolinearoperator(background_covariance).dot(
            observation_operator.T)

        projected_background_covariance = ProductLinearOperator(
            observation_operator, B_HT)

    if ((isinstance(projected_background_covariance, LinearOperator) ^
         (not obs_is_arry))):
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = B_HT.dot(
        solve(
            covariance_sum,
            observation_increment))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    if not calculate_posterior_error_covariance:
        return analysis

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = (B_HT.dot(solve(
                covariance_sum,
                B_HT.T)))
    if (not bg_is_arry) ^ isinstance(decrease, LinearOperator):
        analysis_covariance = (tolinearoperator(background_covariance) -
                               tolinearoperator(decrease))
    else:
        analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance


def scipy_chol(background, background_covariance,
               observations, observation_covariance,
               observation_operator):
    """Use the Cholesky decomposition to solve the inverison problem.

    Assumes all arrays fit in memory with room to spare.
    Uses cholesky decomposition for solving a matrix equation
    rather than using matrix inverses.

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
    background = atleast_1d(background)
    if not isinstance(background_covariance, LinearOperator):
        background_covariance = atleast_2d(background_covariance)

    observations = atleast_1d(observations)
    if not isinstance(observation_covariance, LinearOperator):
        observation_covariance = atleast_2d(observation_covariance)

    if not isinstance(observation_operator, LinearOperator):
        observation_operator = atleast_2d(observation_operator)

    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    B_HT = background_covariance.dot(observation_operator.T)
    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        B_HT)

    if isinstance(observation_covariance, LinearOperator):
        projected_background_covariance = tolinearoperator(
            projected_background_covariance)
    covariance_sum = projected_background_covariance + observation_covariance
    cov_sum_chol_up = scipy.linalg.cho_factor(covariance_sum, overwrite_a=True)
    del covariance_sum

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = B_HT.dot(
        scipy.linalg.cho_solve(
            cov_sum_chol_up,
            observation_increment,
            overwrite_b=True))
    del observation_increment

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = B_HT.dot(
        scipy.linalg.cho_solve(
            cov_sum_chol_up,
            B_HT.T,
            overwrite_b=False))
    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance
