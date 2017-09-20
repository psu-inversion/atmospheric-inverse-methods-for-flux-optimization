"""Inversions using Optimal Interpolaiton.

Also known as Kalman Matrix Inversion or batch inversion.
"""
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from inversion.util import atleast_1d, atleast_2d, solve, tolinearoperator


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
    if isinstance(background_covariance, LinearOperator):
        # Leave this as array earlier
        # Avoiding the indirection may help
        # Not sure how leaving this up there would help clarity

        # Needed for HBHT + R addition
        observation_covariance = tolinearoperator(
            observation_covariance)

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

    covariance_sum = projected_background_covariance + observation_covariance

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
    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)
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
