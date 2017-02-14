"""Inversions using Optimal Interpolaiton.

Also known as Kalman Matrix Inversion or batch inversion.
"""
import numpy as np


def simple(background, background_covariance,
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

    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        background_covariance.dot(observation_operator.T))

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            np.linalg.solve(
                projected_background_covariance +
                observation_covariance,
                observation_increment)))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    analysis_covariance = (background_covariance -
                           background_covariance.dot(
                               observation_operator.T.dot(
                                   np.linalg.solve(
                                       projected_background_covariance +
                                       observation_covariance,
                                       observation_operator).dot(
                                           background_covariance))))

    return analysis, analysis_covariance
