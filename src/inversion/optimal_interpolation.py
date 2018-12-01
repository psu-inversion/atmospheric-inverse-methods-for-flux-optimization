"""Inversions using Optimal Interpolaiton.

Also known as Kalman Matrix Inversion or batch inversion.
"""
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from inversion.util import method_common
from inversion.linalg import (ProductLinearOperator, ARRAY_TYPES,
                              solve, tolinearoperator)


@method_common
def simple(background, background_covariance,
           observations, observation_covariance,
           observation_operator,
           reduced_background_covariance,
           reduced_observation_operator):
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
    reduced_background_covariance: array_like[Nred, Nred], optional
    reduced_observation_operator: array_like[M, Nred], optional

    Returns
    -------
    analysis: np.ndarray[N]
    analysis_covariance: np.ndarray[N,N]
    """
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    observation_increment = observations - projected_obs

    # B_{proj} = HBH^T
    if ((isinstance(observation_operator, LinearOperator) or
         isinstance(background_covariance, LinearOperator))):
        projected_background_covariance = ProductLinearOperator(
            observation_operator, background_covariance, observation_operator.T
        )
    else:
        projected_background_covariance = observation_operator.dot(
            background_covariance.dot(observation_operator.T))

    if isinstance(observation_covariance, LinearOperator):
        projected_background_covariance = tolinearoperator(
            projected_background_covariance)

    covariance_sum = projected_background_covariance + observation_covariance

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            solve(
                covariance_sum,
                observation_increment)))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    if reduced_background_covariance is None:
        decrease = background_covariance.dot(
            observation_operator.T.dot(
                solve(
                    covariance_sum,
                    observation_operator).dot(
                    background_covariance)))

        if isinstance(background_covariance, LinearOperator):
            decrease = tolinearoperator(decrease)

        analysis_covariance = background_covariance - decrease
    else:
        decrease = reduced_background_covariance.dot(
            reduced_observation_operator.T.dot(
                solve(
                    covariance_sum,
                    reduced_observation_operator).dot(
                    reduced_background_covariance)))
        analysis_covariance = reduced_background_covariance - decrease

    return analysis, analysis_covariance


@method_common
def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                reduced_background_covariance,
                reduced_observation_operator):
    """Solve the inversion problem, evaluating sub-expressions only once.

    Assumes all arrays fit in memory with room to spare.

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
    """
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    innovation = (observations - projected_obs)

    # B_{proj} = HBH^T
    if isinstance(observation_operator, LinearOperator):
        B_HT = tolinearoperator(background_covariance).dot(
            observation_operator.T)

        projected_background_covariance = ProductLinearOperator(
            observation_operator, B_HT)
    else:
        B_HT = background_covariance.dot(observation_operator.T)
        projected_background_covariance = observation_operator.dot(
            B_HT)

    if ((isinstance(projected_background_covariance, LinearOperator) ^
         isinstance(observation_covariance, LinearOperator))):
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    # This does repeat work for in memory data, but is perhaps doable
    # for out-of-core computations
    observation_increment = solve(
        covariance_sum, innovation)
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            observation_increment))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    if reduced_background_covariance is None:
        decrease = B_HT.dot(solve(
            covariance_sum,
            B_HT.T))
        if isinstance(decrease, LinearOperator):
            background_covariance = tolinearoperator(
                background_covariance)
        analysis_covariance = background_covariance - decrease
    else:
        B_HT_red = reduced_background_covariance.dot(
            reduced_observation_operator.T)
        decrease = B_HT_red.dot(solve(
            covariance_sum,
            B_HT_red.T))
        analysis_covariance = reduced_background_covariance - decrease

    return analysis, analysis_covariance


@method_common
def save_sum(background, background_covariance,
             observations, observation_covariance,
             observation_operator,
             reduced_background_covariance=None,
             reduced_observation_operator=None):
    """Solve the inversion problem, evaluating sub-expressions only once.

    Assumes all arrays fit in memory with room to spare.

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
    """
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    innovation = (observations - projected_obs)

    # B_{proj} = HBH^T
    if isinstance(observation_operator, ARRAY_TYPES):
        # TODO: test this
        if hasattr(background_covariance, "quadratic_form"):
            projected_background_covariance = (
                background_covariance.quadratic_form(
                    observation_operator.T))
        else:
            projected_background_covariance = observation_operator.dot(
                background_covariance.dot(observation_operator.T))
    else:
        projected_background_covariance = ProductLinearOperator(
            observation_operator, tolinearoperator(background_covariance),
            observation_operator.T)

    if ((isinstance(projected_background_covariance, LinearOperator) ^
         isinstance(observation_covariance, LinearOperator))):
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    observation_increment = solve(covariance_sum, innovation)
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            observation_increment))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    if reduced_background_covariance is None:
        if isinstance(observation_operator, ARRAY_TYPES):
            B_HT = background_covariance.dot(observation_operator.T)
        else:
            B_HT = ProductLinearOperator(background_covariance,
                                         observation_operator.T)
        decrease = B_HT.dot(solve(
            covariance_sum,
            B_HT.T))
        if isinstance(decrease, LinearOperator):
            background_covariance = tolinearoperator(
                background_covariance)
        analysis_covariance = background_covariance - decrease
    else:
        B_HT_red = reduced_background_covariance.dot(
            reduced_observation_operator.T)
        decrease = B_HT_red.dot(solve(
            covariance_sum,
            B_HT_red.T))
        analysis_covariance = reduced_background_covariance - decrease

    return analysis, analysis_covariance


@method_common
def scipy_chol(background, background_covariance,
               observations, observation_covariance,
               observation_operator,
               reduced_background_covariance=None,
               reduced_observation_operator=None):
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
    reduced_background_covariance: array_like[Nred, Nred], optional
    reduced_observation_operator: array_like[M, Nred], optional

    Returns
    -------
    analysis: np.ndarray[N]
    analysis_covariance: np.ndarray[N,N]
    """
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    innovation = observations - projected_obs

    B_HT = background_covariance.dot(observation_operator.T)
    # B_{proj} = HBH^T
    projected_background_covariance = observation_operator.dot(
        B_HT)

    if isinstance(observation_covariance, LinearOperator):
        projected_background_covariance = tolinearoperator(
            projected_background_covariance)
    covariance_sum = projected_background_covariance + observation_covariance
    # TODO: rewrite in terms of things dask has
    cov_sum_chol_up = scipy.linalg.cho_factor(covariance_sum, overwrite_a=True)
    del covariance_sum

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = B_HT.dot(
        scipy.linalg.cho_solve(
            cov_sum_chol_up,
            innovation,
            overwrite_b=True))
    del innovation

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    if reduced_background_covariance is None:
        decrease = B_HT.dot(
            scipy.linalg.cho_solve(
                cov_sum_chol_up,
                B_HT.T,
                overwrite_b=False))
        if isinstance(background_covariance, LinearOperator):
            decrease = tolinearoperator(decrease)
        analysis_covariance = background_covariance - decrease
    else:
        B_HT_red = reduced_background_covariance.dot(
            reduced_observation_operator.T)
        decrease = B_HT_red.dot(
            scipy.linalg.cho_solve(
                cov_sum_chol_up,
                B_HT_red.T,
                overwrite_b=False))
        if isinstance(reduced_background_covariance, LinearOperator):
            decrease = tolinearoperator(decrease)
        analysis_covariance = reduced_background_covariance - decrease

    return analysis, analysis_covariance
