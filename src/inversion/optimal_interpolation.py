"""Inversions using Optimal Interpolaiton.

Also known as Kalman Matrix Inversion or batch inversion.
"""
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

import dask.array as da
from dask.array import asarray

from inversion.util import solve, tolinearoperator, validate_args
from inversion.util import ProductLinearOperator, chunk_sizes, ARRAY_TYPES


@validate_args
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

    covariance_sum = projected_background_covariance + observation_covariance

    if isinstance(covariance_sum, ARRAY_TYPES):
        chunks = chunk_sizes((covariance_sum.shape[0],))
        covariance_sum = covariance_sum.rechunk(chunks[0])

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            solve(
                covariance_sum,
                observation_increment)))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = background_covariance.dot(
        observation_operator.T.dot(
            solve(
                covariance_sum,
                observation_operator).dot(
                background_covariance)))

    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)

    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance


@validate_args
def fold_common(background, background_covariance,
                observations, observation_covariance,
                observation_operator):
    """Solve the inversion problem, evaluating sub-expressions only once.

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
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    innovation = (observations - projected_obs).persist()

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

    if isinstance(covariance_sum, ARRAY_TYPES):
        chunks = chunk_sizes((covariance_sum.shape[0],))
        covariance_sum = covariance_sum.rechunk(chunks[0]).persist()

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    # This does repeat work for in memory data, but is perhaps doable
    # for out-of-core computations
    observation_increment = solve(
        covariance_sum, innovation).persist()
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            observation_increment))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = (B_HT.dot(solve(
                covariance_sum,
                B_HT.T)))
    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance


@validate_args
def save_sum(background, background_covariance,
             observations, observation_covariance,
             observation_operator):
    """Solve the inversion problem, evaluating sub-expressions only once.

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
    # \vec{y}_b = H \vec{x}_b
    projected_obs = observation_operator.dot(background)
    # \Delta\vec{y} = \vec{y} - \vec{y}_b
    innovation = (observations - projected_obs)

    # B_{proj} = HBH^T
    if isinstance(observation_operator, ARRAY_TYPES):
        B_HT = background_covariance.dot(observation_operator.T)

        # TODO: test this
        if hasattr(background_covariance, "quadratic_form"):
            projected_background_covariance = (
                background_covariance.quadratic_form(
                    observation_operator.T))
        else:
            projected_background_covariance = observation_operator.dot(B_HT)
    else:
        B_HT = tolinearoperator(background_covariance).dot(
            observation_operator.T)

        projected_background_covariance = ProductLinearOperator(
            observation_operator, B_HT)

    if ((isinstance(projected_background_covariance, LinearOperator) ^
         isinstance(observation_covariance, LinearOperator))):
        covariance_sum = (tolinearoperator(projected_background_covariance) +
                          tolinearoperator(observation_covariance))
    else:
        covariance_sum = (projected_background_covariance +
                          observation_covariance)

    if isinstance(covariance_sum, ARRAY_TYPES):
        covariance_sum, innovation = map(
            asarray, da.compute(covariance_sum, innovation))
    else:
        innovation = innovation.persist()

    # \Delta\vec{x} = B H^T (B_{proj} + R)^{-1} \Delta\vec{y}
    observation_increment = solve(covariance_sum, innovation)
    analysis_increment = background_covariance.dot(
        observation_operator.T.dot(
            observation_increment))

    # \vec{x}_a = \vec{x}_b + \Delta\vec{x}
    analysis = background + analysis_increment

    # P_a = B - B H^T (B_{proj} + R)^{-1} H B
    decrease = (B_HT.dot(solve(
                covariance_sum,
                B_HT.T)))
    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance


@validate_args
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
    decrease = B_HT.dot(
        scipy.linalg.cho_solve(
            cov_sum_chol_up,
            B_HT.T,
            overwrite_b=False))
    if isinstance(background_covariance, LinearOperator):
        decrease = tolinearoperator(decrease)
    analysis_covariance = background_covariance - decrease

    return analysis, analysis_covariance
