"""Utility functions for compatibility.

These functions mirror :mod:`numpy` functions but produce dask output.
"""
import itertools
import numbers
import math

import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg.interface import MatrixLinearOperator
import dask.array as da
import dask.array.linalg as la
from dask.array import asarray, concatenate

OPTIMAL_ELEMENTS = int(2e4)
"""Optimal elements per chunk in a dask array.

Magic number, arbitrarily chosen.  Dask documentation mentions many
chunks should fit easily in memory, but each should contain at least a
million elements.  This size matrix is fast to allocate and fill, but
:math:`10^5` gives a memory error.
"""


def chunk_sizes(shape, matrix_side=True):
    """Good chunk sizes for the given shape.

    Optimized mostly for matrix operations on covariance matrices on
    this domain.

    Parameters
    ----------
    shape: tuple
    matrix_side: bool
        Whether the shape is one side of a matrix or intended to be
        only a vector.

    Returns
    -------
    chunks: tuple
    """
    chunk_start = len(shape)
    nelements = 1

    here_max = OPTIMAL_ELEMENTS
    if not matrix_side:
        here_max **= 2

    if np.prod(shape) <= here_max:
        # The total number of elements is smaller than the recommended
        # chunksize, so the whole thing is a single chunk
        return tuple(shape)

    for dim_size in reversed(shape):
        nelements *= dim_size
        chunk_start -= 1

        if nelements > here_max:
            nelements //= dim_size
            break

    chunks = [1 if i < chunk_start else dim_size
              for i, dim_size in enumerate(shape)]
    next_dimsize = shape[chunk_start]

    # I happen to like neatness
    # And cholesky requires square chunks.
    if here_max > nelements:
        max_to_check = int(here_max // nelements)
        check_step = int(min(10 ** math.floor(math.log10(max_to_check) - .1),
                             here_max))
        chunks[chunk_start] = max(
            i for i in itertools.chain(
                (1,),
                range(check_step, max_to_check + 1, check_step))
            if next_dimsize % i == 0)
    return tuple(chunks)


def atleast_1d(arry):
    """Ensure `arry` is dask array of rank at least one.

    Parameters
    ----------
    arry: array_like

    Returns
    -------
    new_arry: dask.array.core.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 1:
            return arry
        return arry[np.newaxis]
    if isinstance(arry, (list, tuple, np.ndarray)):
        arry = np.atleast_1d(arry)
    if isinstance(arry, numbers.Number):
        arry = np.atleast_1d(arry)

    array_shape = arry.shape
    return da.from_array(arry, chunks=chunk_sizes(array_shape))


def atleast_2d(arry):
    """Ensure arry is a dask array of rank at least two.

    Parameters
    ----------
    arry: array_like

    Returns
    -------
    new_arry: dask.array.core.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 2:
            return arry
        elif arry.ndim == 1:
            return arry[np.newaxis, :]
        return arry[np.newaxis, np.newaxis]
    if isinstance(arry, (list, tuple, np.ndarray)):
        arry = np.atleast_2d(arry)
    if isinstance(arry, numbers.Number):
        arry = np.atleast_2d(arry)

    array_shape = arry.shape
    return da.from_array(arry, chunks=chunk_sizes(array_shape))


def solve(arr1, arr2):
    """Solve arr1 @ x = arr2 for x.

    Parameters
    ----------
    arr1: array_like[N, N]
    arr2: array_like[N]

    Returns
    -------
    array_like[N]
    """
    if hasattr(arr1, "solve"):
        return arr1.solve(arr2)
    # Shorter method for assuring dask arrays
    return la.solve(da.asarray(arr1), da.asarray(arr2))


def schmidt_decomposition(vector, dim1, dim2):
    """Decompose a state vector into a sum of Kronecker products.

    Parameters
    ----------
    vector: array_like[dim1 * dim2]
    dim1, dim2: int

    Returns
    -------
    tuple of (weights, unit_vecs[dim1], unit_vecs[dim2]
        The rows form the separate vectors.
        The weights are guaranteed to be greater than zero

    Note
    ----
    Algorithm from stackexchange:
    https://physics.stackexchange.com/questions/251522/how-do-you-find-a-schmidt-basis-and-how-can-the-schmidt-decomposition-be-used-f
    Also from Mathematica code I wrote based on description in the green
    Quantum Computation book in the reading library
    """
    if vector.ndim == 2 and vector.shape[1] != 1:
        raise ValueError("Schmidt decomposition only valid for vectors")
    state_matrix = asarray(vector).reshape(dim1, dim2)

    if dim1 > dim2:
        vecs1, lambdas, vecs2 = la.svd(state_matrix)
    else:
        # Transpose, because dask expects tall and skinny
        vecs2T, lambdas, vecs1T = la.svd(state_matrix.T)
        vecs1 = vecs1T.T
        vecs2 = vecs2T.T

    return lambdas, vecs1.T[:len(lambdas), :], vecs2[:len(lambdas), :]


def kronecker_product(operator1, operator2):
    """Form the Kronecker product of the given operators.

    Delegates to ``operator1.kron()`` if possible,
    :class:`inversion.correlations.KroneckerProduct` if not.

    Parameters
    ----------
    operator1, operator2: scipy.sparse.linalg.LinearOperator

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
    """
    if hasattr(operator1, "kron"):
        return operator1.kron(operator2)
    from inversion.correlations import KroneckerProduct
    return KroneckerProduct(operator1, operator2)


def is_odd(num):
    """Return oddity of num.

    Parameters
    ----------
    num

    Returns
    -------
    bool
    """
    return num & 1 == 1


def tolinearoperator(operator):
    """Return operator as a LinearOperator.

    Parameters
    ----------
    operator: array_like or scipy.sparse.linalg.LinearOperator

    Returns
    -------
    scipy.sparse.linalg.LinearOperator

    See Also
    --------
    scipy.sparse.linalg.aslinearoperator
        A similar function without as wide a range of inputs.
        Used for everything but array_likes that are not
        :class:`np.ndarrays` or :class:`scipy.sparse.spmatrix`.
    """
    try:
        return aslinearoperator(operator)
    except TypeError:
        return MatrixLinearOperator(atleast_2d(operator))


def kron(matrix1, matrix2):
    """Kronecker product of two matrices.

    Parameters
    ----------
    matrix1: array_like[M, N]
    matrix2: array_like[J, K]

    Returns
    -------
    array_like[M*J, N*K]

    See Also
    --------
    scipy.linalg.kron
        Where I got the overview of the implementation.
    """
    matrix1 = atleast_2d(matrix1)
    matrix2 = atleast_2d(matrix2)

    total_shape = matrix1.shape + matrix2.shape
    change = matrix1.ndim

    matrix1_index = tuple(slice(None) if i < change else np.newaxis
                          for i in range(len(total_shape)))
    matrix2_index = tuple(np.newaxis if i < change else slice(None)
                          for i in range(len(total_shape)))

    product = matrix1[matrix1_index] * matrix2[matrix2_index]

    return concatenate(concatenate(product, axis=1), axis=1)
