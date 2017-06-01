"""Utility functions for compatibility.

These functions mirror :mod:`numpy` functions but produce dask output.
"""
import numbers

import numpy as np
import dask.array as da
import dask.array.linalg as la

OPTIMAL_ELEMENTS = int(1e4)
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

    for dim_size in reversed(shape):
        nelements *= dim_size
        chunk_start -= 1

        if nelements > here_max:
            nelements /= dim_size
            break

    chunks = [1 if i < chunk_start else dim_size
              for i, dim_size in enumerate(shape)]
    next_dimsize = shape[chunk_start]

    # I happen to like neatness
    if chunk_start > 0:
        chunks[chunk_start - 1] = max(
            i for i in range(
                1,
                here_max // nelements)
            if next_dimsize % i == 0)
    return tuple(chunks)


def atleast_1d(arry):
    """Ensure `arry` is dask array of rank at least one.

    Parameters
    ----------
    arry: array_like

    Returns
    -------
    new_arry: dask.array.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 1:
            return arry
        return arry[np.newaxis]
    if isinstance(arry, (list, tuple)):
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
    new_arry: dask.array.Array
    """
    if isinstance(arry, da.Array):
        if arry.ndim >= 2:
            return arry
        elif arry.ndim == 1:
            return arry[np.newaxis, :]
        return arry[np.newaxis, np.newaxis]
    if isinstance(arry, (list, tuple)):
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
