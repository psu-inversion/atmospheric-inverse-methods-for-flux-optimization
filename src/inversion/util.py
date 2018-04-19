"""Utility functions for compatibility.

These functions mirror :mod:`numpy` functions but produce dask output.
"""
import functools
import itertools
import operator
import warnings
import numbers
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lgmres
from scipy.sparse.linalg.interface import (
    MatrixLinearOperator,
    _CustomLinearOperator, _SumLinearOperator,
    _ScaledLinearOperator)
from scipy.sparse.linalg.eigen import eigsh as linop_eigsh

import dask.array as da
import dask.array.linalg as la
from dask.array import asarray, concatenate, stack, hstack, vstack, zeros

OPTIMAL_ELEMENTS = int(4e4)
"""Optimal elements per chunk in a dask array.

Magic number, arbitrarily chosen.  Dask documentation mentions many
chunks should fit easily in memory, but each should contain at least a
million elements, recommending 10-100MiB per chunk.  This size matrix
is fast to allocate and fill, but :math:`10^5` gives a memory error.
A square matrix of float64 with ten thousand elements on a side is 762
megabytes.

A single level of our domain is 4.6e4 elements.  The calculation
proceeds much more naturally when this fits in a chunk, since it needs
to for the FFTs.  This would be for OPTIMAL_ELEMENTS**2.

Leaving this as 1e4 causes memory errors and deadlocks over an hour
and a half.  5e4 can do the same program twice in ten minutes.
I don't entirely understand how this works.

I'm going to say these problems have little use for previous results,
so this can be larger than the dask advice.  This greatly reduces the
requirements for setting up the graph.

4e4 works for both, I think.  BE VERY CAREFUL CHANGING THIS!!

At least, it works on compute-0-6.  It may not on compute-0-0, where
it likes to dump me.
"""
ARRAY_TYPES = (np.ndarray, da.Array)
"""Array types for determining Kronecker product type.

These are combined for a direct product.
"""
REAL_DTYPE_KINDS = "fiu"
"""The kinds used by dtypes to represent real numbers.

Includes subsets.
"""
MAX_EXPLICIT_ARRAY = 1 << 25
"""Maximum size for an array represented explicitly.

:func:`kronecker_product` will form products smaller than this as an
explicit matrix using :func:`kron`.  Arrays larger than this will use
:class:`DaskKroneckerProduct`.

Currently completely arbitrary.
`2 ** 16` works fine in memory, `2**17` gives a MemoryError.
Hopefully Dask knows not to try this.
"""
DASK_OPTIMIZATIONS = dict(
    inline_functions_fast_functions=(
        # default value
        da.core.getter_inline,
        # recommended in dask #3139
        da.core.getter,
        operator.getitem,
        # recommended in dask#874
        np.ones,
        # extension of previous
        np.zeros,
        np.ones_like,
        np.zeros_like,
        np.full,
        np.full_like,
    ),
)


def chunk_sizes(shape, matrix_side=True):
    """Good chunk sizes for the given shape.

    Optimized mostly for matrix operations on covariance matrices on
    this domain.

    Parameters
    ----------
    shape: tuple
    matrix_side: bool
        Whether the shape will need to be one side of a matrix or
        intended to stay a vector.

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
    return da.from_array(
        arry, chunks=chunk_sizes(array_shape, matrix_side=False))


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
    # Either this is a square matrix and this chunking makes products
    # faster, or it is non-square and I can't optimize.
    return da.from_array(
        arry, chunks=chunk_sizes(array_shape, matrix_side=False))


# TODO: Test
def linop_solve(operator, arr):
    """Solve `operator @ x = arr`.

    deal with arr possibly having multiple columns.

    Parameters
    ----------
    operator: LinearOperator
    arr: array_like

    Returns
    -------
    array_like
    """
    if arr.ndim == 1:
        return asarray(lgmres(operator, np.asarray(arr))[0])
    return asarray(stack([lgmres(operator, np.asarray(col))[0]
                          for col in atleast_2d(arr).T],
                         axis=1))


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
    elif isinstance(arr1, (MatrixLinearOperator, DaskMatrixLinearOperator)):
        return la.solve(asarray(arr1.A), asarray(arr2))
    elif isinstance(arr1, LinearOperator):
        # Linear operators with neither an underlying matrix nor a
        # provided solver. Use iterative sparse solvers.
        # TODO: Get preconditioner working
        # TODO: Test Ax = b for b not column vector
        if isinstance(arr2, ARRAY_TYPES):
            return linop_solve(arr1, arr2)
        elif isinstance(arr2, (MatrixLinearOperator,
                               DaskMatrixLinearOperator)):
            # TODO: test this branch
            return linop_solve(arr1, arr2.A)
        else:
            def solver(vec):
                """Solve `arr1 x = vec`.

                Parameters
                ----------
                vec: array_like

                Returns
                -------
                array_like
                """
                return linop_solve(arr1, vec)
            inverse = DaskLinearOperator(matvec=solver, shape=arr1.shape[::-1])
            return inverse.dot(arr2)
        # # TODO: Figure out dask tasks for this
        # return da.Array(
        #     {(chunkname, 0):
        #      (spsolve, arr1, arr2.rechunk(1, whatever))},
        #     "solve-arr1.name-arr2.name",
        #     chunks)
    # Shorter method for assuring dask arrays
    return la.solve(asarray(arr1), asarray(arr2))


def matrix_sqrt(mat):
    """Find a matrix S such that S.T @ S == mat.

    Parameters
    ----------
    mat: array_like or LinearOperator

    Returns
    -------
    S: array_like or LinearOperator

    Raises
    ------
    ValueError: if mat is not symmetric
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix square root only defined for square arrays")
    if hasattr(mat, "sqrt"):
        return mat.sqrt()
    elif isinstance(mat, (MatrixLinearOperator, DaskMatrixLinearOperator)):
        mat = mat.A

    if isinstance(mat, ARRAY_TYPES):
        return la.cholesky(
            asarray(mat).rechunk(
                chunk_sizes(mat.shape[:1], matrix_side=True)[0]))

    # TODO: test this
    if isinstance(mat, (LinearOperator, DaskLinearOperator)):
        from inversion.covariances import DiagonalOperator
        warnings.warn("The square root will be approximate.")
        vals, vecs = linop_eigsh(
            mat, min(mat.shape[0] // 2, OPTIMAL_ELEMENTS // mat.shape[0]))
        sqrt_valop = DiagonalOperator(np.sqrt(vals))
        vecop = DaskMatrixLinearOperator(vecs)
        return ProductLinearOperator(vecop, sqrt_valop, vecop.T)

    # TODO: test on xarray datasets or iris cubes
    raise TypeError("Don't know how to find square root of {cls!s}".format(
        cls=type(mat)))


# TODO Test for handling of different chunking schemes
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
        # TODO: Test failure mode
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
    :func:`kron` if both are :const:`ARRAY_TYPES`, or
    :class:`inversion.correlations.SchmidtKroneckerProduct` otherwise.

    Parameters
    ----------
    operator1, operator2: scipy.sparse.linalg.LinearOperator
        The component operators of the Kronecker product.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
    """
    if hasattr(operator1, "kron"):
        return operator1.kron(operator2)
    elif isinstance(operator1, ARRAY_TYPES):
        if ((isinstance(operator2, ARRAY_TYPES) and
             # TODO: test this
             operator1.size * operator2.size < MAX_EXPLICIT_ARRAY)):
            return kron(operator1, operator2)
        return DaskKroneckerProductOperator(operator1, operator2)
    from inversion.correlations import SchmidtKroneckerProduct
    return SchmidtKroneckerProduct(operator1, operator2)


def is_odd(num):
    """Return oddity of num.

    Parameters
    ----------
    num: int

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
    DaskLinearOperator
        I want everything to work with dask where possible.

    See Also
    --------
    scipy.sparse.linalg.aslinearoperator
        A similar function without as wide a range of inputs.
        Used for everything but array_likes that are not
        :class:`np.ndarrays` or :class:`scipy.sparse.spmatrix`.
    """
    if isinstance(operator, ARRAY_TYPES):
        return DaskMatrixLinearOperator(atleast_2d(operator))
    try:
        return DaskLinearOperator.fromlinearoperator(
            aslinearoperator(operator))
    except TypeError:
        return DaskMatrixLinearOperator(atleast_2d(operator))


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

    # TODO: choose good chunk sizes
    product = matrix1[matrix1_index] * matrix2[matrix2_index]

    return concatenate(concatenate(product, axis=1), axis=1)


class DaskLinearOperator(LinearOperator):
    """LinearOperator designed to work with dask.

    Does not support :class:`np.matrix` objects.
    """

    @classmethod
    def fromlinearoperator(cls, original):
        """Turn other into a DaskLinearOperator.

        Parameters
        ----------
        original: LinearOperator

        Returns
        -------
        DaskLinearOperator
        """
        if isinstance(original, DaskLinearOperator):
            return original
        # This returns _CustomLinearOperator, not DaskLinearOperator
        result = cls(
            matvec=original._matvec, rmatvec=original._rmatvec,
            matmat=original._matmat,
            shape=original.shape, dtype=original.dtype)
        return result

    # Everything below here essentially copied from the original LinearOperator
    # scipy.sparse.linalg.interface
    def __new__(cls, *args, **kwargs):
        if cls is DaskLinearOperator:
            # Operate as _DaskCustomLinearOperator factory.
            return _DaskCustomLinearOperator(*args, **kwargs)
        else:
            obj = super(DaskLinearOperator, cls).__new__(cls, *args, **kwargs)

            if ((type(obj)._matvec == DaskLinearOperator._matvec and
                 type(obj)._matmat == DaskLinearOperator._matmat)):
                raise TypeError("LinearOperator subclass should implement"
                                " at least one of _matvec and _matmat.")

            return obj

    def _matmat(self, X):
        """Multiply self by matrix X using basic algorithm.

        Default matrix-matrix multiplication handler.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """
        return hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])

    def matvec(self, x):
        """

        Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : da.Array
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : da.Array
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """
        x = asarray(x)

        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        y = asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : da.Array
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : da.Array
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """
        x = asarray(x)

        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        y = asarray(y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError(
                'invalid shape returned by user-defined rmatvec()')

        return y

    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : da.Array
            An array with shape (N,K).

        Returns
        -------
        Y : da.Array
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        X = asarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        M, N = self.shape

        if X.shape[0] != N:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)

        if isinstance(Y, np.matrix):
            Y = np.asmatrix(Y)

        return Y

    # Dask assumes everything has an ndim
    # Hopefully this fixes the problem
    ndim = 2

    def __add__(self, x):
        """Add self and x."""
        if isinstance(x, LinearOperator):
            return _DaskSumLinearOperator(
                self, DaskLinearOperator.fromlinearoperator(x))
        elif isinstance(x, ARRAY_TYPES):
            return _DaskSumLinearOperator(
                self, DaskMatrixLinearOperator(x))
        else:
            return NotImplemented

    __radd__ = __add__

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        if isinstance(x, (LinearOperator, DaskLinearOperator)):
            return ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _DaskScaledLinearOperator(self, x)
        else:
            x = asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        shape = (self.shape[1], self.shape[0])
        return _DaskCustomLinearOperator(shape, matvec=self.rmatvec,
                                         rmatvec=self.matvec,
                                         dtype=self.dtype)


class _DaskCustomLinearOperator(DaskLinearOperator, _CustomLinearOperator):
    """This should let the factory functions above work."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None):
        super(_DaskCustomLinearOperator, self).__init__(
            shape, matvec, rmatvec, matmat, dtype)

        if ((self.dtype.kind in REAL_DTYPE_KINDS and
             not hasattr(self, "_transpose"))):
            self._transpose = self._adjoint


class DaskMatrixLinearOperator(MatrixLinearOperator, DaskLinearOperator):
    """This should help out with the tolinearoperator.

    Should I override __add__, ...?
    """

    def __init__(self, A):
        super(DaskMatrixLinearOperator, self).__init__(A)
        self.__transp = None
        self.__adj = None

    def _transpose(self):
        if self.__transp is None:
            self.__transp = _DaskTransposeLinearOperator(self)
        return self.__transp

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _DaskAdjointLinearOperator(self)
        return self.__adj


class _DaskTransposeLinearOperator(DaskMatrixLinearOperator):
    """Transpose of a DaskMatrixLinearOperator."""

    def __init__(self, transpose):
        super(_DaskTransposeLinearOperator, self).__init__(transpose.A.T)
        self.__transp = transpose

    def _transpose(self):
        return self.__transp


class _DaskAdjointLinearOperator(DaskMatrixLinearOperator):
    """Adjoint of a DaskMatrixLinearOperator."""

    def __init__(self, adjoint):
        super(_DaskAdjointLinearOperator, self).__init__(adjoint.A.H)
        self.__adjoint = adjoint

    def _adjoint(self):
        return self.__adjoint


class _DaskSumLinearOperator(_SumLinearOperator, DaskLinearOperator):
    """Sum of two DaskLinearOperators."""

    pass


class ProductLinearOperator(DaskLinearOperator):
    """Represent a product of linear operators."""

    def __init__(self, *operators):
        """Set up a product of linear operators.

        Parameters
        ----------
        operators: LinearOperator
        """
        operators = tuple(tolinearoperator(op)
                          for op in operators)
        result_type = np.result_type(
            *(op.dtype for op in operators))
        super(ProductLinearOperator, self).__init__(
            result_type, (operators[0].shape[0], operators[-1].shape[1]))
        self._operators = operators
        # self._init_dtype()

        try:
            if all(op is rop.T for op, rop in
                   zip(operators, reversed(operators))):
                self.quadratic_form = self._quadratic_form
                self.sqrt = self._sqrt
        except NotImplementedError:
            # Transpose not implemented for a subclass
            pass

    def _matvec(self, vector):
        """Form matrix-vector product with vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        for op in reversed(self._operators):
            vector = op.matvec(vector)

        return vector

    def _rmatvec(self, vector):
        """Matrix-vector product on the left.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        # TODO: test this
        for op in self._operators:
            vector = op.H.matvec(vector)

        return vector

    def _matmat(self, matrix):
        """The matrix-matrix product.

        Parameters
        ----------
        matrix: array_like

        Returns
        -------
        array_like
        """
        for op in reversed(self._operators):
            matrix = op.matmat(matrix)

        return matrix

    def _adjoint(self):
        """The Hermitian adjoint of the operator."""
        # TODO: test this
        return ProductLinearOperator(
            [op.H for op in reversed(self._operators)])

    def _transpose(self):
        """The transpose of the operator."""
        return ProductLinearOperator(
            *[op.T for op in reversed(self._operators)])

    def solve(self, vector):
        """Solve A @ x == vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
            Solution of self @ x == vec
        """
        for op in self._operators:
            vector = solve(op, vector)

        return vector

    def _quadratic_form(self, mat):
        """Find the quadratic form mat.T @ self @ mat.

        Parameters
        ----------
        mat: array_like[N, M]

        Returns
        -------
        result: array_like[M, M]
        """
        operators = self._operators
        n_ops = len(operators)
        half_n_ops = n_ops // 2

        for op in operators[:half_n_ops]:
            mat = op.T.dot(mat)
        if is_odd(n_ops):
            middle_op = operators[half_n_ops]
            if hasattr(middle_op, "quadratic_form"):
                # TODO: test this
                result = operators[half_n_ops].quadratic_form(mat)
            else:
                result = mat.T.dot(middle_op.dot(mat))
        else:
            result = mat.T.dot(mat)
        return result

    def _sqrt(self):
        """Find S such that S.T @ S == self."""
        operators = self._operators
        n_ops = len(operators)
        half_n_ops = n_ops // 2

        last_operators = operators[-half_n_ops:]

        if is_odd(n_ops):
            middle_operator = operators[half_n_ops]

            return ProductLinearOperator(
                matrix_sqrt(middle_operator), *last_operators)
        return ProductLinearOperator(*last_operators)


# TODO: move to covariances.py and inherit from SelfAdjointOperator
class CorrelationStandardDeviation(ProductLinearOperator):
    """Represent correlation-std product."""

    def __init__(self, correlation, std):
        from inversion.covariances import DiagonalOperator
        std_matrix = DiagonalOperator(std)
        super(CorrelationStandardDeviation, self).__init__(
            std_matrix, correlation, std_matrix)

    def _transpose(self):
        """Return transpose of self."""
        return self

    def _adjoint(self):
        """Return hermetian adjoint of self."""
        return self

    def sqrt(self):
        """Find S such that S.T @ S == self."""
        std_matrix, correlation, _ = self._operators

        return ProductLinearOperator(
            matrix_sqrt(correlation), std_matrix)

    _sqrt = sqrt


class _DaskScaledLinearOperator(_ScaledLinearOperator, DaskLinearOperator):
    """Scaled linear operator."""

    pass


class DaskKroneckerProductOperator(DaskLinearOperator):
    """Operator for Kronecker product.

    Uses the :math:`O(n^{2.5})` algorithm of Yadav and Michalak (2013)
    to make memory and computational requirements practical.

    Left argument to Kronecker product must be array.

    References
    ----------
    V. Yadav and A.M. Michalak. "Improving computational efficiency in
    large linear inverse problems: an example from carbon dioxide flux
    estimation" *Geosci. Model Dev.* 2013. Vol 6, pp. 583--590.
    URL: http://www.geosci-model-dev.net/6/583/2013
    DOI: 10.5194/gmd-6-583-2013.
    """

    def __init__(self, operator1, operator2):
        """Set up the instance.

        Parameters
        ----------
        operator1: array_like
        operator2: array_like or DaskLinearOperator
        """
        if isinstance(operator1, (DaskMatrixLinearOperator,
                                  MatrixLinearOperator)):
            # TODO: test this
            operator1 = asarray(operator1.A)
        else:
            operator1 = asarray(operator1)
        operator2 = tolinearoperator(operator2)

        total_shape = np.multiply(operator1.shape, operator2.shape)
        super(DaskKroneckerProductOperator, self).__init__(
            shape=tuple(total_shape),
            dtype=np.result_type(operator1.dtype, operator2.dtype))
        self._operator1 = operator1
        self._operator2 = operator2
        self._block_size = operator2.shape[1]
        self._n_chunks = operator1.shape[0]

        self.__transp = None

    def _transpose(self):
        """Transpose the operator."""
        if self.__transp is None:
            operator1 = self._operator1
            operator2 = self._operator2
            if ((operator1.shape[0] == operator1.shape[1] and
                 da.allclose(operator1, operator1.T).compute())):
                if operator2.T is operator2:
                    self.__transp = self
                else:
                    # TODO: test this
                    self.__transp = DaskKroneckerProductOperator(
                        operator1, operator2.T)
            else:
                self.__transp = DaskKroneckerProductOperator(
                    operator1.T, operator2.T)
        return self.__transp

    def sqrt(self):
        """Find an operator S such that S.T @ S == self.

        Requires self be symmetric.

        Returns
        -------
        S: KroneckerProductOperator

        Raises
        ------
        ValueError: if operator not self-adjoint
        """
        operator1 = self._operator1
        if ((self.shape[0] != self.shape[1] or
             operator1.shape[0] != operator1.shape[1])):
            # TODO: test this
            raise ValueError(
                "Square root not defined for {shape!s} operators."
                .format(shape=self.shape))
        if self.T is not self:
            # TODO: test this
            raise ValueError(
                "Square root not defined for non-symmetric operators.")
        # Cholesky can be fragile, so delegate to central location for
        # handling that.
        sqrt1 = matrix_sqrt(operator1)
        sqrt2 = matrix_sqrt(self._operator2)
        return DaskKroneckerProductOperator(
            sqrt1, sqrt2)

    def _matmat(self, mat):
        r"""Compute matrix-matrix product.

        Parameters
        ----------
        mat: array_like

        Returns
        -------
        result: array_like

        Note
        ----
        Implementation depends on the structure of the Kronecker
        product:

        .. math::
            A \otimes B = \begin{pmatrix}
                A_{11} B & A_{12} B & \cdots & A_{1n} B \\
                A_{21} B & A_{22} B & \cdots & A_{2n} B \\
                \vdots   & \vdots   & \ddots & \vdots   \\
                A_{m1} B & A_{m2} B & \cdots & A_{mn} B
            \end{pmatrix}

        Matrix-scalar products are commutative, and :math:`B` is the
        same for each block.  When right-multiplying by a matrix
        :math:`C`, we can take advantage of this by splitting
        :math:`C` into chunks, multiplying each by the corresponding
        element of :math:`A`, adding them up, and multiplying by
        :math:`B`.

        """
        chunks = []
        block_size = self._block_size
        chunk_shape = (block_size, mat.shape[1])
        chunk_chunks = (block_size, mat.chunks[1][0])
        chunk_dtype = np.result_type(self.dtype, mat.dtype)
        operator1 = self._operator1
        operator2 = self._operator2
        in_chunk = (operator1.shape[1], block_size, mat.shape[1])

        # each row in the outer operator will produce a chunk of rows
        # in the result
        for row1 in range(self._n_chunks):
            # Each section of block_size rows in mat is multiplied by
            # the same element of operator1, then we move onto the
            # next.  These are then summed over the elements of
            # operator1.  This way is about twice as fast as a python
            # loop with indexing.
            chunk = (operator1[row1, :, np.newaxis, np.newaxis] *
                     mat.reshape(in_chunk)).sum(axis=0)
            chunks.append(operator2.dot(chunk))
        return vstack(tuple(chunks))

    def quadratic_form(self, mat):
        r"""Calculate the quadratic form mat.T @ self @ mat.

        Parameters
        ----------
        mat: array_like[N, M]

        Returns
        -------
        result: array_like[M, M]

        Note
        ----

        Implementation depends on Kronecker structure, using the
        :meth:`_matmat` algorithm for self @ mat.  If mat is a lazy
        dask array, this implementation will load it multiple times to
        avoid dask keeping it all in memory.
        """
        if not isinstance(mat, ARRAY_TYPES) or self.shape[0] != self.shape[1]:
            # TODO: test failure mode
            raise TypeError("Unsupported")
        elif mat.ndim == 1:
            mat = mat[:, np.newaxis]
        mat = asarray(mat)
        if mat.shape[0] != self.shape[1]:
            # TODO: test failure mode
            raise ValueError("Dim mismatch")
        outer_size = mat.shape[-1]
        result_shape = (outer_size, outer_size)
        result_dtype = np.result_type(self.dtype, mat.dtype)
        # I load this into memory, so may as well keep as one chunk
        result = zeros(result_shape, dtype=result_dtype, chunks=result_shape)

        block_size = self._block_size
        chunk_shape = (block_size, mat.shape[1])
        chunk_chunks = (block_size, mat.chunks[1][0])
        operator1 = self._operator1
        operator2 = self._operator2
        # row_chunk_size = mat.chunks[0][0]
        # loops_between_save = row_chunk_size // block_size
        loops_between_save = max(
            # How many blocks there are
            (mat.shape[0] // block_size) //
            # How many blocks it needs to be
            # OPTIMAL_ELEMENTS is one chunk.
            # Each chunk will be the sum of multiple chunks
            # The three is a magic constant that will depend on machine
            # It is roughly how many chunks fit in memory at once.
            # It varies with the size of mat.
            max(mat.size // (OPTIMAL_ELEMENTS**2), 1), 1)
        row_count = 0
        print("Total loops", mat.shape[0] // block_size)
        print("Number of chunks in mat", mat.size / (OPTIMAL_ELEMENTS**2))
        print("Loop chunk:", loops_between_save)
        import sys; sys.stdout.flush(); sys.stderr.flush()
        in_chunk = (operator1.shape[1], block_size, mat.shape[1])

        for row1, row_start in enumerate(range(
                0, mat.shape[0], block_size)):
            # Two function calls and a C loop, instead of python loop
            # with lots of indexing.
            chunk = (operator1[row1, :, np.newaxis, np.newaxis] *
                     mat.reshape(in_chunk)).sum(axis=0)
            result += mat[row_start:(row_start + block_size)].T.dot(
                operator2.dot(chunk))
            # Calculate this bit so we don't run out of memory.
            # Hopefully this pushes the memory barrier well past the
            # flux correlation time.
            # It should at least get closer.
            row_count += 1
            if row_count >= loops_between_save:
                result = result.persist(**DASK_OPTIMIZATIONS)
                row_count = 0
        return result.persist(**DASK_OPTIMIZATIONS)


def method_common(inversion_method):
    """Wrap method to validate args.

    Parameters
    ----------
    inversion_method: function

    Returns
    -------
    wrapped_method: function
    """
    @functools.wraps(inversion_method)
    def wrapper(background, background_covariance,
                observations, observation_covariance,
                observation_operator,
                reduced_background_covariance=None,
                reduced_observation_operator=None):
        """Solve the inversion problem.

        Assumes everything follows a multivariate normal distribution
        with the specified covariance matrices.  Under this assumption
        `analysis_covariance` is exact, and `analysis` is the Maximum
        Likelihood Estimator and the Best Linear Unbiased Estimator
        for the underlying state in the frequentist framework, and
        specify the posterior distribution for the state in the
        Bayesian framework.  If these are not satisfied, these still
        form the Generalized Least Squares estimates for the state and
        an estimated uncertainty.

        Parameters
        ----------
        background: array_like[N]
            The background state estimate.
        background_covariance:  array_like[N, N]
            Covariance of background state estimate across
            realizations/ensemble members.  "Ensemble" is here
            interpreted in the sense used in statistical mechanics or
            frequentist statistics, and may not be derived from a
            sample as in meteorological ensemble Kalman filters
        observations: array_like[M]
            The observations constraining the background estimate.
        observation_covariance: array_like[M, M]
            Covariance of observations across realizations/ensemble
            members.  "Ensemble" again has the statistical meaning.
        observation_operator: array_like[M, N]
            The relationship between the state and the observations.
        reduced_background_covariance: array_like[Nred, Nred], optional
        reduced_observation_operator: array_like[M, Nred], optional

        Returns
        -------
        analysis: array_like[N]
            Analysis state estimate
        analysis_covariance: array_like[Nred, Nred] or array_like[N, N]
            Estimated uncertainty of analysis across
            realizations/ensemble members.  Calculated using
            reduced_background_covariance and
            reduced_observation_operator if possible
        """
        background = atleast_1d(background)
        if not isinstance(background_covariance, LinearOperator):
            background_covariance = atleast_2d(background_covariance)
            chunks = chunk_sizes((background_covariance.shape[0],),
                                 matrix_side=True)
            background_covariance = background_covariance.rechunk(
                chunks[0])

        observations = atleast_1d(observations)
        if not isinstance(observation_covariance, LinearOperator):
            observation_covariance = atleast_2d(observation_covariance)
            chunks = chunk_sizes((observation_covariance.shape[0],),
                                 matrix_side=True)
            observation_covariance = observation_covariance.rechunk(
                chunks[0])

        if not isinstance(observation_operator, LinearOperator):
            observation_operator = atleast_2d(observation_operator)

        if reduced_background_covariance is not None:
            if not isinstance(reduced_background_covariance, _LinearOperator):
                reduced_background_covariance = atleast_2d(
                    reduced_background_covariance)

            if reduced_observation_operator is None:
                raise ValueError("Need reduced versions of both B and H")
            if not isinstance(reduced_observation_operator, _LinearOperator):
                reduced_observation_operator = atleast_2d(
                    reduced_observation_operator)
        elif reduced_observation_operator is not None:
            raise ValueError("Need reduced versions of both B and H")

        analysis_estimate, analysis_covariance = (
            inversion_method(background, background_covariance,
                             observations, observation_covariance,
                             observation_operator,
                             reduced_background_covariance,
                             reduced_observation_operator))

        if analysis_covariance is None:
            B_HT = reduced_background_covariance.dot(
                reduced_observation_operator.T)
            # (I - KH) B
            analysis_covariance = (
                # May need to be a LinearOperator to work properly
                reduced_background_covariance -
                B_HT.dot(solve(
                    reduced_observation_operator.dot(B_HT) +
                    observation_covariance,
                    B_HT.T))
            )

        return analysis_estimate, analysis_covariance
    return wrapper
