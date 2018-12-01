"""All the linear algebra details for the package.

I figured it would be more useful to have it separate
"""
import warnings

import numpy as np
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg.interface import (
    LinearOperator,
    MatrixLinearOperator)

from scipy.sparse.linalg.eigen import eigsh as linop_eigsh
from numpy import newaxis

from numpy import concatenate, zeros, einsum
from numpy import asarray, atleast_2d, stack, where, sqrt
from scipy.linalg import cholesky
import numpy.linalg as la


from .linalg_interface import (
    DaskLinearOperator, DaskMatrixLinearOperator,
    ProductLinearOperator, tolinearoperator,
    ARRAY_TYPES, REAL_DTYPE_KINDS)

NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

See Also
--------
inversion.correlations.NEAR_ZERO
"""


# TODO: Test
# TODO: Get preconditioner working
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
        return cholesky(
            asarray(mat)
        )

    if isinstance(mat, (LinearOperator, DaskLinearOperator)):
        from inversion.util import OPTIMAL_ELEMENTS
        warnings.warn("The square root will be approximate.")
        vals, vecs = linop_eigsh(
            mat, min(mat.shape[0] // 2, OPTIMAL_ELEMENTS // mat.shape[0]))
        sqrt_valop = DiagonalOperator(sqrt(vals))
        vecop = DaskMatrixLinearOperator(vecs)
        return ProductLinearOperator(vecop, sqrt_valop, vecop.T)

    # TODO: test on xarray datasets or iris cubes
    raise TypeError("Don't know how to find square root of {cls!s}".format(
        cls=type(mat)))


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
                 np.allclose(operator1, operator1.T))):
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
        block_size = self._block_size
        operator1 = self._operator1
        operator2 = self._operator2
        in_chunk = (operator1.shape[1], block_size, mat.shape[1])

        chunks = (
            operator2.dot(
                einsum(
                    "ij,jkl->kil", operator1, mat.reshape(in_chunk),
                    # Column-major output should speed the
                    # operator2 @ tmp bit
                    order="F"
                ).reshape(block_size, -1)
            )
            # Reshape to separate out the block dimension from the
            # original second dim of mat
            .reshape(operator2.shape[0], self._n_chunks, mat.shape[1])
            # Transpose back to have block dimension first
            .transpose((1, 0, 2))
        )
        # Reshape back to expected result size
        return chunks.reshape(self.shape[0], mat.shape[1])

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
            raise ValueError("Dim mismatch: {mat:d} != {self:d}".format(
                mat=mat.shape[0], self=self.shape[1]))
        outer_size = mat.shape[-1]
        result_shape = (outer_size, outer_size)
        result_dtype = np.result_type(self.dtype, mat.dtype)
        # I load this into memory, so may as well keep as one chunk
        result = zeros(result_shape, dtype=result_dtype)

        block_size = self._block_size
        operator1 = self._operator1
        operator2 = self._operator2
        in_chunk = (operator1.shape[1], block_size, mat.shape[1])
        # row_chunk_size = mat.chunks[0][0]

        for row1, row_start in enumerate(range(
                0, mat.shape[0], block_size)):
            # Two function calls and a C loop, instead of python loop
            # with lots of indexing.
            # Having the chunk be fortran-contiguous should speed the
            # next steps (operator2 @ chunk)
            chunk = einsum("j,jkl->kl", operator1[row1, :],
                           mat.reshape(in_chunk), order="F")
            result += mat[row_start:(row_start + block_size)].T.dot(
                operator2.dot(chunk))
        return result


class SelfAdjointLinearOperator(DaskLinearOperator):
    """Self-adjoint linear operators.

    Provides :meth:`_rmatvec` and :meth:`_adjoint` methods.
    """

    def __init__(self, dtype, size):
        """Also set up transpose if operator is real."""
        # TODO: Test complex self-adjoint operators
        super(SelfAdjointLinearOperator, self).__init__(dtype, size)

        if self.dtype.kind in REAL_DTYPE_KINDS:
            # Real array; implies symmetric
            self._transpose = self._adjoint

    def _rmatvec(self, vector):
        """self.H.dot(vec).

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        # TODO: Figure out how to test this and do it
        return self._matvec(vector)

    def _adjoint(self):
        """Return transpose.

        Self-adjoint operators are their own transpose.

        Returns
        -------
        SelfAdjointLinearOperator
        """
        return self


class DiagonalOperator(SelfAdjointLinearOperator):
    """Operator with entries only on the diagonal."""

    def __init__(self, array):
        """Set up diagonal operator.

        Parameters
        ----------
        array: array_like
            The array of values to go on the diagonal.
        """
        self._diag = asarray(array).reshape(-1)
        side = self._diag.shape[0]

        super(DiagonalOperator, self).__init__(
            self._diag.dtype,
            (side, side))

        self._diag_near_zero = self._diag < NEAR_ZERO

    def _matvec(self, vector):
        """Multiply self and vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        if vector.ndim == 2:
            return self._diag[:, newaxis] * vector
        return self._diag * vector

    def _matmat(self, other):
        """Multiply self and other.

        Parameters
        ----------
        other: array_like

        Returns
        -------
        array_like
        """
        return self._diag[:, newaxis] * other

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
        result = vector / self._diag
        # result[self._diag_near_zero] = 0
        return where(self._diag_near_zero, 0, result)

    def sqrt(self):
        """Find S such that S.T @ S == self."""
        return DiagonalOperator(sqrt(self._diag))
