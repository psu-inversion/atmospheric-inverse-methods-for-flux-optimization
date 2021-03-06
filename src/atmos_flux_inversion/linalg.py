"""All the linear algebra details for the package.

I figured it would be more useful to have it separate
"""
import warnings
from itertools import islice

import numpy as np
from numpy import newaxis
import numpy.linalg as la
from numpy.linalg import svd, LinAlgError
from numpy import einsum
from numpy import concatenate, zeros, nonzero
from numpy import asarray, atleast_2d, stack, where, sqrt
try:
    from sparse import tensordot
except ImportError:
    pass

from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg.interface import (
    LinearOperator,
    MatrixLinearOperator)

from scipy.sparse.linalg.eigen import eigsh as linop_eigsh
from scipy.sparse.linalg import svds

from scipy.linalg import cholesky

from .linalg_interface import DaskLinearOperator, DaskMatrixLinearOperator
from .linalg_interface import tolinearoperator, ProductLinearOperator
from .linalg_interface import ARRAY_TYPES, REAL_DTYPE_KINDS

NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

See Also
--------
atmos_flux_inversion.correlations.NEAR_ZERO
"""
OPTIMAL_ELEMENTS = 2 ** 16
"""Maximum size for the approximate square root of a LinearOperator.

Bounding the eigenvector matrix by this should keep everything in
memory.
"""
ROUNDOFF = 1e-10
"""Determines how many terms should be kept from SVD."""


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
        The solution to the linear equation.
    """
    _asarray = np.asarray

    def toarray(arr):
        """Make `arr` an array."""
        try:
            return arr.todense()
        except AttributeError:
            return _asarray(arr)

    if arr.ndim == 1:
        return asarray(lgmres(operator, toarray(arr),
                              atol=1e-7)[0])
    return asarray(
        stack(
            [lgmres(operator, toarray(col), atol=1e-7)[0]
             for col in arr.T],
            axis=1))


def solve(arr1, arr2):
    """Solve arr1 @ x = arr2 for x.

    Parameters
    ----------
    arr1: array_like[N, N]
        A square matrix
    arr2: array_like[N]
        A vector

    Returns
    -------
    array_like[N]
        The solution to the linear equation

    Raises
    ------
    ValueError
        if the dimensions do not match up
    LinAlgError
        if `arr1` is not square
    """
    if arr1.shape[0] != arr2.shape[0]:
        print(arr1.shape[1], arr2.shape[0])
        raise ValueError("Dimension mismatch")
    if arr1.shape[0] != arr1.shape[1]:
        raise LinAlgError("arr1 is not square")
    # Get everything in a standard form
    if isinstance(arr2, MatrixLinearOperator):
        arr2 = arr2.A
    # Deal with arr2 being a LinearOperator
    if isinstance(arr2, LinearOperator):
        def solver(vec):
            """Solve `arr1 x = vec`.

            Parameters
            ----------
            vec: array_like
                The vector for which the solution is wanted

            Returns
            -------
            array_like
                The solution of the linear equation
            """
            return solve(arr1, vec)
        inverse = DaskLinearOperator(matvec=solver, shape=arr1.shape)
        return inverse.dot(arr2)

    # arr2 is an array
    if hasattr(arr1, "solve"):
        try:
            return arr1.solve(arr2)
        except NotImplementedError:
            pass

    if isinstance(arr1, MatrixLinearOperator):
        return la.solve(asarray(arr1.A), asarray(arr2))
    if isinstance(arr1, LinearOperator):
        # Linear operators with neither an underlying matrix nor a
        # provided solver. Use iterative sparse solvers.
        return linop_solve(arr1, arr2)
        # return da.Array(
        #     {(chunkname, 0):
        #      (spsolve, arr1, arr2.rechunk(1, whatever))},
        #     "solve-arr1.name-arr2.name",
        #     chunks)
    # Shorter method for assuring dask arrays

    def toarray(arr):
        """Make `arr` an array."""
        try:
            return arr.todense()
        except AttributeError:
            return asarray(arr)

    return la.solve(toarray(arr1), toarray(arr2))


def kron(matrix1, matrix2):
    """Kronecker product of two matrices.

    Parameters
    ----------
    matrix1: array_like[M, N]
        One of the matrixes for the product
    matrix2: array_like[J, K]
        The other matrix for the product

    Returns
    -------
    array_like[M*J, N*K]
        The kronecker product of the matrices

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

    Raises
    ------
    ValueError
        if `vector` isn't actually a vector.

    Notes
    -----
    Algorithm from stackexchange:
    https://physics.stackexchange.com/questions/251522/how-do-you-find-a-schmidt-basis-and-how-can-the-schmidt-decomposition-be-used-f
    Also from Mathematica code I wrote based on description in the green
    Quantum Computation book in the reading library
    """
    if vector.ndim == 2 and vector.shape[1] != 1:
        raise ValueError("Schmidt decomposition only valid for vectors")
    state_matrix = asarray(vector).reshape(dim1, dim2)
    min_dim = min(dim1, dim2)

    if min_dim > 6:
        # svds crashes if we ask for svd output
        # Ask for at least six singular values
        # For very large inputs, ask for at least 1/20 of smaller dimension
        n_singular_vectors = min(max(6, int(0.05 * min_dim)), min_dim - 1)
        vecs1, lambdas, vecs2 = svds(state_matrix, n_singular_vectors)
    else:
        vecs1, lambdas, vecs2 = svd(state_matrix)

    big_lambdas = nonzero(lambdas)[0]

    if not big_lambdas.any():
        return lambdas[:1], vecs1.T[:1, :], vecs2[:1, :]

    return lambdas[big_lambdas], vecs1.T[big_lambdas, :], vecs2[big_lambdas, :]


def matrix_sqrt(mat):
    """Find a matrix S such that S.T @ S == mat.

    Parameters
    ----------
    mat: array_like or LinearOperator
        The square matrix for which the square root is desired

    Returns
    -------
    array_like or LinearOperator
        A matrix such that S.T @ S == mat

    Raises
    ------
    ValueError
        if mat is not symmetric
    TypeError
        if mat is of an unrecognized type
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix square root only defined for square arrays")
    if hasattr(mat, "sqrt"):
        return mat.sqrt()
    if isinstance(mat, MatrixLinearOperator):
        mat = mat.A

    if isinstance(mat, ARRAY_TYPES):
        return cholesky(
            asarray(mat)
        )

    if isinstance(mat, LinearOperator):
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
    URL: https://www.geosci-model-dev.net/6/583/2013
    :doi:`10.5194/gmd-6-583-2013`.
    """

    def __init__(self, operator1, operator2):
        """Set up the instance.

        Parameters
        ----------
        operator1: duck_array
        operator2: duck_array or LinearOperator
        """
        if isinstance(operator1, MatrixLinearOperator):
            operator1 = operator1.A
        elif isinstance(operator1, LinearOperator):
            raise ValueError("operator1 must be an array")
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
        DaskKroneckerProductOperator
            The square root S of the operator
            S.T @ S == self

        Raises
        ------
        ValueError
            if operator not self-adjoint
        ValueError
            if operator not symmetric
        """
        operator1 = self._operator1
        if ((self.shape[0] != self.shape[1] or
             operator1.shape[0] != operator1.shape[1])):
            raise ValueError(
                "Square root not defined for {shape!s} operators."
                .format(shape=self.shape))
        if self.T is not self:
            raise ValueError(
                "Square root not defined for non-symmetric operators.")
        # Cholesky can be fragile, so delegate to central location for
        # handling that.
        sqrt1 = matrix_sqrt(operator1)
        sqrt2 = matrix_sqrt(self._operator2)
        return DaskKroneckerProductOperator(
            sqrt1, sqrt2)

    def _matmat(self, X):
        r"""Compute matrix-matrix product.

        Parameters
        ----------
        X: array_like
            The matrix with which the product is desired.

        Returns
        -------
        array_like
            The product of self with X.

        Notes
        -----
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

        This function uses :mod:`dask` for the splitting, multiplication, and
        addition, which defaults to using all available cores.
        """
        block_size = self._block_size
        operator1 = self._operator1
        operator2 = self._operator2
        in_chunk = (operator1.shape[1], block_size, X.shape[1])

        if isinstance(X, ARRAY_TYPES) and isinstance(operator1, ARRAY_TYPES):
            partial_answer = einsum(
                "ij,jkl->kil", operator1, X.reshape(in_chunk),
                # Column-major output should speed the
                # operator2 @ tmp bit
                order="F"
            ).reshape(block_size, -1)
        else:
            partial_answer = tensordot(
                operator1, X.reshape(in_chunk),
                (1, 0)
            ).transpose((1, 0, 2)).reshape((block_size, -1))
        chunks = (
            operator2.dot(
                partial_answer
            )
            # Reshape to separate out the block dimension from the
            # original second dim of X
            .reshape((operator2.shape[0], self._n_chunks, X.shape[1]))
            # Transpose back to have block dimension first
            .transpose((1, 0, 2))
        )
        # Reshape back to expected result size
        return chunks.reshape((self.shape[0], X.shape[1]))

    def quadratic_form(self, mat):
        r"""Calculate the quadratic form mat.T @ self @ mat.

        Parameters
        ----------
        mat: array_like[N, M]

        Returns
        -------
        array_like[M, M]
            The product mat.T @ self @ mat

        Raises
        ------
        TypeError
            if mat is not an array or if self is not square
        ValueError
            if the shapes of self and mat are not compatible

        Notes
        -----
        Implementation depends on Kronecker structure, using the
        :meth:`._matmat` algorithm for self @ mat.  If mat is a lazy
        dask array, this implementation will load it multiple times to
        avoid dask keeping it all in memory.

        This function uses :mod:`dask` for the splitting, multiplication, and
        addition, which defaults to using all available cores.
        """
        if self.shape[0] != self.shape[1]:
            raise TypeError("quadratic_form only defined for square matrices.")
        elif isinstance(mat, LinearOperator):
            raise TypeError("quadratic_form only supports explicit arrays.")
        elif not isinstance(mat, ARRAY_TYPES):
            warnings.warn("mat not a recognised array type.  "
                          "Proceed with caution.")
        elif mat.ndim == 1:
            mat = mat[:, np.newaxis]
        if mat.shape[0] != self.shape[1]:
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
            if isinstance(mat, np.ndarray):
                chunk = einsum("j,jkl->kl", operator1[row1, :],
                               mat.reshape(in_chunk), order="F")
            else:
                chunk = tensordot(operator1[row1, :], mat.reshape(in_chunk), 1)
            result += mat[row_start:(row_start + block_size)].T.dot(
                operator2.dot(chunk))
        return result


class SchmidtKroneckerProduct(DaskLinearOperator):
    """Kronecker product of two operators using Schmidt decomposition.

    This works best when the input vectors are nearly Kronecker
    products as well, dominated by some underlying structure with
    small variations.  One example would be average net flux + trend
    in net flux + average daily cycle + daily cycle timing variations
    across domain + localized events + ...

    Multiplications are roughly the same time complexity class as with
    an explicit Kronecker Product, perhaps a factor of two or three
    slower in the best case, but the memory requirements are
    :math:`N_1^2 + N_2^2` rather than :math:`(N_1 * N_2)^2`, plus this
    approach works with sparse matrices and other LinearOperators
    which can further reduce the memory requirements and may decrease
    the time complexity.

    Forming the Kronecker product from the component vectors currently
    requires the whole thing to be in memory, so a new implementation
    of kron would be needed to take advantage of this. There may be
    some difficulties with the dask cache getting flushed and causing
    repeat work in this case. I don't know how to get around this.
    """

    def __init__(self, operator1, operator2):
        """Set up the instance.

        Parameters
        ----------
        operator1, operator2: scipy.sparse.linalg.LinearOperator
            The operators input to the Kronecker product.
        """
        operator1 = tolinearoperator(operator1)
        operator2 = tolinearoperator(operator2)
        total_shape = np.multiply(operator1.shape, operator2.shape)

        super(SchmidtKroneckerProduct, self).__init__(
            shape=tuple(total_shape),
            dtype=np.result_type(operator1.dtype, operator2.dtype))

        self._inshape1 = operator1.shape[1]
        self._inshape2 = operator2.shape[1]
        self._operator1 = operator1
        self._operator2 = operator2

    def _transpose(self):
        """Return the transpose of the operator."""
        return type(self)(self._operator1.T, self._operator2.T)

    def _matvec(self, vector):
        """Evaluate the indicated matrix-vector product.

        Parameters
        ----------
        vector: array_like[N]

        Returns
        -------
        array_like[M]
        """
        result_shape = self.shape[0]

        lambdas, vecs1, vecs2 = schmidt_decomposition(
            asarray(vector), self._inshape1, self._inshape2)

        # The vector should fit in memory, and I need specific
        # elements of lambdas
        lambdas = np.asarray(lambdas)
        vecs1 = np.asarray(vecs1)
        vecs2 = np.asarray(vecs2)

        small_lambdas = np.nonzero(lambdas < lambdas[0] * ROUNDOFF)[0]
        if small_lambdas.any():
            last_lambda = int(small_lambdas[0])
        else:
            last_lambda = len(lambdas)

        result = zeros(shape=result_shape,
                       dtype=np.result_type(self.dtype, vector.dtype))
        for lambd, vec1, vec2 in islice(zip(lambdas, vecs1, vecs2),
                                        0, last_lambda):
            result += kron(
                asarray(lambd * self._operator1.dot(vec1).reshape(-1, 1)),
                asarray(self._operator2.dot(vec2).reshape(-1, 1))
            ).reshape(result_shape)

        return asarray(result)


class SelfAdjointLinearOperator(DaskLinearOperator):
    """Self-adjoint linear operators.

    Provides :meth:`_rmatvec` and :meth:`_adjoint` methods.
    """

    def __init__(self, dtype, shape):
        """Also set up transpose if operator is real.

        Raises
        ------
        ValueError
            if operator would not be square
        """
        if shape[0] != shape[1]:
            raise ValueError("Self-adjoint operators must be square")
        super(SelfAdjointLinearOperator, self).__init__(dtype, shape)

        if self.dtype.kind in REAL_DTYPE_KINDS:
            # Real array; implies symmetric
            self._transpose = self._adjoint

    def _rmatvec(self, vector):
        """self.H.dot(vec).

        Since self is self-adjoint, self is self.H

        Parameters
        ----------
        vector: array_like
            The vector with which the product is desired.

        Returns
        -------
        array_like
            The product of self and vector
        """
        # TODO: Figure out how to test this and do it
        return self._matvec(vector)  # pragma: no cover

    def _adjoint(self):
        """Return transpose.

        Self-adjoint operators are their own transpose.

        Returns
        -------
        SelfAdjointLinearOperator
            The self-adjoint Hermitian adjoint of self.
            Since self is self-adjoint, this is self.
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
        if isinstance(array, DiagonalOperator):
            self._diag = array._diag  # noqa: W0212
        else:
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
            The vector with which the product is desired

        Returns
        -------
        array_like
            The product
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
