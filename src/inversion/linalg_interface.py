"""LinearOperator subclasses to avoid np.asarray calls.

Copied from :mod:`scipy.sparse.linalg.interface`
"""
import numpy as np
from numpy import promote_types
from numpy import empty, asarray, atleast_2d

from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg.interface import (
    LinearOperator,
    MatrixLinearOperator,
    _CustomLinearOperator, _SumLinearOperator,
    _ScaledLinearOperator)

try:
    import dask.array as da
    try:
        import sparse
        ARRAY_TYPES = (np.ndarray, da.Array, sparse.COO)
    except ImportError:
        ARRAY_TYPES = (np.ndarray, da.Array)
except ImportError:
    ARRAY_TYPES = (np.ndarray,)
"""Array types for determining Kronecker product type.

These are combined for a direct product.
"""
REAL_DTYPE_KINDS = "fiu"
"""The kinds used by dtypes to represent real numbers.

Includes subsets.
"""


def is_odd(num):
    """Return oddity of num.

    Parameters
    ----------
    num: int
        The number to test.

    Returns
    -------
    bool
        Whether the number is odd
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


############################################################
# Start of copied code
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
        """Create a new DaskLinearOperator.

        If called directly, expects matvec and/or matmul arguments.
        If called from a subclass, checks that one of `_matvec` and
        `_matmat` is defined since the default definitions call each
        other.
        """
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

        Default matrix-matrix multiplication handler.  Optimized for numpy.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """
        # return hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])
        ncols = X.shape[1]
        result = empty((self.shape[0], ncols),
                       order="F",
                       dtype=promote_types(self.dtype, X.dtype))
        for i in range(ncols):
            result[:, i] = self.matvec(X[:, i])
        return result

    def _matvec(self, x):
        """Handle matrix-vector multiplication in default manner.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        return self.matmat(x.reshape((-1, 1)))

    def matvec(self, x):
        """

        Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : array_like
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : array_like
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """
        if not isinstance(x, ARRAY_TYPES):
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
        x : array_like
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : array_like
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
        X : array_like
            An array with shape (N,K).

        Returns
        -------
        Y : array_like
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        if not isinstance(X, ARRAY_TYPES):
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
            if not isinstance(x, ARRAY_TYPES):
                x = asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def _adjoint(self):
        """Return adjoint of self; defers to rmatvec."""
        shape = (self.shape[1], self.shape[0])
        return _DaskCustomLinearOperator(shape, matvec=self.rmatvec,
                                         rmatvec=self.matvec,
                                         dtype=self.dtype)


class _DaskCustomLinearOperator(_CustomLinearOperator, DaskLinearOperator):
    """This should let the factory functions above work."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None):
        super(_DaskCustomLinearOperator, self).__init__(
            shape=shape, matvec=matvec, rmatvec=rmatvec,
            matmat=matmat, dtype=dtype
        )

        if ((self.dtype.kind in REAL_DTYPE_KINDS and
             not hasattr(self, "_transpose"))):
            self._transpose = self._adjoint


class DaskMatrixLinearOperator(MatrixLinearOperator, DaskLinearOperator):
    """This should help out with the tolinearoperator.

    Should I override __add__, ...?
    """

    def __init__(self, A):
        """Wrap A in a LinearOperator."""
        super(DaskMatrixLinearOperator, self).__init__(A)
        self.__transp = None
        self.__adj = None

    def _transpose(self):
        """Return the transpose."""
        if self.__transp is None:
            self.__transp = _DaskTransposeLinearOperator(self)
        return self.__transp

    def _adjoint(self):
        """Return the Hermitian adjoint of self."""
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
        super(_DaskAdjointLinearOperator, self).__init__(adjoint.A.T.conj())
        self.__adjoint = adjoint

    def _adjoint(self):
        return self.__adjoint


class _DaskSumLinearOperator(_SumLinearOperator, DaskLinearOperator):
    """Sum of two DaskLinearOperators."""

    pass


class _DaskScaledLinearOperator(_ScaledLinearOperator, DaskLinearOperator):
    """Scaled linear operator."""

    pass


############################################################
# Not copied from scipy: I needed added methods
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
        for op1, op2 in zip(operators[:-1], operators[1:]):
            if op1.shape[1] != op2.shape[0]:
                raise ValueError("Incompatible dimensions")
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
        except AttributeError:
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

    def _rmatvec(self, vector):  # pragma: nocover
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
        """Calculate the matrix-matrix product.

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
        """Return the Hermitian adjoint of the operator."""
        return ProductLinearOperator(
            *[op.H for op in reversed(self._operators)])

    def _transpose(self):
        """Return the transpose of the operator."""
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

        See Also
        --------
        inversion.util.linop_solve
            Will likely be faster if there are multiple linear
            operators in the chain
        """
        from .linalg import solve
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
            from .linalg import matrix_sqrt
            middle_operator = operators[half_n_ops]

            return ProductLinearOperator(
                matrix_sqrt(middle_operator), *last_operators)
        return ProductLinearOperator(*last_operators)
