"""Support classes for covariances.

See Also
--------
inversion.correlations
"""
from numpy import newaxis

from inversion.util import tolinearoperator, solve, REAL_DTYPE_KINDS
from inversion.util import DaskLinearOperator as LinearOperator


NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

See Also
--------
inversion.correlations.NEAR_ZERO
"""


class SelfAdjointLinearOperator(LinearOperator):
    """Self-adjoint linear operators.

    Provides :meth:`_rmatvec` and :meth:`_adjoint` methods.
    """

    def __init__(self, dtype, size):
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
        self._diag = array.reshape(-1)
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
        result[self._diag_near_zero] = 0


class ProductLinearOperator(LinearOperator):
    """Represent a product of linear operators."""

    def __init__(self, *operators):
        """Set up a product on linear operators.

        Parameters
        ----------
        operators: LinearOperator
        """
        super(ProductLinearOperator, self).__init__(
            None, (operators[0].shape[0], operators[-1].shape[1]))
        self._operators = tuple(tolinearoperator(op)
                                for op in operators)
        self._init_dtype()

    def _matvec(self, vector):
        """The matrix-vector product with vector.

        Parameters
        ----------
        vector: array_like

        Returns
        -------
        array_like
        """
        for op in reversed(self._operators):
            vector = op.dot(vector)

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
        for op in self._operators:
            vector = op.dot(vector)

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
            matrix = op.dot(matrix)

        return matrix

    def _adjoint(self):
        """The Hermitian adjoint of the operator."""
        return ProductLinearOperator(
            [op.H for op in reversed(self._operators)])

    def _transpose(self):
        """The transpose of the operator."""
        return ProductLinearOperator(
            [op.T for op in reversed(self._operators)])

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
