"""Support classes for covariances.

See Also
--------
inversion.correlations
"""
from numpy import newaxis

from dask.array import where

from inversion.util import REAL_DTYPE_KINDS
from inversion.util import DaskLinearOperator

NEAR_ZERO = 1e-20
"""Where correlations are rounded to zero.

The method of assuring positive definiteness increases some values
away from zero due to roundoff. Values that were originally smaller
than this are reset to zero.

See Also
--------
inversion.correlations.NEAR_ZERO
"""


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
        # TODO: test
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
