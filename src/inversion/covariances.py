"""Support classes for covariances.

See Also
--------
inversion.correlations
"""
from __future__ import absolute_import
from .linalg import (
    ProductLinearOperator, SelfAdjointLinearOperator,
    DiagonalOperator, matrix_sqrt)


# TODO: move to covariances.py and inherit from SelfAdjointOperator
class CorrelationStandardDeviation(ProductLinearOperator,
                                   SelfAdjointLinearOperator):
    """Represent correlation-std product."""

    def __init__(self, correlation, std):
        """Set up instance to use given parameters.

        Parameters
        ----------
        correlation: LinearOperator[N, N]
            Correlations
        std: array_like[N]
            Standard deviations
        """
        std_matrix = DiagonalOperator(std)
        super(CorrelationStandardDeviation, self).__init__(
            std_matrix, correlation, std_matrix)

    def _transpose(self):
        """Return transpose of self."""
        return self

    def _adjoint(self):
        """Return adjoint of self."""
        return self

    def sqrt(self):
        """Find S such that S.T @ S == self."""
        std_matrix, correlation, _ = self._operators

        return ProductLinearOperator(
            matrix_sqrt(correlation), std_matrix)

    _sqrt = sqrt
