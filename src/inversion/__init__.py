r"""Root package for inversion methods.

Import subpackages for specific versions.

The general idea is to take two bits of data about some state, one
direct and one related by means of a known function, together with
information about their distribution, to produce a best guess as to
the state.

The state is typically referred to by the variable :math:`\vec{x}`.  The
indirect information is called an observation and denoted :math:`\vec{y}`.
The state provided to the inversion is called the background, a
priori, or prior state, while that given by the inversion is called
the analysis, a posteriori, or posterior state. The function relating
:math:`\vec{x}` and :math:`\vec{y}` is called the observation operator and is
called :math:`h`.

These methods assume that :math:`h` is differentiable near the states of
interest, so we may take the Taylor series expansion

.. math::

    h(\vec{x}) \approx h(\vec{x}_0) + H (\vec{x} - \vec{x}_0) +
    O((\vec{x} - \vec{x}_0)^2)

as being a close approximation for :math:`\vec{x}` near
:math:`\vec{x}_0`, where :math:`H` is the derivative of :math:`h` at
:math:`\vec{x}_0`, defined by

.. math::

    H := \frac{d h}{d\vec{x}} \big|_{\vec{x}=\vec{x}_0}

.. Note::

    The subpackages will generally be taking a frequentist approach to
    the problem in the descriptions.

.. Note::

    The many submodules use a `from module import names` pattern to
    get needed functions. This is to allow a substitution from other
    libraries (:func:`np.exp` instead of :func:`math.exp` for
    vectorization, :func:`dask.array.exp` instead for out-of-core
    evaluation, ...)

"""

MAX_ITERATIONS = 40
"""Max. iterations allowed during minimizations.

I think 40 is what the operational centers use.

Used by variational and PSAS schemes to constrain iterative
minimization.

Note
----
Must change test tolerances if this changes
"""
GRAD_TOL = 1e-5
"""How small the gradient norm must be to declare convergence.

From `gtol` option to the BFGS method of
:func:`scipy.optimize.minimize`

Used by variational and PSAS schemes to constrain iterative
minimization.

Note
----
Must change test tolerances if this changes.
"""


class ConvergenceError(ValueError):
    """An iterative scheme did not reach convergence.

    The idea is that those who want good answers or nothing will get
    them, and those who want to take a chance with a bad one can do
    so.  I feel good or nothing is the better default.

    """

    def __init__(self, msg, result, guess=None, hess_inv=None):
        """Save useful attributes.

        Parameters
        ----------
        msg: str
        result:
            current state of scheme
        guess: array_like[N], optional
            Last state estimate
        hess_inv: array_like[N,N], optional
            Estimate of inverse hessian
        """
        super(ConvergenceError, self).__init__(self, msg)

        self.result = result

        if guess is None:
            self.guess = result.x
        else:
            self.guess = guess

        if hess_inv is None and hasattr(result, "hess_inv"):
            self.hess_inv = result.hess_inv
        else:
            self.hess_inv = hess_inv
