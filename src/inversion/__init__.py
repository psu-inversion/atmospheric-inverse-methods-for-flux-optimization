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

as being a close approximation for :math:`\vec{x}` near :math:`\vec{x}_0`, where
:math:`H` is the derivative of :math:`h` at :math:`\vec{x}_0`, defined by

.. math::

    H := \frac{d h}{d\vec{x}} \big|_{\vec{x}=\vec{x}_0}

.. Note::

    The subpackages will generally be taking a frequentist approach to
    the problem in the descriptions.

.. Note::

    The many submodules use a `from module import names` pattern to
    get needed functions. This is to allow a substitution from other
    libraries (:func:`np.exp` instead of :func:`math.exp` for
    factorization, :func:`dask.array.exp` instead for out-of-core
    evaluation, ...)

"""
