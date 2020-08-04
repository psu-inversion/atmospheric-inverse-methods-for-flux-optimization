==================
Bayesian Estimator
==================

Our prior information about the surface fluxes :math:`\vec{x}` can be
represented by a multivariate normal prior distribution:

.. math::

   \vec{x} \sim \mathcal{N}(\vec{x}_b, B)

The likelihood of observing a set of atmospheric measurements
:math:`\vec{y}` given surface fluxes :math:`\vec{x}` is another
multivariate normal:

.. math::

   \vec{y} | \vec{x} \sim \mathcal{N}(H \vec{x}, R)

We can combine these to obtain the posterior distribution for the
surface fluxes :math:`\vec{x}`.

Set up Bayes's rule:

.. math::

   P(\vec{x} | \vec{y})
   &= \frac{P(\vec{y} | \vec{x}) P(\vec{x})}{\int_{\vec{x}} P(\vec{y} | \vec{x}) d\vec{x}} \\
   &\propto P(\vec{y} | \vec{x}) P(\vec{x})

Substitute in the prior and likelihood from above:

.. math::

   P(\vec{x} | \vec{y})
   &\propto (2\pi)^{-\frac{M}{2}} \det(R)^{-\frac{1}{2}}
   \exp[-\frac{1}{2} (\vec{y} - H \vec{x})^T R^{-1} (\vec{y} - H \vec{x})] \\
   &\quad \times (2\pi)^{-\frac{N}{2}} \det(B)^{-\frac{1}{2}}
   \exp[-\frac{1}{2} (\vec{x} - \vec{x}_b)^T B^{-1} (\vec{x} - \vec{x}_b)]

Rearrange:

.. math::

   P(\vec{x} | \vec{y})
   &\propto (2\pi)^{-\frac{M + N}{2}} \det(R)^{-\frac{1}{2}} \det(B)^{-\frac{1}{2}} \\
   &\quad \times \exp[-\frac{1}{2} (\vec{y} - H \vec{x})^T R^{-1} (\vec{y} - H \vec{x}_b)
   - \frac{1}{2} (\vec{x} - \vec{x}_b)^T B^{-1} (\vec{x} - \vec{x}_b)]]

Drop the constant factors out front and expand the quadratic forms:

.. math::

   P(\vec{x} | \vec{y})
   &\propto \exp[-\frac{1}{2} (
   \vec{y}^T R^{-1} \vec{y} - \vec{x}^T H^T R^{-1} \vec{y}
   - \vec{y}^T R^{-1} H \vec{x} + \vec{x}^T H^T R^{-1} H \vec{x} \\
   &\qquad\qquad\qquad + \vec{x}^T B^{-1} \vec{x} - \vec{x}^T B^{-1} \vec{x_b}
   - \vec{x}_b^T B^{-1} \vec{x} + \vec{x}_b^T B^{-1} \vec{x}_b)]

Then rearrange the terms in the exponential until they look familiar.

.. math::

   P(\vec{x} | \vec{y})
   &\propto \exp\{-\frac{1}{2} [
   \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x} + \vec{y}^T R^{-1} \vec{y}
   - \vec{x}^T H^T R^{-1} \vec{y} - \vec{y}^T R^{-1} H \vec{x} \\
   &\qquad\qquad\qquad - \vec{x}^T B^{-1} \vec{x}_b
   - \vec{x}_b^T B^{-1} \vec{x} + \vec{x}_b^T B^{-1} \vec{x}_b]\}

At this point, it is apparent the posterior is a multivariate Gaussian
with covariance :math:`A = (B^{-1} + H R^{-1} H^T)^{-1}`, but the mean
is not yet evident [1]_.  Since the mean of a multivariate Gaussian is
also the mode, the location of the maximum of the posterior
probability is the posterior mean.  The exponential function is
monotonic increasing and the negation is monotonic decreasing, so this
is equivalent to finding the minimum of

.. math::

   J(x) &= \frac{1}{2} [
   \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x} + \vec{y}^T R^{-1} \vec{y}
   - \vec{x}^T H^T R^{-1} \vec{y} - \vec{y}^T R^{-1} H \vec{x} \\
   &\qquad\qquad - \vec{x}^T B^{-1} \vec{x}_b
   - \vec{x}_b^T B^{-1} \vec{x} + \vec{x}_b^T B^{-1} \vec{x}_b]

The derivative of this function with respect to :math:`\vec{x}` is

.. math::

   \frac{dJ}{d \vec{x} } = (B^{-1} + H R^{-1} H^T) \vec{x} - H^T R^{-1} \vec{y} - B^{-1} \vec{x}_b

Setting this equal to zero to find the maximum of the posterior
probability density :math:`\vec{x}_a` and solving for
:math:`\vec{x}_a` gives

.. math::

   0 = (B^{-1} + H R^{-1} H^T) \vec{x}_a - H^T R^{-1} \vec{y} - B^{-1} \vec{x}_b \\
   (B^{-1} + H R^{-1} H^T) \vec{x}_a = H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b \\
   \vec{x}_a = (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)

This can be rearranged to look more like the answers from other methods:

.. math::

   \vec{x}_a &= (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} - H^T R^{-1} H \vec{x}_b
       + H^T R^{-1} H \vec{x}_b + B^{-1} \vec{x}_b) \\
   &= (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} - H^T R^{-1} H \vec{x}_b)
       + (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} H \vec{x}_b + B^{-1} \vec{x}_b) \\
   &= (B^{-1} + H R^{-1} H^T)^{-1} H^T R^{-1} (\vec{y} - H \vec{x}_b)
       + (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} H + B^{-1}) \vec{x}_b \\
   &= (B^{-1} + H R^{-1} H^T)^{-1} H^T R^{-1} (\vec{y} - H \vec{x}_b) + \vec{x}_b \\
   &= \vec{x}_b + (B^{-1} + H R^{-1} H^T)^{-1} H^T R^{-1} (\vec{y} - H \vec{x}_b).

Given this, the full posterior probability density is

.. math::

   P(\vec{x} | \vec{y}) &= (2 \pi)^{-\frac{N}{2}} \det(A)^{-\frac{1}{2}}
       \exp[-\frac{1}{2} (\vec{x} - \vec{x}_a)^T A^{-1} (\vec{x} - \vec{x}_a)] \\
   &\propto \exp\{-\frac{1}{2} [\vec{x} - (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)]^T \\
   &\qquad\qquad (B^{-1} + H R^{-1} H^T) [\vec{x} - (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)] \\
   &= \exp\{-\frac{1}{2} [
       \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x} + \\
   &\qquad\qquad (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T  (B^{-1} + H R^{-1} H^T)^{-T} (B^{-1} + H R^{-1} H^T) \\
   &\qquad\qquad (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b) - \\
   &\qquad\qquad (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T  (B^{-1} + H R^{-1} H^T)^{-T} (B^{-1} + H R^{-1} H^T) \vec{x} - \\
   &\qquad\qquad \vec{x} (B^{-1} + H R^{-1} H^T) (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)
   ]\}

Simplifying this and using the symmetry of :math:`B` and :math:`R` gives

.. math::

   P(\vec{x} | \vec{y}) &\propto \exp\{-\frac{1}{2} [
       \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x} + \\
   &\qquad\qquad (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b) - \\
   &\qquad\qquad (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T \vec{x} - \\
   &\qquad\qquad \vec{x} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)
   ]\} \\
   &= \exp[-\frac{1}{2}
       (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T (B^{-1} + H R^{-1} H^T)^{-1} (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)
   ] \\
   &\quad \exp\{-\frac{1}{2} [
       \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x} -
       (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)^T \vec{x} -
       \vec{x}^T (H^T R^{-1} \vec{y} + B^{-1} \vec{x}_b)
  ]\} \\
  &\propto \exp\{-\frac{1}{2} [
      \vec{x}^T (B^{-1} + H R^{-1} H^T) \vec{x}
      - \vec{y}^T R^{-1} H \vec{x} - \vec{x}_b^T B^{-1} \vec{x}
      - \vec{x}^T H^T R^{-1} \vec{y} - \vec{x}^T B^{-1} \vec{x}_b
  ]\},

which you will recognize from the line that was recognized as a
multivariate Gaussian.

.. [1] The highest-order term in the exponential is a quadratic of the
       form :math:`\vec{x}^T A^{-1} \vec{x}`.  The linear terms will
       shift the mean around a bit.
