===================================
Generalized Least Squares Estimator
===================================

Given a previous estimate :math:`\vec{x}_b` of the fluxes, with
uncertainty expressed as a covariance matrix :math:`B`, together with
independent atmospheric measurements :math:`\vec{y}` related to the
fluxes by :math:`\vec{y} \approx H \vec{x}_b`, with uncertainty
expressed as a covariance matrix :math:`R`, we seek the estimate
:math:`\vec{x}_a` that minimizes the distance from each of them given
their respective uncertainties.

.. math::

   J(\vec{x}_a) = (\vec{x}_a - \vec{x}_b)^T B^{-1} (\vec{x}_a - \vec{x}_b) + (\vec{y} - H \vec{x}_a)^T R^{-1} (\vec{y} - H\vec{x}_a)

To find the minimum of this quadratic expression, we need the derivative.

.. math::

   \frac{d J(\vec{x}_a)}{d\vec{x}_a} &= 2 B^{-1} (\vec{x}_a - \vec{x}_b) - 2 H^T R^{-1} (\vec{y} - H \vec{x}_a) \\
   \frac{1}{2} \frac{d J(\vec{x}_a)}{d\vec{x}_a} &= B^{-1} \vec{x}_a - B^{-1} \vec{x}_b - H^T R^{-1} \vec{y} + H^T R^{-1} H \vec{x_a} \\
   &= (B^{-1} + H^T R^{-1} H) \vec{x}_a - B^{-1} \vec{x}_b - H^T R^{-1} \vec{y}

Seting this derivative equal to zero and solving for :math:`\vec{x}_a`
will give the location of the minimum of the cost function
:math:`J(\vec{x}_a)`.

.. math::

   0 &= (B^{-1} + H^T R^{-1} H) \vec{x}_a - B^{-1} \vec{x}_b - H^T R^{-1} \vec{y} \\
   B^{-1} \vec{x}_b + H^T R^{-1} \vec{y} &= (B^{-1} + H^T R^{-1} H) \vec{x}_a \\
   (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} \vec{x}_b + H^T R^{-1} \vec{y}) &= (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} + H^T R^{-1} H) \vec{x}_a \\
   \vec{x}_a &= (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} \vec{x}_b + H^T R^{-1} \vec{y})

References
==========

Aitken, A. (1936).  IV.---On Least Squares and Linear Combination of
Observations.  *Proceedings of the Royal Society of Edinburgh*, 55,
42--48.  :doi:`10.1017/S0370164600014346`
