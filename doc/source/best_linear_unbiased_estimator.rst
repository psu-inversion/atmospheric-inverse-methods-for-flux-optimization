==============================
Best Linear Unbiased Estimator
==============================

This derivation follows the Grubbs and Weaver (1947) definition of the
Best Linear Unbiased Estimator.

We have two estimates related to the true flux :math:`\vec{x}_t`: some
flux estimate :math:`\vec{x}_b` and an independent set of atmospheric
measurements :math:`\vec{y}`.  We think the difference
:math:`\epsilon_b = \vec{x}_b - \vec{x}_a` has mean zero and
variance-covariance matrix :math:`B`.  Similarly, we think the
measurement errors :math:`\epsilon_o = \vec{y} - H \vec{x}_t` have
mean zero and variance-covariance :math:`R`.  We would like to find
the linear combination of these estimates that is an unbiased
estimator for :math:`\vec{x}_t` and has the lowest variance of such
estimators.

That is, we seek :math:`\vec{x}_a` such that:

.. math::

   \vec{x}_a = A_1 \vec{x}_b + A_2 \vec{y} + A_3 \\

   E[\vec{x}_a - \vec{x}_t] = 0 \\

   \vec{x}_a = \operatorname{argmin}( E[\vec{x}_a \cdot \vec{x}_a] )

for some matrices :math:`A_1` and :math:`A_2` and some vector :math:`A_3`.

Let us start with the first requirement.  To obtain an unbiased
estimator, we must have:

.. math::

   \vec{x}_t &= E[\vec{x}_a] \\
   &= E[A_1 \vec{x}_b + A_2 \vec{y} + A_3] \\
   &= E[A_1 (\vec{x}_t + \epsilon_b)] + E[A_2 (H \vec{x}_t + \epsilon_o)] + E[A_3] \\
   &= A_1 E[\vec{x}_t] + A_1 E[\epsilon_b] + A_2 H E[\vec{x}_t] + A_2 H E[\epsilon_o] + A_3 \\
   &= A_1 \vec{x}_t + 0 + A_2 H \vec{x}_t + 0 + A_3 \\
   &= (A_1 + A_2 H) \vec{x}_t + A_3

Since we want this equality to hold for all values of
:math:`\vec{x}_t`, we must have :math:`A_3 = 0` and :math:`A_1 + A_2 H = I`.

With this relationship between :math:`A_1` and :math:`A_2`, we can
start on the second requirement.  We want to minimize
:math:`E[\vec{x}_a \cdot \vec{x}_a]`, where :math:`\vec{x}_a` is now
given by :math:`(I - A_2 H) \vec{x}_b + A_2 \vec{y}`.  The quantity we
are trying to minimize is the trace of the covariance matrix of
:math:`\vec{x}_a`, which is given by the covariance of
:math:`\vec{x}_a` with itself:

.. math::

   \DeclareMathOperator{\Cov}{Cov}
   \Cov[\vec{x}_a, \vec{x}_a] &= \Cov[(I - A_2 H) \vec{x}_b + A_2 \vec{y}, (I - A_2 H) \vec{x}_b + A_2 \vec{y}] \\
   &= \Cov[(I - A_2 H) \vec{x}_b, (I - A_2 H) \vec{x}_b] + \Cov[(I - A_2 H) \vec{x}_b, A_2 \vec{y}] + \Cov[A_2 \vec{y}, (I - A_2 H) \vec{x}_b] + \Cov[A_2 \vec{y}, A_2 \vec{y}] \\
   &= (I - A_2 H) \Cov[\vec{x}_b, \vec{x}_b] (I - A_2 H)^T + (I - A_2 H) \Cov[\vec{x}_b, \vec{y}] A_2^T + A_2 \Cov[\vec{y}, \vec{x}_b] (I - A_2 H)^T + A_2 \Cov[\vec{y}, \vec{y}] A_2^T \\
   &= (I - A_2 H) B (I - H^T A_2^T) + 0 + 0 + A_2 T A_2^T \\
   &= B - B H^T A_2^T - A_2 H B + A_2 H B H^T A_2^T + A_2 R A_2^T

To find the minimum, we will need to find the derivative of the trace
of this covariance matrix with respect to A_2.

.. math::

   \DeclareMathOperator{\trace}{tr}
   \frac{d}{dA_2} \trace(B - B H^T A_2^T - A_2 H B + A_2 H B H^T A_2^T + A_2 R A_2^T) = \\
   \frac{d}{dA_2} [\trace(B) - \trace(B H^T A_2^T) - \trace(A_2 H B) + \trace(A_2 H B H^T A_2^T) + \trace(A_2 R A_2^T) = \\
   \frac{d}{dA_2} \trace(B) - \frac{d}{dA_2} 2 \trace(B H^T A_2^T) + \frac{d}{dA_2} \trace(A_2 H B H^T A_2^T) + \frac{d}{d A_2} \trace(A_2 R A_2^T) = \\
   0 - 2 \trace(B H^T) + 2 \trace(A_2 H B H^T) + 2 \trace(A_2 R) = \\
   2 \trace[-B H^T + A_2 (H B H^T + R)]

We find the minimum where the derivative is equal to zero.  The
simplest way to find that is to set everything inside the trace to be
equal to zero and solve for :math:`A_2`.

.. math::

   0 &= -B H^T + A_2 (HBH^T + R) \\
   B H^T &= A_2 (H B H^T + R) \\
   B H^T (H B H^T + R)^{-1} &= A_2 (H B H^T + R) (H B H^T + R)^{-1} \\
   A_2 &= B H^T (H B H^T + R)^{-1}

That is, the best linear unbiased estimate :math:`\vec{x}_a` of the
true fluxes :math:`\vec{x}_t` given a previous estimate
:math:`\vec{x}_b` and observations :math:`\vec{y}` with uncertainties
:math:`B` and :math:`R`, respectively, is

.. math::

   \vec{x}_a &= (I - A_2 H) \vec{x}_b + A_2 \vec{y} \\
   &= \vec{x}_b + A_2 (\vec{y} - H \vec{x}_b) \\
   &= \vec{x}_b + B H^T (H B H^T + R)^{-1} (\vec{y} - H \vec{x}_b)

with covariance matrix given by

.. math::

   A &= (I - A_2 H) B (I - H^T A_2^T) + A_2 R A_2^T \\
     &= (I - A_2 H) B - (I - A_2 H) B H^T A_2^T + A_2 R A_2^T \\
     &= (I - A_2 H) B - [(I - A_2 H) B H^T - A_2 R] A_2^T \\
     &= (I - A_2 H) B - [B H^T - A_2 (H B H^T + R)] A_2^T \\
     &= (I - A_2 H) B - [B H^T - B H^T (H B H^T + R)^{-1} (H B H^T + R)] A_2^T \\
     &= (I - A_2 H) B - [B H^T - B H^T I] A_2^T \\
     &= (I - A_2 H) B - 0 \\
     &= B - B H^T (H B H^T + R)^{-1} H B

References
==========

Grubbs, F., and Weaver, C. (1947). The Best Unbiased Estimate of
Population Standard Deviation Based on Group Ranges.  *Journal of the
American Statistical Association*, 42(238), 224--241.
:doi:`10.2307/2280652`, :jstor:`2280652`
