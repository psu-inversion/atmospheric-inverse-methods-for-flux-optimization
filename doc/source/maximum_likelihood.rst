============================
Maximum Likelihood Estimator
============================

We have a background estimate :math:`\vec{x}_b` of the :math:`N`
fluxes drawn from a multivariate normal distribution such that

.. math::

   \epsilon_b := \vec{x}_b - \vec{x}_t \sim \mathcal{N}(0, B)

We also have :math:`M` measurements :math:`\vec{y}` drawn from an
independent multivariate normal distribution such that

.. math::

   \epsilon_o := \vec{y} - H \vec{x}_t \sim \mathcal{N}(0, R)

We seek the estimator :math:`\vec{x}_a` of :math:`\vec{x}_t` that
maximizes the likelihood of both events occurring.  The likelihood of
:math:`\vec{x}_a` given :math:`\vec{x}_b` and :math:`\vec{y}` is the
joint probability of :math:`\vec{x}_b` and :math:`\vec{y}` given
:math:`\vec{x}_t = \vec{x}_a`:

.. math::

   L(\vec{x}_a; \vec{x}_b, \vec{y}) &= P(\vec{x}_b, \vec{y} | \vec{x}_a) \\
   &= (2 \pi)^{\frac{N}{2}} det(B)^{-\frac{1}{2}} exp[-\frac{1}{2} (\vec{x}_b - \vec{x}_a)^T B^{-1} (\vec{x}_b - \vec{x}_a)] \\
   &\quad\times (2 \pi)^{\frac{M}{2}} det(R)^{-\frac{1}{2}} exp[-\frac{1}{2} (\vec{y} - H \vec{x}_a)^T R^{-1} (\vec{y} - H \vec{x}_a)] \\
   &= (2 \pi)^{\frac{N + M}{2}} det(B)^{-\frac{1}{2}} det(R)^{-\frac{1}{2}} \\
   &\quad\times exp\{-\frac{1}{2} [(\vec{x}_b - \vec{x}_a)^T B^{-1} (\vec{x}_b - \vec{x}_a) + (\vec{y} - H \vec{x}_a)^T R^{-1} (\vec{y} - H \vec{x}_a)]\}

Since the logarithm is an increasing function, maximizing the
logarithm of a function is equivalent to maximizing the original
function.  It is therefore convenient to work with the log-likelihood
:math:`\ell(\vec{x}_a; \vec{x}_b, \vec{y})`

.. math::

   \ell(\vec{x}_a; \vec{x}_b, \vec{y})
   &= \frac{N + M}{2} ln (2 \pi) - \frac{1}{2} ln[det(B)] - \frac{1}{2} ln[det(R)] \\
   &\quad - \frac{1}{2} [(\vec{x}_b - \vec{x}_a)^T B^{-1} (\vec{x}_b - \vec{x}_a) + (\vec{y} - H \vec{x}_a)^T R^{-1} (\vec{y} - H \vec{x}_a)]

To maximize the log-likelihood, and therefore the likelihood, we need
to find the derivative of the log-likelihood and set it equal to zero.

.. math::

   0 &= \frac{d}{d \vec{x}_a} \ell(\vec{x}_a; \vec{x}_b, \vec{y}) \\
   &= -\frac{1}{2} \frac{d}{d \vec{x}_a} [(\vec{x}_b - \vec{x}_a)^T B^{-1} (\vec{x}_b - \vec{x}_a) + (\vec{y} - H \vec{x}_a)^T R^{-1} (\vec{y} - H \vec{x}_a)] \\
   &= -\frac{1}{2} [-2 B^{-1} (\vec{x}_b - \vec{x}_a) - 2 H^T R^{-1} (\vec{y} - H \vec{x}_a)] \\
   &= B^{-1} (\vec{x}_b - \vec{x}_a) + H^T R^{-1} (\vec{y} - H \vec{x}_a) \\
   &= B^{-1} \vec{x}_b + H^T R^{-1} \vec{y} - (B^{-1} + H^T R^{-1} H) \vec{x}_a \\
   (B^{-1} + H^T R^{-1} H) \vec{x}_a &= B^{-1} \vec{x}_b + H^T R^{-1} \vec{y} \\
   \vec{x}_a &= (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} \vec{x}_b + H^T R^{-1} \vec{y})


The covariance matrix of this estimator is then:

.. math::

   Cov[\vec{x}_a, \vec{x}_a]
   &= Cov[(B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} \vec{x}_b + H^T R^{-1} \vec{y}), \\
   &\qquad\qquad (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} \vec{x}_b + H^T R^{-1} \vec{y})]

We the exploit the bilinearity of the covariance operator to get

.. math::

   Cov[\vec{x}_a, \vec{x}_a]
   &= Cov[(B^{-1} + H^T R^{-1} H)^{-1} B^{-1} \vec{x}_b, (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} \vec{x}_b] \\
   &\quad + Cov[(B^{-1} + H^T R^{-1} H)^{-1} B^{-1} \vec{x}_b, (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} \vec{y}] \\
   &\quad + Cov[(B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} \vec{y}, (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} \vec{x}_b] \\
   &\quad + Cov[(B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} \vec{y}, (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} \vec{y}]

At this point, using the property that :math:`Cov[A X, B Y] = A Cov[X,
Y] B^T` allows us to simplify this to

.. math::

   Cov[\vec{x}_a, \vec{x}_a]
   &= (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} Cov[\vec{x}_b, \vec{x}_b] B^{-T} (B^{-1} + H^T R^{-1} H)^{-T} \\
   &\quad + (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} Cov[\vec{x}_b, \vec{y}] R^{-T} H (B^{-1} + H^T R^{-1} H)^{-T} \\
   &\quad + (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} Cov[\vec{y}, \vec{x}_b] B^{-T} (B^{-1} + H^T R^{-1} H)^{-T} \\
   &\quad + (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} Cov[\vec{y}, \vec{y}] R^{-T} H (B^{-1} + H^T R^{-1} H)^{-T} \\
   &= (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} B B^{-T} (B^{-1} + H^T R^{-1} H)^{-T} + 0 + 0 \\
   &\quad + (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} R R^{-T} H (B^{-1} + H^T R^{-1} H)^{-T}

At this point, we can use the symmetry of the covariance matrices
:math:`B` and :math:`R` to further simplify this, obtaining

.. math::

   Cov[\vec{x}_a, \vec{x}_a]
   &= (B^{-1} + H^T R^{-1} H)^{-1} B^{-1} (B^{-1} + H^T R^{-1} H)^{-T} \\
   &\quad + (B^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1} H (B^{-1} + H^T R^{-1} H)^{-T} \\
   &= (B^{-1} + H^T R^{-1} H)^{-1} (B^{-1} + H^T R^{-1} H) (B^{-1} + H^T R^{-1} H)^{-1} \\
   &= (B^{-1} + H^T R^{-1} H)^{-1}
