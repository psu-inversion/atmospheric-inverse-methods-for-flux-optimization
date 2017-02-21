#!/usr/bin/env python
from __future__ import division, print_function
import fractions
import sys
import os

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import inversion.optimal_interpolation

bg = np.array((18., 15., 22.))
bg_var = np.array((2., 2., 2.))
bg_corr = np.array(((1, .5, .25),
                    (.5, 1, .5),
                    (.25, .5, 1)))

obs = np.array((19., 14.))
obs_var = np.array((1., 1.))

obs_op = np.array(((1., 0., 0.),
                   (0., 1., 0.)))

bg_std = np.sqrt(bg_var)
bg_cov = np.diag(bg_std).dot(bg_corr.dot(np.diag(bg_std)))

obs_std = np.sqrt(obs_var)
# Assume no correlations between observations.
obs_cov = np.diag(obs_var)

############################################################
# problem 3
state_college_index = 1
post, post_cov = inversion.optimal_interpolation.simple(
    bg[state_college_index], bg_cov[state_college_index, state_college_index],
    obs[state_college_index], obs_cov[state_college_index, state_college_index],
    obs_op[state_college_index, state_college_index])

print("Problem 3", post)
# compare to what I got by hand
assert np.allclose(post, np.asanyarray(14 + fractions.Fraction(1, 3), dtype=np.float128))

############################################################
# problem 4
state_college_index = 1

post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov,
    obs[state_college_index], obs_cov[state_college_index, state_college_index],
    obs_op[state_college_index, :])

print("Problem 4", post)
# compare to what I got by hand
assert np.allclose(
    post, np.asanyarray((17 + fractions.Fraction(2, 3),
                         14 + fractions.Fraction(1, 3),
                         21 + fractions.Fraction(2, 3)),
                        dtype=np.float128))

############################################################
# Problem 5
pittsburgh_index = 0

post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov,
    obs[pittsburgh_index], obs_cov[pittsburgh_index, pittsburgh_index],
    obs_op[pittsburgh_index, :])

print("Problem 5", post)
# compare to what I got by hand
assert np.allclose(
    post,
    np.asanyarray((18 + fractions.Fraction(2, 3),
                   15 + fractions.Fraction(1, 3),
                   22 + fractions.Fraction(1, 6)),
                  np.float128))

############################################################
# Problem 7
state_college_index = 1

post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov,
    obs[state_college_index], 2 * obs_cov[state_college_index, state_college_index] * 2,
    obs_op[state_college_index, :])

print("Problem 7", post)
# compare to what I would expect by hand
assert np.allclose(
    post, np.asanyarray((17 + fractions.Fraction(5, 6),
                         14 + fractions.Fraction(2, 3),
                         21 + fractions.Fraction(5, 6)),
                        dtype=np.float128))

############################################################
# Problem 8
post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov, obs, obs_cov, obs_op)

print("Problem 8", post)
plt.figure("Problem 8")
plt.plot(np.arange(3), bg, 'b-',
         np.arange(2), obs, 'r.',
         np.arange(3), post, 'k--')
plt.legend(("Background", "Observation", "Analysis"))

# problem 9
# change the observation error variances
obs_std_new = np.array((.5, 4))
post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov, obs, np.diag(np.square(obs_std_new)), obs_op)

print("Problem 9a", post)

# change the background error correlation structures
bg_corr_new = np.array(((1, .5, .4),
                        (.5, 1, -.5),
                        (.4, -.5, 1)))
# make sure it's symmetric positive definite.
np.linalg.cholesky(bg_corr_new)
bg_cov_new = np.diag(bg_std).dot(bg_corr_new.dot(np.diag(bg_std)))

post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov_new, obs, obs_cov, obs_op)

print("Problem 9b", post)

# Problem 10
obs_op_new = np.array(((.5, .5, 0),
                       (0, .5, .5)))

post, post_cov = inversion.optimal_interpolation.simple(
    bg, bg_cov, obs, obs_cov, obs_op_new)

print("Problem 10", post)
plt.figure("Problem 10")
plt.plot(np.arange(3), bg, 'b-',
         (.5, 1.5), obs, 'r.',
         np.arange(3), post, 'k--')
plt.legend(("Background", "Observation", "Analysis"))

plt.show()
