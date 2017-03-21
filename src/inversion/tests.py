"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.
"""
import fractions
import itertools
import unittest

import numpy as np
import numpy.testing
import unittest2

import inversion.optimal_interpolation
import inversion.variational
import inversion.psas
import inversion.noise
import inversion.correlations
import inversion.integrators

ALL_METHODS = (inversion.optimal_interpolation.simple,
               inversion.optimal_interpolation.fold_common,
               inversion.optimal_interpolation.scipy_chol,
               inversion.variational.simple,
               inversion.variational.incremental,
               inversion.variational.incr_chol,
               inversion.psas.simple,
               inversion.psas.fold_common,
)
PRECISE_DTYPE = np.float128

def getname(method):
    module = method.__module__
    group = module.split(".")[-1]
    variant = method.__name__

    return "{group:s} ({variant:s})".format(group=group,
                                            variant=variant)


class TestSimple(unittest2.TestCase):
    """Test inversions using simple cases."""

    def test_scalar_equal_variance(self):
        """Test a direct measurement of a scalar state."""
        bg = np.atleast_1d(2.)
        bg_cov = np.atleast_2d(1.)

        obs = np.atleast_1d(3.)
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                post, post_cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np.testing.assert_allclose(post, 2.5)
                #self.assertTrue(np.allclose(post_cov, .5))
                np.testing.assert_allclose(post_cov, .5)

    def test_scalar_unequal_variance(self):
        """Test the a direct measurement fo a scalar state.

        Variances not equal.
        """
        bg = np.atleast_1d(15.)
        bg_cov = np.atleast_2d(2.)

        obs = np.atleast_1d(14.)
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                post, post_cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np.testing.assert_allclose(
                    post, PRECISE_DTYPE(14 + fractions.Fraction(1, 3)))

    def test_homework_one(self):
        """Verify that this can reproduce the answers to HW1.

        Make sure the answers here are within roundoff of the analytic
        solutions.

        """
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

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                with self.subTest(problem=3):
                    state_college_index = 1
                    post, post_cov = method(
                        bg[state_college_index],
                        bg_cov[state_college_index, state_college_index],
                        obs[state_college_index],
                        obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, state_college_index])

                    np.testing.assert_allclose(
                        post, np.asanyarray(14 + fractions.Fraction(1, 3),
                                            dtype=PRECISE_DTYPE))

                with self.subTest(problem=4):
                    state_college_index = 1

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[state_college_index],
                        obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, :])

                    np.testing.assert_allclose(
                        post, np.asanyarray((17 + fractions.Fraction(2, 3),
                                             14 + fractions.Fraction(1, 3),
                                             21 + fractions.Fraction(2, 3)),
                                            dtype=PRECISE_DTYPE))

                with self.subTest(problem=5):
                    pittsburgh_index = 0

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[pittsburgh_index],
                        obs_cov[pittsburgh_index, pittsburgh_index],
                        obs_op[pittsburgh_index, :])

                    np.testing.assert_allclose(
                        post,
                        np.asanyarray((18 + fractions.Fraction(2, 3),
                                       15 + fractions.Fraction(1, 3),
                                       22 + fractions.Fraction(1, 6)),
                                      PRECISE_DTYPE))

                with self.subTest(problem=7):
                    state_college_index = 1

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[state_college_index],
                        4 * obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, :])

                    np.testing.assert_allclose(
                        post, np.asanyarray((17 + fractions.Fraction(5, 6),
                                             14 + fractions.Fraction(2, 3),
                                             21 + fractions.Fraction(5, 6)),
                                            dtype=PRECISE_DTYPE))

                with self.subTest(problem=8):
                    post, post_cov = method(
                        bg, bg_cov, obs, obs_cov, obs_op)

                    # background correlations make this problem not
                    # strictly linear, at least without doing
                    # sequential inversions. Have not verified by hand
                    np.testing.assert_allclose(
                        post, np.asanyarray(
                            (18 + fractions.Fraction(1, 2),
                             14 + fractions.Fraction(1, 2),
                             21 + fractions.Fraction(3, 4)),
                            dtype=PRECISE_DTYPE))

    def test_sequential_assimilations(self):
        """Make sure this follows Bayes' rule."""
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

        for method in ALL_METHODS:
            name = getname(method)
            if "var" in name.lower() or "psas" in name.lower():
                state_rtol = 1e-3
                cov_rtol = 1e-1
            else:
                # The default for assert_allclose
                cov_rtol = state_rtol = 1e-7
            with self.subTest(method=name):
                inter1, inter_cov1 = method(
                    bg, bg_cov, obs[0], obs_cov[0, 0],
                    obs_op[0, :])
                post1, post_cov1 = method(
                    inter1, inter_cov1, obs[1], obs_cov[1, 1],
                    obs_op[1, :])

                post2, post_cov2 = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np.testing.assert_allclose(
                    post1, post2, rtol=state_rtol)
                np.testing.assert_allclose(
                    post_cov1, post_cov2, rtol=cov_rtol)


class TestGaussianNoise(unittest.TestCase):
    """Test the properties of the gaussian noise."""

    def test_ident_cov(self):
        """Test generation with identity as covariance."""
        sample_shape = 3
        cov = np.eye(sample_shape)
        noise = inversion.noise.gaussian_noise(cov, int(1e6))

        self.assertTrue(np.allclose(noise.mean(axis=0),
                                    np.zeros((sample_shape,)),
                                    rtol=1e-2, atol=1e-2))
        self.assertTrue(np.allclose(np.cov(noise.T), cov,
                                    rtol=1e-2, atol=1e-2))


class TestCorrelations(unittest2.TestCase):
    """Test the generation of correlation matrices."""

    def test_far_correl(self):
        """Test the correlation between points far apart.

        Should be zero.
        """
        for corr_class in (
                inversion.correlations.SpatialCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=corr_class):
                corr_fun = corr_class(1e-8)

                corr = corr_fun(0, 0, 1e8, 1e8)
                self.assertAlmostEqual(corr, 0)

    def test_near_correl(self):
        """Test correlation between near points.

        Should be one.
        """
        for corr_class in (
                inversion.correlations.SpatialCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=corr_class):
                corr_fun = corr_class(1e8)

                corr = corr_fun(0, 0, 1e-8, 1e-8)
                self.assertAlmostEqual(corr, 1)

    def test_np_fromfunction(self):
        """Test that the structure works with np.fromfunction.

        This is how the integration tests will get background
        covariances, so this needs to work.
        """
        test_size = (int(.2e2), int(.3e2))
        for corr_class in (
                inversion.correlations.SpatialCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=getname(corr_class)):
                corr_fun = corr_class(1.)

                corr = np.fromfunction(corr_fun, test_size*2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),)*2)

                # test postitive definite
                chol_upper = np.linalg.cholesky(corr_mat)

                # test symmetry
                np.testing.assert_allclose(chol_upper.dot(chol_upper.T),
                                           corr_mat,
                                           rtol=1e-4, atol=1e-4)

    def test_make_matrix(self):
        """Test that make_matrix gives a positive definite matrix.

        Checks against original value.
        """
        # 30x25 Gaussian 10 not close
        test_nx = 30
        test_ny = 20
        test_points = test_ny * test_nx

        for corr_class in (
                inversion.correlations.SpatialCorrelationFunction.
                __subclasses__()):
            for dist in (1, 5, 10, 15):
                with self.subTest(corr_class=getname(corr_class),
                                  dist=dist):
                    corr_fun = corr_class(dist)

                    corr_mat = corr_fun.make_matrix(test_ny, test_nx)

                    # Make sure diagonal elements are ones
                    np.testing.assert_allclose(np.diag(corr_mat), 1)

                    # check if it matches the original
                    np.testing.assert_allclose(
                        corr_mat,
                        np.fromfunction(
                            corr_fun, (test_ny, test_nx, test_ny, test_nx)
                        ).reshape((test_points, test_points)),
                        # rtol=1e-13: Gaussian 10 and 15 fail
                        # atol=1e-15: Gaussian 1 and 5 fail
                        rtol=1e-12, atol=1e-14)

                    # check if it actually is positive definite
                    chol_lower = np.linalg.cholesky(corr_mat)


class TestIntegrators(unittest2.TestCase):
    """Test the integrators."""

    def test_exp(self):
        for integrator in (inversion.integrators.forward_euler,
                           inversion.integrators.scipy_odeint):
            with self.subTest(integrator=getname(integrator)):
                solns = integrator(
                    lambda y, t: y,
                    1.,
                    (0, 1),
                    1e-6)

                np.testing.assert_allclose(solns[1,:],
                                           np.exp(1.), rtol=1e-5)
