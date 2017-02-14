"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.
"""
import fractions
import unittest

import numpy as np
import unittest2

import inversion.optimal_interpolation
import inversion.noise
import inversion.correlations


class TestOISimple(unittest2.TestCase):
    """Test simple OI."""

    def test_scalar_equal_variance(self):
        """Test a direct measurement of a scalar state."""
        bg = np.atleast_1d(2.)
        bg_cov = np.atleast_2d(1.)

        obs = np.atleast_1d(3.)
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        post, post_cov = inversion.optimal_interpolation.simple(
            bg, bg_cov, obs, obs_cov, obs_op)

        self.assertAlmostEqual(post, 2.5)
        self.assertAlmostEqual(post_cov, .5)

    def test_scalar_unequal_variance(self):
        """Test the a direct measurement fo a scalar state.

        Variances not equal.
        """
        bg = np.atleast_1d(15.)
        bg_cov = np.atleast_2d(2.)

        obs = np.atleast_1d(14.)
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        post, post_cov = inversion.optimal_interpolation.simple(
            bg, bg_cov, obs, obs_cov, obs_op)

        self.assertTrue(np.allclose(
            post, np.float128(14 + fractions.Fraction(1, 3))))

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

        with self.subTest(problem=3):
            state_college_index = 1
            post, post_cov = inversion.optimal_interpolation.simple(
                bg[state_college_index], bg_cov[state_college_index, state_college_index],
                obs[state_college_index], obs_cov[state_college_index, state_college_index],
                obs_op[state_college_index, state_college_index])

            self.assertTrue(np.allclose(
                post, np.asanyarray(14 + fractions.Fraction(1, 3), dtype=np.float128)))

        with self.subTest(problem=4):
            state_college_index = 1

            post, post_cov = inversion.optimal_interpolation.simple(
                bg, bg_cov,
                obs[state_college_index], obs_cov[state_college_index, state_college_index],
                obs_op[state_college_index, :])

            self.assertTrue(np.allclose(
                post, np.asanyarray((17 + fractions.Fraction(2, 3),
                                     14 + fractions.Fraction(1, 3),
                                     21 + fractions.Fraction(2, 3)),
                                    dtype=np.float128)))

        with self.subTest(problem=5):
            pittsburgh_index = 0

            post, post_cov = inversion.optimal_interpolation.simple(
                bg, bg_cov,
                obs[pittsburgh_index], obs_cov[pittsburgh_index, pittsburgh_index],
                obs_op[pittsburgh_index, :])

            self.assertTrue(np.allclose(
                post,
                np.asanyarray((18 + fractions.Fraction(2, 3),
                               15 + fractions.Fraction(1, 3),
                               22 + fractions.Fraction(1, 6)),
                              np.float128)))

        with self.subTest(problem=7):
            state_college_index = 1

            post, post_cov = inversion.optimal_interpolation.simple(
                bg, bg_cov,
                obs[state_college_index], 2 * obs_cov[state_college_index, state_college_index] * 2,
                obs_op[state_college_index, :])

            self.assertTrue(np.allclose(
                post, np.asanyarray((17 + fractions.Fraction(5, 6),
                                     14 + fractions.Fraction(2, 3),
                                     21 + fractions.Fraction(5, 6)),
                                    dtype=np.float128)))

        with self.subTest(problem=8):
            post, post_cov = inversion.optimal_interpolation.simple(
                bg, bg_cov, obs, obs_cov, obs_op)

            self.assertTrue(np.allclose(
                post, np.asanyarray(
                    (18 + fractions.Fraction(1, 3),
                     14 + fractions.Fraction(2, 3),
                     21 + fractions.Fraction(5, 6)),
                    dtype=np.float128)))


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
        test_size = (int(.5e2), int(1e2))
        for corr_class in (
                inversion.correlations.SpatialCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=corr_class):
                corr_fun = corr_class(1.)

                corr = np.fromfunction(corr_fun, test_size*2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),)*2)

                # test postitive definite
                chol_upper = np.dual.cholesky(corr_mat).T

                # test symmetry
                self.assertTrue(np.allclose(chol_upper.T.dot(chol_upper),
                                            corr_mat,
                                            rtol=1e-4, atol=1e-4))
