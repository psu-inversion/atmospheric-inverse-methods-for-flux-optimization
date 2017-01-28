"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.
"""
import unittest

import numpy as np
import unittest2

import inversion.optimal_interpolation
import inversion.noise
import inversion.correlations


class TestOISimple(unittest.TestCase):
    """Test simple OI."""

    def test_direct(self):
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
