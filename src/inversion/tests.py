"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.

"""
from __future__ import print_function, division
import itertools
import fractions
import math
import sys
try:
    from functools import reduce
except ImportError:
    # reduce used to be a builtin
    pass

import numpy as np
import numpy.linalg as np_la
import numpy.random as np_rand
import numpy.testing as np_tst
import scipy.linalg
import scipy.sparse
import scipy.optimize
import unittest2

import dask
import dask.array as da
import numpy.linalg as la
# Import from scipy.linalg if not using dask
from scipy.linalg import cholesky

import inversion.covariance_estimation
import inversion.optimal_interpolation
import inversion.ensemble.integrators
import inversion.correlations
import inversion.covariances
import inversion.integrators
import inversion.variational
import inversion.ensemble
import inversion.linalg
import inversion.models
import inversion.noise
import inversion.psas
import inversion.util
from inversion.linalg import tolinearoperator

dask.set_options(get=dask.get)

# If adding other inexact methods to the list tested, be sure to add
# those to the `if "var" in name or "psas" in name` and
# `if "psas" in name` tests as applicable.
ALL_METHODS = (
    inversion.optimal_interpolation.simple,
    inversion.optimal_interpolation.fold_common,
    inversion.optimal_interpolation.save_sum,
    inversion.optimal_interpolation.scipy_chol,
    inversion.variational.simple,
    inversion.variational.incremental,
    inversion.variational.incr_chol,
    inversion.psas.simple,
    inversion.psas.fold_common,
)
ITERATIVE_METHOD_START = 4
"""Where the iterative methods start in the above list.

Used to test failure modes for these solvers.
"""

PRECISE_DTYPE = np.float128
"""The dtype used to represent analytic results.

These are initialized as :class:`fractions.Fraction` then converted to
this dtype for the comparison.
"""

ITERATIVE_STATE_TOLERANCE = 1e-3
ITERATIVE_COVARIANCE_TOLERANCE = 1e-1
EXACT_TOLERANCE = 1e-7

DTYPE = np.float64
"""Default dtype for certain tests."""


def getname(method):
    """Descriptive name for the function.

    A name combining the function name and module.

    Parameters
    ----------
    method: callable

    Returns
    -------
    name: str

    """
    module = method.__module__
    group = module.split(".")[-1]
    variant = method.__name__

    return "{group:s} ({variant:s})".format(group=group,
                                            variant=variant)


class TestInversionSimple(unittest2.TestCase):
    """Test inversions using simple cases."""

    def test_scalar_equal_variance(self):
        """Test a direct measurement of a scalar state."""
        bg = np.atleast_1d(2.)
        bg_cov = np.atleast_2d(1.)

        obs = np.atleast_1d(3.)
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        for method in ALL_METHODS:
            name = getname(method)

            with self.subTest(method=name):
                post, post_cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np_tst.assert_allclose(post, 2.5)
                np_tst.assert_allclose(post_cov, .5)

    def test_scalar_unequal_variance(self):
        """Test assimilation of a direct measurement fo a scalar state.

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

                np_tst.assert_allclose(
                    post, PRECISE_DTYPE(14 + fractions.Fraction(1, 3)))
                np_tst.assert_allclose(
                    post_cov, PRECISE_DTYPE(fractions.Fraction(2, 3)))

    def test_multiple_priors(self):
        """Test doing multiple assimilations at once.

        Simple test.
        """
        bg = np.array([[2., 3.]])
        bg_cov = np.atleast_2d(1.)

        obs = np.array([[3., 4.]])
        obs_cov = np.atleast_2d(1.)

        obs_op = np.atleast_2d(1.)

        for method in ALL_METHODS[:ITERATIVE_METHOD_START]:
            name = getname(method)

            with self.subTest(method=name):
                post, post_cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np_tst.assert_allclose(post, [[2.5, 3.5]])
                np_tst.assert_allclose(post_cov, .5)

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

        # obs_std = np.sqrt(obs_var)

        # Assume no correlations between observations.
        obs_cov = np.diag(obs_var)

        for method in ALL_METHODS:
            # Setup for expected degradation of solutions
            name = getname(method)
            # The default for assert_allclose
            cov_rtol = state_rtol = EXACT_TOLERANCE

            with self.subTest(method=name):
                # Also tested above in scalar_unequal_variance
                with self.subTest(problem=3):
                    state_college_index = 1
                    post, post_cov = method(
                        bg[state_college_index],
                        bg_cov[state_college_index, state_college_index],
                        obs[state_college_index],
                        obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, state_college_index])

                    np_tst.assert_allclose(
                        post, np.asanyarray(14 + fractions.Fraction(1, 3),
                                            dtype=PRECISE_DTYPE),
                        rtol=state_rtol)
                    np_tst.assert_allclose(
                        post_cov, np.asanyarray(fractions.Fraction(2, 3),
                                                dtype=PRECISE_DTYPE),
                        rtol=cov_rtol)

                with self.subTest(problem=4):
                    state_college_index = 1

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[state_college_index],
                        obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, :])

                    np_tst.assert_allclose(
                        post, np.asanyarray((17 + fractions.Fraction(2, 3),
                                             14 + fractions.Fraction(1, 3),
                                             21 + fractions.Fraction(2, 3)),
                                            dtype=PRECISE_DTYPE),
                        rtol=state_rtol)

                with self.subTest(problem=5):
                    pittsburgh_index = 0

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[pittsburgh_index],
                        obs_cov[pittsburgh_index, pittsburgh_index],
                        obs_op[pittsburgh_index, :])

                    np_tst.assert_allclose(
                        post,
                        np.asanyarray((18 + fractions.Fraction(2, 3),
                                       15 + fractions.Fraction(1, 3),
                                       22 + fractions.Fraction(1, 6)),
                                      PRECISE_DTYPE),
                        rtol=state_rtol)

                with self.subTest(problem=7):
                    state_college_index = 1

                    post, post_cov = method(
                        bg, bg_cov,
                        obs[state_college_index],
                        4 * obs_cov[state_college_index, state_college_index],
                        obs_op[state_college_index, :])

                    np_tst.assert_allclose(
                        post, np.asanyarray((17 + fractions.Fraction(5, 6),
                                             14 + fractions.Fraction(2, 3),
                                             21 + fractions.Fraction(5, 6)),
                                            dtype=PRECISE_DTYPE),
                        rtol=state_rtol)

                with self.subTest(problem=8):
                    post, post_cov = method(
                        bg, bg_cov, obs, obs_cov, obs_op)

                    # background correlations make this problem not
                    # strictly linear, at least without doing
                    # sequential inversions. Have not verified by hand
                    np_tst.assert_allclose(
                        post, np.asanyarray(
                            (18 + fractions.Fraction(1, 2),
                             14 + fractions.Fraction(1, 2),
                             21 + fractions.Fraction(3, 4)),
                            dtype=PRECISE_DTYPE),
                        rtol=state_rtol)

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

        # obs_std = np.sqrt(obs_var)

        # Assume no correlations between observations.
        obs_cov = np.diag(obs_var)

        for method in ALL_METHODS:
            name = getname(method)
            if "var" in name.lower() or "psas" in name.lower():
                state_rtol = ITERATIVE_STATE_TOLERANCE
                cov_rtol = ITERATIVE_COVARIANCE_TOLERANCE
            else:
                # The default for assert_allclose
                cov_rtol = state_rtol = EXACT_TOLERANCE

            with self.subTest(method=name):
                inter1, inter_cov1 = method(
                    bg, bg_cov, obs[0], obs_cov[0, 0],
                    obs_op[0, :])
                post1, post_cov1 = method(
                    inter1, inter_cov1, obs[1], obs_cov[1, 1],
                    obs_op[1, :])

                post2, post_cov2 = method(
                    bg, bg_cov, obs, obs_cov, obs_op)

                np_tst.assert_allclose(
                    post1, post2, rtol=state_rtol)

                if "psas" in name.lower():
                    # The second covariance isn't positive definite (one
                    # positive entry) and no entry shares the order of
                    # magnitude between the two.
                    raise unittest2.SkipTest("Known Failure: PSAS Covariances")

                np_tst.assert_allclose(
                    post_cov1, post_cov2, rtol=cov_rtol)

    def test_iterative_failures(self):
        """Test failure modes of iterative solvers."""
        bg_stds = np.logspace(-8, 1, 10)
        bg_corr = scipy.linalg.toeplitz(
            np.arange(1, .9, -.01))
        bg_cov = np.diag(bg_stds).dot(bg_corr).dot(np.diag(bg_stds))

        bg_vals = np.arange(10)

        obs_op = np.eye(3, 10)
        obs_vals = 10 - np.arange(3)
        obs_cov = np.diag((10, 1e-3, 1e-6)) / 8

        for method in ALL_METHODS[ITERATIVE_METHOD_START:]:
            name = getname(method)

            with self.subTest(method=name):
                with self.assertRaises(inversion.ConvergenceError) as cxt_mgr:
                    method(bg_vals, bg_cov, obs_vals, obs_cov, obs_op)

                conv_err = cxt_mgr.exception
                self.assertTrue(hasattr(conv_err, "guess"))
                self.assertTrue(hasattr(conv_err, "result"))
                self.assertIsInstance(conv_err.result,
                                      scipy.optimize.OptimizeResult)
                self.assertTrue(hasattr(conv_err, "hess_inv"))


class TestGaussianNoise(unittest2.TestCase):
    """Test the properties of the gaussian noise."""

    def test_ident_cov(self):
        """Test generation with identity as covariance."""
        sample_shape = 3
        cov = np.eye(sample_shape)
        noise = inversion.noise.gaussian_noise(cov, int(1e6))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros((sample_shape,)),
                               rtol=1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T), cov,
                               rtol=1e-2, atol=1e-2)

    def test_shape(self):
        """Make sure the returned shapes are correct."""
        sample_shape = (3,)
        sample_cov = np.eye(sample_shape[0])

        for shape in ((), (6,), (2, 3)):
            with self.subTest(shape=shape):
                res = inversion.noise.gaussian_noise(
                    sample_cov, shape)

                self.assertEqual(res.shape, shape + sample_shape)

        with self.subTest(shape=5):
            res = inversion.noise.gaussian_noise(
                sample_cov, 5)

            self.assertEqual(res.shape, (5,) + sample_shape)

        with self.subTest(shape=None):
            res = inversion.noise.gaussian_noise(
                sample_cov, None)
            self.assertEqual(res.shape, sample_shape)

    def test_operator(self):
        """Test that the code works with operator covariances."""
        diagonal = (1, .5, .3, .2, .1)
        sample_cov = inversion.covariances.DiagonalOperator(diagonal)
        sample_shape = (len(diagonal),)
        noise = inversion.noise.gaussian_noise(sample_cov, int(1e6))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros(sample_shape),
                               rtol=1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T), np.diag(diagonal),
                               rtol=1e-2, atol=1e-2)

    def test_kron_op(self):
        """Test that large kronecker operators don't break the handling."""
        op1 = scipy.linalg.toeplitz(.6 ** np.arange(15))
        diag = (1, .9, .8, .7, .6, .5, .4, .3, .2, .1)
        op2 = inversion.covariances.DiagonalOperator(diag)

        combined = inversion.util.kronecker_product(op1, op2)

        noise = inversion.noise.gaussian_noise(combined, int(1e6))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros(combined.shape[0]),
                               rtol=1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T),
                               scipy.linalg.kron(op1, np.diag(diag)),
                               rtol=1e-2, atol=1e-2)

    def test_off_diagonal(self):
        """Test that the code works with off-diagonal elements."""
        sample_cov = scipy.linalg.toeplitz((1, .5, .25, .125))
        sample_shape = (4,)
        noise = inversion.noise.gaussian_noise(sample_cov, int(1e6))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros(sample_shape),
                               rtol=1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T), sample_cov,
                               rtol=1e-2, atol=1e-2)

    def test_slow_decay(self):
        """Test that the code handles slowly-decaying covariances."""
        sample_cov = scipy.linalg.toeplitz(.8 ** np.arange(10))
        sample_shape = (10,)
        noise = inversion.noise.gaussian_noise(sample_cov, int(1e6))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros(sample_shape),
                               rtol=1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T), sample_cov,
                               rtol=1e-2, atol=1e-2)


class TestCorrelations(unittest2.TestCase):
    """Test the generation of correlation matrices."""

    def test_far_correl(self):
        """Test the correlation between points far apart.

        Should be zero.
        """
        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=corr_class.__name__):
                corr_fun = corr_class(1e-8)

                corr = corr_fun(1e8)
                self.assertAlmostEqual(corr, 0)

    def test_near_correl(self):
        """Test 2D correlation between near points.

        Should be one.
        """
        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=corr_class.__name__):
                corr_fun = corr_class(1e8)

                corr = corr_fun(1e-8)
                self.assertAlmostEqual(corr, 1)

    def test_2d_np_fromfunction(self):
        """Test that the structure works with np.fromfunction.

        This is how the integration tests will get background
        covariances, so this needs to work.
        """
        test_size = (int(15), int(20))
        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=getname(corr_class)):
                corr_fun = corr_class(2.)

                corr = np.fromfunction(corr_fun.correlation_from_index,
                                       shape=test_size * 2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),) * 2)

                # test postitive definite
                chol_upper = cholesky(corr_mat)

                # test symmetry
                np_tst.assert_allclose(chol_upper.T.dot(chol_upper),
                                       corr_mat,
                                       rtol=1e-4, atol=1e-4)

    def test_2d_make_matrix(self):
        """Test make_matrix for 2D correlations.

        Checks against original value.
        """
        # 30x25 Gaussian 10 not close
        test_nx = 30
        test_ny = 20
        test_points = test_ny * test_nx

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 5, 10, 15):
                with self.subTest(corr_class=getname(corr_class),
                                  dist=dist):
                    corr_fun = corr_class(dist)

                    corr_mat = inversion.correlations.make_matrix(
                        corr_fun, (test_ny, test_nx))

                    # Make sure diagonal elements are ones
                    np_tst.assert_allclose(np.diag(corr_mat), 1)

                    # check if it matches the original
                    np_tst.assert_allclose(
                        corr_mat,
                        np.fromfunction(
                            corr_fun.correlation_from_index,
                            (test_ny, test_nx, test_ny, test_nx)
                        ).reshape((test_points, test_points)),
                        # rtol=1e-13: Gaussian 10 and 15 fail
                        # atol=1e-15: Gaussian 1 and 5 fail
                        rtol=1e-11, atol=1e-13)

                    # check if it actually is positive definite
                    cholesky(corr_mat)

    def test_1d_np_fromfunction(self):
        """Test that the structure works with np.fromfunction.

        This is how the integration tests will get background
        covariances, so this needs to work.
        """
        test_size = (200,)
        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction
                .__subclasses__()):
            with self.subTest(corr_class=getname(corr_class)):
                # This fails with a correlation length of 5
                corr_fun = corr_class(2.)

                corr = np.fromfunction(corr_fun.correlation_from_index,
                                       shape=test_size * 2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),) * 2)

                # test postitive definite
                chol_upper = cholesky(corr_mat)

                # test symmetry
                np_tst.assert_allclose(chol_upper.T.dot(chol_upper),
                                       corr_mat,
                                       rtol=1e-4, atol=1e-4)

    def test_1d_make_matrix(self):
        """Test make_matrix for 1D correlations.

        Checks against original value.
        """
        test_nt = 200

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 5, 10, 30):
                with self.subTest(corr_class=getname(corr_class),
                                  dist=dist):
                    corr_fun = corr_class(dist)

                    corr_mat = inversion.correlations.make_matrix(corr_fun,
                                                                  test_nt)

                    # Make sure diagonal elements are ones
                    np_tst.assert_allclose(np.diag(corr_mat), 1)

                    # check if it matches the original
                    np_tst.assert_allclose(
                        corr_mat,
                        np.fromfunction(
                            corr_fun.correlation_from_index, (test_nt, test_nt)
                        ).reshape((test_nt, test_nt)),
                        # rtol=1e-13: Gaussian 10 and 15 fail
                        # atol=1e-15: Gaussian 1 and 5 fail
                        rtol=1e-12, atol=1e-14)

                    # check if it actually is positive definite
                    chol_upper = cholesky(corr_mat)

                    # test symmetry
                    np_tst.assert_allclose(chol_upper.T.dot(chol_upper),
                                           corr_mat,
                                           rtol=1e-4, atol=1e-4)

    def test_fft_correlation_structure(self):
        """Ensure the FFT-based operators satisfy conditions of correlation matrices.

        Checks for symmetry and ones on the diagonal.
        """
        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for test_shape in ((300,), (20, 30)):
                test_size = int(np.prod(test_shape, dtype=int))
                for dist in (1, 3, 10, 30):
                    corr_fun = corr_class(dist)

                    corr_op = (
                        inversion.correlations.HomogeneousIsotropicCorrelation.
                        from_function(corr_fun, test_shape))
                    # This is the fastest way to get column-major
                    # order from da.eye.
                    corr_mat = corr_op.dot(np.eye(test_size).T)

                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_shape=test_shape,
                                      test="symmetry"):
                        np_tst.assert_allclose(corr_mat, corr_mat.T,
                                               rtol=1e-14, atol=1e-15)
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_shape=test_shape,
                                      test="self-correlation"):
                        np_tst.assert_allclose(np.diag(corr_mat), 1)

    def test_1d_fft_correlation(self):
        """Test HomogeneousIsotropicCorrelation for 1D arrays.

        Check against `make_matrix` and ignore values near the edges
        of the domain where the two methods are different.
        """
        test_nt = 512
        test_lst = (np.zeros(test_nt), np.ones(test_nt), np.arange(test_nt),
                    np.eye(100, test_nt)[-1])

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 3, 10):
                # Magic numbers
                # May need to increase for larger test_nt
                noncorr_dist = 20 + 8 * dist
                corr_fun = corr_class(dist)

                corr_mat = inversion.correlations.make_matrix(
                    corr_fun, test_nt)
                corr_op = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_nt))

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      inverse="no"):
                        np_tst.assert_allclose(
                            corr_op.dot(test_vec)[noncorr_dist:-noncorr_dist],
                            corr_mat.dot(test_vec)[noncorr_dist:-noncorr_dist],
                            rtol=1e-3, atol=1e-5)

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      inverse="yes"):
                        if ((corr_class is inversion.correlations.
                             GaussianCorrelation and
                             dist >= 3)):
                            # Gaussian(3) has FFT less
                            # well-conditioned than make_matrix
                            raise unittest2.SkipTest(
                                "Gaussian({0:d}) correlations ill-conditioned".
                                format(dist))
                        elif ((corr_class is inversion.correlations.
                               BalgovindCorrelation and
                               dist == 10)):
                            # This one distance is problematic
                            # Roughly 3% of the points disagree
                            # for the last half of the tests
                            # I have no idea why
                            raise unittest2.SkipTest(
                                "Balgovind(10) correlations weird")
                        np_tst.assert_allclose(
                            corr_op.solve(
                                test_vec)[noncorr_dist:-noncorr_dist],
                            la.solve(
                                corr_mat,
                                test_vec)[noncorr_dist:-noncorr_dist],
                            rtol=1e-3, atol=1e-5)

    def test_2d_fft_correlation(self):
        """Test HomogeneousIsotropicCorrelation for 2D arrays.

        Check against `make_matrix` and ignore values near the edges
        where the two methods differ.
        """
        test_shape = (20, 30)
        test_size = np.prod(test_shape)
        test_lst = (np.zeros(test_size),
                    np.ones(test_size),
                    np.arange(test_size),
                    np.eye(10 * test_shape[0], test_size)[-1])

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 3):
                # Magic numbers
                # May need to increase for larger domains
                noncorr_dist = 20 + 8 * dist
                corr_fun = corr_class(dist)

                corr_mat = inversion.correlations.make_matrix(
                    corr_fun, test_shape)
                corr_op = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_shape))

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      direction="forward"):
                        np_tst.assert_allclose(
                            corr_op.dot(test_vec).reshape(test_shape)
                            [noncorr_dist:-noncorr_dist,
                             noncorr_dist:-noncorr_dist],
                            corr_mat.dot(test_vec).reshape(test_shape)
                            [noncorr_dist:-noncorr_dist,
                             noncorr_dist:-noncorr_dist],
                            rtol=1e-3, atol=1e-5)

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      direction="backward"):
                        if ((corr_class is inversion.correlations.
                             GaussianCorrelation and
                             dist >= 3)):
                            # Gaussian(3) has FFT less
                            # well-conditioned than make_matrix
                            raise unittest2.SkipTest(
                                "Gaussian({0:d}) correlations ill-conditioned".
                                format(dist))
                        np_tst.assert_allclose(
                            corr_op.solve(
                                test_vec).reshape(test_shape)
                            [noncorr_dist:-noncorr_dist,
                             noncorr_dist:-noncorr_dist],
                            la.solve(
                                corr_mat,
                                test_vec).reshape(test_shape)
                            [noncorr_dist:-noncorr_dist,
                             noncorr_dist:-noncorr_dist],
                            rtol=1e-3, atol=1e-5)

    def test_homogeneous_from_array(self):
        """Make sure from_array can be roundtripped.

        Also tests that odd state sizes work.
        """
        test_size = 25

        corr_class = inversion.correlations.ExponentialCorrelation
        for dist in (1, 3, 5):
            with self.subTest(dist=dist):
                corr_fun = corr_class(dist)
                corr_op1 = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_size))
                first_column = corr_op1.dot(np.eye(test_size, 1)[:, 0])

                corr_op2 = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_array(first_column))

                np_tst.assert_allclose(
                    corr_op1.dot(np.eye(test_size)),
                    corr_op2.dot(np.eye(test_size)))

    def test_kron_composition(self):
        """Test that `kron` works similar to composition of the domains."""
        from inversion.correlations import HomogeneousIsotropicCorrelation
        corr_class = inversion.correlations.GaussianCorrelation
        corr_fun = corr_class(5)

        shape1 = (5,)
        shape2 = (7,)

        corr_op1 = (HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, shape1))
        corr_op2 = (HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, shape2))
        kron_corr = corr_op1.kron(corr_op2)
        direct_corr = (HomogeneousIsotropicCorrelation.
                       from_function(corr_fun, shape1 + shape2))

        self.assertEqual(kron_corr.shape, direct_corr.shape)
        self.assertEqual(kron_corr._underlying_shape,
                         direct_corr._underlying_shape)
        np_tst.assert_allclose(kron_corr._corr_fourier,
                               direct_corr._corr_fourier)
        np_tst.assert_allclose(kron_corr._fourier_near_zero,
                               direct_corr._fourier_near_zero)

    def test_kron_results(self):
        """Test the Kronecker product implementation."""
        HomogeneousIsotropicCorrelation = (
            inversion.correlations.HomogeneousIsotropicCorrelation)
        corr_class = inversion.correlations.ExponentialCorrelation
        test_shapes = (20, 25, (5, 6))
        distances = (3, 5,)

        for dist1, shape1, dist2, shape2 in itertools.product(
                distances, test_shapes, repeat=2):
            with self.subTest(dist1=dist1, dist2=dist2):
                corr_fun1 = corr_class(dist1)
                corr_fun2 = corr_class(dist2)

                corr_op1 = (
                    HomogeneousIsotropicCorrelation.
                    from_function(corr_fun1, shape1))
                corr_op2 = (
                    HomogeneousIsotropicCorrelation.
                    from_function(corr_fun2, shape2))

                size1 = np.prod(shape1)
                size2 = np.prod(shape2)
                corr_mat1 = corr_op1.dot(np.eye(size1))
                corr_mat2 = corr_op2.dot(np.eye(size2))

                full_corr1 = corr_op1.kron(corr_op2)
                full_corr2 = scipy.linalg.kron(np.asarray(corr_mat1),
                                               np.asarray(corr_mat2))

                self.assertIsInstance(
                    corr_op1, HomogeneousIsotropicCorrelation)

                test_vec = np.arange(size1 * size2)
                np_tst.assert_allclose(
                    full_corr1.dot(test_vec),
                    full_corr2.dot(test_vec))

                test_mat = np.eye(size1 * size2)
                np_tst.assert_allclose(
                    full_corr1.dot(test_mat),
                    full_corr2.dot(test_mat))

    def test_kron_delegate(self):
        """Test that kron delegates where appropriate."""
        op1 = (inversion.correlations.HomogeneousIsotropicCorrelation.
               from_array((1, .5, .25)))
        mat2 = np.eye(5)

        combined_op = op1.kron(mat2)

        self.assertIsInstance(combined_op,
                              inversion.correlations.SchmidtKroneckerProduct)

    def test_sqrt_direct(self):
        """Test the square root in the most direct manner possible.

        Checks whether matrices corresponding to sqrt.T@sqrt and the
        original matrix are approximately equal.
        """
        operator = (inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_array((1, .5, .25, .125)))
        sqrt = operator.sqrt()
        sqrt_squared = sqrt.T.dot(sqrt)

        mat = np.eye(4)

        np_tst.assert_allclose(operator.dot(mat),
                               sqrt_squared.dot(mat))


class TestSchmidtKroneckerProduct(unittest2.TestCase):
    """Test the Schmidt Kronecker product implementation for LinearOperators.

    This class tests the implementation based on the Schmidt decomposition.
    """

    def test_identity(self):
        """Test that the implementation works with identity matrices."""
        test_sizes = (4, 5)
        SchmidtKroneckerProduct = (
            inversion.correlations.SchmidtKroneckerProduct)

        # I want to be sure either being smaller works.
        # Even versus odd also causes problems occasionally
        for size1, size2 in itertools.product(test_sizes, repeat=2):
            with self.subTest(size1=size1, size2=size2):
                mat1 = np.eye(size1)
                mat2 = np.eye(size2)

                full_mat = SchmidtKroneckerProduct(
                    mat1, mat2)
                big_ident = np.eye(size1 * size2)

                np_tst.assert_allclose(
                    full_mat.dot(big_ident),
                    big_ident)

    def test_identical_submatrices(self):
        """Test whether the implementation will generate identical blocks."""
        mat1 = np.ones((3, 3))
        mat2 = ((1, .5, .25), (.5, 1, .5), (.25, .5, 1))

        np_tst.assert_allclose(
            inversion.correlations.SchmidtKroneckerProduct(
                mat1, mat2).dot(np.eye(9)),
            np.tile(mat2, (3, 3)))

    def test_constant_blocks(self):
        """Test whether the implementation will produce constant blocks."""
        mat1 = ((1, .5, .25), (.5, 1, .5), (.25, .5, 1))
        mat2 = np.ones((3, 3))

        np_tst.assert_allclose(
            inversion.correlations.SchmidtKroneckerProduct(
                mat1, mat2).dot(np.eye(9)),
            np.repeat(np.repeat(mat1, 3, 0), 3, 1))

    def test_entangled_state(self):
        """Test whether the implementation works with entangled states."""
        sigmax = np.array(((0, 1), (1, 0)))
        sigmaz = np.array(((1, 0), (0, -1)))

        operator = inversion.correlations.SchmidtKroneckerProduct(
            sigmax, sigmaz)
        matrix = scipy.linalg.kron(sigmax, sigmaz)

        # (k01 - k10) / sqrt(2)
        epr_state = (0, .7071, -.7071, 0)

        np_tst.assert_allclose(
            operator.dot(epr_state),
            matrix.dot(epr_state))


class TestYMKroneckerProduct(unittest2.TestCase):
    """Test the YM13 Kronecker product implementation for LinearOperators.

    This tests the :class:`~inversion.linalg.DaskKroneckerProduct`
    implementation based on the algorithm in Yadav and Michalak (2013)
    """

    def test_identity(self):
        """Test that the implementation works with identity matrices."""
        test_sizes = (4, 5)
        DaskKroneckerProductOperator = (
            inversion.linalg.DaskKroneckerProductOperator)

        # I want to be sure either being smaller works.
        # Even versus odd also causes problems occasionally
        for size1, size2 in itertools.product(test_sizes, repeat=2):
            with self.subTest(size1=size1, size2=size2):
                mat1 = np.eye(size1)
                mat2 = np.eye(size2)

                full_mat = DaskKroneckerProductOperator(
                    mat1, mat2)
                big_ident = np.eye(size1 * size2)

                np_tst.assert_allclose(
                    full_mat.dot(big_ident),
                    big_ident)

    def test_identical_submatrices(self):
        """Test whether the implementation will generate identical blocks."""
        mat1 = np.ones((3, 3))
        mat2 = ((1, .5, .25), (.5, 1, .5), (.25, .5, 1))

        np_tst.assert_allclose(
            inversion.linalg.DaskKroneckerProductOperator(
                mat1, mat2).dot(np.eye(9)),
            np.tile(mat2, (3, 3)))

    def test_constant_blocks(self):
        """Test whether the implementation will produce constant blocks."""
        mat1 = ((1, .5, .25), (.5, 1, .5), (.25, .5, 1))
        mat2 = np.ones((3, 3))

        np_tst.assert_allclose(
            inversion.linalg.DaskKroneckerProductOperator(
                mat1, mat2).dot(np.eye(9)),
            np.repeat(np.repeat(mat1, 3, 0), 3, 1))

    def test_entangled_state(self):
        """Test whether the implementation works with entangled states."""
        sigmax = np.array(((0, 1), (1, 0)))
        sigmaz = np.array(((1, 0), (0, -1)))

        operator = inversion.linalg.DaskKroneckerProductOperator(
            sigmax, sigmaz)
        matrix = scipy.linalg.kron(sigmax, sigmaz)

        # (k01 - k10) / sqrt(2)
        epr_state = (0, .7071, -.7071, 0)

        np_tst.assert_allclose(
            operator.dot(epr_state),
            matrix.dot(epr_state))

    def test_transpose(self):
        """Test whether the transpose is properly implemented."""
        mat1 = np.eye(3)
        mat2 = inversion.covariances.DiagonalOperator((1, 1))
        mat3 = np.eye(4)
        DaskKroneckerProductOperator = (
            inversion.linalg.DaskKroneckerProductOperator)

        with self.subTest(check="symmetric"):
            product = DaskKroneckerProductOperator(
                mat1, mat2)

            self.assertIs(product.T, product)

        with self.subTest(check="asymmetric"):
            mat1[0, 1] = 1
            product = DaskKroneckerProductOperator(
                mat1, mat2)
            transpose = product.T

            self.assertIsNot(transpose, product)
            np_tst.assert_allclose(transpose._operator1,
                                   mat1.T)

        with self.subTest(check="rectangular"):
            product = DaskKroneckerProductOperator(
                mat1[:2], mat3[:3])
            transpose = product.T

            np_tst.assert_allclose(transpose._operator1,
                                   mat1[:2].T)
            np_tst.assert_allclose(transpose._operator2.A,
                                   mat3[:3].T)

    def test_sqrt(self):
        """Test whether the sqrt method works as intended."""
        matrix1 = np.eye(2)
        matrix2 = inversion.covariances.DiagonalOperator((1, 2, 3))
        tester = np.eye(6)

        product = inversion.linalg.DaskKroneckerProductOperator(matrix1, matrix2)
        sqrt = product.sqrt()
        proposed = sqrt.T.dot(sqrt)

        np_tst.assert_allclose(proposed.dot(tester), product.dot(tester))
        # Should I check the submatrices or assume that's covered?

    def test_quadratic_form(self):
        """Test whether quadratic_form returns the intended result."""
        matrix1 = scipy.linalg.toeplitz((1., 1/3., 1/9., 1/27., 1/81.))
        matrix2 = scipy.linalg.toeplitz((1., .5, .25, .125, .0625, .03125))

        product = inversion.linalg.DaskKroneckerProductOperator(
            matrix1, matrix2)

        tester = np.eye(product.shape[0])

        np_tst.assert_allclose(product.quadratic_form(tester),
                               scipy.linalg.kron(matrix1, matrix2))


class TestUtilKroneckerProduct(unittest2.TestCase):
    """Test inversion.util.kronecker_product."""

    def test_delegation(self):
        """Test that it delegates to subclasses where appropriate."""
        HomogeneousIsotropicCorrelation = (
            inversion.correlations.HomogeneousIsotropicCorrelation)
        corr_class = inversion.correlations.GaussianCorrelation
        corr_fun = corr_class(5)

        op1 = HomogeneousIsotropicCorrelation.from_function(corr_fun, 15)
        op2 = HomogeneousIsotropicCorrelation.from_function(corr_fun, 20)

        combined_op = inversion.util.kronecker_product(op1, op2)
        proposed_result = HomogeneousIsotropicCorrelation.from_function(
            corr_fun, (15, 20))

        self.assertIsInstance(combined_op, HomogeneousIsotropicCorrelation)
        self.assertSequenceEqual(combined_op.shape,
                                 tuple(np.multiply(op1.shape, op2.shape)))
        self.assertEqual(combined_op._underlying_shape,
                         proposed_result._underlying_shape)
        np_tst.assert_allclose(combined_op._fourier_near_zero,
                               proposed_result._fourier_near_zero)
        np_tst.assert_allclose(combined_op._corr_fourier,
                               proposed_result._corr_fourier,
                               rtol=1e-5, atol=1e-6)

    def test_array_array(self):
        """Test array-array Kronecker product."""
        mat1 = np.eye(2)
        mat2 = np.eye(3)

        combined_op = inversion.util.kronecker_product(mat1, mat2)

        self.assertIsInstance(combined_op, np.ndarray)
        self.assertSequenceEqual(combined_op.shape,
                                 tuple(np.multiply(mat1.shape, mat2.shape)))
        np_tst.assert_allclose(combined_op, scipy.linalg.kron(mat1, mat2))

    def test_array_sparse(self):
        """Test array-sparse matrix Kronecker products."""
        mat1 = np.eye(3)
        mat2 = scipy.sparse.eye(10)

        combined_op = inversion.util.kronecker_product(mat1, mat2)
        big_ident = np.eye(30)

        self.assertIsInstance(
            combined_op, inversion.linalg.DaskKroneckerProductOperator)
        self.assertSequenceEqual(combined_op.shape,
                                 tuple(np.multiply(mat1.shape, mat2.shape)))
        np_tst.assert_allclose(combined_op.dot(big_ident),
                               big_ident)

    def test_linop_array(self):
        """Test linop-sparse Kronecker products."""
        HomogeneousIsotropicCorrelation = (
            inversion.correlations.HomogeneousIsotropicCorrelation)
        corr_class = inversion.correlations.GaussianCorrelation
        corr_fun = corr_class(5)

        op1 = HomogeneousIsotropicCorrelation.from_function(corr_fun, 15)
        mat2 = np.eye(10)
        combined_op = inversion.util.kronecker_product(op1, mat2)

        self.assertIsInstance(
            combined_op, inversion.correlations.SchmidtKroneckerProduct)
        self.assertSequenceEqual(combined_op.shape,
                                 tuple(np.multiply(op1.shape, mat2.shape)))


class TestIntegrators(unittest2.TestCase):
    """Test the integrators."""

    def test_exp(self):
        """Test that the integrators can integrate y'=y for one unit.

        Uses a very small integration step to get similarity from
        one-step forward Euler.

        """
        for integrator in (inversion.integrators.forward_euler,
                           inversion.integrators.scipy_odeint):
            with self.subTest(integrator=getname(integrator)):
                solns = integrator(
                    lambda y, t: y,
                    1.,
                    (0, 1),
                    1e-6)

                np_tst.assert_allclose(solns[1, :],
                                       np.exp(1.), rtol=1e-5)


class TestEnsembleIntegrators(unittest2.TestCase):
    """Test the ensemble integrators."""

    IMPLEMENTATIONS = (inversion.ensemble.integrators.
                       EnsembleIntegrator.__subclasses__())

    if sys.version_info >= (3, 5):
        OS_ISSUES = ()
    else:
        OS_ISSUES = (inversion.ensemble.integrators.
                     MultiprocessEnsembleIntegrator,)

    @staticmethod
    def trial_function(y, t):
        r"""Evaluate derivative given y and t.

        This is f(y, t) in :math:`y^\prime = f(y, t)`.
        Here :math:`f(y, t) = y`.

        Parameters
        ----------
        y: array_like[N]
        t: float

        Returns
        -------
        yprime: array_like[N]
        """
        return y

    def test_working(self):
        """Test if the integrators work."""
        start_state = np.arange(2.).reshape(2, 1)
        end_state = np.array((0, np.exp(1))).reshape(2, 1)

        for int_cls in self.IMPLEMENTATIONS:
            if int_cls in self.OS_ISSUES:
                raise unittest2.SkipTest(
                    "OS has trouble with {name:s}"
                    .format(name=getname(int_cls)))

            with self.subTest(int_cls=getname(int_cls)):
                int_inst = int_cls(inversion.integrators.forward_euler)
                solns = int_inst(self.trial_function, start_state,
                                 (0, 1), 1e-5)

                np_tst.assert_allclose(solns[1, :, :], end_state, rtol=1e-4)


class TestCovarianceEstimation(unittest2.TestCase):
    """Test the background error covariance estimators.

    Not quite a true unit test, since it uses inversion.correlations.
    """

    def test_nmc_identity(self):
        """Test the NMC method for a simple case.

        Uses stationary noise on top of a forecast of zero.
        """
        test_sample = int(1e6)
        state_size = 8
        sample_cov = np.eye(state_size)

        sim_forecasts = np_rand.multivariate_normal(
            np.zeros(state_size),
            sample_cov,
            (test_sample, 2))

        for assume_homogeneous in (False, True):
            with self.subTest(assume_homogeneous=assume_homogeneous):
                estimated_cov = (
                    inversion.covariance_estimation.nmc_covariances(
                        sim_forecasts, 4, assume_homogeneous))
                np_tst.assert_allclose(estimated_cov, sample_cov,
                                       rtol=1e-2, atol=3e-3)

    def test_nmc_generated(self):
        """Test NMC method for a more complicated case.

        Gaussian correlations are still a bad choice.
        """
        test_sample = int(1e6)
        state_size = 5

        corr_class = inversion.correlations.ExponentialCorrelation
        for dist in (1, 3):
            corr_fun = corr_class(dist)

            # corr_mat = inversion.correlations.make_matrix(corr_fun,
            #                                               state_size)
            corr_op = (
                inversion.correlations.HomogeneousIsotropicCorrelation.
                from_function(
                    corr_fun, state_size))
            corr_mat = corr_op.dot(np.eye(state_size))

            sim_forecasts = np_rand.multivariate_normal(
                np.zeros(state_size),
                corr_mat,
                (test_sample, 2))

            for assume_homogeneous in (False, True):
                with self.subTest(assume_homogeneous=assume_homogeneous,
                                  corr_class=getname(corr_class),
                                  dist=dist):
                    estimated_cov = (
                        inversion.covariance_estimation.nmc_covariances(
                            sim_forecasts, 4, assume_homogeneous))

                    # 1/sqrt(test_sample) would be roughly the
                    # standard error for the mean. I don't know the
                    # characteristics of the distribution for
                    # covariance matrices, but they tend to be more
                    # finicky.
                    np_tst.assert_allclose(estimated_cov, corr_mat,
                                           rtol=1e-2, atol=3e-3)

    def test_cq_identity(self):
        """Test Canadian Quick covariances for identity."""
        state_size = 7
        sample_size = int(1e6)
        state_cov = np.eye(state_size) / 2

        forecast_tendencies = np_rand.standard_normal(
            (sample_size, state_size))
        climatology = np.cumsum(forecast_tendencies, axis=0)

        for assume_homogeneous in (False, True):
            with self.subTest(assume_homogeneous=assume_homogeneous):
                estimated_covariances = (
                    inversion.covariance_estimation.canadian_quick_covariances(
                        climatology, assume_homogeneous))

                np_tst.assert_allclose(estimated_covariances, state_cov,
                                       rtol=2e-3, atol=2e-3)

    def test_cq_expon(self):
        """Test Canadian Quick covariances with exponential truth."""
        state_size = 5
        sample_size = int(1e6)

        corr_cls = inversion.correlations.ExponentialCorrelation
        for dist in (1, 3):
            corr_fun = corr_cls(dist)

            # assume_homogeneous forces this structure
            # using it simplifies the tests
            corr_op = (
                inversion.correlations.HomogeneousIsotropicCorrelation.
                from_function(corr_fun, state_size))
            corr_mat = corr_op.dot(np.eye(state_size))

            forecast_tendencies = np_rand.multivariate_normal(
                np.zeros(state_size), corr_mat * 2,
                sample_size)
            climatology = np.cumsum(forecast_tendencies, axis=0)

            for assume_homogeneous in (False, True):
                with self.subTest(assume_homogeneous=assume_homogeneous,
                                  dist=dist):
                    estimated_covariances = (
                        inversion.covariance_estimation.
                        canadian_quick_covariances(
                            climatology, assume_homogeneous))

                    np_tst.assert_allclose(estimated_covariances, corr_mat,
                                           rtol=3e-3, atol=3e-3)


class TestEnsembleBase(unittest2.TestCase):
    """Test the utility functions in inversion.ensemble."""

    sample_size = int(1e6)
    state_size = 10

    def test_mean(self):
        """Test whether ensemble mean is close."""
        sample_data = np_rand.standard_normal((self.sample_size,
                                               self.state_size))

        np_tst.assert_allclose(inversion.ensemble.mean(sample_data),
                               np.zeros(self.state_size),
                               # Use 5 * standard error
                               # 3 fails a tad too often for my taste.
                               atol=5 / np.sqrt(self.sample_size))

    def test_spread(self):
        """Test whether ensemble spread is reasonable."""
        sample_data = np_rand.standard_normal((self.sample_size,
                                               self.state_size))
        self.assertAlmostEqual(inversion.ensemble.spread(sample_data),
                               # rtol is 10**(-places)/state_size
                               self.state_size, places=1)

    def test_mean_and_perturbations(self):
        """Test if mean and perturbations combine to give original."""
        sample_data = np_rand.standard_normal((self.sample_size,
                                               self.state_size))

        mean, perturbations = inversion.ensemble.mean_and_perturbations(
            sample_data)

        np_tst.assert_allclose(mean + perturbations, sample_data)


class TestUtilSchmidtDecomposition(unittest2.TestCase):
    """Test the Schimdt decomposition code in inversion.linalg."""

    def setUp(self):
        """Set up the test vectors."""
        from scipy.linalg import kron
        # The notation here is borrowed from quantum computation.  I
        # use the k prefix to indicate the vector has precisely one
        # nonzero entry, a one.  The digits following are the binary
        # representation of the zero-based index of that one.
        self.k0 = np.array((1, 0)).reshape(-1, 1)
        self.k1 = np.array((0, 1)).reshape(-1, 1)

        self.k00 = kron(self.k0, self.k0)
        self.k01 = kron(self.k0, self.k1)
        self.k10 = kron(self.k1, self.k0)
        self.k11 = kron(self.k1, self.k1)

        self.k000 = kron(self.k0, self.k00)
        self.k001 = kron(self.k0, self.k01)
        self.k010 = kron(self.k0, self.k10)
        self.k011 = kron(self.k0, self.k11)
        self.k100 = kron(self.k1, self.k00)
        self.k101 = kron(self.k1, self.k01)
        self.k110 = kron(self.k1, self.k10)
        self.k111 = kron(self.k1, self.k11)

    def test_simple_combinations(self):
        """Test many combinations of vectors."""
        possibilities = (
            self.k0, self.k1,
            self.k00, self.k01, self.k10, self.k11)

        for vec1, vec2 in itertools.product(possibilities, possibilities):
            with self.subTest(vec1=vec1[:, 0], vec2=vec2[:, 0]):
                composite_state = scipy.linalg.kron(vec1, vec2)

                reported_decomposition = inversion.linalg.schmidt_decomposition(
                    composite_state, vec1.shape[0], vec2.shape[0])
                lambdas, vecs1, vecs2 = da.compute(*reported_decomposition)

                np_tst.assert_allclose(np.nonzero(lambdas),
                                       [[0]])
                np_tst.assert_allclose(np.abs(vecs1[0]),
                                       vec1[:, 0])
                np_tst.assert_allclose(np.abs(vecs2[0]),
                                       vec2[:, 0])
                np_tst.assert_allclose(
                    reported_decomposition[0][0] *
                    scipy.linalg.kron(
                        np.asarray(reported_decomposition[1][:1].T),
                        np.asarray(reported_decomposition[2][:1].T)),
                    composite_state)

    def test_composite_compination(self):
        """Test composite combinations."""
        sqrt2 = math.sqrt(2)
        rsqrt2 = 1 / sqrt2
        # b00 = (k00 + k11) / sqrt2
        # b01 = (k00 - k11) / sqrt2
        # b10 = (k01 + k10) / sqrt2
        # b11 = (k01 - k10) / sqrt2
        composite_state = (
            scipy.linalg.kron(self.k0, self.k00) +
            scipy.linalg.kron(self.k1, self.k01)) / sqrt2
        res_lambda, res_vec1, res_vec2 = inversion.linalg.schmidt_decomposition(
            composite_state, 2, 4)

        self.assertEqual(res_vec1.shape, (2, 2))
        self.assertEqual(res_vec2.shape, (2, 4))
        np_tst.assert_allclose(res_lambda, (rsqrt2, rsqrt2))
        np_tst.assert_allclose(
            sum(lambd * scipy.linalg.kron(
                np.asarray(vec1).reshape(-1, 1),
                np.asarray(vec2).reshape(-1, 1))
                for lambd, vec1, vec2 in zip(res_lambda, res_vec1, res_vec2)),
            composite_state)

    def test_epr_state(self):
        """Test that it correctly decomposes the EPR state."""
        sqrt2o2 = math.sqrt(2) / 2
        epr_state = (self.k01 - self.k10) * sqrt2o2

        lambdas, vecs1, vecs2 = inversion.linalg.schmidt_decomposition(
            epr_state, 2, 2)

        lambdas = np.asarray(lambdas)
        vecs1 = np.asarray(vecs1)
        vecs2 = np.asarray(vecs2)

        # This will not recover the original decomposition
        np_tst.assert_allclose(lambdas, (sqrt2o2, sqrt2o2))
        self.assertAlmostEqual(np.prod(lambdas), .5)

        for vec1, vec2 in zip(vecs1, vecs2):
            if np.allclose(np.abs(vec1), self.k0[:, 0]):
                sign = 1
            else:
                sign = -1
            np_tst.assert_allclose(vec1, sign * vec2[-1::-1])

        np_tst.assert_allclose(
            sum(lambd * scipy.linalg.kron(
                np.asarray(vec1).reshape(-1, 1),
                np.asarray(vec2).reshape(-1, 1))
                for lambd, vec1, vec2 in zip(lambdas, vecs1, vecs2)),
            epr_state)


class TestUtilIsOdd(unittest2.TestCase):
    """Test inversion.linalg.is_odd."""

    MAX_TO_TEST = 100

    def test_known_odd(self):
        """Test known odd numbers."""
        is_odd = inversion.linalg.is_odd

        for i in range(1, self.MAX_TO_TEST, 2):
            with self.subTest(i=i):
                self.assertTrue(is_odd(i))

    def test_known_even(self):
        """Test known even numbers."""
        is_odd = inversion.linalg.is_odd

        for i in range(0, self.MAX_TO_TEST, 2):
            with self.subTest(i=i):
                self.assertFalse(is_odd(i))


class TestUtilToLinearOperator(unittest2.TestCase):
    """Test inversion.linalg.tolinearoperator."""

    def test_tolinearoperator(self):
        """Test that tolinearoperator returns LinearOperators."""
        tolinearoperator = inversion.linalg.tolinearoperator
        LinearOperator = inversion.linalg.DaskLinearOperator

        for trial in (0, 1., (0, 1), [0, 1], ((1, 0), (0, 1)),
                      [[0, 1.], [1., 0]], np.arange(5),
                      scipy.sparse.identity(8), da.arange(10, chunks=10)):
            with self.subTest(trial=trial):
                self.assertIsInstance(tolinearoperator(trial),
                                      LinearOperator)


class TestUtilKron(unittest2.TestCase):
    """Test inversion.linalg.kron against scipy.linalg.kron."""

    def test_util_kron(self):
        """Test my kronecker implementation against scipy's."""
        trial_inputs = (1, (1,), [0], np.arange(10), np.eye(5),
                        da.arange(10, chunks=10), da.eye(5, chunks=5))
        my_kron = inversion.linalg.kron
        scipy_kron = scipy.linalg.kron

        for input1, input2 in itertools.product(trial_inputs, repeat=2):
            my_result = my_kron(input1, input2)
            scipy_result = scipy_kron(
                np.atleast_2d(input1), np.atleast_2d(input2))

            np_tst.assert_allclose(my_result, scipy_result)


class TestHomogeneousInversions(unittest2.TestCase):
    """Ensure inversion functions work with HomogeneousIsotropicCorrelation.

    Integration test to ensure things work together as intended.

    TODO: Check that the answers are reasonable.
    """

    CURRENTLY_BROKEN = frozenset(
        (inversion.optimal_interpolation.simple,  # Invalid addition
         inversion.optimal_interpolation.scipy_chol,  # cho_factor/solve
         inversion.variational.incr_chol))  # cho_factor/solve

    def setUp(self):
        """Define values for use in test methods."""
        self.bg_vals = np.zeros(10, dtype=DTYPE)
        self.obs_vals = np.ones(3, dtype=DTYPE)

        corr_class = inversion.correlations.ExponentialCorrelation
        corr_fun = corr_class(2)

        bg_corr = (inversion.correlations.HomogeneousIsotropicCorrelation.
                   from_function(corr_fun, self.bg_vals.shape))
        obs_corr = (inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, self.obs_vals.shape))
        obs_op = scipy.sparse.diags(
            (.5, 1, .5),
            (-1, 0, 1),
            (self.obs_vals.shape[0], self.bg_vals.shape[0]))

        self.bg_corr = (bg_corr, bg_corr.dot(np.eye(*bg_corr.shape)))
        self.obs_corr = (obs_corr, obs_corr.dot(np.eye(*obs_corr.shape)))
        self.obs_op = (inversion.linalg.tolinearoperator(obs_op.toarray()),
                       # Dask requires subscripting; diagonal sparse
                       # matrices don't do this.
                       obs_op.toarray())

    def test_combinations_produce_answer(self):
        """Test that background error as a LinearOperator doesn't crash."""
        for inversion_method in ALL_METHODS:
            for bg_corr, obs_corr, obs_op in (itertools.product(
                    self.bg_corr, self.obs_corr, self.obs_op)):
                if inversion_method in self.CURRENTLY_BROKEN:
                    # TODO: XFAIL
                    continue
                with self.subTest(method=getname(inversion_method),
                                  bg_corr=getname(type(bg_corr)),
                                  obs_corr=getname(type(obs_corr)),
                                  obs_op=getname(type(obs_op))):
                    post, post_cov = inversion_method(
                        self.bg_vals, bg_corr,
                        self.obs_vals, obs_corr,
                        obs_op)


class TestKroneckerQuadraticForm(unittest2.TestCase):
    """Test that DaskKroneckerProductOperator.quadratic_form works."""

    def test_simple(self):
        """Test for identity matrix."""
        mat1 = da.eye(2, chunks=2)
        vectors = da.eye(4, chunks=4)

        product = inversion.linalg.DaskKroneckerProductOperator(mat1, mat1)

        for i, vec in enumerate(vectors):
            with self.subTest(i=i):
                np_tst.assert_allclose(
                    product.quadratic_form(vec),
                    1)

    def test_shapes(self):
        """Test for different shapes of input."""
        mat1 = np.eye(2)
        vectors = np.eye(4)

        product = inversion.linalg.DaskKroneckerProductOperator(mat1, mat1)

        for i in range(4):
            stop = i + 1

            with self.subTest(num=stop):
                result = product.quadratic_form(vectors[:, :stop])
                np_tst.assert_allclose(result, vectors[:stop, :stop])

    def test_off_diagonal(self):
        """Test a case with off-diagonal elements."""
        mat1 = scipy.linalg.toeplitz(3.**-np.arange(5))
        mat2 = scipy.linalg.toeplitz(2.**-np.arange(10))

        scipy_kron = scipy.linalg.kron(mat1, mat2)
        linop_kron = inversion.linalg.DaskKroneckerProductOperator(mat1, mat2)

        test_arry = np.eye(50, 20)

        np_tst.assert_allclose(
            linop_kron.quadratic_form(test_arry),
            test_arry.T.dot(scipy_kron.dot(test_arry)))


class TestUtilProduct(unittest2.TestCase):
    """Test that quadratic_form works properly for ProductLinearOperator."""

    def test_symmetric_methods_added(self):
        """Test that the method is added or not as appropriate."""
        op1 = tolinearoperator(np.eye(2))
        op2 = inversion.linalg.DiagonalOperator(np.ones(2))
        ProductLinearOperator = inversion.linalg.ProductLinearOperator

        with self.subTest(num=2, same=True):
            product = ProductLinearOperator(op1.T, op1)

            self.assertTrue(hasattr(product, "quadratic_form"))

        with self.subTest(num=2, same=False):
            product = ProductLinearOperator(op1.T, op2)

            self.assertFalse(hasattr(product, "quadratic_form"))

        with self.subTest(num=3, same=True):
            product = ProductLinearOperator(op1.T, op2, op1)

            self.assertTrue(hasattr(product, "quadratic_form"))

        with self.subTest(num=3, same=False):
            product = ProductLinearOperator(op1.T, op1, op2)

            self.assertFalse(hasattr(product, "quadratic_form"))

    def test_returned_shape(self):
        """Test that the shape of the result is correct."""
        op1 = tolinearoperator(np.eye(3))
        op2 = inversion.linalg.DiagonalOperator(np.ones(3))
        ProductLinearOperator = inversion.linalg.ProductLinearOperator

        vectors = np.eye(3)

        with self.subTest(num=2):
            product = ProductLinearOperator(op1.T, op1)

            for i in range(vectors.shape[0]):
                stop = i + 1
                with self.subTest(shape=stop):
                    result = product.quadratic_form(vectors[:, :stop])
                    self.assertEqual(result.shape, (stop, stop))
                    np_tst.assert_allclose(result, vectors[:stop, :stop])

        with self.subTest(num=3):
            product = ProductLinearOperator(op1.T, op2, op1)

            for i in range(vectors.shape[0]):
                stop = i + 1

                with self.subTest(shape=stop):
                    result = product.quadratic_form(vectors[:, :stop])
                    self.assertEqual(result.shape, (stop, stop))
                    np_tst.assert_allclose(result, vectors[:stop, :stop])

    def test_product_sqrt(self):
        """Test the square root of a ProductLinearOperator."""
        mat1 = np.eye(3)
        mat1[1, 0] = 1
        op1 = tolinearoperator(mat1)
        op2 = inversion.linalg.DiagonalOperator((1, .25, .0625))
        ProductLinearOperator = inversion.linalg.ProductLinearOperator

        tester = np.eye(3)

        with self.subTest(num=2):
            product = ProductLinearOperator(op1.T, op1)
            mat_sqrt = product.sqrt()
            test = mat_sqrt.T.dot(mat_sqrt)

            np_tst.assert_allclose(test.dot(tester), product.dot(tester))

        with self.subTest(num=3):
            product = ProductLinearOperator(op1.T, op2, op1)
            mat_sqrt = product.sqrt()
            test = mat_sqrt.T.dot(mat_sqrt)

            np_tst.assert_allclose(test.dot(tester), product.dot(tester))

    def test_transpose(self):
        """Test that transpose works."""
        mat1 = np.eye(3)
        mat1[1, 0] = 1
        op1 = tolinearoperator(mat1)
        op2 = inversion.linalg.DiagonalOperator((1, .25, .0625))
        ProductLinearOperator = inversion.linalg.ProductLinearOperator

        product = ProductLinearOperator(op1, op2)
        result = product.T
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result._operators, (op2.T, op1.T))

    def test_adjoinst(self):
        """Test that adjoint works."""
        mat1 = np.eye(3)
        mat1[1, 0] = 1
        op1 = tolinearoperator(mat1)
        op2 = inversion.linalg.DiagonalOperator((1, .25, .0625))
        ProductLinearOperator = inversion.linalg.ProductLinearOperator

        product = ProductLinearOperator(op1, op2)
        result = product.H
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result._operators, (op2.H, op1.H))


class TestCorrelationStandardDeviation(unittest2.TestCase):
    """Test that this sub-class works as intended."""

    def test_transpose(self):
        """Test transpose works as expected."""
        correlations = np.eye(2)
        stds = np.ones(2)

        covariances = inversion.covariances.CorrelationStandardDeviation(
            correlations, stds)

        self.assertIs(covariances, covariances.T)

    def test_values(self):
        """Test that the combined operator is equivalent."""
        correlations = np.array(((1, .5), (.5, 1)))
        stds = (2, 1)

        linop_cov = inversion.covariances.CorrelationStandardDeviation(
            correlations, stds)
        np_cov = np.diag(stds).dot(correlations.dot(np.diag(stds)))

        np_tst.assert_allclose(linop_cov.dot(np.eye(2)), np_cov)

    def test_adjoint(self):
        """Test that the adjoint works as expected."""
        correlations = np.eye(2)
        stds = np.ones(2)

        covariances = inversion.covariances.CorrelationStandardDeviation(
            correlations, stds)

        self.assertIs(covariances, covariances.H)


class TestOddChunks(unittest2.TestCase):
    """Test that input with odd chunks still works.

    The chunking required for influence functions to load into memory
    might not work all the way through the inversion, since
    :func:`dask.array.linalg.solve` needs square chunks.  Make sure
    inversion functions provide this.
    """

    N_BG = 50
    BG_CHUNK = 30
    N_OBS = 30
    OBS_CHUNK = 20

    def test_unusual(self):
        """Test unusual chunking schemes."""
        bg_cov = da.eye(self.N_BG, chunks=self.BG_CHUNK)
        background = da.zeros(self.N_BG, chunks=self.BG_CHUNK)
        observations = da.ones(self.N_OBS, chunks=self.OBS_CHUNK)
        obs_cov = da.eye(self.N_OBS, chunks=self.OBS_CHUNK)
        obs_op = da.eye(N=self.N_OBS, M=self.N_BG,
                        chunks=30).rechunk(
            (self.OBS_CHUNK, self.BG_CHUNK))

        for inversion_method in ALL_METHODS:
            with self.subTest(method=getname(inversion_method)):
                post, post_cov = inversion_method(
                    background, bg_cov, observations, obs_cov, obs_op)


class TestCovariances(unittest2.TestCase):
    """Test the covariance classes."""

    # SelfAdjointLinearOperator isn't really a concrete class

    def test_diagonal_operator_from_domain_stds(self):
        """Test DiagonalOperator creation from array of values."""
        stds = np.arange(20).reshape(4, 5)

        inversion.covariances.DiagonalOperator(stds)

    def test_diagonal_operator_behavior(self):
        """Test behavior of DiagonalOperator."""
        diag = np.arange(10.) + 1.

        op = inversion.covariances.DiagonalOperator(diag)
        arry = np.diag(diag)

        test_vecs = (np.arange(10.),
                     np.ones(10),
                     np.array((0., 1) * 5))
        test_mats = (np.eye(10, 4),
                     np.hstack(test_vecs))

        for vec in test_vecs:
            with self.subTest(test_vec=vec):
                with self.subTest(direction="forward"):
                    np_tst.assert_allclose(op.dot(vec), arry.dot(vec))
                with self.subTest(direction="inverse"):
                    np_tst.assert_allclose(np.asarray(op.solve(vec)),
                                           np_la.solve(arry, vec))

        for mat in test_mats:
            with self.subTest(test_mat=mat):
                np_tst.assert_allclose(op.dot(vec), arry.dot(vec))

    def test_diagonal_2d_vector(self):
        """Test DiagonalOperator works with Nx1 vector."""
        diag = np.arange(10.)
        op = inversion.covariances.DiagonalOperator(diag)
        vec = np.arange(10.)[:, np.newaxis]
        result = op.dot(vec)
        self.assertEqual(da.squeeze(result).ndim, 1)
        self.assertEqual(result.shape, (10, 1))

    def test_diagonal_self_adjoint(self):
        """Test the self-adjoint methods of DiagonalOperator."""
        operator = inversion.covariances.DiagonalOperator(np.arange(10.))

        self.assertIs(operator, operator.H)
        self.assertIs(operator, operator.T)

    def test_diagonal_sqrt(self):
        """Test that DiagonalOperator.sqrt works as expected."""
        DiagonalOperator = inversion.covariances.DiagonalOperator
        diagonal = np.arange(10.)
        operator = DiagonalOperator(diagonal)
        sqrt = operator.sqrt()

        self.assertIsInstance(sqrt, DiagonalOperator)
        np_tst.assert_allclose(sqrt._diag, np.sqrt(diagonal))

    def test_product(self):
        """Test that the product operator works as expected."""
        # TODO: move this somewhere appropriate.
        test_vecs = (np.arange(5.),
                     np.ones(5, dtype=DTYPE),
                     np.array((0, 1, 0, 1, 0.)))
        test_mats = (np.eye(5, 4, dtype=DTYPE),
                     np.vstack(test_vecs).T)

        operator_list = (np.arange(25.).reshape(5, 5) + np.diag((2.,) * 5),
                         np.eye(5, dtype=DTYPE),
                         np.ones((5, 5), dtype=DTYPE) + np.diag((1.,) * 5))
        operator = inversion.linalg.ProductLinearOperator(*operator_list)

        arry = reduce(lambda x, y: x.dot(y), operator_list)

        for vec in test_vecs:
            with self.subTest(test_vec=vec):
                with self.subTest(direction="forward"):
                    np_tst.assert_allclose(operator.dot(vec),
                                           arry.dot(vec))
                with self.subTest(direction="inverse"):
                    np_tst.assert_allclose(np.asanyarray(operator.solve(vec)),
                                           np_la.solve(arry, vec))

        for mat in test_mats:
            with self.subTest(test_mat=mat):
                np_tst.assert_allclose(operator.dot(mat),
                                       arry.dot(mat))


class TestUtilMatrixSqrt(unittest2.TestCase):
    """Test that inversion.linalg.sqrt works as planned."""

    def test_array(self):
        """Test that matrix_sqrt works with arrays."""
        matrix_sqrt = inversion.linalg.matrix_sqrt

        with self.subTest(trial="identity"):
            mat = np.eye(3)
            proposed = matrix_sqrt(mat)
            expected = cholesky(mat)

            np_tst.assert_allclose(proposed, expected)

        with self.subTest(trial="toeplitz"):
            mat = scipy.linalg.toeplitz((1, .5, .25, .125))

            proposed = matrix_sqrt(mat)
            expected = cholesky(da.asarray(mat))

            np_tst.assert_allclose(proposed, expected)

    def test_matrix_op(self):
        """Test that matrix_sqrt recognizes MatrixLinearOperator."""
        mat = np.eye(10)
        mat_op = inversion.linalg.DaskMatrixLinearOperator(mat)

        result1 = inversion.linalg.matrix_sqrt(mat_op)
        self.assertIsInstance(result1, np.ndarray)

        result2 = inversion.linalg.matrix_sqrt(mat)
        tester = np.eye(*result1.shape)
        np_tst.assert_allclose(result1.dot(tester), result2.dot(tester))

    @unittest2.expectedFailure
    def test_semidefinite_array(self):
        """Test that matrix_sqrt works for semidefinite arrays.

        This currently fails due to lazy evaluation.
        """
        mat = np.eye(2)
        mat[1, 1] = 0

        proposed = inversion.linalg.matrix_sqrt(mat)
        # Fun with one and zero
        np_tst.assert_allclose(proposed, mat)

    def test_delegate(self):
        """Test that matrix_sqrt delegates where possible."""
        operator = (inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_array((1, .5, .25, .125, .25, .5, 1)))

        proposed = inversion.linalg.matrix_sqrt(operator)

        self.assertIsInstance(
            proposed, inversion.correlations.HomogeneousIsotropicCorrelation)

    def test_nonsquare(self):
        """Test matrix_sqrt raises for non-square input."""
        with self.assertRaises(ValueError):
            inversion.linalg.matrix_sqrt(np.eye(4, 3))

    # TODO: test arbitrary linear operators
    # TODO: Test odd chunking


class TestPointVortex(unittest2.TestCase):
    """Test the point-vortex model."""

    def test_equilateral(self):
        """Test derivative for equilateral triangle."""
        model = inversion.models.PointVortex((1, 1, 1.))

        triangle = ((0, 0),
                    (2, 0),
                    (1, math.sqrt(3)))
        velocities = model(np.asarray(triangle))
        np_tst.assert_allclose(
            velocities,
            ((.25 * math.sqrt(3), -.75),
             (.25 * math.sqrt(3), .75),
             (-.5 * math.sqrt(3), 0)))

    def test_line(self):
        """Test derivative for a line."""
        model = inversion.models.PointVortex((1, 1, 1.))

        with self.subTest(orientation="horiz"):
            line = ((1, 0),
                    (0, 0),
                    (-1, 0.))
            velocities = model(np.asarray(line))
            np_tst.assert_allclose(
                velocities,
                ((0, 1.5),
                 (0, 0),
                 (0, -1.5)))

        with self.subTest(orientation="vert"):
            line = ((0, 1),
                    (0, 0),
                    (0, -1.))
            velocities = model(np.asarray(line))
            np_tst.assert_allclose(
                velocities,
                ((-1.5, 0),
                 (0, 0),
                 (1.5, 0)))

    def test_self_advection(self):
        """Test advection for a pair of vortices."""
        model = inversion.models.PointVortex((1, -1.))

        locs = ((0, 0),
                (1, 0.))
        velocities = model(np.asarray(locs))
        np_tst.assert_allclose(
            velocities,
            ((0, 1),
             (0, 1)))

    def test_uneven_orbit(self):
        """Test orbit with unequal vortices."""
        model = inversion.models.PointVortex((2, 1))

        with self.subTest(orientation="+x"):
            locs = ((0, 0),
                    (1, 0.))
            velocities = model(np.asarray(locs))
            np_tst.assert_allclose(
                velocities,
                ((0, -1),
                 (0, 2)))

        with self.subTest(orientation="+y"):
            locs = ((0, 0),
                    (0, 1.))
            velocities = model(np.asarray(locs))
            np_tst.assert_allclose(
                velocities,
                ((1, 0),
                 (-2, 0)))

        with self.subTest(orientation="-x"):
            locs = ((0, 0),
                    (-1, 0.))
            velocities = model(np.asarray(locs))
            np_tst.assert_allclose(
                velocities,
                ((0, 1),
                 (0, -2)))

        with self.subTest(orientation="-y"):
            locs = ((0, 0),
                    (0, -1.))
            velocities = model(np.asarray(locs))
            np_tst.assert_allclose(
                velocities,
                ((-1, 0),
                 (2, 0)))


class TestLorenz96(unittest2.TestCase):
    """Test Lorenz96 system."""

    def test_zero(self):
        """Test model with state of all zeros."""
        model = inversion.models.Lorenz96(8., 40)

        state = np.zeros(40)
        deriv = model(state)
        np_tst.assert_allclose(
            deriv,
            np.full(40, 8))

    def test_equilibrium(self):
        """Test model at equilibrium."""
        model = inversion.models.Lorenz96(8., 40)

        state = np.full(40, 8.)
        deriv = model(state)
        np_tst.assert_allclose(
            deriv,
            np.zeros(40))

    def test_single_perturbation(self):
        """Test simple perturbation from eq."""
        model = inversion.models.Lorenz96(8., 40)

        for i in range(40):
            with self.subTest(i=i):
                state = np.full(40, 8.)
                state[i] += 1
                deriv = model(state)

                analytic = np.zeros(40)
                analytic[i - 1] = 8
                analytic[i] = -1
                analytic[(i + 2) % 40] = -8

                np_tst.assert_allclose(deriv, analytic)


class TestReducedUncertainties(unittest2.TestCase):
    """Test that inversion methods properly treat requested uncertainties."""

    def test_identical_simple(self):
        """Test that the result is the same when requesting such."""
        bg = 1.
        obs = 2.
        bg_cov = 1.
        obs_cov = 1.
        obs_op = 1.

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                directval, directcov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)
                altval, altcov = method(
                    bg, bg_cov, obs, obs_cov, obs_op,
                    bg_cov, obs_op)
                np_tst.assert_allclose(directval, altval)
                np_tst.assert_allclose(directcov, altcov)

    def test_identical_complicated(self):
        """Test that the result remains the same with harder problem."""
        bg = np.zeros(10)
        obs = np.ones(5)
        bg_cov = scipy.linalg.toeplitz(3.**-np.arange(10.))
        obs_cov = np.eye(5)
        obs_op = np.eye(5, 10)

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                directval, directcov = method(
                    bg, bg_cov, obs, obs_cov, obs_op)
                altval, altcov = method(
                    bg, bg_cov, obs, obs_cov, obs_op,
                    bg_cov, obs_op)
                np_tst.assert_allclose(directval, altval,
                                       rtol=1e-5, atol=1e-5)
                np_tst.assert_allclose(directcov, altcov,
                                       rtol=ITERATIVE_COVARIANCE_TOLERANCE,
                                       atol=ITERATIVE_COVARIANCE_TOLERANCE)

    @unittest2.expectedFailure
    def test_reduced_uncorrelated(self):
        """Test reduced uncertainties for uncorrelated background.

        HBHT changes a lot in this case.
        """
        bg = (0, 0.)
        bg_cov = np.eye(2)
        obs = (1.,)
        obs_cov = 1.
        obs_op = (.5, .5)

        # Using mean for bg, not sum
        bg_cov_red = 2./4
        obs_op_red = 1.

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                value, cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op,
                    bg_cov_red, obs_op_red)
                np_tst.assert_allclose(
                    value, (1/3., 1/3.))
                # ((5/6., 1/6.), (1/6., 5/6.))
                np_tst.assert_allclose(
                    cov,
                    2./3)

    def test_reduced_correlated(self):
        """Test reduced uncertainties for a simple case."""
        bg = (0, 0.)
        bg_cov = [[1, .9], [.9, 1]]
        obs = (1.,)
        obs_cov = 1.
        obs_op = (.5, .5)

        # Using mean for bg, not sum
        bg_cov_red = 3.8/4
        obs_op_red = 1.

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                value, cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op,
                    bg_cov_red, obs_op_red)
                np_tst.assert_allclose(
                    value, (.48717949, .48717949))
                # ((.53, .43), (.43, .53))
                # analytic: 1.9
                # reduced: .79167
                np_tst.assert_allclose(
                    cov,
                    .48717948717948717)


if __name__ == "__main__":
    unittest2.main()
