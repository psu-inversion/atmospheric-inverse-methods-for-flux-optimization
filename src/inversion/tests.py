"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.

"""
from __future__ import print_function, division
import fractions
import itertools
import operator
import os.path
import atexit
import pickle
import math
try:
    from functools import reduce
except ImportError:
    # reduce used to be a builtin
    pass

import numpy as np
import numpy.linalg as np_la
import numpy.linalg as la
import numpy.testing as np_tst
import scipy.linalg
import scipy.sparse
import scipy.optimize
# Import from scipy.linalg if not using dask
from scipy.linalg import cholesky
from scipy.sparse.linalg.interface import LinearOperator, MatrixLinearOperator

import unittest2
import pyfftw

import pandas as pd
import xarray
try:
    import sparse
    HAVE_SPARSE = True
except ImportError:
    HAVE_SPARSE = False

import inversion.optimal_interpolation
import inversion.correlations
import inversion.covariances
import inversion.variational
import inversion.remapper
import inversion.wrapper
import inversion.linalg
import inversion.noise
import inversion.psas
import inversion.util
from inversion.linalg import tolinearoperator


if os.path.exists(".pyfftw.pickle"):
    with open(".pyfftw.pickle", "rb") as wis_in:
        WISDOM = pickle.load(wis_in)

    if isinstance(WISDOM[0], str):
        WISDOM = [wis.encode("ascii")
                  for wis in WISDOM]
    pyfftw.import_wisdom(WISDOM)
    del WISDOM, wis_in

    def save_wisdom():
        """Save accumulated pyfftw wisdom.

        Saves in hidden file in current directory.
        Should help speed up subsequent test runs.
        """
        with open(".pyfftw.pickle", "wb") as wis_out:
            pickle.dump(pyfftw.export_wisdom(), wis_out, 2)
    atexit.register(save_wisdom)
    del save_wisdom


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

        noise = inversion.noise.gaussian_noise(combined, int(1e5))

        np_tst.assert_allclose(noise.mean(axis=0),
                               np.zeros(combined.shape[0]),
                               rtol=1.1e-2, atol=1e-2)
        np_tst.assert_allclose(np.cov(noise.T),
                               scipy.linalg.kron(op1, np.diag(diag)),
                               rtol=3e-2, atol=3e-2)

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

    def test_fails(self):
        """Test that construction fails on invalid input."""
        self.assertRaises(ValueError, inversion.noise.gaussian_noise,
                          np.ones(10))
        self.assertRaises(ValueError, inversion.noise.gaussian_noise,
                          np.eye(3, 2))


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
                try:
                    chol_upper = cholesky(corr_mat)
                except la.LinAlgError:
                    self.fail("corr_mat not positive definite")

                # test symmetry
                np_tst.assert_allclose(chol_upper.T.dot(chol_upper),
                                       corr_mat,
                                       rtol=1e-4, atol=1e-4)

    def test_2d_make_matrix(self):
        """Test make_matrix for 2D correlations.

        Checks against original value.

        This test is really slow.
        """
        # 30x25 Gaussian 10 not close
        test_nx = 30
        test_ny = 20
        test_points = test_ny * test_nx

        # TODO: speed up
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
                    for is_cyclic in (True, False):
                        corr_fun = corr_class(dist)

                        corr_op = (
                            inversion.correlations.
                            HomogeneousIsotropicCorrelation.
                            from_function(corr_fun, test_shape, is_cyclic))
                        # This is the fastest way to get column-major
                        # order from da.eye.
                        corr_mat = corr_op.dot(np.eye(test_size).T)

                        with self.subTest(
                                corr_class=getname(corr_class), dist=dist,
                                test_shape=test_shape, is_cyclic=is_cyclic,
                                test="symmetry"):
                            np_tst.assert_allclose(corr_mat, corr_mat.T,
                                                   rtol=1e-14, atol=1e-15)
                        with self.subTest(
                                corr_class=getname(corr_class), dist=dist,
                                test_shape=test_shape, is_cyclic=is_cyclic,
                                test="self-correlation"):
                            np_tst.assert_allclose(np.diag(corr_mat), 1)

    def test_1d_fft_correlation_cyclic(self):
        """Test HomogeneousIsotropicCorrelation for cyclic 1D arrays.

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

    def test_1d_fft_correlation_acyclic(self):
        """Test HomogeneousIsotropicCorrelation for acyclic 1D arrays.

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
                corr_fun = corr_class(dist)

                corr_mat = inversion.correlations.make_matrix(
                    corr_fun, test_nt)
                corr_op = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_nt, False))

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      inverse="no"):
                        np_tst.assert_allclose(
                            corr_op.dot(test_vec),
                            corr_mat.dot(test_vec),
                            rtol=1e-3, atol=1e-5)

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      inverse="yes"):
                        self.assertRaises(
                            NotImplementedError, corr_op.solve, test_vec)

    def test_2d_fft_correlation_cyclic(self):
        """Test HomogeneousIsotropicCorrelation for cyclic 2D arrays.

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

    def test_2d_fft_correlation_acyclic(self):
        """Test HomogeneousIsotropicCorrelation for acyclic 2D arrays.

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
                corr_fun = corr_class(dist)

                corr_mat = inversion.correlations.make_matrix(
                    corr_fun, test_shape)
                corr_op = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_shape, False))

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      direction="forward"):
                        np_tst.assert_allclose(
                            corr_op.dot(test_vec).reshape(test_shape),
                            corr_mat.dot(test_vec).reshape(test_shape),
                            rtol=1e-3, atol=1e-5)

                for i, test_vec in enumerate(test_lst):
                    with self.subTest(corr_class=getname(corr_class),
                                      dist=dist, test_num=i,
                                      direction="backward"):
                        self.assertRaises(
                            NotImplementedError, corr_op.solve, test_vec)

    def test_homogeneous_from_array_cyclic(self):
        """Make sure cyclic from_array can be roundtripped.

        Also tests that odd state sizes work.
        """
        test_size = 25

        corr_class = inversion.correlations.ExponentialCorrelation
        for dist in (1, 3, 5):
            with self.subTest(dist=dist):
                corr_fun = corr_class(dist)
                corr_op1 = (
                    inversion.correlations.HomogeneousIsotropicCorrelation.
                    from_function(corr_fun, test_size, True))
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

    def test_from_function_direct(self):
        """Directly test the output of from_function."""
        corr_func = (inversion.correlations.
                     ExponentialCorrelation(1 / np.log(2)))
        from_function = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_function)
        toeplitz = scipy.linalg.toeplitz

        with self.subTest(is_cyclic=False, nd=1):
            corr_op = from_function(corr_func, [10], False)
            np_tst.assert_allclose(
                corr_op.dot(np.eye(10)),
                toeplitz(0.5 ** np.arange(10)))

        with self.subTest(is_cyclic=False, nd=2):
            corr_op = from_function(corr_func, [2, 3], False)
            same_row = toeplitz(0.5 ** np.array([0, 1, 2]))
            other_row = toeplitz(
                0.5 ** np.array([1, np.sqrt(2), np.sqrt(5)]))
            np_tst.assert_allclose(
                corr_op.dot(np.eye(6)),
                np.block([[same_row, other_row],
                          [other_row, same_row]]))

            corr_op = from_function(corr_func, [4, 6], False)
            same_row = toeplitz(0.5 ** np.arange(6))
            next_row = toeplitz(
                0.5 ** np.array([1, np.sqrt(2), np.sqrt(5),
                                 np.sqrt(10), np.sqrt(17),
                                 np.sqrt(26)]))
            row_after_next = toeplitz(
                0.5 ** np.array([2, np.sqrt(5), np.sqrt(8),
                                 np.sqrt(13), np.sqrt(20),
                                 np.sqrt(29)]))
            two_rows_on = toeplitz(
                0.5 ** np.array([3, np.sqrt(10), np.sqrt(13),
                                 np.sqrt(18), 5, np.sqrt(34)]))
            np_tst.assert_allclose(
                corr_op.dot(np.eye(24)),
                np.block([[same_row, next_row, row_after_next, two_rows_on],
                          [next_row, same_row, next_row, row_after_next],
                          [row_after_next, next_row, same_row, next_row],
                          [two_rows_on, row_after_next, next_row, same_row]]))

        with self.subTest(is_cyclic=True, nd=1):
            corr_op = from_function(corr_func, [10], True)
            np_tst.assert_allclose(
                corr_op.dot(np.eye(10)),
                toeplitz(
                    0.5 ** np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])))

        with self.subTest(is_cyclic=True, nd=2):
            corr_op = from_function(corr_func, [4, 6])
            same_row = toeplitz(
                0.5 ** np.array([0, 1, 2, 3, 2, 1]))
            next_row = toeplitz(
                0.5 ** np.array([1, np.sqrt(2), np.sqrt(5),
                                 np.sqrt(10), np.sqrt(5), np.sqrt(2)]))
            row_after_next = toeplitz(
                0.5 ** np.array([2, np.sqrt(5), np.sqrt(8),
                                 np.sqrt(13), np.sqrt(8), np.sqrt(5)]))

            np_tst.assert_allclose(
                corr_op.dot(np.eye(24)),
                np.block([[same_row, next_row, row_after_next, next_row],
                          [next_row, same_row, next_row, row_after_next],
                          [row_after_next, next_row, same_row, next_row],
                          [next_row, row_after_next, next_row, same_row]]))

    def test_inv(self):
        """Test inverse matches linalg."""
        corr_func = (inversion.correlations.
                     ExponentialCorrelation(1 / np.log(2)))
        from_function = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_function)

        for test_shape in (10, 11, (3, 3), (4, 4)):
            with self.subTest(test_shape=test_shape):
                corr_op = from_function(corr_func, test_shape)
                test_size = np.prod(test_shape)
                ident = np.eye(test_size)
                np_tst.assert_allclose(
                    corr_op.inv().dot(ident),
                    la.inv(corr_op.dot(ident)),
                    rtol=1e-5, atol=1e-5)

    def test_acyclic_inv_fails(self):
        """Test inverse fails for acyclic correlations."""
        corr_func = (inversion.correlations.
                     ExponentialCorrelation(1 / np.log(2)))
        from_function = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_function)

        for test_shape in (10, 11, (3, 3), (4, 4)):
            with self.subTest(test_shape=test_shape):
                corr_op = from_function(corr_func, test_shape,
                                        is_cyclic=False)
                self.assertRaises(
                    NotImplementedError,
                    corr_op.inv)

    def test_wrong_shape_fails(self):
        """Test that a vector of the wrong shape fails noisily."""
        corr_func = (inversion.correlations.
                     ExponentialCorrelation(2))
        corr_op = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_function(corr_func, (3, 4)))

        self.assertRaises(
            ValueError,
            corr_op.solve,
            np.arange(5))

    def test_cyclic_from_array(self):
        """Test from_array with assumed cyclic correlations."""
        array = [1, .5, .25, .125, .0625, .125, .25, .5]
        op = (inversion.correlations.HomogeneousIsotropicCorrelation.
              from_array(array))
        mat = scipy.linalg.toeplitz(array)

        np_tst.assert_allclose(op.dot(np.eye(*mat.shape)),
                               mat)

    def test_acyclic_from_array(self):
        """Test from_array with correlations assumed acyclic."""
        array = [1, .5, .25, .125, .0625, .03125]
        op = (inversion.correlations.HomogeneousIsotropicCorrelation.
              from_array(array, False))
        mat = scipy.linalg.toeplitz(array)

        np_tst.assert_allclose(op.dot(np.eye(*mat.shape)),
                               mat)

    @unittest2.skipUnless(HAVE_SPARSE, "sparse not installed")
    def test_sparse(self):
        """Test HomogeneousIsotropicCorrelations work on sparse.COO."""
        array = 2. ** -np.arange(6)
        op = (inversion.correlations.HomogeneousIsotropicCorrelation.
              from_array(array, False))
        mat = scipy.linalg.toeplitz(array)

        np_tst.assert_allclose(op.dot(sparse.eye(*mat.shape)),
                               mat)


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

    def test_drop_small(self):
        """Test that the implementation properly drops small components."""
        SchmidtKroneckerProduct = (
            inversion.correlations.SchmidtKroneckerProduct)

        # I want to be sure either being smaller works.
        # Even versus odd also causes problems occasionally
        mat1 = np.eye(2)
        mat2 = np.eye(3)

        full_mat = SchmidtKroneckerProduct(
            mat1, mat2)
        test_vec = np.array([1, 0, 0,
                             0, 1e-15, 0])

        np_tst.assert_allclose(
            full_mat.dot(test_vec),
            np.eye(6, 1)[:, 0])

    def test_transpose(self):
        """Test that SchmidtKroneckerProduct can be transposed."""
        mat1 = np.eye(2)
        mat2 = np.eye(3)

        op = inversion.correlations.SchmidtKroneckerProduct(mat1, mat2)

        op_transpose = op.T

        np_tst.assert_allclose(
            op_transpose.dot(np.eye(6)),
            np.eye(6))


class TestYMKroneckerProduct(unittest2.TestCase):
    """Test the YM13 Kronecker product implementation for LinearOperators.

    This tests the :class:`~inversion.linalg.DaskKroneckerProductOperator`
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

    @unittest2.skipUnless(HAVE_SPARSE, "sparse not installed")
    def test_sparse(self):
        """Test that DaskKroneckerProductOperator works on sparse.COO."""
        sigmax = np.array(((0, 1), (1, 0)))
        sigmaz = np.array(((1, 0), (0, -1)))

        operator = inversion.linalg.DaskKroneckerProductOperator(
            sigmax, sigmaz)
        matrix = scipy.linalg.kron(sigmax, sigmaz)
        epr_state = np.array((0, .7071, -.7071, 0))

        np_tst.assert_allclose(
            operator.dot(sparse.COO(epr_state)),
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

        with self.subTest(check="asymmetric1"):
            mat1[0, 1] = 1
            product = DaskKroneckerProductOperator(
                mat1, mat2)
            transpose = product.T

            self.assertIsNot(transpose, product)
            np_tst.assert_allclose(transpose._operator1,
                                   mat1.T)

        with self.subTest(check="asymmetric2"):
            product = DaskKroneckerProductOperator(
                mat3, mat1)
            transpose = product.T

            self.assertIsNot(transpose, product)
            self.assertIs(transpose._operator1, mat3)
            np_tst.assert_allclose(transpose._operator2.A,
                                   mat1.T)

        with self.subTest(check="asymmetric3"):
            product = DaskKroneckerProductOperator(
                mat1, mat1)
            transpose = product.T

            np_tst.assert_allclose(transpose._operator1,
                                   mat1.T)
            np_tst.assert_allclose(transpose._operator2.A,
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

        product = inversion.linalg.DaskKroneckerProductOperator(
            matrix1, matrix2)
        sqrt = product.sqrt()
        proposed = sqrt.T.dot(sqrt)

        np_tst.assert_allclose(proposed.dot(tester), product.dot(tester))
        # Should I check the submatrices or assume that's covered?

    def test_quadratic_form(self):
        """Test whether quadratic_form returns the intended result."""
        matrix1 = scipy.linalg.toeplitz((1., 1/3., 1/9., 1/27., 1/81.))  # noqa
        matrix2 = scipy.linalg.toeplitz((1., .5, .25, .125, .0625, .03125))

        product = inversion.linalg.DaskKroneckerProductOperator(
            matrix1, matrix2)

        tester = np.eye(product.shape[0])

        dense_product = scipy.linalg.kron(matrix1, matrix2)
        test_vec = np.arange(product.shape[0])

        np_tst.assert_allclose(product.quadratic_form(tester),
                               dense_product)
        np_tst.assert_allclose(product.quadratic_form(test_vec),
                               test_vec.dot(dense_product.dot(test_vec)))

        test_op = inversion.linalg.DiagonalOperator(test_vec)
        self.assertRaises(
            TypeError,
            product.quadratic_form,
            test_op)
        self.assertRaises(
            ValueError,
            product.quadratic_form,
            test_vec[:-1])

    @unittest2.skipUnless(HAVE_SPARSE, "sparse not installed")
    def test_quadratic_form_sparse(self):
        """Test that quadratic_form works on sparse.COO."""
        matrix1 = scipy.linalg.toeplitz(3. ** -np.arange(4))
        matrix2 = scipy.linalg.toeplitz(5. ** -np.arange(5))

        product = inversion.linalg.DaskKroneckerProductOperator(
            matrix1, matrix2)
        tester = sparse.eye(product.shape[0])
        dense_product = scipy.linalg.kron(matrix1, matrix2)
        np_tst.assert_allclose(product.quadratic_form(tester),
                               dense_product)

    def test_matrix_linop(self):
        """Test that the implementation works with MatrixLinearOperator."""
        test_sizes = (4, 5)
        DaskKroneckerProductOperator = (
            inversion.linalg.DaskKroneckerProductOperator)

        # I want to be sure either being smaller works.
        # Even versus odd also causes problems occasionally
        for size1, size2 in itertools.product(test_sizes, repeat=2):
            with self.subTest(size1=size1, size2=size2):
                mat1 = tolinearoperator(np.eye(size1))
                mat2 = np.eye(size2)

                full_mat = DaskKroneckerProductOperator(
                    mat1, mat2)
                big_ident = np.eye(size1 * size2)

                np_tst.assert_allclose(
                    full_mat.dot(big_ident),
                    big_ident)

    def test_fails_not_array(self):
        """Test for failure if the first operator is not an array.

        The implementation requires it.  The implementation should
        fail quickly, not slowly.
        """
        mat1 = inversion.linalg.DiagonalOperator(np.arange(10))
        mat2 = np.eye(3)

        self.assertRaises(
            ValueError,
            inversion.linalg.DaskKroneckerProductOperator,
            mat1, mat2)

    def test_sqrt_fails(self):
        """Test that the square root fails for bad inputs.

        Specifically, non-square arrays and asymmetric arrays.
        """
        kron_op = inversion.linalg.DaskKroneckerProductOperator

        self.assertRaises(
            ValueError,
            kron_op(np.eye(3, 2), np.eye(3)).sqrt)
        self.assertRaises(
            ValueError,
            kron_op(np.eye(3), np.eye(2, 3)).sqrt)
        self.assertRaises(
            ValueError,
            kron_op(np.array([[1, 1], [0, 1]]), np.eye(3)).sqrt)


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

    def test_large_array_array(self):
        """Test large array-array Kronecker products.

        At some point it becomes faster to use Y&M kronecker
        representation than the dense one.
        """
        mat1 = np.eye(1 << 5)
        mat2 = np.eye(1 << 6)

        combined = inversion.util.kronecker_product(mat1, mat2)

        self.assertIsInstance(
            combined, inversion.linalg.DaskKroneckerProductOperator)
        self.assertSequenceEqual(combined.shape,
                                 tuple(np.multiply(mat1.shape, mat2.shape)))

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
        op1 = inversion.linalg.DiagonalOperator(np.arange(15))
        mat2 = np.eye(10)
        combined_op = inversion.util.kronecker_product(op1, mat2)

        self.assertIsInstance(
            combined_op, inversion.correlations.SchmidtKroneckerProduct)
        self.assertSequenceEqual(combined_op.shape,
                                 tuple(np.multiply(op1.shape, mat2.shape)))


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

                lambdas, vecs1, vecs2 = (
                    inversion.linalg.schmidt_decomposition(
                        composite_state, vec1.shape[0], vec2.shape[0]))

                np_tst.assert_allclose(np.nonzero(lambdas),
                                       [[0]])
                np_tst.assert_allclose(np.abs(vecs1[0]),
                                       vec1[:, 0])
                np_tst.assert_allclose(np.abs(vecs2[0]),
                                       vec2[:, 0])
                np_tst.assert_allclose(
                    lambdas[0] *
                    scipy.linalg.kron(
                        np.asarray(vecs1[:1].T),
                        np.asarray(vecs2[:1].T)),
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
        res_lambda, res_vec1, res_vec2 = (
            inversion.linalg.schmidt_decomposition(
                composite_state, 2, 4))

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

    def test_failure(self):
        """Test that schmidt_decomposition fails on invalid input."""
        schmidt_decomp = inversion.linalg.schmidt_decomposition
        schmidt_decomp(np.eye(6, 1), 2, 3)
        schmidt_decomp(np.arange(6), 2, 3)
        self.assertRaises(
            ValueError, schmidt_decomp, np.eye(6, 2), 2, 3)

    def test_big_vector(self):
        """Test size of results for large vectors."""
        vec = np.arange(1000, dtype=float)
        lambdas, uvecs, vvecs = (
            inversion.linalg.schmidt_decomposition(vec, 10, 100))
        self.assertLessEqual(len(lambdas), 10)
        self.assertNotIn(0, lambdas)
        np_tst.assert_allclose(
            sum(lambd[...] * scipy.linalg.kron(
                vec1.reshape(-1, 1),
                vec2.reshape(-1, 1))[:, 0]
                for lambd, vec1, vec2 in zip(lambdas, uvecs, vvecs)),
            vec, atol=1e-10)

    def test_small_nonzero(self):
        """Test that all returned data is significant."""
        vec = np.eye(20, 1)
        lambdas, uvecs, vvecs = (
            inversion.linalg.schmidt_decomposition(vec, 4, 5))
        self.assertNotIn(0, lambdas)


class TestUtilIsOdd(unittest2.TestCase):
    """Test inversion.linalg.is_odd."""

    MAX_TO_TEST = 100

    def test_known_odd(self):
        """Test known odd numbers."""
        is_odd = inversion.linalg_interface.is_odd

        for i in range(1, self.MAX_TO_TEST, 2):
            with self.subTest(i=i):
                self.assertTrue(is_odd(i))

    def test_known_even(self):
        """Test known even numbers."""
        is_odd = inversion.linalg_interface.is_odd

        for i in range(0, self.MAX_TO_TEST, 2):
            with self.subTest(i=i):
                self.assertFalse(is_odd(i))


class TestUtilToLinearOperator(unittest2.TestCase):
    """Test inversion.linalg.tolinearoperator."""

    def test_tolinearoperator(self):
        """Test that tolinearoperator returns LinearOperators."""
        tolinearoperator = inversion.linalg.tolinearoperator

        for trial in (0, 1., (0, 1), [0, 1], ((1, 0), (0, 1)),
                      [[0, 1.], [1., 0]], np.arange(5),
                      scipy.sparse.identity(8), np.arange(10)):
            with self.subTest(trial=trial):
                self.assertIsInstance(tolinearoperator(trial),
                                      LinearOperator)


class TestUtilKron(unittest2.TestCase):
    """Test inversion.linalg.kron against scipy.linalg.kron."""

    def test_util_kron(self):
        """Test my kronecker implementation against scipy's."""
        trial_inputs = (1, (1,), [0], np.arange(10), np.eye(5))
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

    CURRENTLY_BROKEN = frozenset((
        inversion.optimal_interpolation.scipy_chol,  # cho_factor/solve
        inversion.variational.incr_chol,  # cho_factor/solve
    ))

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

        if HAVE_SPARSE:
            self.obs_op += (sparse.COO(obs_op.toarray()),)

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
                with self.subTest(method=getname(inversion_method),
                                  bg_corr=getname(type(bg_corr)),
                                  obs_corr=getname(type(obs_corr)),
                                  obs_op=getname(type(obs_op)),
                                  request_reduced=True):
                    post, post_cov = inversion_method(
                        self.bg_vals, bg_corr,
                        self.obs_vals, obs_corr,
                        obs_op, bg_corr, obs_op)


class TestKroneckerQuadraticForm(unittest2.TestCase):
    """Test that DaskKroneckerProductOperator.quadratic_form works."""

    def test_simple(self):
        """Test for identity matrix."""
        mat1 = np.eye(2)
        vectors = np.eye(4)

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

    @unittest2.skipUnless(HAVE_SPARSE, "sparse not installed")
    def test_coo(self):
        """Test that `sparse.COO` works for the operator."""
        mat1 = scipy.linalg.toeplitz(3.**-np.arange(5))
        mat2 = scipy.linalg.toeplitz(2.**-np.arange(10))

        scipy_kron = scipy.linalg.kron(mat1, mat2)
        linop_kron = inversion.linalg.DaskKroneckerProductOperator(mat1, mat2)

        test_arry = sparse.eye(50, 20)

        np_tst.assert_allclose(
            linop_kron.quadratic_form(test_arry),
            test_arry.T.dot(scipy_kron.dot(test_arry.todense())))

    def test_failure_modes(self):
        """Test the failure modes of YMKron.quadratic_form."""
        mat1 = np.eye(3, 2)

        op1 = inversion.linalg.DaskKroneckerProductOperator(
            mat1, mat1)

        self.assertRaises(
            TypeError,
            op1.quadratic_form,
            np.arange(4))

        mat2 = np.eye(3)
        op2 = inversion.linalg.DaskKroneckerProductOperator(
            mat2, mat2)

        self.assertRaises(
            TypeError,
            op2.quadratic_form,
            op1)

        self.assertRaises(
            ValueError,
            op2.quadratic_form,
            np.arange(4))


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
        op3 = (inversion.correlations.HomogeneousIsotropicCorrelation
               .from_array(
                   (1, .5, .25), is_cyclic=False))
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

        with self.subTest(num=3, quadratic_form=True):
            product = ProductLinearOperator(op1.T, op3, op1)

            for i in range(vectors.shape[0]):
                stop = i + 1

                with self.subTest(shape=stop):
                    result = product.quadratic_form(vectors[:, :stop])
                    self.assertEqual(result.shape, (stop, stop))

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

    def test_adjoint(self):
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

    def test_bad_shapes(self):
        """Test that the product fails if the shapes are incompatible."""
        self.assertRaises(
            ValueError, inversion.linalg.ProductLinearOperator,
            np.eye(10, 3), np.eye(4, 10))
        self.assertRaises(
            ValueError, inversion.linalg.ProductLinearOperator,
            np.eye(10, 3), np.eye(3, 6), np.eye(5, 10))
        self.assertRaises(
            ValueError, inversion.linalg.ProductLinearOperator,
            np.eye(10, 4), np.eye(3, 6), np.eye(6, 10))

    def test_product_without_transpose(self):
        """Test ProductLinearOperator of non-transposing operators."""
        op = inversion.linalg_interface.DaskLinearOperator(
            shape=(10, 10),
            dtype=np.complex128,
            matvec=lambda vec: vec,
            matmat=lambda mat: mat)

        self.assertRaises(
            AttributeError,
            operator.attrgetter("T"),
            op)
        self.assertRaises(
            AttributeError,
            op.transpose)
        product = inversion.linalg_interface.ProductLinearOperator(
            op, op)
        self.assertRaises(
            AttributeError,
            product.transpose)


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

    def test_sqrt(self):
        """Test the sqrt method."""
        corr_sqrt = np.array([[1, .5, .25],
                              [0, 1, .5],
                              [0, 0, 1]])
        correlations = corr_sqrt.T.dot(corr_sqrt)
        stds = [1, .5, .25]

        covariance = inversion.covariances.CorrelationStandardDeviation(
            correlations, stds)
        sqrt = covariance.sqrt()
        self.assertEqual(len(sqrt._operators), 2)
        np_tst.assert_allclose(sqrt._operators[0].A, corr_sqrt)
        np_tst.assert_allclose(sqrt._operators[1]._diag, stds)


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
        self.assertEqual(np.squeeze(result).ndim, 1)
        self.assertEqual(result.shape, (10, 1))

    def test_diagonal_self_adjoint(self):
        """Test the self-adjoint methods of DiagonalOperator."""
        operator = inversion.covariances.DiagonalOperator(np.arange(10.))

        self.assertIs(operator, operator.H)
        self.assertIs(operator, operator.T)

    def test_diagonal_from_diagonal(self):
        """Test that creating a DiagonalOperator from another works."""
        op1 = inversion.linalg.DiagonalOperator(np.arange(10))
        op2 = inversion.linalg.DiagonalOperator(op1)

        np_tst.assert_allclose(
            op1.dot(np.arange(10)),
            op2.dot(np.arange(10)))

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


class TestLinalgSolve(unittest2.TestCase):
    """Test the general solve function."""

    def test_array_array(self):
        """Test solving a system with two arrays."""
        test_op = np.eye(2)
        test_vec = np.arange(2)

        np_tst.assert_allclose(
            inversion.linalg.solve(test_op, test_vec),
            la.solve(test_op, test_vec))

    def test_method_array(self):
        """Test that solve delegates."""
        test_op = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_array([1, .5, .25, .5]))
        test_vec = np.arange(4)

        np_tst.assert_allclose(
            inversion.linalg.solve(
                test_op, test_vec),
            la.solve(test_op.dot(np.eye(4)),
                     test_vec),
            atol=1e-10)

    def test_linop_array(self):
        """Test solve for a linear operator."""
        test_diag = np.ones(4)
        test_op = (
            LinearOperator(
                matvec=lambda x: x * test_diag, shape=(4, 4)))
        test_vec = np.arange(4)

        np_tst.assert_allclose(
            inversion.linalg.solve(test_op, test_vec),
            test_vec / test_diag)

        test_mat = np.eye(4)
        np_tst.assert_allclose(
            inversion.linalg.solve(test_op, test_mat),
            test_mat / test_diag[np.newaxis, :])

    def test_array_linop(self):
        """Test solve with a linear operator as rhs."""
        test_diag = 1 + np.arange(4)
        test_op = (
            inversion.linalg.DiagonalOperator(
                test_diag))
        test_arry = np.diag(test_diag)

        result = inversion.linalg.solve(
            test_arry, test_op)
        self.assertIsInstance(
            result, LinearOperator)
        np_tst.assert_allclose(
            result.dot(np.eye(4)),
            np.eye(4),
            atol=1e-10)

    def test_matop_matop(self):
        """Test solve with a MatrixOperator as rhs."""
        test_op = MatrixLinearOperator(
            np.eye(4))
        test_vec = MatrixLinearOperator(
            np.arange(4).reshape(4, 1))

        np_tst.assert_allclose(
            inversion.linalg.solve(
                test_op, test_vec),
            la.solve(test_op.A, test_vec.A))

    def test_bad_shape(self):
        """Test solve fails for bad input."""
        test_op = np.eye(4)
        test_vec = np.arange(5)

        self.assertRaises(
            ValueError,
            inversion.linalg.solve,
            test_op, test_vec)
        self.assertRaises(
            la.LinAlgError,
            inversion.linalg.solve,
            test_op[:, :-1],
            test_vec[:-1])

    def test_solve_method_fails(self):
        """Test that solve still works if a solve method fails."""
        test_op = (
            inversion.correlations.HomogeneousIsotropicCorrelation.
            from_array([1, .5, .25, .125, .0625], is_cyclic=False))
        ident = np.eye(*test_op.shape)
        test_mat = test_op.dot(ident)

        for vec in ident:
            with self.subTest(test_vec=vec):
                np_tst.assert_allclose(
                    inversion.linalg.solve(test_op, vec),
                    np_la.solve(test_mat, vec),
                    atol=1e-10)


class TestLinopSolve(unittest2.TestCase):
    """Test the abilities of linop_solve."""

    def test_single(self):
        """Test with single vector."""
        test_op = np.eye(4)
        test_vec = np.arange(4)

        np_tst.assert_allclose(
            inversion.linalg.linop_solve(
                test_op, test_vec),
            la.solve(test_op, test_vec))

    def test_multi(self):
        """Test with multiple vectors."""
        test_op = np.eye(4)
        test_vecs = np.arange(12).reshape(4, 3)

        np_tst.assert_allclose(
            inversion.linalg.linop_solve(
                test_op, test_vecs),
            la.solve(test_op, test_vecs))


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
            expected = cholesky(np.asarray(mat))

            np_tst.assert_allclose(proposed, expected)

    def test_matrix_op(self):
        """Test that matrix_sqrt recognizes MatrixLinearOperator."""
        mat = np.eye(10)
        mat_op = MatrixLinearOperator(mat)

        result1 = inversion.linalg.matrix_sqrt(mat_op)
        self.assertIsInstance(result1, np.ndarray)

        result2 = inversion.linalg.matrix_sqrt(mat)
        tester = np.eye(*result1.shape)
        np_tst.assert_allclose(result1.dot(tester), result2.dot(tester))

    def test_semidefinite_array(self):
        """Test that matrix_sqrt works for semidefinite arrays.

        This currently fails due to use of cholesky decomposition.  I
        would need to rewrite matrix_sqrt to catch the error and use
        scipy's matrix_sqrt: I'm already assuming symmetric inputs.

        """
        mat = np.diag([1, 0])

        with self.assertRaises(la.LinAlgError):
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

    def test_linop(self):
        """Test matrix_sqrt works for linear operators."""
        diag = np.arange(100, 0, -1)
        operator = LinearOperator(
            matvec=lambda x: diag * x, shape=(100, 100))
        sqrt = inversion.linalg.matrix_sqrt(operator)

        self.assertIsInstance(
            sqrt, inversion.linalg.ProductLinearOperator)
        self.assertEqual(len(sqrt._operators), 3)
        np_tst.assert_allclose(sqrt._operators[1]._diag,
                               0.07 + np.sqrt(np.arange(50, 100)),
                               rtol=1e-2, atol=1e-5)
        # np_tst.assert_allclose(sqrt._operators[0].A,
        #                        np.eye(100, 50)[:, ::-1],
        #                        rtol=1e-2, atol=1e-3)
        diag[50:] = 0
        np_tst.assert_allclose(sqrt.dot(np.eye(100)),
                               np.diag(np.sqrt(diag)),
                               rtol=1e-2, atol=1e-3)


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
                if "optimal_interpolation" in getname(method):
                    cov_tol = EXACT_TOLERANCE
                elif "variational" in getname(method):
                    cov_tol = 1.1 * ITERATIVE_COVARIANCE_TOLERANCE
                elif "psas" in getname(method):
                    # This uses the same code as Var for the reduced
                    # covariance.  My only guess is PSAS and the
                    # reduced covariance code have errors in
                    # offsetting directions.
                    raise unittest2.SkipTest(
                        "PSAS and reduced covariances do not play well")
                np_tst.assert_allclose(directcov, altcov,
                                       rtol=cov_tol,
                                       atol=cov_tol)

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
        bg_cov_red = 2. / 4
        obs_op_red = 1.

        for method in ALL_METHODS:
            with self.subTest(method=getname(method)):
                value, cov = method(
                    bg, bg_cov, obs, obs_cov, obs_op,
                    bg_cov_red, obs_op_red)
                np_tst.assert_allclose(
                    value, (1 / 3., 1 / 3.))
                # ((5/6., -1/6.), (-1/6., 5/6.))
                # var of sum is 4 / 3
                # var of mean is 1 / 3.
                np_tst.assert_allclose(
                    cov,
                    1. / 3)

    def test_reduced_correlated(self):
        """Test reduced uncertainties for a simple case."""
        bg = (0, 0.)
        bg_cov = [[1, .9], [.9, 1]]
        obs = (1.,)
        obs_cov = 1.
        obs_op = (.5, .5)

        # Using mean for bg, not sum
        bg_cov_red = 3.8 / 4
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

    def test_fail(self):
        """Test failure modes.

        These tests are handled in the wrapper, so I only test once.
        """
        bg = (0, 0.)
        bg_cov = [[1, .9], [.9, 1]]
        obs = (1.,)
        obs_cov = 1.
        obs_op = (.5, .5)

        # Using mean for bg, not sum
        bg_cov_red = 3.8 / 4
        obs_op_red = 1.

        with self.subTest(red_bg_cov=False, red_obs_op=True):
            self.assertRaises(
                ValueError, ALL_METHODS[0],
                bg, bg_cov, obs, obs_cov, obs_op,
                reduced_observation_operator=obs_op_red)

        with self.subTest(red_bg_cov=True, red_obs_op=False):
            self.assertRaises(
                ValueError, ALL_METHODS[0],
                bg, bg_cov, obs, obs_cov, obs_op,
                reduced_background_covariance=bg_cov_red)

    @unittest2.expectedFailure
    def test_multi_dim_correlated(self):
        """Test that reduced uncertainties are close even for multidimensional systems.

        The process of coarsening the resolutions of the state and
        observation operators is causing problems.
        """
        bg = np.zeros((7, 3, 5), dtype=DTYPE)
        obs = np.ones((5, 3), dtype=DTYPE)
        times_in_first_group = 3
        times_in_second_group = bg.shape[0] - times_in_first_group
        test_bg = np.arange(bg.size, dtype=DTYPE).reshape(bg.shape)

        temp_cov = scipy.linalg.toeplitz(
            (1 - 1. / 14) ** np.arange(bg.shape[0]))
        spatial_cov = (
            inversion.correlations.HomogeneousIsotropicCorrelation
            .from_function(inversion.correlations.ExponentialCorrelation(3.),
                           bg.shape[1:],
                           False))

        bg_cov = inversion.linalg.DaskKroneckerProductOperator(
            temp_cov, spatial_cov)

        same_tower_corr = scipy.linalg.toeplitz(
            np.exp(-np.arange(obs.shape[0], dtype=DTYPE)))
        other_tower_corr = np.zeros_like(same_tower_corr, dtype=DTYPE)
        obs_corr = np.block(
            [[same_tower_corr, other_tower_corr, other_tower_corr],
             [other_tower_corr, same_tower_corr, other_tower_corr],
             [other_tower_corr, other_tower_corr, same_tower_corr]])

        obs_op = np.zeros(obs.shape + bg.shape, dtype=DTYPE)
        for i in range(obs.shape[0]):
            for j in range(bg.shape[0] - i):
                obs_op[i, :, i + j, :, :] = np.exp(-j)

        spatial_remapper = np.full(
            bg.shape[1:],
            1. / np.product(bg.shape[1:]),
            dtype=DTYPE
        ).reshape(-1)
        spatial_cov_reduced = spatial_remapper.dot(
            spatial_cov.dot(spatial_remapper))

        temp_cov_reduced = np.block(
            [[temp_cov[:times_in_first_group, :times_in_first_group].mean(),
              temp_cov[:times_in_first_group, times_in_first_group:].mean()],
             [temp_cov[times_in_first_group:, :times_in_first_group].mean(),
              temp_cov[times_in_first_group:, times_in_first_group:].mean()]])
        bg_cov_red = inversion.util.kron(temp_cov_reduced,
                                         spatial_cov_reduced)

        obs_op_part_red = obs_op.sum(axis=-1).sum(axis=-1)
        obs_op_red = np.stack(
            [obs_op_part_red[:, :, :times_in_first_group].sum(axis=-1),
             obs_op_part_red[:, :, times_in_first_group:].sum(axis=-1)],
            axis=2)

        fluxes_in_first_group = (
            spatial_remapper.shape[0] * times_in_first_group)
        fluxes_in_second_group = (
            spatial_remapper.shape[0] * times_in_second_group)
        cov_remapper = np.block(
            [[np.full(fluxes_in_first_group, 1. / fluxes_in_first_group,
                      dtype=DTYPE),
              np.zeros(fluxes_in_second_group, dtype=DTYPE)],
             [np.zeros(fluxes_in_first_group, dtype=DTYPE),
              np.full(fluxes_in_second_group, 1. / fluxes_in_second_group,
                      dtype=DTYPE)]]
        )
        np_tst.assert_allclose(cov_remapper.dot(bg_cov.dot(cov_remapper.T)),
                               bg_cov_red)

        np_tst.assert_allclose(cov_remapper.dot(test_bg.reshape(-1)),
                               [test_bg[:times_in_first_group, :, :].mean(),
                                test_bg[times_in_first_group:, :, :].mean()])

        np_tst.assert_allclose(
            obs_op_red.reshape(obs_corr.shape[0],
                               bg_cov_red.shape[0]).dot(
                cov_remapper.dot(
                    test_bg.reshape(-1))),
            obs_op.reshape(obs_corr.shape[0],
                           bg_cov.shape[0]).dot(
                test_bg.reshape(-1))
        )

        for method in ALL_METHODS[:4]:
            with self.subTest(method=getname(method)):
                print(getname(method))
                post, post_cov = method(
                    bg.reshape(-1), bg_cov,
                    obs.reshape(-1), obs_corr,
                    obs_op.reshape(obs_corr.shape[0], bg_cov.shape[0]))
                post, post_cov_red = method(
                    bg.reshape(-1), bg_cov,
                    obs.reshape(-1), obs_corr,
                    obs_op.reshape(obs_corr.shape[0], bg_cov.shape[0]),
                    bg_cov_red,
                    obs_op_red.reshape(obs_corr.shape[0], bg_cov_red.shape[0]))

                la.cholesky(post_cov_red)
                reduced_post_cov = cov_remapper.dot(
                    post_cov.dot(cov_remapper.T))
                np_tst.assert_allclose(reduced_post_cov, post_cov_red)


class TestWrapperMetadata(unittest2.TestCase):
    """Test the metadata provided for the wrapper."""

    def test_cf(self):
        """Test metadata for CF attributes I can guess."""
        metadata = inversion.wrapper.global_attributes_dict()

        self.assertIn("Conventions", metadata)
        self.assertIn("CF", metadata.get("Conventions", ""))
        self.assertIn("history", metadata)

    def test_acdd(self):
        """Test metadata for ACDD attributes I can guess."""
        metadata = inversion.wrapper.global_attributes_dict()

        self.assertIn("Conventions", metadata)
        self.assertIn("standard_name_vocabulary", metadata)
        self.assertIn("date_created", metadata)
        self.assertIn("date_modified", metadata)
        self.assertIn("date_metadata_modified", metadata)
        self.assertIn("creator_name", metadata)

    def test_modules_list(self):
        """Test the list of installed modules.

        Will fail if neither pip nor conda is installed.
        """
        metadata = inversion.wrapper.global_attributes_dict()

        self.assertIn("installed_modules", metadata)
        installed_modules = metadata["installed_modules"]
        self.assertGreater(len(installed_modules), 0)

        for name_version in installed_modules:
            self.assertIn("=", name_version)


class TestWrapperUniform(unittest2.TestCase):
    """Test the wrapper functions."""

    def test_simple_site(self):
        """Test the wrapper for a temporally uniform inversion."""
        prior_fluxes = xarray.DataArray(
            np.zeros((40, 10, 20), dtype=DTYPE),
            coords=dict(
                flux_time=pd.date_range(start="2010-06-01", periods=40,
                                        freq="1D"),
                dim_y=np.arange(10, dtype=DTYPE),
                dim_x=np.arange(20, dtype=DTYPE),
            ),
            dims=("flux_time", "dim_y", "dim_x"),
            name="prior_fluxes",
            attrs=dict(units="umol/m^2/s"),
        )
        observations = xarray.DataArray(
            np.ones((20, 3), dtype=DTYPE),
            coords=dict(
                observation_time=prior_fluxes.coords["flux_time"][-20:].values,
                site=["here", "there", "somewhere"],
            ),
            dims=("observation_time", "site"),
            name="observations",
            attrs=dict(units="ppm"),
        )
        influence_function = xarray.DataArray(
            np.full((20, 3, 40, 10, 20), 1. / 8e3, dtype=DTYPE),
            coords=dict(
                observation_time=observations.coords["observation_time"],
                site=observations.coords["site"],
                flux_time=prior_fluxes.coords["flux_time"],
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("observation_time", "site",
                  "flux_time", "dim_y", "dim_x"),
            name="influence_functions",
            attrs=dict(units="ppm/(umol/m^2/s)"),
        )
        prior_flux_standard_deviations = xarray.DataArray(
            np.ones((10, 20), dtype=DTYPE),
            coords=dict(
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("dim_y", "dim_x"),
            name="prior_flux_standard_deviations",
            attrs=dict(units="umol/m^2/s"),
        )
        result = inversion.wrapper.invert_uniform(
            prior_fluxes,
            observations,
            influence_function,
            5,
            inversion.correlations.ExponentialCorrelation,
            10,
            3,
            prior_flux_standard_deviations,
            3,
            inversion.optimal_interpolation.save_sum,
        )

        self.assertIn("prior", result)
        self.assertIn("increment", result)
        self.assertIn("posterior", result)
        self.assertIn("posterior_covariance", result)

        for dim in result.dims:
            self.assertIn(dim, result.coords)

    def test_site_more_data(self):
        """Test the wrapper for a temporally uniform inversion."""
        prior_fluxes = xarray.DataArray(
            np.zeros((40, 10, 20), dtype=DTYPE),
            coords=dict(
                flux_time=pd.date_range(start="2010-06-01", periods=40,
                                        freq="1D"),
                dim_y=np.arange(10, dtype=DTYPE),
                dim_x=np.arange(20, dtype=DTYPE),
            ),
            dims=("flux_time", "dim_y", "dim_x"),
            name="prior_fluxes",
            attrs=dict(units="umol/m^2/s"),
        )
        observations = xarray.DataArray(
            np.ones((20, 3), dtype=DTYPE),
            coords=dict(
                observation_time=prior_fluxes.coords["flux_time"][-20:].values,
                site=["here", "there", "somewhere"],
                site_heights=(("site",), [100, 110, 120]),
            ),
            dims=("observation_time", "site"),
            name="observations",
            attrs=dict(units="ppm"),
        )
        influence_function = xarray.DataArray(
            np.full((20, 3, 40, 10, 20), 1. / 8e3, dtype=DTYPE),
            coords=dict(
                observation_time=observations.coords["observation_time"],
                site=observations.coords["site"],
                flux_time=prior_fluxes.coords["flux_time"],
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("observation_time", "site",
                  "flux_time", "dim_y", "dim_x"),
            name="influence_functions",
            attrs=dict(units="ppm/(umol/m^2/s)"),
        )
        prior_flux_standard_deviations = xarray.DataArray(
            np.ones((10, 20), dtype=DTYPE),
            coords=dict(
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("dim_y", "dim_x"),
            name="prior_flux_standard_deviations",
            attrs=dict(units="umol/m^2/s"),
        )
        result = inversion.wrapper.invert_uniform(
            prior_fluxes,
            observations,
            influence_function,
            5,
            inversion.correlations.ExponentialCorrelation,
            10,
            3,
            prior_flux_standard_deviations,
            3,
            inversion.optimal_interpolation.save_sum,
        )

        self.assertIn("prior", result)
        self.assertIn("increment", result)
        self.assertIn("posterior", result)
        self.assertIn("posterior_covariance", result)

        for dim in result.dims:
            self.assertIn(dim, result.coords)

    def test_site_as_aux_coord(self):
        """Test the wrapper for a temporally uniform inversion."""
        prior_fluxes = xarray.DataArray(
            np.zeros((40, 10, 20), dtype=DTYPE),
            coords=dict(
                flux_time=pd.date_range(start="2010-06-01", periods=40,
                                        freq="1D"),
                dim_y=np.arange(10, dtype=DTYPE),
                dim_x=np.arange(20, dtype=DTYPE),
            ),
            dims=("flux_time", "dim_y", "dim_x"),
            name="prior_fluxes",
            attrs=dict(units="umol/m^2/s"),
        )
        observations = xarray.DataArray(
            np.ones((20, 3), dtype=DTYPE),
            coords=dict(
                observation_time=prior_fluxes.coords["flux_time"][-20:].values,
                site_names=(("site",), ["here", "there", "somewhere"]),
                site_heights=(("site",), [100, 110, 120]),
            ),
            dims=("observation_time", "site"),
            name="observations",
            attrs=dict(units="ppm"),
        ).set_index(site="site_names")
        influence_function = xarray.DataArray(
            np.full((20, 3, 40, 10, 20), 1. / 8e3, dtype=DTYPE),
            coords=dict(
                observation_time=observations.coords["observation_time"],
                site=observations.coords["site"],
                flux_time=prior_fluxes.coords["flux_time"],
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("observation_time", "site",
                  "flux_time", "dim_y", "dim_x"),
            name="influence_functions",
            attrs=dict(units="ppm/(umol/m^2/s)"),
        )
        prior_flux_standard_deviations = xarray.DataArray(
            np.ones((10, 20), dtype=DTYPE),
            coords=dict(
                dim_y=prior_fluxes.coords["dim_y"],
                dim_x=prior_fluxes.coords["dim_x"],
            ),
            dims=("dim_y", "dim_x"),
            name="prior_flux_standard_deviations",
            attrs=dict(units="umol/m^2/s"),
        )

        observations = observations.stack(dict(
            observation=("observation_time", "site")
        ))
        influence_function = influence_function.stack(dict(
            observation=("observation_time", "site")
        ))

        result = inversion.wrapper.invert_uniform(
            prior_fluxes,
            observations,
            influence_function,
            5,
            inversion.correlations.ExponentialCorrelation,
            10,
            3,
            prior_flux_standard_deviations,
            3,
            inversion.optimal_interpolation.save_sum,
        )

        self.assertIn("prior", result)
        self.assertIn("increment", result)
        self.assertIn("posterior", result)
        self.assertIn("posterior_covariance", result)

        for dim in result.dims:
            self.assertIn(dim, result.coords)


class TestRemapper(unittest2.TestCase):
    """Test that the remappers are working properly."""

    def test_simple(self):
        """Test for the simplest possible case."""
        extensive, intensive = inversion.remapper.get_remappers(
            (6, 6), 3)

        old_data = np.arange(36, dtype=float).reshape(6, 6)
        test_sum = extensive.reshape(4, 36).dot(
            old_data.reshape(36)).reshape(2, 2)

        np_tst.assert_allclose(
            test_sum,
            [[63, 90],
             [225, 252]])

        test_mean = intensive.reshape(4, 36).dot(
            old_data.reshape(36)).reshape(2, 2)
        np_tst.assert_allclose(
            test_mean,
            [[7, 10],
             [25, 28]])

    def test_harder(self):
        """Test for domains that do not divide evenly."""
        extensive, intensive = inversion.remapper.get_remappers(
            (7, 7), 3)

        old_data = np.arange(49, dtype=float).reshape(7, 7)

        test_sum = extensive.reshape(9, 49).dot(
            old_data.reshape(49)).reshape(3, 3)
        np_tst.assert_allclose(
            test_sum,
            [[72, 99, 39],
             [261, 288, 102],
             [129, 138, 48]])

        test_mean = intensive.reshape(9, 49).dot(
            old_data.reshape(49)).reshape(3, 3)
        np_tst.assert_allclose(
            test_mean,
            [[8, 11, 13],
             [29, 32, 34],
             [43, 46, 48]])


if __name__ == "__main__":
    unittest2.main()
