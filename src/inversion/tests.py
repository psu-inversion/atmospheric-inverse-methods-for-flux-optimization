"""Tests for the inversion package.

Includes tests using random data, analytic solutions, and checks that
different methods agree for simple problems.

"""
from __future__ import print_function, division
import fractions

import numpy as np
import numpy.linalg as la
import numpy.random as np_rand
import numpy.testing as np_tst
import scipy.optimize
import unittest2

import inversion.covariance_estimation
import inversion.optimal_interpolation
import inversion.correlations
import inversion.integrators
import inversion.variational
import inversion.ensemble
import inversion.noise
import inversion.psas

# If adding other inexact methods to the list tested, be sure to add
# those to the `if "var" in name or "psas" in name` and
# `if "psas" in name` tests as applicable.
ALL_METHODS = (
    inversion.optimal_interpolation.simple,
    inversion.optimal_interpolation.fold_common,
    inversion.optimal_interpolation.scipy_chol,
    inversion.variational.simple,
    inversion.variational.incremental,
    inversion.variational.incr_chol,
    inversion.psas.simple,
    inversion.psas.fold_common,
)
ITERATIVE_METHOD_START = 3
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

                np_tst.assert_allclose(
                    post, PRECISE_DTYPE(14 + fractions.Fraction(1, 3)))
                np_tst.assert_allclose(
                    post_cov, PRECISE_DTYPE(fractions.Fraction(2, 3)))

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
                                       test_size * 2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),) * 2)

                # test postitive definite
                chol_upper = la.cholesky(corr_mat)

                # test symmetry
                np_tst.assert_allclose(chol_upper.dot(chol_upper.T),
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
                        rtol=1e-12, atol=1e-14)

                    # check if it actually is positive definite
                    la.cholesky(corr_mat)

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
                                       test_size * 2, dtype=float)
                corr_mat = corr.reshape((np.prod(test_size),) * 2)

                # test postitive definite
                chol_upper = la.cholesky(corr_mat)

                # test symmetry
                np_tst.assert_allclose(chol_upper.dot(chol_upper.T),
                                       corr_mat,
                                       rtol=1e-4, atol=1e-4)

    def test_1d_make_matrix(self):
        """Test make_matrix for 1D correlations.

        Checks against original value.
        """
        test_nt = 300

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 5, 10, 30, 100):
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
                    chol_upper = la.cholesky(corr_mat)

                    # test symmetry
                    np_tst.assert_allclose(chol_upper.dot(chol_upper.T),
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
                for dist in (1, 3, 10, 30):
                    corr_fun = corr_class(dist)

                    corr_op = (
                        inversion.correlations.HomogeneousIsotropicCorrelation.
                        from_function(corr_fun, test_shape))
                    corr_mat = corr_op.dot(np.eye(np.prod(test_shape)))

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
        test_nt = 500
        test_lst = (np.zeros(test_nt), np.ones(test_nt), np.arange(test_nt),
                    np.eye(100, test_nt)[-1])

        for corr_class in (
                inversion.correlations.DistanceCorrelationFunction.
                __subclasses__()):
            for dist in (1, 3, 10, 30):
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


class TestCovarianceEstimation(unittest2.TestCase):
    """Test the background error covariance estimators.

    Not quite a true unit test, since it uses inversion.correlations.
    """

    def test_nmc_identity(self):
        """Test that the NMC method for simple case.

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
                                       rtol=1e-2, atol=1e-2)

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
                                           rtol=1e-2, atol=1e-2)


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
                               # Use 3 * standard error
                               atol=3 / np.sqrt(self.sample_size))

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


if __name__ == "__main__":
    unittest2.main()
