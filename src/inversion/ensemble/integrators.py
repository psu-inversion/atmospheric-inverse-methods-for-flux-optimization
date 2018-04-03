"""Functions to integrate multiple initial states with the same ODE.

Various degrees of parallelism.

"""
from __future__ import absolute_import

import multiprocessing
import threading
import abc
import sys

import numpy as np
import six

# Should this module be set up to use dask?
from numpy import transpose
from numpy import atleast_1d, atleast_2d

MAX_PARALLEL = None
"""Passed to :class:`multiprocessing.Pool` constructor

None means use CPU count.
"""
if ((hasattr(multiprocessing, "set_start_method") and
     sys.platform == "cygwin")):
    multiprocessing.set_start_method("spawn")


class EnsembleIntegrator(six.with_metaclass(abc.ABCMeta)):
    """Metaclass for ensemble integrators."""

    def __init__(self, integrator):
        """Set up to integrate.

        Parameters
        ----------
        integrator: callable
            The integrator to use. Assumed to follow the signature of
            :func:`inversion.integrators.forward_euler`

        """
        self._integrator = integrator

    @abc.abstractmethod
    def __call__(self, func, y0s, tseq, dt, Dfunc=None):
        """Integrate each y0 in `y0s` to times in `tseq`.

        Parameters
        ----------
        func: callable(y, t)
        y0s: array_like[K, N]
        tseq: sequence[M]
            tseq[0] := t0
        Dfunc: callable(y, t)
            Dfunc[i, j] = d func[i]/d y[j]

        Returns
        -------
        sol_seqs: array_like[M, K, N]
            Array of solutions to y' = func(y, t)
        """
        pass


class SerialEnsembleIntegrator(EnsembleIntegrator):
    """Integrate the ensemble members one at a time."""

    def __call__(self, func, y0s, tseq, dt, Dfunc=None):
        """Integrate each y0 in `y0s` to times in `tseq`.

        Parameters
        ----------
        func: callable(y, t)
        y0s: array_like[K, N]
        tseq: sequence[M]
            tseq[0] := t0
        Dfunc: callable(y, t)
            Dfunc[i, j] = d func[i]/d y[j]

        Returns
        -------
        sol_seqs: array_like[M, K, N]
            Array of solutions to y' = func(y, t)
        """
        tseq = atleast_1d(tseq)
        y0s = atleast_2d(y0s)

        res = np.empty((len(tseq),) + y0s.shape, y0s.dtype)

        integrator = self._integrator
        for i in range(res.shape[1]):
            res[:, i, :] = integrator(func, y0s[i, :], tseq, dt, Dfunc)

        return res


class MultiprocessEnsembleIntegrator(EnsembleIntegrator):
    """Integrate the ensemble members in different processes.

    Unrelaible on cygwin due to fork() failures.
    """

    def __call__(self, func, y0s, tseq, dt, Dfunc=None):
        """Integrate each y0 in `y0s` to times in `tseq`.

        Break up the work across :const:`MAX_PARALLEL` processes.  Not
        sure if :func:`scipy.integrators.odeint` is threadsafe enough for
        this application.  I know the object oriented solvers are not.

        Parameters
        ----------
        func: callable(y, t)
        y0s: array_like[K, N]
        tseq: sequence[M]
            tseq[0] := t0
        Dfunc: callable(y, t)
            Dfunc[i, j] = d func[i]/d y[j]
        integrator: callable(func, y0s, tseq, dt, Dfunc
            The integrator to use

        Returns
        -------
        sol_seqs: array_like[M, K, N]
            Array of solutions to y' = func(y, t)

        """
        y0s = atleast_2d(y0s)

        # Need the function to be picklable,
        # so need to set these here.
        self._func = func
        self._tseq = tseq
        self._dt = dt
        self._Dfunc = Dfunc

        pool = multiprocessing.Pool(MAX_PARALLEL)
        res_list = pool.map(
            self._integrate_one,
            y0s)
        pool.close()
        pool.join()
        # res_list is currently K, M, N

        return transpose(res_list, (1, 0, 2))

    def _integrate_one(self, y0):
        """Integrate a single member.

        Parameters
        ----------
        y0: array_like[N]

        Returns
        -------
        array_like[M, N]
        """
        return self._integrator(self._func, y0, self._tseq,
                                self._dt, self._Dfunc)


class ThreadedEnsembleIntegrator(EnsembleIntegrator):
    """Integrate the ensemble members in different threads."""

    def __call__(self, func, y0s, tseq, dt, Dfunc=None):
        """Integrate each y0 in `y0s` to times in `tseq`.

        Break up the work across :const:`MAX_PARALLEL` threads.  Not
        sure if :func:`scipy.integrators.odeint` is threadsafe enough for
        this application.  I know the object oriented solvers are not.

        Parameters
        ----------
        func: callable(y, t)
        y0s: array_like[K, N]
        tseq: sequence[T]
            tseq[0] := t0
        Dfunc: callable(y, t)
            Dfunc[i, j] = d func[i]/d y[j]
        integrator: callable(func, y0s, tseq, dt, Dfunc
            The integrator to use

        Returns
        -------
        sol_seqs: array_like[T, K, N]
            Array of solutions to y' = func(y, t)
        """
        y0s = atleast_2d(y0s)
        res = np.empty((len(tseq),) + y0s.shape, y0s.dtype)

        integrator = self._integrator

        def thread_fun(i):
            """Integrate ensemble member `i`.

            Assumes anything not thread-local is global.

            Parameters
            ----------
            i: the thread id
            """
            res[:, i, :] = integrator(func, y0s[i, :], tseq, dt, Dfunc)

        pool = [threading.Thread(target=thread_fun, args=(i,))
                for i in range(y0s.shape[0])]

        for thread in pool:
            thread.start()

        for thread in pool:
            thread.join()

        return res
