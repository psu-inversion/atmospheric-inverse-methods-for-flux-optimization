"""Functions to integrate multiple initial states with the same ODE.

Various degrees of parallelism.

"""
from __future__ import absolute_import

import multiprocessing
import threading

import numpy as np

import inversion.integrators

MAX_PARALLEL = None
"""Passed to :class:`multiprocessing.Pool` constructor

None means use CPU count.
"""


def serial(func, y0s, tseq, dt, Dfunc=None,
           integrator=inversion.integrators.scipy_odeint):
    """Integrate each y0 in `y0s` to times in `tseq`.

    Parameters
    ----------
    func: callable(y, t)
    y0s: np.ndarray[K, N]
    tseq: sequence[M]
        tseq[0] := t0
    Dfunc: callable(y, t)
        Dfunc[i, j] = d func[i]/d y[j]
    integrator: callable(func, y0s, tseq, dt, Dfunc
        The integrator to use

    Returns
    -------
    sol_seqs: np.ndarray[M, K, N]
        Array of solutions to y' = func(y, t)
    """
    tseq = np.atleast_1d(tseq)
    y0s = np.atleast_2d(y0s)

    res = np.empty((len(tseq),) + y0s.shape, y0s.dtype)

    for i in range(res.shape[1]):
        res[:, i, :] = integrator(func, y0s[i, :], tseq, dt, Dfunc)

    return res


def processes(func, y0s, tseq, dt, Dfunc=None,
              integrator=inversion.integrators.scipy_odeint):
    """Integrate each y0 in `y0s` to times in `tseq`.

    Break up the work across :const:`MAX_PARALLEL` processes.  Not
    sure if :func:`scipy.integrators.odeint` is threadsafe enough for
    this application.  I know the object oriented solvers are not.

    Parameters
    ----------
    func: callable(y, t)
    y0s: np.ndarray[K, N]
    tseq: sequence[M]
        tseq[0] := t0
    Dfunc: callable(y, t)
        Dfunc[i, j] = d func[i]/d y[j]
    integrator: callable(func, y0s, tseq, dt, Dfunc
        The integrator to use

    Returns
    -------
    sol_seqs: np.ndarray[M, K, N]
        Array of solutions to y' = func(y, t)

    """
    y0s = np.atleast_2d(y0s)

    pool = multiprocessing.Pool(MAX_PARALLEL)
    res_list = pool.map(
        lambda y0: integrator(func, y0, tseq, dt, Dfunc),
        y0s)
    pool.close()
    pool.join()

    return np.atleast_3d(res_list)


def threads(func, y0s, tseq, dt, Dfunc=None,
            integrator=inversion.integrators.scipy_odeint):
    """Integrate each y0 in `y0s` to times in `tseq`.

    Break up the work across :const:`MAX_PARALLEL` threads.  Not
    sure if :func:`scipy.integrators.odeint` is threadsafe enough for
    this application.  I know the object oriented solvers are not.

    Parameters
    ----------
    func: callable(y, t)
    y0s: np.ndarray[K, N]
    tseq: sequence[M]
        tseq[0] := t0
    Dfunc: callable(y, t)
        Dfunc[i, j] = d func[i]/d y[j]
    integrator: callable(func, y0s, tseq, dt, Dfunc
        The integrator to use

    Returns
    -------
    sol_seqs: np.ndarray[M, K, N]
        Array of solutions to y' = func(y, t)
    """
    y0s = np.atleast_2d(y0s)
    res = np.empty((len(tseq),) + y0s.shape, y0s.dtype)

    def thread_fun(i):
        """The function for the threads.

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
