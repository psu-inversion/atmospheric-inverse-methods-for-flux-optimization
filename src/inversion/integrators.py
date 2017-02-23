"""Integrators.

Some I wrote, some dispatch to other functions
"""
import numpy as np
import scipy.integrate

def scipy_odeint(func, y0, tseq, dt=None, Dfunc=None):
    """Integrate y to times in `tseq`.

    Parameters
    ----------
    func: callable(y, t)
    y0: np.ndarray[N]
    tseq: sequence[M]
       tseq[0] := t0
    Dfunc: callable(y, t)
      Dfunc[i, j] = d func[i]/ d y[j]

    Returns
    -------
    sol_seq: np.ndarray[M, N]
        Array of solutions to y' = func(y, t)
    """
    if dt is None:
        dt = 0.0
    return scipy.integrate.odeint(func, y0, tseq, Dfun=Dfunc, h0=dt)

def forward_euler(func, y0, tseq, dt, Dfunc=None):
    """Integrate y to times in `tseq`.

    Parameters
    ----------
    func: callable(y, t)
    y0: np.ndarray[N]
    tseq: sequence[M]
        tseq[0] := t0
    Dfunc: callable(y, t)
        Dfunc[i, j] = d func[i]/ d y[j]
        ignored

    Returns
    -------
    sol_seq: np.ndarray[M, N]
        Array of solutions to y' = func(y, t)
    """
    tseq = np.atleast_1d(tseq)
    y0 = np.atleast_1d(y0)

    currt = tseq[0]
    curry = y0.copy()

    sol_seq = np.empty((len(tseq), len(y0)), dtype=y0.dtype)
    for i, tadd in enumerate(tseq):

        while currt + dt < tadd:
            deriv = func(curry, currt)
            curry += deriv * dt
            currt += dt

        sol_seq[i,:] = curry
    return sol_seq
