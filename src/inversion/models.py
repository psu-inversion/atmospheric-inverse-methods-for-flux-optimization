"""Models for identical twin experiments and similar."""

import numpy as np

class Lorenz63:
    """Classic Lorenz '63 model."""

    def __init__(self, sigma=10, r=28, b=8./3):
        """Set up instance with parameters.

        Parameters
        ----------
        sigma: float
        r: float
        b: float
        """
        self._sigma = sigma
        self._r = r
        self._b = b

    def __call__(self, state):
        """The derivative at state.

        Parameters
        ----------
        state: np.ndarray[3]

        Returns
        -------
        deriv: np.ndarray[3]
        """
        return np.array((
            self._sigma * (state[1] - state[0]),
            self._r * state[0] - state[1] - state[0] * state[2],
            state[0] * state[1] - self._b * state[2]
        ))

class Lorenz96:
    """Lorenz '96 model."""

    def __init__(self, forcing=8, size=40):
        """Set up instance with parameters.

        Parameters
        ----------
        forcing: float
        size: int
        """
        self._forcing = forcing
        self._size = size

    def __call__(self, state):
        """The derivative at `state`

        Parameters
        ----------
        state: np.ndarray[`size`]

        Returns
        -------
        deriv: np.ndarray[`size`]
        """
        xkm1 = np.roll(state, 1)
        xkm2 = np.roll(state, 2)
        xkp1 = np.roll(state, -1)
        res = xkm1 * (xkp1 - xkm2) - state + self._forcing

        return res
