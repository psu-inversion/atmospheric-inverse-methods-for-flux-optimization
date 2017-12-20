"""Models for identical twin experiments and similar."""
from __future__ import division

from numpy import empty_like, array


class Lorenz63:
    """Classic Lorenz '63 model."""

    def __init__(self, sigma=10, r=28, b=8/3):  # noqa: E226
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
        """Get the derivative at state.

        Parameters
        ----------
        state: array_like[3]

        Returns
        -------
        deriv: array_like[3]
        """
        return array((
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
        """Get the derivative at `state`.

        Parameters
        ----------
        state: np.ndarray[`size`]

        Returns
        -------
        deriv: np.ndarray[`size`]
        """
        res = empty_like(state)
        res[2:-1] = state[1:-2] * (state[3:] - state[:-3])

        # fill in -1, 0, 1
        for i in range(-1, 2):
            res[i] = state[i - 1] * (state[i + 1] - state[i - 2])
        res -= state
        res += self._forcing

        return res


class ArgsYTWrapper:
    """Wrap an instance to accept args y, t."""

    def __init__(self, model):
        """Wrap model to take y, t as args.

        Parameters
        ----------
        model: callable
        """
        self._model = model

    def __call__(self, y, t):
        """Call the model with arg `y`.

        Parameters
        ----------
        y: np.ndarray
        t: float
        """
        return self._model(y)
