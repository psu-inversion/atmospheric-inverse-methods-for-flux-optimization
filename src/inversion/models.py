"""Models for identical twin experiments and similar."""
# TODO: Test
from __future__ import division

from numpy import empty_like, array, atleast_1d, newaxis
from scipy.spatial import distance_matrix
try:
    from bottleneck import nansum
except ImportError:
    from numpy import nansum


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

    def __init__(self, forcing=8., size=40):
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
        state: array_like[`size`]

        Returns
        -------
        deriv: array_like[`size`]
        """
        res = empty_like(state)
        res[2:-1] = state[1:-2] * (state[3:] - state[:-3])

        # fill in -1, 0, 1
        for i in range(-1, 2):
            res[i] = state[i - 1] * (state[i + 1] - state[i - 2])
        res -= state
        res += self._forcing

        return res


class PointVortex:
    """Point-vortex model."""

    def __init__(self, strengths):
        """Set up an instance for n-vortex problem.

        Parameters
        ----------
        strengths: array_like[nvortices]
            The strength of each vortex
        """
        self._strengths = atleast_1d(strengths)
        self._nvortices = len(self._strengths)

    def __call__(self, state):
        """Describe the velocities of the vortices.

        Parameters
        ----------
        state: array_like[nvortices, 3]
            Rows are (x, y, strength)

        Returns
        -------
        velocities: array_like[nvortices, 2]
            Rows are (xvel, yvel)
        """
        distances = distance_matrix(state, state)
        displacements = state[:, newaxis, :] - state[newaxis, :, :]
        vel_components = (
            self._strengths[newaxis, :, newaxis] /
            distances[:, :, newaxis]**2 *
            displacements[:, :, ::-1])
        vortex_vel = nansum(vel_components, axis=1)
        vortex_vel[:, 0] *= -1
        return vortex_vel


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
