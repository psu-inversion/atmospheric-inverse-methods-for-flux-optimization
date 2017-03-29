#!/usr/bin/env python
r"""Run a flux inversion with a simple H and B.

Welcome to the problem that combines all the best features of GARCH,
VAR, SARIMA, and HMM models, all in one place. Bad estimates of the
HMM observation operator! Bad initial estimates of the VAR noise
process covariance matrix! Seasonal variations in variance from the
annual cycle! Systematic problems in the prior estimates! And much,
much more!

For the fraternal twin setup, we use the different covariance matrices
in the noise generation and the inversion.
"""

import os.path
import sys

import numpy as np
import numpy.linalg as la
import scipy.linalg

import iris.cube
import iris.util
import iris.coords
import iris.analysis.maths

try:
    sys.path.append(os.path.join(os.path.dirname(__file__),
                                 "..", "src"))
except NameError:
    sys.path.append(os.path.join(os.getcwd(), "..", "src"))

import inversion.optimal_interpolation
import inversion.correlations
import inversion.noise

isqrt = iris.analysis.maths.IFunc(
    np.sqrt, lambda cube: cube.units.root(2))

NX = 45
NY = 35
N_FLUX_TIMES = 1

N_TIMES_BACK = 1
N_SITES = 6

N_GRID_POINTS = NX * NY
N_OBS_TIMES = N_FLUX_TIMES - N_TIMES_BACK + 1

N_RUNS = 200

TRUE_CORR_LEN = 2
ASSUMED_CORR_LEN = 5
TRUE_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.Gaussian2DCorrelation(TRUE_CORR_LEN))
ASSUMED_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.Gaussian2DCorrelation(ASSUMED_CORR_LEN))
TM_ERROR_CORRELATION_FUN = (
    inversion.correlations.Exponential1DCorrelation(2))
STDS = np.ones((N_FLUX_TIMES, NY, NX))

COORD_ADJOINT_STR = "adjoint_"
CUBE_ADJOINT_STR = "adjoint_of_"


def adjointize_coord(coord):
    """Make an adjoint of coord.

    Adds or removes "adjoint_" to the start of `coord.long_name`,
    borrowing from :meth:`coord.name` if necessary.

    Parameters
    ----------
    coord: iris.coords.Coord

    Returns
    iris.coords.Coord

    """
    result = coord.copy()
    base_name = coord.long_name
    if base_name is None:
        base_name = coord.name()

    if base_name.startswith(COORD_ADJOINT_STR):
        result.long_name = base_name[len(COORD_ADJOINT_STR):]
    else:
        result.long_name = COORD_ADJOINT_STR + base_name

    return result


def adjointize_cube(cube):
    """Adjoint of the cube.

    :module:`iris` takes care of aligning axes, so it only really runs
    :func:`adjointize_coord` on all the coordinates.

    Parameters
    ----------
    cube: iris.cube.Cube

    Returns
    -------
    iris.cube.Cube

    """
    result = cube.copy()

    for coord in result.dim_coords:
        new_coord = adjointize_coord(coord)
        result.add_dim_coord(new_coord, cube.coord_dims(coord))
        result.remove_coord(coord)

    for coord in result.aux_coords:
        new_coord = adjointize_coord(coord)
        result.add_dim_coord(new_coord, cube.coord_dims(coord))
        result.remove_coord(coord)

    base_name = cube.long_name

    if base_name is None:
        base_name = cube.name()

    if base_name.startswith(CUBE_ADJOINT_STR):
        result.long_name = base_name[len(CUBE_ADJOINT_STR):]
    else:
        result.long_name = COORD_ADJOINT_STR + base_name

    return result


DX = iris.cube.Cube(1, units="km")
X_COORD = iris.coords.DimCoord(np.arange(NX), units="km",
                               standard_name="projection_x_coordinate")
Y_COORD = iris.coords.DimCoord(np.arange(NY), units="km",
                               standard_name="projection_y_coordinate")
DT = iris.cube.Cube(1, units="days")
FLUX_TIME_COORD = iris.coords.DimCoord(
    np.arange(N_FLUX_TIMES),
    units="days since 2010-01-01 00:00:00+0000",
    standard_name="time",
    long_name="flux_time")
OBS_TIME_COORD = iris.coords.DimCoord(
    FLUX_TIME_COORD.points[-N_OBS_TIMES:],
    units=FLUX_TIME_COORD.units,
    standard_name="forecast_reference_time",
    long_name="observation_time")
TIME_BACK_COORD = iris.coords.DimCoord(
    np.arange(N_TIMES_BACK),
    units="days",
    long_name="time_before_observation")

#: thickness of the bottom layer in Pa
THICKNESS = iris.cube.Cube(720., units="Pa")
# source: wolfram alpha
CO2_AIR_MASS_RATIO = iris.cube.Cube(
    (.78084 * 2 * 14.0067 +
     .20948 * 2 * 15.9994 +
     # # this bit varys a lot.
     # ~0% in deserts to ~5%? in tropics
     # .01 * (2 * 1.00794 + 15.9994) +
     .00934 * 39.948 +
     .000380 * (12.0107 + 2*15.9994)) / (12.0107 + 2*15.9994),
    units="(g/mol)/(g/mol)")
# I think this is also from wolfram alpha
EARTH_GRAVITY = iris.cube.Cube(9.807, units="m/s^2")
GRAMS_TO_PPM = CO2_AIR_MASS_RATIO / (THICKNESS/EARTH_GRAVITY)
"""Conversion from flux units to mixing ratio units

Assumes fluxes are in g/m^2/hr;
I think this is independent of the actual area units,
but I don't know
Converts to flux tendency in ppmv/hr

Notes
-----
.. math::

    F/M_{CO2}/dz = \\Delta n_{CO2} \\\\
    dz = -dP/\\rho g \\\\
    \\Delta X_{CO2} = \\Delta n_{CO2}/n_{air}
                    = \\Delta n_{CO2}/ (\\rho_{air} / M_{air}) \\\\
    \\Delta X_{CO2} = F/(M_{CO2} * -dP/(\\rho_{air} g) * M_{air}/\\rho_{air} \\\\
    \\Delta X_{CO2} = F/M_{CO2} / (-dP/g) * M_{air}

Need to convert F to kg if dP uses Pa
X_{CO2} is here in units of 1; multiply by 1e6 to get ppmv
"""


# I'm pretty sure this is the proper memory order for products to work
# Unfortunately, this conflicts with the intuition I get from the
# $E(\ce \ce^T)$ definition
TRUE_SP_ERROR_CORRELATION = TRUE_SP_ERROR_CORRELATION_FUN.make_matrix(NY, NX)
ASSUMED_SP_ERROR_CORRELATION = (
    ASSUMED_SP_ERROR_CORRELATION_FUN.make_matrix(NY, NX))
TM_ERROR_CORRELATION = TM_ERROR_CORRELATION_FUN.make_matrix(N_FLUX_TIMES)
TRUE_ERROR_CORRELATION = scipy.linalg.kron(TM_ERROR_CORRELATION,
                                           TRUE_SP_ERROR_CORRELATION)
ASSUMED_ERROR_CORRELATION = scipy.linalg.kron(TM_ERROR_CORRELATION,
                                              ASSUMED_SP_ERROR_CORRELATION)

# Do this part before or after the kronecker product?
STDS = np.ones(N_FLUX_TIMES * N_GRID_POINTS)
DIAG_STDS = np.diag(STDS)
TRUE_ERROR_COVARIANCE = DIAG_STDS.dot(TRUE_ERROR_CORRELATION.dot(DIAG_STDS))
ASSUMED_ERROR_COVARIANCE = DIAG_STDS.dot(
    ASSUMED_ERROR_CORRELATION.dot(DIAG_STDS))
print(TRUE_ERROR_COVARIANCE.shape)
OBS_STDS = np.ones((N_OBS_TIMES, N_SITES))
OBSERVATION_COVARIANCE = np.diag(np.square(OBS_STDS.reshape(-1)))

TRUE_ERR_COV_CUBE = iris.cube.Cube(
    TRUE_ERROR_COVARIANCE.reshape(N_FLUX_TIMES, NY, NX,
                                  N_FLUX_TIMES, NY, NX),
    long_name="error_covariances",
    units="(g/km^2/hr)^2",
    dim_coords_and_dims=(
        (FLUX_TIME_COORD, 0),
        (Y_COORD, 1),
        (X_COORD, 2),
        (adjointize_coord(FLUX_TIME_COORD), 3),
        (adjointize_coord(Y_COORD), 4),
        (adjointize_coord(X_COORD), 5),
    ),
)
ASSUMED_ERR_COV_CUBE = iris.cube.Cube(
    ASSUMED_ERROR_COVARIANCE.reshape(N_FLUX_TIMES, NY, NX,
                                     N_FLUX_TIMES, NY, NX),
    long_name="error_covariances",
    units="(g/km^2/hr)^2",
    dim_coords_and_dims=(
        (FLUX_TIME_COORD, 0),
        (Y_COORD, 1),
        (X_COORD, 2),
        (adjointize_coord(FLUX_TIME_COORD), 3),
        (adjointize_coord(Y_COORD), 4),
        (adjointize_coord(X_COORD), 5),
    ),
)
print(TRUE_ERR_COV_CUBE)

# This one goes forward in time
OBSERVATION_OPERATOR = iris.cube.Cube(
    np.zeros((N_OBS_TIMES, N_SITES, N_TIMES_BACK, NY, NX)),
    long_name="influence_function",
    units="ppmv/(g/km^2/hr)",
    dim_coords_and_dims=(
        (OBS_TIME_COORD, 0),
        (TIME_BACK_COORD, 2),
        (Y_COORD, 3),
        (X_COORD, 4),
    ),
    aux_coords_and_dims=(
        (iris.coords.AuxCoord(
            np.random.uniform(X_COORD.points[0], X_COORD.points[-1], N_SITES),
            units=X_COORD.units,
            standard_name="projection_x_coordinate",
            long_name="tower_x"), (1,)),
        (iris.coords.AuxCoord(
            np.random.uniform(Y_COORD.points[0], Y_COORD.points[-1], N_SITES),
            units=Y_COORD.units,
            standard_name="projection_y_coordinate",
            long_name="tower_y"), (1,)),
        (iris.coords.AuxCoord(
            scipy.linalg.hankel(FLUX_TIME_COORD[:N_OBS_TIMES].points,
                                FLUX_TIME_COORD[-N_TIMES_BACK:].points),
            units=FLUX_TIME_COORD.units,
            standard_name=FLUX_TIME_COORD.standard_name,
            long_name=FLUX_TIME_COORD.long_name), (0, 2)),
    )
)

# The next block fills this with gaussian puffs
# This is the advection velocity and spread rate.
WIND = iris.cube.Cube(5, units="m/s")
DIFFUSIVITY = iris.cube.Cube(1e5, units="km^2/day")


def coord_to_cube(coord):
    """Turn the coord into a cube.

    Parameters
    ----------
    coord: iris.coord.Coord

    Returns
    -------
    iris.cube.Cube
    """
    result = iris.cube.Cube(
        coord.points,
        standard_name=coord.standard_name,
        long_name=coord.long_name,
        var_name=coord.var_name,
        units=coord.units,
        attributes=coord.attributes,
        aux_coords_and_dims=(
            (coord, range(len(coord.shape))),
        ),
    )
    if len(coord.shape) == 1:
        iris.util.promote_aux_coord_to_dim_coord(
            result, coord.name())
    return result


# def pr(thing):
#     print(thing)
#     if isinstance(thing, iris.cube.Cube):
#         print(thing.data)
#     return thing

tower_x = OBSERVATION_OPERATOR.coord(long_name="tower_x")
tower_y = OBSERVATION_OPERATOR.coord(long_name="tower_y")
# Filling obs_op
for site in range(N_SITES):
    site_x = coord_to_cube(tower_x[site])
    site_y = coord_to_cube(tower_y[site])

    site_x.coord("projection_x_coordinate").long_name = None
    site_y.coord("projection_y_coordinate").long_name = None

    y_dist_part = (coord_to_cube(Y_COORD).data - site_y.data) ** 2

    for curr_time in range(N_TIMES_BACK):
        advection_time = coord_to_cube(TIME_BACK_COORD[curr_time]) + 1
        # indexing apparently doesn't preserve name.
        # Code says it does.
        x_shift = advection_time * WIND
        x_shift.convert_units(site_x.units)

        x_dist_part = (coord_to_cube(X_COORD).data -
                       (site_x.data - x_shift.data)) ** 2

        # The sign is important
        # Second time I've forgotten this
        exponent = -1 * iris.cube.Cube(
            y_dist_part[np.newaxis, :, np.newaxis] +
            x_dist_part[np.newaxis, np.newaxis, :],
            units=Y_COORD.units**2,
            dim_coords_and_dims=(
                (advection_time.coord("time_before_observation"), 0),
                (Y_COORD, 1),
                (X_COORD, 2),
            ),
        ) / (4 * advection_time * DIFFUSIVITY)
        exponent.convert_units("1")
        infl_fun = (
            .5 * iris.analysis.maths.exp(exponent) *
            isqrt(np.pi * advection_time * DIFFUSIVITY) ** -1 *
            # Use proper flux units
            GRAMS_TO_PPM *
            # integrate over streamwise space and time
            DX * DT
        )
        infl_fun.convert_units(OBSERVATION_OPERATOR.units)
        OBSERVATION_OPERATOR.data[:, site, curr_time, :, :] = infl_fun.data
        del infl_fun, x_dist_part, x_shift, exponent
    del y_dist_part
del site_x, site_y, site
del curr_time, advection_time


if __name__ == "__main__":
    truth_shaped = iris.cube.Cube(
        np.zeros((N_FLUX_TIMES, NY, NX)),
        long_name="true_fluxes",
        units="g/km^2/hr",
        dim_coords_and_dims=(
            (FLUX_TIME_COORD, 0),
            (Y_COORD, 1),
            (X_COORD, 2),
        ),
    )
    truth_vec = truth_shaped.data.reshape(-1)

    # This will need to be refined for N_TIMES_BACK != N_FLUX_TIMES
    true_obs_shaped = (OBSERVATION_OPERATOR * truth_shaped).collapsed(
        (X_COORD, Y_COORD, FLUX_TIME_COORD),
        iris.analysis.SUM)
    true_obs_vec = true_obs_shaped.data.reshape(-1)

    chi2s = []
    innovations = iris.cube.Cube(
        np.empty((N_RUNS, N_OBS_TIMES, N_SITES)),
        long_name="innovations",
        dim_coords_and_dims=(
            (OBS_TIME_COORD, 1),
        )
    )
    increments = iris.cube.Cube(
        np.empty((N_RUNS, N_TIMES_BACK, NY, NX)),
        long_name="increments",
        dim_coords_and_dims=(
            (TIME_BACK_COORD, 1),
            (Y_COORD, 2),
            (X_COORD, 3),
        )
    )

    for i in range(N_RUNS):
        noise = inversion.noise.gaussian_noise(TRUE_ERROR_COVARIANCE)

        prior_shaped = truth_shaped + noise.reshape(truth_shaped.shape)
        prior_vec = prior_shaped.data.reshape(-1)
        prior_cov = ASSUMED_ERROR_COVARIANCE[:N_TIMES_BACK * N_GRID_POINTS,
                                             :N_TIMES_BACK * N_GRID_POINTS]

        obs_noise = inversion.noise.gaussian_noise(OBSERVATION_COVARIANCE)
        observations = true_obs_vec + obs_noise
        observations_shaped = true_obs_shaped.copy()
        observations_shaped.data = observations.reshape(true_obs_shaped.shape)
        observations_shaped.long_name = "pseudo-observations"

        # For N_TIMES_BACK != N_FLUX_TIMES, need to figure out how to loop.
        # With OI, iterating mean and covariance should work,
        # 3D-Var will work with relaxation to prior errors
        # PSAS: only iterate mean. Keep static covariances and evaluate after.
        # Will need to get :math:`M/M^T` figured out.
        # Note: Will need to store part of posterior as we go
        obs_op = OBSERVATION_OPERATOR.data.reshape(
            (N_OBS_TIMES * N_SITES, N_TIMES_BACK * N_GRID_POINTS))
        posterior, posterior_cov = inversion.optimal_interpolation.scipy_chol(
            prior_vec, prior_cov, observations, OBSERVATION_COVARIANCE,
            obs_op)

        prior_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                                prior_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)

        post_shaped = iris.cube.Cube(
            posterior.reshape((N_TIMES_BACK, NY, NX)),
            long_name="posterior_flux",
            units=prior_shaped.units,
            dim_coords_and_dims=(
                (FLUX_TIME_COORD, 0),
                (Y_COORD, 1),
                (X_COORD, 2),
            ),
        )

        post_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                               post_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)

        error_proj = obs_op.dot(prior_cov.dot(obs_op.T))
        total_err_cov = error_proj + OBSERVATION_COVARIANCE

        chisq = prior_mismatch.data.dot(
            la.solve(total_err_cov, prior_mismatch.data))
        df_expected = np.prod(prior_mismatch.shape)
        # print("Chi squared statistic:", chisq)
        # print("Expected value:       ", df_expected)

        # print("Chi squared reduced:", chisq / df_expected)
        chi2s.append(chisq)
        innovations.data[i, :, :] = prior_mismatch.data[np.newaxis, :]
        increments.data[i] = (post_shaped - prior_shaped).data

    # print("To increase this statistic, decrease the flux variances\n"
    #       "To decrease this statistic, increase the flux variances\n"
    #       "If this is not close to one for this perfect-model setup,\n"
    #       "we have big problems.")
    iris.save([post_shaped, innovations, increments],
              "fraternal_gaussian_actual_{true:d}_assumed_{assumed:d}.nc".format(
                  true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN),
              zlib=True)
