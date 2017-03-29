#!/usr/bin/env python
r"""Rerun flux_osse_all_systems.py inversions with a different correlation function.

Get observations and prior from netCDF file saved at the end of that script.
"""

import os.path
import sys

import numpy as np
import numpy.linalg as la
import scipy.linalg
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import iris.cube
import iris.util
import iris.coords
import iris.analysis.maths
import iris.plot as iplt
import iris.quickplot as qplt

try:
    sys.path.append(os.path.join(os.path.dirname(__file__),
                                 "..", "src"))
except NameError:
    sys.path.append(os.path.join(os.getcwd(), "..", "src"))

import inversion.optimal_interpolation
import inversion.correlations
import inversion.noise
import inversion.tests

isqrt = iris.analysis.maths.IFunc(
    np.sqrt, lambda cube: cube.units.root(2))

lst = iris.load("crosscheck_exponential_actual_5_assumed_5.nc")

observations = lst.extract("pseudo-observations")[0]
OBSERVATION_OPERATOR = lst.extract("influence_function")[0]
FLUX_PRIOR = lst.extract("Prior flux")[0]

N_OBS_TIMES, N_SITES, N_TIMES_BACK, NY, NX = OBSERVATION_OPERATOR.shape

N_GRID_POINTS = NY * NX
N_FLUX_TIMES = N_OBS_TIMES + N_TIMES_BACK - 1

TRUE_CORR_LEN = 5
ASSUMED_CORR_LEN = 10

ASSUMED_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.Exponential2DCorrelation(ASSUMED_CORR_LEN))
TM_ERROR_CORRELATION_FUN = (
    inversion.correlations.Exponential1DCorrelation(2))
STDS = np.ones((N_FLUX_TIMES, NY, NX))

COORD_ADJOINT_STR = "adjoint_"
CUBE_ADJOINT_STR = "adjoint_of_"

INVERSION_FUNCTIONS = inversion.tests.ALL_METHODS
N_FUNCTIONS = len(INVERSION_FUNCTIONS)
FUNCTION_COORD = iris.coords.AuxCoord(
    [inversion.tests.getname(func)
     for func in INVERSION_FUNCTIONS],
    long_name="inversion_function_name")

DIVERGING_CMAP = plt.get_cmap("RdBu_r")

observations.convert_units("ppmv")
for cube in (OBSERVATION_OPERATOR, observations, FLUX_PRIOR):
    for coord in cube.coords():
        coord.var_name = None


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

X_COORD = iris.coords.DimCoord(np.arange(0, NX, 1), units="km",
                               standard_name="projection_x_coordinate")
Y_COORD = iris.coords.DimCoord(np.arange(0, NY), units="km",
                               standard_name="projection_y_coordinate")
DT = iris.cube.Cube(1, units="days")
FLUX_TIME_COORD = FLUX_PRIOR.coord("time")
OBS_TIME_COORD = OBSERVATION_OPERATOR.coord("forecast_reference_time")
TIME_BACK_COORD = OBSERVATION_OPERATOR.coord("time_before_observation")
TOWER_COORD = OBSERVATION_OPERATOR.coord("tower_number")

OBSERVATION_OPERATOR.add_dim_coord(Y_COORD, 3)
OBSERVATION_OPERATOR.add_dim_coord(X_COORD, 4)
FLUX_PRIOR.add_dim_coord(Y_COORD, 1)
FLUX_PRIOR.add_dim_coord(X_COORD, 2)

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


ASSUMED_SP_ERROR_CORRELATION = (
    ASSUMED_SP_ERROR_CORRELATION_FUN.make_matrix(NY, NX))
TM_ERROR_CORRELATION = TM_ERROR_CORRELATION_FUN.make_matrix(N_FLUX_TIMES)
ASSUMED_ERROR_CORRELATION = scipy.linalg.kron(TM_ERROR_CORRELATION,
                                              ASSUMED_SP_ERROR_CORRELATION)

# Do this part before or after the kronecker product?
STDS = np.ones(N_FLUX_TIMES * N_GRID_POINTS)
DIAG_STDS = np.diag(STDS)
ASSUMED_ERROR_COVARIANCE = DIAG_STDS.dot(
    ASSUMED_ERROR_CORRELATION.dot(DIAG_STDS))

OBS_STDS = np.ones((N_OBS_TIMES, N_SITES)) / 5
OBSERVATION_COVARIANCE = np.diag(np.square(OBS_STDS.reshape(-1)))

# ASSUMED_ERR_COV_CUBE = iris.cube.Cube(
#     ASSUMED_ERROR_COVARIANCE.reshape(N_FLUX_TIMES, NY, NX,
#                                      N_FLUX_TIMES, NY, NX),
#     long_name="error_covariances",
#     units="(g/km^2/hr)^2",
#     dim_coords_and_dims=(
#         (FLUX_TIME_COORD, 0),
#         (Y_COORD, 1),
#         (X_COORD, 2),
#         (adjointize_coord(FLUX_TIME_COORD), 3),
#         (adjointize_coord(Y_COORD), 4),
#         (adjointize_coord(X_COORD), 5),
#     ),
# )

# The next block fills this with gaussian puffs
# This is the advection velocity and spread rate.
WIND = iris.cube.Cube(.1, units="m/s")
DIFFUSIVITY = iris.cube.Cube(3e2, units="km^2/day")


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
    fig = plt.figure()
    qplt.contourf(truth_shaped[0])
    fig.savefig("crosscheck_truth.png")
    plt.close(fig)

    chi2s = iris.cube.Cube(
        np.empty((N_FUNCTIONS)),
        long_name="chi_squared_statistics",
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    flux_totals = iris.cube.Cube(
        np.empty(N_FUNCTIONS),
        long_name="flux_posterior_totals",
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    innovations = iris.cube.Cube(
        np.empty((N_FUNCTIONS, N_OBS_TIMES, N_SITES)),
        long_name="innovations",
        dim_coords_and_dims=(
            (OBS_TIME_COORD, 1),
        ),
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )
    increments = iris.cube.Cube(
        np.empty((N_FUNCTIONS, N_TIMES_BACK, NY, NX)),
        long_name="increments",
        dim_coords_and_dims=(
            (TIME_BACK_COORD, 1),
            (Y_COORD, 2),
            (X_COORD, 3),
        ),
        aux_coords_and_dims=(
            (FUNCTION_COORD, 0),
        ),
    )

    prior_shaped = FLUX_PRIOR
    prior_vec = prior_shaped.data.reshape(-1)
    prior_cov = ASSUMED_ERROR_COVARIANCE[:N_TIMES_BACK * N_GRID_POINTS,
                                         :N_TIMES_BACK * N_GRID_POINTS]

    prior_tot = prior_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                       iris.analysis.SUM)

    print("Prior total")
    print(prior_tot.data)

    observations_shaped = observations
    observations = observations = observations_shaped.data.reshape(-1)

    # Set figure size where qplt plots look good.
    # TODO borrow code from WRF/LPDM validation to generalize this
    prior_obs = (OBSERVATION_OPERATOR[0] * prior_shaped).collapsed(
        (X_COORD, Y_COORD, TIME_BACK_COORD),
        iris.analysis.SUM)
    prior_obs.convert_units(observations_shaped.units)
    prior_mismatch = observations_shaped - prior_obs
    #print(prior_mismatch)
    print(prior_mismatch.data)

    mpl.rcParams["figure.figsize"] = (6, 5)

    for i, inversion_func in enumerate(INVERSION_FUNCTIONS):
        # For N_TIMES_BACK != N_FLUX_TIMES, need to figure out how to loop.
        # With OI, iterating mean and covariance should work,
        # 3D-Var will work with relaxation to prior errors
        # PSAS: only iterate mean. Keep static covariances and evaluate after.
        # Will need to get :math:`M/M^T` figured out.
        # Note: Will need to store part of posterior as we go
        obs_op = OBSERVATION_OPERATOR.data.reshape(
            (N_OBS_TIMES * N_SITES, N_TIMES_BACK * N_GRID_POINTS))

        try:
            posterior, posterior_cov = inversion_func(
                prior_vec, prior_cov, observations, OBSERVATION_COVARIANCE,
                obs_op)
        except inversion.ConvergenceError as err:
            print("Convergence Not Achieved:", inversion_func)
            posterior = err.guess
            posterior_cov = err.hess_inv

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
        fig = plt.figure()
        qplt.contourf(post_shaped[0], cmap=DIVERGING_CMAP)
        plt.tight_layout()
        fig.savefig("crosscheck_{true:d}_{assumed:d}_posterior_{name:s}.png".format(
            name=FUNCTION_COORD.points[i].replace(" ", "_").replace("(", "").replace(")", ""),
            true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN))
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False,
                                 gridspec_kw=dict(wspace=.1, hspace=.1),
                                 figsize=(12, 5), )
        contours = axes[0, 0].contourf(
            X_COORD.points, Y_COORD.points,
            np.diag(posterior_cov).reshape(
                N_TIMES_BACK, NY, NX)[0],
            vmin=.5, vmax=1)
        axes[0, 0].set_title("Posterior Variances")
        fig.colorbar(contours, ax=axes[0, 0], orientation="horizontal", pad=.1)
        fig.suptitle("Results for method {name:s}".format(
            name=FUNCTION_COORD.points[i]))

        # TODO borrow code from WRF/LPDM validation to generalize this
        post_obs = (OBSERVATION_OPERATOR[0] * post_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)
        post_obs.convert_units(observations_shaped.units)
        post_mismatch = observations_shaped - post_obs

        gain = (1 -
                (iris.analysis.maths.abs(prior_shaped) /
                 iris.analysis.maths.abs(post_shaped)).data)

        # plt.sca(axes[0, 1])
        contours = axes[0, 1].contourf(
            X_COORD.points, Y_COORD.points,
            gain[0], np.linspace(-1, 1, 9), vmin=-1, vmax=1,
            extend="both",
            norm=mpl.colors.Normalize(-1, 1))
        axes[0, 1].set_title("Gain")
        fig.colorbar(contours, ax=axes[0, 1], orientation="horizontal", pad=.1)
        fig.savefig(
            "crosscheck_{true:d}_{assumed:d}_posterior_var_gain_{name:s}.png".
            format(
                name=FUNCTION_COORD.points[i].replace(
                    " ", "_").replace("(", "").replace(")", ""),
                true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN))
        plt.close(fig)

        error_proj = obs_op.dot(prior_cov.dot(obs_op.T))
        total_err_cov = error_proj + OBSERVATION_COVARIANCE

        chisq = prior_mismatch.data.dot(
            la.solve(total_err_cov, prior_mismatch.data))
        chi2s.data[i] = chisq
        innovations.data[i, :, :] = prior_mismatch.data[np.newaxis, :]
        # For some reason having `coord1 is coord2` isn't similar
        # enough for compatibility.
        increments.data[i] = (post_shaped.data - prior_shaped.data)
        flux_tot = post_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                         iris.analysis.SUM)
        flux_totals.data[i] = flux_tot.data
        print(FUNCTION_COORD.points[i])
        # print(flux_tot)
        print(flux_tot.data)
    iris.save([innovations, increments, chi2s, flux_totals,
               truth_shaped, OBSERVATION_OPERATOR, prior_shaped,
               observations_shaped],
              "crosscheck_exponential_actual_{true:d}_assumed_{assumed:d}.nc".
              format(
                  true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN),
              zlib=True)
