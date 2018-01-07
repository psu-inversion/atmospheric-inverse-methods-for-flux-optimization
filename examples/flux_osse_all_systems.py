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
import inversion.covariances
import inversion.noise
import inversion.tests

isqrt = iris.analysis.maths.IFunc(
    np.sqrt, lambda cube: cube.units.root(2))

NX = 60
NY = 40
N_FLUX_TIMES = 24 * 7

N_TIMES_BACK = 24 * 5
N_SITES = 4

N_GRID_POINTS = NX * NY
N_OBS_TIMES = N_FLUX_TIMES - N_TIMES_BACK + 1

TRUE_CORR_LEN = 5
ASSUMED_CORR_LEN = 5
TRUE_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(TRUE_CORR_LEN))
ASSUMED_SP_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(ASSUMED_CORR_LEN))
TM_ERROR_CORRELATION_FUN = (
    inversion.correlations.ExponentialCorrelation(2))
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
X_COORD = iris.coords.DimCoord(np.arange(0, NX, 1), units="km",
                               standard_name="projection_x_coordinate")
Y_COORD = iris.coords.DimCoord(np.arange(0, NY), units="km",
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
TOWER_COORD = iris.coords.DimCoord(
    np.arange(N_SITES),
    long_name="tower_number",
)

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
TRUE_SP_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        TRUE_SP_ERROR_CORRELATION_FUN, (NY, NX)))
ASSUMED_SP_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        ASSUMED_SP_ERROR_CORRELATION_FUN, (NY, NX)))
TM_ERROR_CORRELATION = (
    inversion.correlations.HomogeneousIsotropicCorrelation.from_function(
        TM_ERROR_CORRELATION_FUN, (N_FLUX_TIMES)))
TRUE_ERROR_CORRELATION = inversion.util.kronecker_product(
    TM_ERROR_CORRELATION, TRUE_SP_ERROR_CORRELATION)
ASSUMED_ERROR_CORRELATION = inversion.util.kronecker_product(
    TM_ERROR_CORRELATION, ASSUMED_SP_ERROR_CORRELATION)

# Do this part before or after the kronecker product?
STDS = np.ones(N_FLUX_TIMES * N_GRID_POINTS)
DIAG_STDS = inversion.covariances.DiagonalOperator(STDS)
TRUE_ERROR_COVARIANCE = DIAG_STDS.dot(TRUE_ERROR_CORRELATION.dot(DIAG_STDS))
ASSUMED_ERROR_COVARIANCE = DIAG_STDS.dot(
    ASSUMED_ERROR_CORRELATION.dot(DIAG_STDS))
print(TRUE_ERROR_COVARIANCE.shape)
OBS_STDS = np.ones((N_OBS_TIMES, N_SITES)) / 5
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
INFLUENCE_FUNCTION = iris.load_cube("/mc1s2/s4/dfw5129/data/LPEMLPDM_2010_01_03hrly_footprints_collapsed.nc4")
OBSERVATION_OPERATOR = iris.cube.Cube(
    INFLUENCE_FUNCTION[:N_OBS_TIMES, :N_SITES, :N_TIMES_BACK].lazy_data(),
    long_name="influence_function",
    units="ppmv/(g/km^2/hr)",
    dim_coords_and_dims=(
        (OBS_TIME_COORD, 0),
        (TOWER_COORD, 1),
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


tower_x = OBSERVATION_OPERATOR.coord(long_name="tower_x")
tower_y = OBSERVATION_OPERATOR.coord(long_name="tower_y")

fig = plt.figure()
# Currently goes 0 to .05
qplt.contourf(OBSERVATION_OPERATOR.collapsed(
    (OBS_TIME_COORD, TOWER_COORD, TIME_BACK_COORD),
    iris.analysis.SUM),
              vmin=0, vmax=.06)
iplt.plot(tower_x, tower_y, "*")
fig.savefig("crosscheck_obs_op.png")
plt.close(fig)
print(TRUE_CORR_LEN, ASSUMED_CORR_LEN)

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

    # This will need to be refined for N_TIMES_BACK != N_FLUX_TIMES
    true_obs_shaped = (OBSERVATION_OPERATOR * truth_shaped).collapsed(
        (X_COORD, Y_COORD, FLUX_TIME_COORD),
        iris.analysis.SUM)
    true_obs_vec = true_obs_shaped.data.reshape(-1)

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

    noise = inversion.noise.gaussian_noise(TRUE_ERROR_COVARIANCE)

    prior_shaped = truth_shaped + noise.reshape(truth_shaped.shape)
    prior_shaped.long_name = "Prior flux"
    prior_vec = prior_shaped.data.reshape(-1)
    prior_cov = ASSUMED_ERROR_COVARIANCE[:N_TIMES_BACK * N_GRID_POINTS,
                                         :N_TIMES_BACK * N_GRID_POINTS]
    fig = plt.figure()
    qplt.contourf(prior_shaped[0], cmap=DIVERGING_CMAP)
    fig.savefig("crosscheck_prior.png")
    plt.close(fig)
    prior_tot = prior_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                       iris.analysis.SUM)
    print("Prior total:")
    print(prior_tot)
    print(prior_tot.data)


    obs_noise = inversion.noise.gaussian_noise(OBSERVATION_COVARIANCE)
    observations = true_obs_vec + obs_noise
    observations_shaped = true_obs_shaped.copy()
    observations_shaped.data = observations.reshape(true_obs_shaped.shape)
    observations_shaped.long_name = "pseudo-observations"

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

        prior_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                                prior_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)
        #print(prior_mismatch)
        print(prior_mismatch.data)

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
        fig.savefig("crosscheck_posterior_{name:s}.png".format(
            name=FUNCTION_COORD.points[i].replace(" ", "_").replace("(", "").replace(")", "")))
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False,
                                 gridspec_kw=dict(wspace=.1, hspace=.1),
                                 figsize=(8, 4), )
        contours = axes[0, 0].contourf(
            X_COORD.points, Y_COORD.points,
            np.diag(posterior_cov).reshape(
                N_TIMES_BACK, NY, NX)[0],
            vmin=.5, vmax=1)
        axes[0, 0].set_title("Posterior Variances")
        fig.colorbar(contours, ax=axes[0, 0], orientation="horizontal", pad=.1)
        fig.suptitle("Results for method {name:s}".format(
            name=FUNCTION_COORD.points[i]))

        post_mismatch = observations_shaped - (OBSERVATION_OPERATOR *
                                               post_shaped).collapsed(
            (X_COORD, Y_COORD, FLUX_TIME_COORD),
            iris.analysis.SUM)

        gain = (1 -
                (iris.analysis.maths.abs(prior_shaped) /
                 iris.analysis.maths.abs(post_shaped)).data)
        plt.sca(axes[0, 1])
        contours = plt.contourf(
            X_COORD.points, Y_COORD.points,
            gain[0], np.linspace(-1, 1, 7), vmin=-1, vmax=1,
            extend="both",
            norm=mpl.colors.Normalize(-1, 1, clip=True))
        axes[0, 1].set_title("Gain")
        fig.colorbar(contours, ax=axes[0, 1], orientation="horizontal", pad=.1)
        fig.savefig("crosscheck_posterior_var_gain_{name:s}.png".format(
            name=FUNCTION_COORD.points[i].replace(" ", "_").replace("(", "").replace(")", "")))
        plt.close(fig)
        # error_reduction = (1 -
        #                    np.sqrt(np.diag(prior_cov) / np.diag(posterior_cov)))
        

        error_proj = obs_op.dot(prior_cov.dot(obs_op.T))
        total_err_cov = error_proj + OBSERVATION_COVARIANCE

        chisq = prior_mismatch.data.dot(
            la.solve(total_err_cov, prior_mismatch.data))
        df_expected = np.prod(prior_mismatch.shape)
        # print("Chi squared statistic:", chisq)
        # print("Expected value:       ", df_expected)

        # print("Chi squared reduced:", chisq / df_expected)
        chi2s.data[i] = chisq
        innovations.data[i, :, :] = prior_mismatch.data[np.newaxis, :]
        increments.data[i] = (post_shaped - prior_shaped).data
        flux_tot = post_shaped.collapsed((FLUX_TIME_COORD, Y_COORD, X_COORD),
                                    iris.analysis.SUM)
        flux_totals.data[i] = flux_tot.data

        print(FUNCTION_COORD.points[i])
        #print(flux_tot)
        print(flux_tot.data)

    # print("To increase this statistic, decrease the flux variances\n"
    #       "To decrease this statistic, increase the flux variances\n"
    #       "If this is not close to one for this perfect-model setup,\n"
    #       "we have big problems.")
    iris.save([post_shaped, innovations, increments, chi2s, flux_totals,
               truth_shaped, OBSERVATION_OPERATOR, prior_shaped, observations_shaped],
              "crosscheck_exponential_actual_{true:d}_assumed_{assumed:d}.nc".format(
                  true=TRUE_CORR_LEN, assumed=ASSUMED_CORR_LEN),
              zlib=True)
