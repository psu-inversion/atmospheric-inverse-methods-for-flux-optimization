===================
Influence Functions
===================

Influence functions describe how a unit change in any given element of
the state vector (often, a unit change in the flux in a certain area)
will change all of the observations.  For the instance where the state
vector is the fluxes in each grid cell, one might calculate this as a
number of observation sites by number of observation times by number
of flux times by number of grid cells in the y direction by number of
grid cells in the x direction array.  Many of the elements of this
array will be zero, as no flux can affect a measurement before it.
Similarly, if the observation time is much later than the flux time,
the influence function will not vary much between sites [1]_.

There are two approaches to calculating influence functions: one using
a forward model and one using an adjoint model.

Foreward Model
==============

The most straightforward method to obtain influence functions is to do
many runs of the tracer transport model, one for each element of the
state vector.  The surface fluxes are those corresponding to a state
vector with one element set to one and all the rest zero.  Sampling
the simulated concentration fields at the time and place of the
measurements then produces the influence functions.

It is very easy to run experiments with different observation networks
using this method, if the simulated concentration fields are saved.
However, the cost of a separate run for each element of the state
vector can become quickly prohibitive for high-resolution inversions.

Adjoint Model
=============

Another method is to find the adjoint of the tracer transport model,
then do adjoint runs back from each observation.  The adjoint model
gives the sensitivity of a quantity of interest (a measurement at a
particular place and time) to conditions before that (in particular,
to the preceeding fluxes).  This method requires one run for each
measurment at each location, and works well when the total number of
measurements is much smaller than the total number of elements in the
state vector.

Running experiments with different observation networks with this
method is somewhat difficult, since each new measurement requires a
new run of the adjoint model.

One method used to avoid finding the adjoint of the full tracer
transport model is to release many imaginary particles at the time and
place of each measurement, then use the winds from the tracer
transport model to move the particles backward in time, finding where
the particles would have to have been at each prior time to become
part of the measurement.  Tracking where and how often these particles
interact with the ground then gives the influence functions.  This is
the approach taken by Lagrangian Particle Dispersion Models, such as
the [LPDM]_ of Uliasz (1994), the Stochastic Time-Inverted
Lagrangian Transport [STILT]_ model, the HYbrid Single-Particle
Lagrangian Integrated Trajectory [HYSPLIT]_ model, and the FLEXible
PARTicle dispersion model [FLEXPART]_.

This method requires relating the change in mole fraction for one of
these particles to the change in mole fraction of the measurement: if
all the particles composing the measurement experience an increase in
mole fraction of 1 ppm, then the measurement will be 1 ppm higher.  If
only half the particles experience this 1 ppm increase and the other
half are unaffected, the measurement will only be 0.5 ppm higher.
Extending this, if a single particle experiences an increase in mole
fraction of 1 (in arbitrary units), the measurement will be larger by
the reciprocal of the number of particles composing the measurement
(in the same units).

.. [1] If all the measurements are in the troposphere, "much later" is
       two to three years.  If all the measurements are additionally
       in the same hemisphere, "much later" is on the order of a
       month, perhaps less.  If the inversion is regional, with a
       spatial domain that does not cover the whole globe, "much
       later" is related to how long it takes air parcels to leave the
       domain.

References
==========

.. [LPDM] Uliasz, M. (1993). "The atmospheric Mesoscale Dispersion
          Modeling System". *Journal of Applied Meteorology*, 32 (1),
          139–149.  Retrieved 2016-07-15, from
	  https://journals.ametsoc.org/doi/abs/10.1175/1520-0450%281993%29032%3C0139%3ATAMDMS%3E2.0.CO%3B2
          :doi:`10.1175/1520-0450(1993)032<0139:TAMDMS>2.0.CO;2`

	  Uliasz, M. (1994).  "Lagrangian particle dispersion modeling
	  in mesoscale applications". *Environ Model: Comput Methods
	  and Softw for Simulat Environ Pollut and Its Adverse Effects
	  (CMP)*, 2 , 71–102. Retrieved from
	  http://indico.ictp.it/event/a02274/contribution/22/material/0/0.pdf

.. [STILT] http://stilt-model.org/index.php/Main/HomePage

	   Lin, J.C., C. Gerbig, S.C. Wofsy, A.E. Andrews, B.C. Daube,
	   K.J. Davis, and C.A. Grainger, "A near-field tool for
	   simulating the upstream influence of atmospheric
	   observations: The Stochastic Time-Inverted Lagrangian
	   Transport (STILT) model". *Journal of Geophysical
	   Research-Atmospheres*, (2003) 108(D16): 4493,
	   :doi:`10.1029/2002JD003161`.

.. [HYSPLIT] https://ready.arl.noaa.gov/HYSPLIT.php

	     Stein, A.F., R.R. Draxler, G.D. Rolph, B.J. Stunder,
	     M.D. Cohen, and F. Ngan, 2015: "NOAA’s HYSPLIT
	     Atmospheric Transport and Dispersion Modeling
	     System". *Bull. Amer. Meteor. Soc.*, 96, 2059–2077,
	     :doi:`10.1175/BAMS-D-14-00110.1`

.. [FLEXPART] https://www.flexpart.eu/

	      Ignacio Pisso, Espen Sollum, Henrik Grythe,
              Nina I. Kristiansen, Massimo Cassiani, Sabine Eckhardt,
              Delia Arnold, Don Morton, Rona L. Thompson,
              Christine D. Groot Zwaaftink, Nikolaos Evangeliou,
              Harald Sodemann, Leopold Haimberger, Stephan Henne,
              Dominik Brunner, John F. Burkhart, Anne Fouilloux,
              Jerome Brioude, Anne Philipp, Petra Seibert, and Andreas
              Stohl (2019), "The Lagrangian particle dispersion model
              FLEXPART version 10.4".  *Geosci. Model Dev.*, 12,
              4955–4997, :doi:`10.5194/gmd-12-4955-2019`.

	      Seibert, P., & Frank, A. (2004, 1). "Source-receptor
              matrix calculation with a lagrangian particle dispersion model in
              backward mode." *Atmospheric Chemistry and Physics*, 4 (1),
              51–63. :doi:`10.5194/acp-4-51-2004`
