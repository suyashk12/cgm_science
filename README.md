Codes for analyzing the kinematics, chemistry, and thermodynamics of the circumgalactic media of distant galaxies. Below is a guide for this repository -

1. Data reduction routines - for reducing telescope data into science-friendly format. Relevant folders are -

   a. STIS_stitching/pabs0_transform - Hubble Space Telescope data 

   b. ldss_catalog/ldss_data_redn - Magellan Telescope data

2. Galaxy survey routines - for identifying ~100 galaxy candidates for which scientific analysis can be performed. Relevant folders -

   a. light_subtraction - for uncovering very faint galaxies using the Very Large telescope

   b. redshift_fitting/halo_finder - for determining galaxy distances and star formation properties

3. Indirect gas detection routines - for identifying properties of gas far away from the galactic center of galaxies identified above. Relevant folder -

   a. voigt_fit - for making direct measurements from Hubble Space Telescope and Keck Telescope data. Includes MCMC uncertainty measurements

   b. lsf_transform - for incorporating telescope properties in measurements above

   c. upper_limits - statistical framework for including undetected features in analysis

   d. velocity_stacks - composite panels made for proposals

4. Simulation routines - for comparing measurements from data with a simulation grid and infer gas properties like chemistry and thermodynamics. Relevant folders -

   a. cloudy_fit - for building and comparing photoionization simulations with gas measurements under a Bayesian framework

   b. TDP_utilities - novel simulations for explaining subset of data not compatible with traditional simulations

For scientific details, please refer to Kumar et al. (2024) accepted to the Open Journal of Astrophysics (https://www.arxiv.org/abs/2408.15824). Please reach out with any questions or bug reports to suyashk@uchicago.edu.
