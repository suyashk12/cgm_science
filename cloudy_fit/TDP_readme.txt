
-----------------
Installing CLOUDY
-----------------

Introduction - CLOUDY is a photoionization code that can simulate astrophysical plasmas. To install this software, check out the “Getting started with Cloudy section” on the home page (https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home).

Note about versions - The models in the paper were built using C22.01. More recent versions are available. Note that the atomic physics may vary between different versions, among other things.

-------------------
Run CLOUDY scripts
-------------------
Running scripts - Make sure you can run basic scripts like the Smoke Test (https://gitlab.nublado.org/cloudy/cloudy/-/wikis/Smoke%20Test) as well as scripts in Hazy1 (available under c2x.xx/docs/hazy1.pdf under your installed CLOUDY folder). The general principle for running a script is to first save it as “x.in”, where x is your script name, and also have a shell script file in the same folder (see run.sh in https://github.com/suyashk12/cgm_science/blob/main/cloudy_fit/run.sh). The script will then be run using the command ./run.sh x in the Command Line (make sure you are in the same folder as your script when executing this command). The script may save output (with extensions of “.out”, etc.) in the folder where the script was executed. 

Sample TDP script - A sample time-dependent photoionization (TDP) script can be found on this publicly hosted GitHub repository under the name isochoric_cool.in (https://github.com/suyashk12/cgm_science/blob/main/cloudy_fit/isochoric_cool.in). Make sure you also have the shell script run.sh in the same directory when you execute CLOUDY!
Note about UVB - available options for the extragalactic UVB in CLOUDY include HM05 (Haardt and Madau 2001), HM12 (Haardt and Madau 2012), KS19 (Khaire and Srianand 2019). However, this work also considers photoionization models built with FG20 (Faucher-Giguere 2020), which is not offered by CLOUDY as a default option. To use the FG20 UVB in photoionization calculations, see 

Building a model grid - After confirming that you can run the sample TDP script, choose a directory where you’d like to build your model grid for a range of densities and metallicities. Then, run the Jupyter Notebook non_eqm_scripts.ipynb (https://github.com/suyashk12/cgm_science/blob/main/cloudy_fit/non_eqm_scripts.ipynb). Make sure rootdir is set to the directory where you’d like to build your model grid. You can specify the list of metallicities and densities for which you’d like to run TDP models. You can also specify the UVB(s) and initial temperature(s) for which you’d like to run the script. You need only concern yourself with the “TDP (with fixed densities)” code cell, which will create CLOUDY scripts for individual grid points. If your rootdir was set to r, these scripts will be stored under r/PI/TDP_isochoric/u/z=1/log_metals=m/log_hdens=n/T0=xK”, where u, m, n, and x are the considered UVB, metallicity, density, and initial temperature. You can switch the redshift to be of your preference depending on the relevant absorption sample. To actually get the model grid, you will need to run scripts for each grid point. For convenience, the code cell also prints executable commands that can conveniently allow you to do so. Switch your command line directory to r/PI/TDP_isochoric, and copy and paste these commands into your terminal. It will take a while to finish (~2 min per script), but at the end, you should have output ion fractions (saved with extension “.elem”, with elem = carb, nit, oxy, etc.) in the directory for each grid point.

---------------------------------
Parse CLOUDY output using Python
---------------------------------

Parsing and interpolating the output files - having obtained ion fractions for each grid point, you can use the notebook TDP_grid_compile.ipynb (https://github.com/suyashk12/cgm_science/blob/main/cloudy_fit/TDP_grid_compile.ipynb) to create a pickled version of ion fraction grids across a range of densities and metallicities.

Comparing with observations - see the notebook z=1.26_TDP_fit.ipynb (https://github.com/suyashk12/cgm_science/blob/main/cloudy_fit/z%3D1.26_TDP_fit.ipynb) for a tutorial on how to compare TDP model grids with Voigt profile measurements.
