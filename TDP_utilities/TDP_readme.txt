
This is a tutorial for creating time-dependent photoionization scripts for a range of gas densities and metallicities using CLOUDY.
If you run into errors or have any questions, please reach out to me at suyashk@uchicago.edu!

---------------------
| Installing CLOUDY |
---------------------

Introduction - CLOUDY is a photoionization code that can simulate astrophysical plasmas. 
To install this software, check out the “Getting started with Cloudy section” on the home page (https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home).

Note about versions - The models in the paper were built using C22.01. 
More recent versions are available. Note that the atomic physics may vary between different versions, among other things.

----------------------
| Run CLOUDY scripts |
----------------------

Running scripts - Make sure you can run basic scripts like the Smoke Test (https://gitlab.nublado.org/cloudy/cloudy/-/wikis/Smoke%20Test) 
as well as scripts in Hazy1 (available under c2x.xx/docs/hazy1.pdf under your installed CLOUDY folder). 
The general principle for running a script is to first save it as “x.in”, where x is your script name, 
and also have a shell script file in the same folder (see run.sh in https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/run.sh). 
Be sure to modify the path in run.sh to the location where you have installed CLOUDY.
The script will then be run using the command "./run.sh x" in the Command Line (make sure you are in the same folder as your script when executing this command). 
The script may save output (with extensions of “.out”, etc.) in the folder where the script was executed. 

Sample TDC script - A time-dependent collisional (TDC) ionization script can be found on a publicly hosted GitHub repository under the name 
"cp-cool-1keV.in" (https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/cp-cool-1keV.in), and was provided in Section 14.3 of Hazy1 (page 160).
Please note that you do not have to download the entire GitHub repo! You can just download individual files by clicking on the download icon
on the top-right (saying "Download raw file if you hover over it").
Make sure you also have the shell script run.sh in the same directory when you execute CLOUDY, and that you can run this script
before running a TDP script!

Sample TDP script - A sample time-dependent photoionization (TDP) script can be found 
"isochoric_cool.in" (https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/isochoric_cool.in).
Make sure you can run this before proceeding to build a model grid!
Note about UVB - available options for the extragalactic UVB in CLOUDY include HM05 (Haardt and Madau 2001), HM12 (Haardt and Madau 2012), KS19 (Khaire and Srianand 2019). 
However, this work also considers photoionization models built with FG20 (Faucher-Giguere 2020), which is not offered by CLOUDY as a default option. 
To use the FG20 UVB in photoionization calculations, see https://galaxies.northwestern.edu/uvb-fg20/.

---------------------------------------------
| Build and parse CLOUDY grids using Python |
---------------------------------------------

Building a model grid - After confirming that you can run the sample TDP script, choose a directory where you’d like to build your model grid for a range of densities 
and metallicities. Then, run the Jupyter Notebook "non_eqm_scripts.ipynb" (https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/TDP_scripts_write.ipynb). 
Make sure rootdir is set to the directory where you’d like to build your model grid. You can specify the list of metallicities and densities 
for which you’d like to run TDP models. You can also specify the UVB(s) and initial temperature(s) for which you’d like to run the script. 
If running TDP models is all you are concerned with, you need only run the first five code cells - i.e. till the cell below
“TDP (with fixed densities)”. Everything below that are other ionization models which you may or may not be interested in.

Anyways, running these five lines will create CLOUDY scripts for individual grid points. 
If your rootdir was set to r, these scripts will be stored under r/PI/TDP_isochoric/u/z=1/log_metals=m/log_hdens=n/T0=xK”, where u, m, n, and x are the considered UVB,
metallicity, density, and initial temperature. You can switch the redshift to be of your preference depending on the relevant absorption sample. 
To actually get the model grid, you will need to run scripts for each grid point. For convenience, the code cell also prints executable commands that can conveniently 
allow you to do so. Switch your command line directory to r/PI/TDP_isochoric, and copy and paste these commands into your terminal. It will take a while to 
finish (~2 min per script on a Macbook Pro M2 chip), but at the end, you should have output ion fractions (saved with extension “.elem”, 
with elem = carb, nit, oxy, etc.) in the directory for each grid point.

Parsing and interpolating the output files - having obtained ion fractions for each grid point, you can use the notebook 
"TDP_grid_compile.ipynb" (https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/TDP_grid_compile.ipynb) to create a pickled version 
of ion fraction grids across a range of densities and metallicities. 
Find a sample saved grid used for this paper at https://drive.google.com/drive/folders/1crTCXT0ydxPIBccDyS9nuJMD5E2rbY41?usp=sharing.

Comparing with observations - see the notebook z=1.26_TDP_fit.ipynb (https://github.com/suyashk12/cgm_science/blob/main/TDP_utilities/z%3D1.26_TDP_fit.ipynb) 
for a tutorial on how to compare TDP model grids with Voigt profile measurements. You'll have to ensure that the Python emcee library 
is installed (https://emcee.readthedocs.io/en/stable/user/install/).