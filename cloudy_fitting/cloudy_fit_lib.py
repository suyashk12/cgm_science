import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from astropy import constants
import matplotlib.pyplot as plt

############################################
#### Useful constants for CLOUDY models ####
############################################

# Bounds for CLOUDY model grid

# Neutral hydrogen column density
logN_HI_min = 12
logN_HI_max = 17

# Neutral hydrogen number density
log_hdens_min = -5
log_hdens_max = 1

# Metal abundance
log_metals_min = -3
log_metals_max = 1

# Dictionary of ionization potentials (in eV) of CREATION, not destruction
# 13.6 eV is for destruction, assume HI to be present from big bang nucleosynthesis
# Similarly, assume HeI to be present already too
IP_dict = {'HI': -1.,
            'HeI': 0., 
            'AlII': 5.986,
            'MgII': 7.646,
            'FeII': 7.87,
            'SiII': 8.151,
            'CII': 11.26,
            'OII': 13.618,
            'NII': 14.534,
            'SiIII': 16.345,
            'AlIII': 18.826,
            'CIII': 24.383,
            'NIII': 29.601,
            'SiIV': 33.492,
            'SIV': 34.83,
            'OIII': 35.117,
            'SV': 47.30,
            'NIV': 47.448,
            'CIV': 47.887,
            'OIV': 54.934,
            'SVI': 72.68,
            'NV': 77.472,
            'OVI': 113.9,
            'NeVI': 126.21,
            'NeVIII': 207.26,
            'MgX': 328.0}

# This is to convert ions from VP fit into species for the CLOUDY interpolated grid
ion_species_dict  = {'HI': '#column density H',
                     'HeI': 'He',
                     'AlII': 'Al+',
                    'MgII': 'Mg+',
                    'FeII': 'Fe+',
                    'SiII': 'Si+',
                    'CII': 'C+',
                    'OII': 'O+',
                    'NII': 'N+',
                    'SiIII': 'Si+2',
                    'AlIII': 'Al+2',
                    'CIII': 'C+2',
                    'NIII': 'N+2',
                    'SiIV': 'Si+3',
                    'SIV': 'S+3',
                    'OIII': 'O+2',
                    'SV': 'S+4',
                    'NIV': 'N+3',
                    'CIV': 'C+3',
                    'OIV': 'O+3',
                    'SVI': 'S+5',
                    'NV': 'N+4',
                    'OVI': 'O+5',
                    'NeVI': 'Ne+5',
                    'NeVIII': 'Ne+7',
                    'MgX': 'Mg+9'}

# Number densities of various elements in the Sun relative to hydrogen
solar_rel_dens_dict = {'Hydrogen': 1.0,
                    'Helium': 0.1,
                    'Lithium': 2.04e-09,
                    'Beryllium': 2.63e-11,
                    'Boron': 6.17e-10,
                    'Carbon': 0.000245,
                    'Nitrogen': 8.51e-05,
                    'Oxygen': 0.00049,
                    'Fluorine': 3.02e-08,
                    'Neon': 0.0001,
                    'Sodium': 2.14e-06,
                    'Magnesium': 3.47e-05,
                    'Aluminium': 2.95e-06,
                    'Silicon': 3.47e-05,
                    'Phosphorus': 3.2e-07,
                    'Sulphur': 1.84e-05,
                    'Chlorine': 1.91e-07,
                    'Argon': 2.51e-06,
                    'Potassium': 1.32e-07,
                    'Calcium': 2.29e-06,
                    'Scandium': 1.48e-09,
                    'Titanium': 1.05e-07,
                    'Vanadium': 1e-08,
                    'Chromium': 4.68e-07,
                    'Manganese': 2.88e-07,
                    'Iron': 2.82e-05,
                    'Cobalt': 8.32e-08,
                    'Nickel': 1.78e-06,
                    'Copper': 1.62e-08,
                    'Zinc': 3.98e-08}

# List of element names
element_names_dict = {'Hydrogen': 'H',
                      'Helium': 'He',
                      'Lithium': 'Li',
                      'Beryllium': 'Be',
                      'Boron': 'B',
                      'Carbon': 'C',
                      'Nitrogen': 'N',
                      'Oxygen': 'O',
                      'Fluorine': 'F',
                      'Neon': 'Ne',
                      'Sodium': 'Na',
                      'Magnesium': 'Mg',
                      'Aluminium': 'Al',
                      'Silicon': 'Si',
                      'Phosphorus': 'P',
                      'Sulphur': 'S',
                      'Chlorine': 'Cl',
                      'Argon': 'Ar',
                      'Potassium': 'K',
                      'Calcium': 'Ca',
                      'Scandium': 'Sc',
                      'Titanium': 'Ti',
                      'Vanadium': 'V',
                      'Chromium': 'Cr',
                      'Manganese': 'Mn',
                      'Iron': 'Fe',
                      'Cobalt': 'Co',
                      'Nickel': 'Ni',
                      'Copper': 'Cu',
                      'Zinc': 'Zn'}

# List of alpha elements
alpha_elements = ['Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur', 'Argon', 'Calcium', 'Titanium']

#####################################################################
#### Utilities for processing and plotting the extragalactic UVB #### 
#####################################################################

def read_uvb(rootdir='', filename='fg20_galaxy.ascii'):

    '''
    Function to read in an extragalactic UVB

    rootdir: Root directory where the UVB is stored (NOTE: modified somewhat from the CLOUDY input for better readability)
    filename: Name of the file containing the UVB (default is FG20)
    '''
    
    # First read in the file, line by line
    with open(rootdir+filename) as file:
        uvb_lines = [line.rstrip() for line in file]
        
    # Next, isolate a grid of redshifts
    
    # Specify default delimiter
    # Necessary for processing later
    
    # HM05 and 12
    if filename == 'hm05_galaxy.ascii' or filename == 'hm12_galaxy.ascii':
        dm = '  '
    # FG20
    else:
        dm = ' '

    # Read all lines relevant to redshift
    uvb_lines_z_grid = uvb_lines[uvb_lines.index('# z_grid')+1: uvb_lines.index('# wav_grid')]
    # Split apart the strings
    uvb_lines_z_grid_split = [z_grid_str.strip().replace(dm,',').split(',') for z_grid_str in uvb_lines_z_grid]
    # Flatten the list
    uvb_lines_z_grid_flat = [item for sublist in uvb_lines_z_grid_split for item in sublist]
    # Numpy-fy and flatten
    uvb_z_grid = np.array(uvb_lines_z_grid_flat, dtype='float')
   
    # Next, get the grid of wavelengths
    
    # Read in all lines relevant to wavelength
    uvb_lines_wav_grid = uvb_lines[uvb_lines.index('# wav_grid')+1: uvb_lines.index('# f_nu_list')]
    # Split apart the strings
    uvb_lines_wav_grid_split = [wav_grid_str.strip().replace(dm, ',').split(',') for wav_grid_str in uvb_lines_wav_grid]
    # Flatten the list
    uvb_lines_wav_grid_flat = [item for sublist in uvb_lines_wav_grid_split for item in sublist]
    #print(uvb_lines_wav_grid_split)
    # Numpy-fy and flatten
    uvb_wav_grid = np.array(uvb_lines_wav_grid_flat, dtype='float').flatten()    
    
    # Read in the list of J_nu and reshape
    uvb_lines_J_nu_list = uvb_lines[uvb_lines.index('# f_nu_list')+1:]
    # Split apart the strings
    uvb_lines_J_nu_split = [J_nu_str.strip().replace(dm, ',').split(',') for J_nu_str in uvb_lines_J_nu_list]
    # Flatten and reshape
    uvb_J_nu_grid = np.array([item for sublist in uvb_lines_J_nu_split for item in sublist], dtype='float').reshape((len(uvb_z_grid), len(uvb_wav_grid))) 
    
    return uvb_z_grid, uvb_wav_grid, uvb_J_nu_grid

def fetch_sed(z, uvb_z_grid, uvb_J_nu_grid):

    '''
    Function to fetch the SED at a particular redshift from a given grid

    z: Redshift to fetch UVB at
    uvb_z_grid: Grid of redshifts where the UVB is defined
    uvb_J_nu_grid: UVB at redshifts that are part of uvb_z_grid
    '''
    
    # First find the redshift closest to the desired one 
    # Search the redshift grid for the same
    idx = np.argmin(np.abs(uvb_z_grid-z))
    
    # Isolate J_nu
    uvb_J_nu = uvb_J_nu_grid[idx]
    
    return uvb_J_nu

def calc_ionizing_flux(uvb_wav_grid, uvb_J_nu):

    '''
    Function to calculate the H-ionizing flux (in photons/s/cm^2) for a given SED

    uvb_wav_grid: Wavelengths (in Angstrom) where the SED is defined
    uvb_J_nu: The SED itself (in ergs/s/cm^2/Hz/sr)
    '''
    
    # First get energies corresponding to ionizing H-flux, >=1 Ryd
    ion_idx = uvb_wav_grid*1e-10<=constants.Ryd.value**-1
    
    # Next, convert Ang to Hz, using nu = c/lam
    uvb_nu_grid = 1e+10*constants.c.value/uvb_wav_grid
    
    # J_nu is the average intensity over 4*pi sr in ergs/s/cm^2/Hz/sr, first convert to J/s/cm^2/Hz/sr
    # Next, to go from energy to photon count, divide by h*nu, the photon energy, this puts units as photon/s/cm^2/Hz/sr
    # Multiply by 4*pi sr to account for photons arriving per cm^2 from all 4*pi sr, this puts units as photon/s/cm^2/Hz
    uvb_phot_dens_nu = 4*np.pi*uvb_J_nu*1e-7/(constants.h.value*uvb_nu_grid)
    
    # Calculate the ionizing flux of photons >= 1 Ryd, units are photon/s/cm^2
    # Need to flip b/c frequency array is decreasing
    phi = integrate.simpson(y=np.flip(uvb_phot_dens_nu[ion_idx]), x=np.flip(uvb_nu_grid[ion_idx]))
    
    return phi

def calc_U(uvb_wav_grid, uvb_J_nu, n_H):

    '''
    Function to calculate the ionization parameter for a given SED and neutral hydrogen density

    uvb_wav_grid: Wavelengths (in Angstrom) where the SED is defined
    uvb_J_nu: The SED itself (in ergs/s/cm^2/Hz/sr)
    n_H: The neutral hydrogen density (in cm^-3)
    '''
    
    # First calculate the ionizing photon flux in units photon/s/cm^2
    phi = calc_ionizing_flux(uvb_wav_grid, uvb_J_nu)
    
    # Assuming a ionizing photon flux in units photon/s/cm^2
    # Divide by c in cm/s to get photon number density in photon/cm^3
    n_gamma = phi/(constants.c.value*1e+2)
    
    # Calculate the ionizing parameter
    # Divide by n_H in cm^-3
    U = n_gamma/n_H
    
    return U

######################################################################
#### Utilities for processing and plotting the CLOUDY model grids ####
######################################################################

def create_grid_file_list(logN_HI_arr):

    '''
    Function to return filenames of different grids (with different stopping column densities of neutral hydrogen)

    N_HI_arr: List of N_HI grid points (dex)
    '''

    file_list = ['igm_lalpha_hm12_grid_{:.2f}'.format(logN_HI).replace('.','') for logN_HI in logN_HI_arr]

    return file_list

def read_grd_file(rootdir, filename, delimiter='\t', dtype=str):

    '''
    Function to read in tabulated summary of a grid run (for fixed N_HI) stored in a file with a .grd extension

    rootdir: Root directory where the .grd file is stored
    filename: Name of the .grd file
    delimiter: Delimiter that separates different column entries
    dtype: Format in which to read in entries of the .grd file
    '''

    grid_df = pd.read_csv(rootdir+filename+'.grd', delimiter='\t', dtype=str)

    # Isolate densities and metallicities
    # Note that there are repetitions, each entry corresponds to a separate grid point
    log_hdens = np.array(grid_df['HDEN=%f L'], dtype=float)
    log_metals = np.array(grid_df['METALS= %'], dtype=float)
    
    # A NOTE about the grid structure
    # The densities are first kept constant, and the metallicities are varied
    return log_hdens, log_metals

def read_avr_file(rootdir, filename):

    '''
    Function to read in average HI temperature, saved in a file with .avr extension

    rootdir: Root directory where the .avr file is stored
    filename: Name of the .avr file
    '''

    with open(rootdir+filename+'.avr') as file:
        tem_lines = [line.rstrip() for line in file]

    # Split the file into lines
    tem_lines_split = [line.split(' ') for line in tem_lines]    

    # Get just the temperatures
    grid_temps = [float(line_split[1]) for line_split in tem_lines_split[1::2]] 

    # Convert to log10
    log_temps = np.log10(np.array(grid_temps))

    return log_temps

def read_col_file(rootdir, filename):

    '''
    Function to read in column densities (in log10) for multiple species, saved in a file with .col extension

    rootdir: Root directory where the .col file is stored
    filename: Name of the .col file
    '''    

    # Load in the file of column densities
    with open(rootdir+filename+'.col') as file:
        col_lines = [line.rstrip() for line in file]

    # Split the file into lines
    col_lines_split = [line.split('\t') for line in col_lines]

    # Get species names
    species_names = col_lines_split[0]
    # Get column densities for all species for all points
    log_col_dens = np.log10(np.array(col_lines_split[1::2], dtype=float))

    return species_names, log_col_dens

############################################################################
#### Utilities for dealing with depatures from solar abundance patterns ####
############################################################################

def get_metal_abundance(O_H, M_O_dict = {}):
    
    '''
    Function to generate [M/H] using [O/H] and [X/O], where X can span other metals.
    This allows access to the interpolated grid of CLOUDY models

    O_H: [O/H], the relative abundance (to solar) of oxygen to hydrogen
    M_O_dict: The relative abundances, [M/O] of various metals to oxygen. If a metal is not present in this dictionary, the relative abundance is assumed to be zero.
    '''

    # First generate the relative density of O to H
    # Recall that [O/H] is on a log scale
    # Add it onto the relative metal density to hydrogen
    n_O_H = solar_rel_dens_dict['Oxygen']*10**O_H
    n_M_H = n_O_H

    # Meanwhile, also build up the relative metal density in the sun
    n_M_H_solar = solar_rel_dens_dict['Oxygen']

    # Now generate the other metals (excluding oxygen)
    
    for i in range(len(solar_rel_dens_dict)):

        # Isolate elements present in the Sun
        element = list(solar_rel_dens_dict.keys())[i]

        # Exclude H and He (b/c they aren't metals) and oxygen b/c it's already been accounted for 
        if element != 'Hydrogen' and element != 'Helium' and element != 'Oxygen':

            # Append to the solar relative metal density
            n_M_H_solar += solar_rel_dens_dict[element]

            # First get the number density of this element relative to oxygen
            n_M_O = solar_rel_dens_dict[element]/solar_rel_dens_dict['Oxygen']

            # Check if a relative abundance is provided for this element
            # NOTE: For convenience, we'll be using short forms of elements in M_O_dict
            if element_names_dict[element] in list(M_O_dict.keys()):
                # If present, correct the relative density compared to solar
                # Again, remember that relative abundances are on a log scale
                n_M_O *= 10**M_O_dict[element_names_dict[element]]

            # Convert the relative density of metal to oxygen into metal to hydrogen
            n_M_H += n_M_O*n_O_H

    # Finally, we have the relative abundance of metals
    M_H =  np.log10(n_M_H)-np.log10(n_M_H_solar)

    return M_H

def get_alpha_abundance(O_H, alpha_O_dict = {}):

    '''
    Function to get the abundance of alpha elements relative to hydrogen

    O_H: [O/H], the relative abundance (to solar) of oxygen to hydrogen
    alpha_O_dict: The relative abundances, [alpha/O] of various alpha elements to oxygen.
    '''

    # First generate the relative density of O to H
    # Recall that [O/H] is on a log scale
    # Add it onto the relative metal density to hydrogen
    n_O_H = solar_rel_dens_dict['Oxygen']*10**O_H
    n_alpha_H = n_O_H

    # Meanwhile, also build up the relative metal density in the sun
    n_alpha_H_solar = solar_rel_dens_dict['Oxygen']

    # Now generate the other alpha elements (excluding oxygen)
    
    for i in range(len(alpha_elements)):

        # Isolate elements present in the Sun
        element = alpha_elements[i]

        # Exclude oxygen b/c it's already been accounted for 
        if element != 'Oxygen':

            # Append to the solar relative metal density
            n_alpha_H_solar += solar_rel_dens_dict[element]

            # First get the number density of this element relative to oxygen
            n_alpha_O = solar_rel_dens_dict[element]/solar_rel_dens_dict['Oxygen']

            # Check if a relative abundance is provided for this element
            # NOTE: For convenience, we'll be using short forms of elements in alpha_O_dict
            if element_names_dict[element] in list(alpha_O_dict.keys()):
                # If present, correct the relative density compared to solar
                # Again, remember that relative abundances are on a log scale
                n_alpha_O *= 10**alpha_O_dict[element_names_dict[element]]

            # Convert the relative density of metal to oxygen into metal to hydrogen
            n_alpha_H += n_alpha_O*n_O_H

    # Finally, we have the relative abundance of metals
    # NOTE: under a solar abundance pattern, [alpha/H] = [O/H]! Important sanity check
    alpha_H =  np.log10(n_alpha_H)-np.log10(n_alpha_H_solar)

    return alpha_H    

def get_X_alpha(X_O, O_H, alpha_O_dict = {}):

    '''
    Function to obtain the relative abundance of a non-alpha metal to alpha elements

    X/O: The relative abundance of non-alpha metal X, to O
    O/H: The relative abundancce of oxygen to hydrogen
    alpha_O_dict: The relative abundances of other alpha elements to oxygen
    '''

    # First, get the relative abundance of X to H
    # You can expand this formula out to see it works :)
    # Again, under a solar abundance pattern, you can see that X_H = O_H!
    X_H = X_O + O_H

    # Then get [alpha/H]
    alpha_H = get_alpha_abundance(O_H, alpha_O_dict)

    # Finally, get [X/alpha]
    # Again, expand the formula to see it works
    X_alpha = X_H - alpha_H

    return X_alpha

#################################################
#### Utilities for plotting column densities ####
#################################################

def plot_column_densities_obs(logN_dict, fig = None, ax = None):

    '''
    Method to plot observed column densities of species from VP fit

    logN_dict: Dictionary of column densities to be plotted, contains significant detections as well as non-detections (upper/ lower limits). NOTE: lower limits are not handled yet.
    '''

    # If figure and axes object doesn't exist, create it
    # This boolean variable keeps track of whether or not a figure object was created within the scope of this function
    create_fig_ax = False
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, figsize=(15,4))
        create_fig_ax = True

    # Order the ions according to their ionization potential
    # NOTE: the assumption is that the dictionary of ionization potentials is already sorted. If not, sort it
    ions_ordered = [s for s in list(IP_dict.keys()) if s in list(logN_dict.keys())]

    for i in range(len(ions_ordered)):
        
        ion = ions_ordered[i]
        
        logN_str = logN_dict[ion]
        
        # Detection
        if logN_str[0] != '<' and logN_str[0] != '>':
            logN_arr = np.array(logN_str.split(','), dtype=float)
            ax.scatter(i, logN_arr[0], s=3, color='black')
            ax.errorbar(x=i, y=logN_arr[0], yerr=logN_arr[1], color='black', linestyle='None',
                    fmt='o', markersize=3, capsize=4)
            ax.text(x=i-.2, y=logN_arr[0]+logN_arr[1]+.45, s=ion)
        
        # Upper limit
        elif logN_str[0] == '<':
            logN_lim = float(logN_str[1:])
            ax.errorbar(x=i, y=logN_lim, yerr=0.3, uplims=True, color='black', fmt='o', markersize=3)
            ax.text(x=i-.2, y=logN_lim+0.3+.15, s=ion)
        
        # Lower limit
        # Not implemented yet
        elif logN_str[0] == '>':
            logN_arr = np.array(logN_str[1:].split(','), dtype=float)
            ax.errorbar(x=i, y=logN_arr[0], yerr=0.3, lolims=True, color='black', fmt='o', markersize=3)
            ax.text(x=i-.2, y=logN_arr[0]-.85, s=ion)

    # Turn off ticks and label axes
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_ylabel(r'$\log (N_{\mathrm{ion}}/\mathrm{cm}^2)$')
    ax.set_xlabel(r'Increasing Ionization Potential $\rightarrow$')

    # Set a limit on the y-axis
    ax.set_ylim(10,18)

    # Return figure object if one was created
    if create_fig_ax == True:
        return fig, ax

def predict_col_dens(logN_dict, logN_HI_test, log_hdens_test, log_metals_test, species_logN_interp, M_O_dict = {}):

    '''
    Predict column densities for an ordered list of ions given an interpolated CLOUDY grid across N_HI, n_H, and [M/H].
    If there are departures from solar abundance patterns, the column densities will be shifted appropriately
    '''

    # Generate sorted list of ions according to ionization potential
    ions_ordered = [s for s in list(IP_dict.keys()) if s in list(logN_dict.keys())]

    logN_species_test = []

    # Respect ordering of ions
    # The hope is that the HI column density from the grid will just be the test value, if interpolation proceeded properly
    for i in range(len(ions_ordered)):
            
        ion = ions_ordered[i]   
        s = ion_species_dict[ion]

        # Get predicted column density for the species from CLOUDY
        logN_s = species_logN_interp[s]([logN_HI_test, log_hdens_test, log_metals_test])[0]

        # If there is departure from solar abundance, shift the predicted column density accordingly
        # s.split('+')[0] is supposed to be the element name. This won't work for hydrogen, or helium, but they're not metals anyway
        if s.split('+')[0] in M_O_dict:
            logN_s += M_O_dict[s.split('+')[0]]

        # Get interpolated column density from CLOUDY grid
        logN_species_test.append(logN_s)

    return logN_species_test

###################################################################################
#### Utilities for constraining HI column density, gas density, and abundances ####
###################################################################################

def log_prior(params):
    
    '''
    Priors for an MCMC search. 

    params_dict: Dictionary of parameters being fitted. Will contain logN_HI, n_H, [O/H], and [X/H]
    '''

    # Grid parameters being varied
    logN_HI, log_hdens, O_H, M_O_dict = params

    # First generate the metal abundance using [O/H] and [X/H]
    log_metals = get_metal_abundance(O_H=O_H, M_O_dict=M_O_dict)
    
    # If the sampled density is within the CLOUDY limits
    # Avoid edges?
    if logN_HI_min<logN_HI<logN_HI_max and log_hdens_min<log_hdens<log_hdens_max and log_metals_min<log_metals<log_metals_max:
        return 0.0
    return -np.inf

def log_likelihood(params, logN_dict, species_logN_interp):

    '''
    Likelihood function for comparing CLOUDY predicted column densities to the observed values from VP fit.
    If only some of the parameters need to be fit, the log_likelihood function can be overridden with a lambda function
    which calls this likelihood function with only some parameters being varied.

    params: parameters needed to generate CLOUDY predicted column densities
    logN_dict: dictionary of measured column densities from VP fit
    species_logN_interp: interpolated CLOUDY grid
    '''
    
    # Grid parameters being varied
    logN_HI, log_hdens, O_H, M_O_dict = params

    # Generate metal abundance using [O/H] and [M/O]
    log_metals = get_metal_abundance(O_H=O_H, M_O_dict=M_O_dict)
    
    # Likelihood function
    ll = 0
    
    ions = list(logN_dict.keys())
    
    # Ignore first entry since it's HI
    for i in range(len(ions)):
        
        # This is from VP fit
        ion = ions[i]
        logN_str = logN_dict[ion]
        
        # This is from CLOUDY
        s = ion_species_dict[ion]
        
        # Get interpolated column density from CLOUDY grid
        y_bar = species_logN_interp[s]([logN_HI, log_hdens, log_metals])[0]

        # If there is departure from solar abundance, shift the predicted column density accordingly
        if s.split('+')[0] in M_O_dict:
            y_bar += M_O_dict[s.split('+')[0]]

        # Based on detection or non-detection, compute the likelihood term
        
        # Detection
        if logN_str[0] != '<' and logN_str[0] != '>':
            
            logN_arr = np.array(logN_str.split(','), dtype=float)
            
            # Observed column density
            y = logN_arr[0]

            # Use max of lower and upper error for defining Gaussian distribution of column density
            sig_y = max(-logN_arr[1], logN_arr[2])

            # Gaussian likelihood
            ll += -.5*(y-y_bar)**2/sig_y**2

        # Upper limit
        elif logN_str[0] == '<':
            
            # Upper limit of column density
            # This is 3-sigma
            y = float(logN_str[1:])
            
            # Define an integration range for the reported value, from "-inf" to upper limit
            # Use a step size of 0.1 dex
            y_range_min = -10
            y_range_step = 0.05

            y_range = np.arange(y_range_min, y+y_range_step, y_range_step)
            # Get the range in linear space
            lin_y_range = 10**y_range

            # The uncertainty on log scale is proportional to the fractional error in the linear scale
            sig_lin_y = 10**y/3

            # Confusing notation :(
            # CDF - marginalize over the reported value
            # Use 10^y_bar, not y_bar! Comparison for upper limits takes place in the linear scale
            ll += np.log(integrate.simpson(x=lin_y_range, y=np.exp(-.5*(lin_y_range-10**y_bar)**2/sig_lin_y**2)))
            
        # Lower limit
        # NOTE: not implemented yet
        elif logN_str[0] == '>':

            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = logN_arr[1]

            y_range_max = 21.5 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

            # "Q"-function, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))

    # Return log likelihood for MCMC
    return ll
