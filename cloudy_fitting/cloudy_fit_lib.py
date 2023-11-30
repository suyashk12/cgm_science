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

# Significance levels
# Sigma values taken from Gehrels 1986
cdf_1sig = 0.8413
cdf_2sig = 0.9772
cdf_3sig = 0.9987

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
            'OV': 77.413,
            'NV': 77.472,
            'NeV': 97.11,
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
                    'OV': 'O+4',
                    'NV': 'N+4',
                    'NeV': 'Ne+4',
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
    ion_idx = uvb_wav_grid<=(constants.Ryd.value**-1)*1e+10
    
    # Next, convert Ang to Hz, using nu = c/lam
    uvb_nu_grid = 1e+10*constants.c.value/uvb_wav_grid
    
    # J_nu is the average intensity over 4*pi sr in ergs/s/cm^2/Hz/sr, first convert to J/s/cm^2/Hz/sr
    # Next, to go from energy to photon count, divide by h*nu, the photon energy, this puts units as photon/s/cm^2/Hz/sr
    # Multiply by 4*pi sr to account for photons arriving per cm^2 from all 4*pi sr, this puts units as photon/s/cm^2/Hz
    uvb_phot_dens_nu = 4*np.pi*uvb_J_nu*1e-7/(constants.h.value*uvb_nu_grid)
    
    # Calculate the ionizing flux of photons >= 1 Ryd, units are photon/s/cm^2
    # Need to flip b/c frequency array is decreasing
    #phi_nu = interpolate.interp1d(np.flip(uvb_nu_grid), np.flip(uvb_phot_dens_nu), fill_value='extrapolate')

    #print(phi_nu(constants.c.value*constants.Ryd.value), phi_nu(np.max(uvb_nu_grid)))

    #phi = integrate.quad(phi_nu, constants.c.value*constants.Ryd.value, np.max(uvb_nu_grid))[0]
    phi = np.trapz(np.flip(uvb_phot_dens_nu[ion_idx]), x=np.flip(uvb_nu_grid[ion_idx]))
    
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

def calc_n_H(uvb_wav_grid, uvb_J_nu, U):

    '''
    Function to calculate gas density for a given SED and ionization parameter

    uvb_wav_grid: Wavelengths (in Angstrom) where the SED is defined
    uvb_J_nu: The SED itself (in ergs/s/cm^2/Hz/sr)
    U: Ionization parameter
    '''
    
    # First calculate the ionizing photon flux in units photon/s/cm^2
    phi = calc_ionizing_flux(uvb_wav_grid, uvb_J_nu)
    
    # Assuming a ionizing photon flux in units photon/s/cm^2
    # Divide by c in cm/s to get photon number density in photon/cm^3
    n_gamma = phi/(constants.c.value*1e+2)
    
    #print(n_gamma)

    # Divide by U
    n_H = n_gamma/U
    
    return n_H


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

#################################################
#### Utilities for plotting column densities ####
#################################################

def plot_column_densities_obs(logN_dict, fig = None, ax = None, gray_out = []):

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

        if ion in gray_out:
            c='darkgray'
        else:
            c='black'
        
        logN_str = logN_dict[ion]
        
        # Detection
        if logN_str[0] != '<' and logN_str[0] != '>':
            logN_arr = np.array(logN_str.split(','), dtype=float)
            ax.scatter(i, logN_arr[0], s=3, color=c)
            ax.errorbar(x=i, y=logN_arr[0], yerr=logN_arr[1], color=c, linestyle='None',
                    fmt='o', markersize=3, capsize=4)
            ax.text(x=i-.2, y=logN_arr[0]+logN_arr[1]+.45, s=ion, color=c)
        
        # Upper limit
        elif logN_str[0] == '<':
            logN_lim = float(logN_str[1:])
            ax.errorbar(x=i, y=logN_lim, yerr=0.3, uplims=True, color=c, fmt='o', markersize=3)
            ax.text(x=i-.2, y=logN_lim+0.3+.15, s=ion, color=c)
        
        # Lower limit
        # Not implemented yet
        elif logN_str[0] == '>':
            logN_arr = np.array(logN_str[1:].split(','), dtype=float)
            ax.errorbar(x=i, y=logN_arr[0], yerr=0.3, lolims=True, color=c, fmt='o', markersize=3)
            ax.text(x=i-.2, y=logN_arr[0]-.85, s=ion, color=c)

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

def predict_col_dens(logN_dict, logN_HI_test, log_hdens_test, log_metals_test, species_logN_interp, X_alpha_dict = {}):

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
        if s.split('+')[0] in X_alpha_dict:
            logN_s += X_alpha_dict[s.split('+')[0]]

        # Get interpolated column density from CLOUDY grid
        logN_species_test.append(logN_s)

    return logN_species_test

#########################################################
#### Utilities for processing column density strings ####
#########################################################

def process_data_str(s, non_det_err):
    
    # Check if the point should be plotted or not
    bool_plot = False
    # Coordinate value
    coord_value = np.nan
    # Lower error
    err_lo = 0
    # Upper error
    err_hi = 0
    # Boolean for upper limit
    uplim = False
    # Boolean for lower limit
    lolim = False
    
    # If we don't have an empty string 
    if len(s)!=0:
        
        # Then it needs to be plotted
        bool_plot = True
        
        # Now, check for a non-detection first
        
        # Upper limit
        if s[0] == '<':
            # Set coordinate value
            coord_value = float(s[1:])
            # Only lower error
            uplim = True
            err_lo = non_det_err
            
        # Lower limit
        elif s[0] == '>':
            # Set coordinate value
            coord_value = float(s[1:])
            # Only upper error
            lolim = True
            err_hi = non_det_err
            
        # Detection
        else:
            s_arr = np.array(s.split(','), dtype=float)
            
            # Errors not provided
            if len(s_arr) == 1:
                coord_value = s_arr[0]
                err_lo = 0
                err_hi = 0
                
            # Errors provided
            else:
                coord_value = s_arr[0]
                err_lo = -s_arr[1] # Negative sign to flip the already present negative sign
                err_hi = s_arr[2]
        
    return bool_plot, coord_value, err_lo, err_hi, uplim, lolim

def gen_logN_ratio_str(logN_str_1, logN_str_2, non_det_err=0.3):
    
    # First process each data string
    bool_plot_logN1, coord_value_logN1, err_lo_logN1, err_hi_logN1, uplim_logN1, lolim_logN1 = process_data_str(logN_str_1, non_det_err)
    bool_plot_logN2, coord_value_logN2, err_lo_logN2, err_hi_logN2, uplim_logN2, lolim_logN2 = process_data_str(logN_str_2, non_det_err)
    
    # Classify detection
    det_logN1 = bool_plot_logN1 and (not uplim_logN1) and (not lolim_logN1)
    det_logN2 = bool_plot_logN2 and (not uplim_logN2) and (not lolim_logN2)
    
    # Relevant quantities
    logN_ratio = coord_value_logN1-coord_value_logN2
    err_lo_logN_ratio = -np.sqrt(err_lo_logN1**2 + err_lo_logN2**2) # Put negative sign for consistency w/ plotting functions
    err_hi_logN_ratio = np.sqrt(err_hi_logN1**2 + err_hi_logN2**2)  
    
    # Classify cases as booleans
        
    # Variables to store 
    logN_ratio_str = ''
    
    # Cases that create upper limits
    
    # N1 upper limit and N2 lower limit
    # N1 upper limit and N2 detection
    # N1 detection and N2 lower limit - WHAT is the probabilistic meaning of this?
    if (uplim_logN1 == True and lolim_logN2 == True) or (uplim_logN1 == True and det_logN2 == True) or (det_logN1 == True and lolim_logN2 == True):
        logN_ratio_str = '<{:.2f}, {:.2f}'.format(logN_ratio, np.sqrt(max(err_lo_logN_ratio,err_hi_logN_ratio)**2 + 0.14**2))
    
    # Cases that create lower limits
    
    # N1 lower limit and N2 upper limit
    # N1 lower limit and N2 detection
    # N1 detection and N2 upper limit - WHAT is the probabilistic meaning of this?
    elif (lolim_logN1 == True and uplim_logN2 == True) or (lolim_logN1 == True and det_logN2 == True) or (det_logN1 == True and uplim_logN2 == True):
        logN_ratio_str = '>{:.2f}, {:.2f}'.format(logN_ratio, np.sqrt(max(err_lo_logN_ratio,err_hi_logN_ratio)**2 + 0.14**2))  
        
    # Both are detections
    elif det_logN1 == True and det_logN2 == True:
        logN_ratio_str = '{:.2f}, {:.2f}, {:.2f}'.format(logN_ratio,
                                            err_lo_logN_ratio, # Errors add in quadrature
                                            err_hi_logN_ratio)
        
    return logN_ratio_str

###############################################
#### Utilities for visualizing gas density ####
###############################################

def gen_logN_ratio_pred(species_logN_interp, ion_pairs_list, logN_HI_ref, log_metals_ref,
                        log_hdens_min, log_hdens_max, log_hdens_grid_size):
    
    '''
    Method to generate predicted column density ratios for ionic pairs

    species_logN_interp: Interpolated CLOUDY grid
    ion_pairs_list: list of ionic pairs
    logN_HI_ref: fiducial HI column density to access CLOUDY grid
    log_metals_ref: fiducial metallicity to access CLOUDY grid
    log_hdens_min/max/size: end points and size of gas density grid to evaluate column density ratios
    '''

    # Establish the gas density grid
    log_hdens_grid = np.linspace(log_hdens_min, log_hdens_max, log_hdens_grid_size)

    # Create grid points
    grid_points = np.array([[logN_HI_ref, log_hdens, log_metals_ref] for log_hdens in log_hdens_grid])

    logN_ratio_pred = {}

    for ion_pair_str in ion_pairs_list:

        ion_pair = ion_pair_str.split('/')
        
        # Get corresponding species
        s1 = ion_species_dict[ion_pair[0]]
        s2 = ion_species_dict[ion_pair[1]]
        
        # Extract column density ratio from grid
        logN_ratio_pred[ion_pair_str] = species_logN_interp[s1](grid_points)-species_logN_interp[s2](grid_points)

    return log_hdens_grid, logN_ratio_pred

def plot_logN_ratio(ax, species_logN_interp, logN_ratio_dict, logN_HI_ref = 12, log_metals_ref = -3,
                             log_hdens_min = -5, log_hdens_max = 1, log_hdens_grid_size = 5000):

    '''
    Method to overplot predicted and observed column density ratios as a function of gas density

    ax: the axes object
    species_logN_interp: interpolated CLOUDY grid
    logN_ratio_dict: dictionary of observed column densities
    '''

    # Generate grid of gas density and column density ratio predictions
    log_hdens_grid, logN_ratio_pred = gen_logN_ratio_pred(species_logN_interp, list(logN_ratio_dict.keys()), 
                                   
                        logN_HI_ref, log_metals_ref,
                        log_hdens_min, log_hdens_max, log_hdens_grid_size)
    
    # List of colors
    colors = ['black', 'darkgray', 'firebrick', 'tan', 'deepskyblue', 
              'cyan', 'green', 'yellowgreen', 'darkmagenta', 'orchid',
              'darkorange', 'orangered', 'goldenrod', 'indigo', 'blueviolet',
              'crimson']

    # Plot the predicted column density ratios
    for i in range(len(logN_ratio_pred)):
        
        ion_pair_str = list(logN_ratio_pred.keys())[i]
        
        # Plot model predictions
        ax.plot(log_hdens_grid, logN_ratio_pred[ion_pair_str], label=ion_pair_str, c=colors[i])
        
    # Plot the observed column density ratios
    for i in range(len(logN_ratio_dict)):

        ion_pair_str = list(logN_ratio_dict.keys())[i]
        
        # Observed column density ratio
        logN_ratio_str = logN_ratio_dict[ion_pair_str]
        
        # If upper limit
        if logN_ratio_str[0] == '<':
            idx = (logN_ratio_pred[ion_pair_str] < float(logN_ratio_str[1:].split(',')[0]))
        
        # If lower limit
        elif logN_ratio_str[0] == '>':
            idx = (logN_ratio_pred[ion_pair_str] > float(logN_ratio_str[1:].split(',')[0]))   
            
        # For detection
        else:
            logN_ratio_list = logN_ratio_str.split(',')
            logN_lo = float(logN_ratio_list[0]) + float(logN_ratio_list[1]) # Mind the negative sign
            logN_hi = float(logN_ratio_list[0]) + float(logN_ratio_list[2])
            
            idx = (logN_ratio_pred[ion_pair_str] > logN_lo) & (logN_ratio_pred[ion_pair_str] < logN_hi)

        mask = np.ones(len(log_hdens_grid))
        mask[~idx] = np.nan
            
        ax.plot(log_hdens_grid*mask, logN_ratio_pred[ion_pair_str]*mask, lw=4, c=colors[i])
    
    ax.set_xlabel(r'$\log (n_{\mathrm{H}}/\mathrm{cm}^{-3})$')
    ax.set_ylabel(r'$\log (\mathrm{Column Density Ratio})$')

    ax.legend(loc='right', bbox_to_anchor=(1.38, 0.5),
            fancybox=True, shadow=True, ncol=1)
    
###########################################
#### Utilities for fitting gas density ####
###########################################

def log_prior_hdens(log_hdens):
    
    '''
    Priors for an MCMC search. 
    '''
   
    # If the sampled density is within the CLOUDY limits
    # Avoid edges?
    if log_hdens_min<log_hdens<log_hdens_max:
        return np.log(10**log_hdens)
    return -np.inf

def log_likelihood_hdens(log_hdens, logN_ratio_dict, species_logN_interp):

    '''
    Likelihood function for comparing CLOUDY predicted column densities to the observed values from VP fit.
    If only some of the parameters need to be fit, the log_likelihood function can be overridden with a lambda function
    which calls this likelihood function with only some parameters being varied.

    log_hdens: gas density to be fit
    logN_ratio_dict: dictionary of measured column density ratios
    species_logN_interp: interpolated CLOUDY grid
    '''

    # Likelihood function
    ll = 0

    # Ignore first entry since it's HI
    for i in range(len(logN_ratio_dict)):
        
        # This is from VP fit
        ion_pair_str = list(logN_ratio_dict.keys())[i]

        ion_pair = ion_pair_str.split('/')
        
        # Get corresponding species
        s1 = ion_species_dict[ion_pair[0]]
        s2 = ion_species_dict[ion_pair[1]]

        # Observed column density ratio
        logN_ratio_str = logN_ratio_dict[ion_pair_str]
        
        # Get interpolated column density from CLOUDY grid
        y_bar = species_logN_interp[s1]([12, log_hdens, -3])[0]-species_logN_interp[s2]([12, log_hdens, -3])[0]

        # Based on detection or non-detection, compute the likelihood term
        
        # Detection
        if logN_ratio_str[0] != '<' and logN_ratio_str[0] != '>':
            
            logN_ratio_arr = np.array(logN_ratio_str.split(','), dtype=float)
            
            # Observed column density
            y = logN_ratio_arr[0]

            # Use max of lower and upper error for defining Gaussian distribution of column density
            sig_y = max(-logN_ratio_arr[1], logN_ratio_arr[2])

            # Gaussian likelihood
            ll += -.5*(y-y_bar)**2/sig_y**2

        # Upper limit
        elif logN_ratio_str[0] == '<':
            
            logN_ratio_arr = np.array(logN_ratio_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_ratio_arr[0]
            sig_y = logN_ratio_arr[1]

            y_range_min = -99 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y_range_min, y+y_range_step, y_range_step)

            # CDF, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))
            
        # Lower limit
        # NOTE: not implemented yet
        elif logN_ratio_str[0] == '>':

            logN_ratio_arr = np.array(logN_ratio_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_ratio_arr[0]
            sig_y = logN_ratio_arr[1]

            y_range_max = 99 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

            # "Q"-function, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))

    # Return log likelihood for MCMC
    return ll

##################################################################
#### Utilities for fitting column densities and metallicities ####
##################################################################

def get_cloud_size(logN_HI, log_hdens, species_logN_interp, log_metals=-3):

    logN_HII = species_logN_interp['H+']([logN_HI, log_hdens, log_metals])[0]
    N_H = 10**logN_HI + 10**logN_HII

    l = (N_H/10**log_hdens)*3.24078e-19*1e-3 # in kpc

    return l

###################################################################################
#### Utilities for constraining HI column density, gas density, and abundances ####
###################################################################################

def log_prior(params):
    
    '''
    Priors for an MCMC search. 

    params_dict: Dictionary of parameters being fitted. Will contain logN_HI, n_H, [O/H], and [X/H]
    '''

    # Grid parameters being varied
    logN_HI, log_hdens, log_metals, non_solar_dict = params
    
    # If the sampled density is within the CLOUDY limits
    # Avoid edges?
    if logN_HI_min<logN_HI<logN_HI_max and log_hdens_min<log_hdens<log_hdens_max and log_metals_min<log_metals<log_metals_max:

        relative_abund = True

        # Check if other elemental abundances are within bounds
        for k in list(non_solar_dict.keys()):
            relative_abund *= (log_metals_min<non_solar_dict[k]+log_metals<log_metals_max)

        if relative_abund == True:
            return np.log(10**logN_HI) + np.log(10**log_hdens)
        else:
            return -np.inf
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
    logN_HI, log_hdens, log_metals, non_solar_dict = params

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
        if s.split('+')[0] in non_solar_dict:
            y_bar += non_solar_dict[s.split('+')[0]]

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
            
            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_min = -10 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y_range_min, y+y_range_step, y_range_step)

            # CDF, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))
            
        # Lower limit
        # NOTE: not implemented yet
        elif logN_str[0] == '>':

            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_max = 21.5 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

            # "Q"-function, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))

    # Return log likelihood for MCMC
    return ll

#########################################################
#### Utilities for constraining a two phase solution ####
#########################################################

def log_prior_two_phase(params, species_logN_interp):
    
    '''
    Priors for an MCMC search. 

    params_dict: Dictionary of parameters being fitted. Will contain logN_HI, n_H, [O/H], and [X/H]
    '''

    # Grid parameters being varied
    logN_HI_p1, log_hdens_p1, log_metals_p1, non_solar_dict_p1, logN_HI_p2, log_hdens_p2, log_metals_p2, non_solar_dict_p2 = params
    
    # If the sampled density is within the CLOUDY limits
    # Avoid edges?
    if logN_HI_min<logN_HI_p1<logN_HI_max and log_hdens_min<log_hdens_p1<log_hdens_max and log_metals_min<log_metals_p1<log_metals_max:
        if logN_HI_min<logN_HI_p2<logN_HI_max and log_hdens_min<log_hdens_p2<log_hdens_max and log_metals_min<log_metals_p2<log_metals_max:

            relative_abund = True

            # Check if other elemental abundances are within bounds
            for k in list(non_solar_dict_p1.keys()):
                relative_abund *= (log_metals_min<non_solar_dict_p1[k]+log_metals_p1<log_metals_max)

            for k in list(non_solar_dict_p2.keys()):
                relative_abund *= (log_metals_min<non_solar_dict_p2[k]+log_metals_p2<log_metals_max)     

            if relative_abund == True:       
                if log_hdens_p1>log_hdens_p2:
                    if logN_HI_p1>logN_HI_p2:
                        l_p1 = get_cloud_size(logN_HI_p1, log_hdens_p1, species_logN_interp, log_metals_p1)
                        l_p2 = get_cloud_size(logN_HI_p2, log_hdens_p2, species_logN_interp, log_metals_p2)
                        if l_p1<l_p2<100:
                            return np.log(10**logN_HI_p1) + np.log(10**log_hdens_p1) + np.log(10**logN_HI_p2) + np.log(10**log_hdens_p2) # Convert log10 to linear, then take natural log
                        else:
                            return -np.inf
                    else:
                        return -np.inf
                else:
                    return -np.inf
            else:
                return -np.inf
        else:
            return -np.inf
    else:
        return -np.inf

def log_likelihood_two_phase(params, logN_dict, species_logN_interp):

    '''
    Likelihood function for comparing CLOUDY predicted column densities to the observed values from VP fit.
    If only some of the parameters need to be fit, the log_likelihood function can be overridden with a lambda function
    which calls this likelihood function with only some parameters being varied.

    params: parameters needed to generate CLOUDY predicted column densities
    logN_dict: dictionary of measured column densities from VP fit
    species_logN_interp: interpolated CLOUDY grid
    '''
    
    # Grid parameters being varied
    logN_HI_p1, log_hdens_p1, log_metals_p1, non_solar_dict_p1, logN_HI_p2, log_hdens_p2, log_metals_p2, non_solar_dict_p2 = params

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
        if ion != 'HI':
            y_bar_1 = species_logN_interp[s]([logN_HI_p1, log_hdens_p1, log_metals_p1])[0]
            y_bar_2 = species_logN_interp[s]([logN_HI_p2, log_hdens_p2, log_metals_p2])[0]
        else:
            y_bar_1 = logN_HI_p1
            y_bar_2 = logN_HI_p2

        # If there is departure from solar abundance, shift the predicted column density accordingly
        if s.split('+')[0] in non_solar_dict_p1:
            y_bar_1 += non_solar_dict_p1[s.split('+')[0]]

        if s.split('+')[0] in non_solar_dict_p2:
            y_bar_2 += non_solar_dict_p2[s.split('+')[0]]

        y_bar = np.log10(10**y_bar_1 + 10**y_bar_2)

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
            
            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_min = -10 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y_range_min, y+y_range_step, y_range_step)

            # CDF, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))
            
        # Lower limit
        # NOTE: not implemented yet
        elif logN_str[0] == '>':

            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_max = 21.5 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

            # "Q"-function, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))

    # Return log likelihood for MCMC
    return ll

#########################################################
#### Utilities for constraining a three phase solution ##
#########################################################

def log_prior_three_phase(params, species_logN_interp):
    
    '''
    Priors for an MCMC search. 

    params_dict: Dictionary of parameters being fitted. Will contain logN_HI, n_H, [O/H], and [X/H]
    '''

    # Grid parameters being varied
    logN_HI_p1, log_hdens_p1, log_metals_p1, non_solar_dict_p1, logN_HI_p2, log_hdens_p2, log_metals_p2, non_solar_dict_p2, logN_HI_p3, log_hdens_p3, log_metals_p3, non_solar_dict_p3 = params
    
    # If the sampled density is within the CLOUDY limits
    # Avoid edges?
    if logN_HI_min<logN_HI_p1<logN_HI_max and log_hdens_min<log_hdens_p1<log_hdens_max and log_metals_min<log_metals_p1<log_metals_max:
        if logN_HI_min<logN_HI_p2<logN_HI_max and log_hdens_min<log_hdens_p2<log_hdens_max and log_metals_min<log_metals_p2<log_metals_max:
            if logN_HI_min<logN_HI_p3<logN_HI_max and log_hdens_min<log_hdens_p3<log_hdens_max and log_metals_min<log_metals_p3<log_metals_max:

                relative_abund = True

                # Check if other elemental abundances are within bounds
                for k in list(non_solar_dict_p1.keys()):
                    relative_abund *= (log_metals_min<non_solar_dict_p1[k]+log_metals_p1<log_metals_max)

                for k in list(non_solar_dict_p2.keys()):
                    relative_abund *= (log_metals_min<non_solar_dict_p2[k]+log_metals_p2<log_metals_max)   

                for k in list(non_solar_dict_p3.keys()):
                    relative_abund *= (log_metals_min<non_solar_dict_p3[k]+log_metals_p3<log_metals_max)   

                if relative_abund == True:              

                    if log_hdens_p1>log_hdens_p2>log_hdens_p3:
                        if logN_HI_p1>logN_HI_p2>logN_HI_p3:
                            l_p1 = get_cloud_size(logN_HI_p1, log_hdens_p1, species_logN_interp, log_metals_p1)
                            l_p2 = get_cloud_size(logN_HI_p2, log_hdens_p2, species_logN_interp, log_metals_p2)
                            l_p3 = get_cloud_size(logN_HI_p3, log_hdens_p3, species_logN_interp, log_metals_p3)
                            if l_p1<l_p2<l_p3<100:
                                return np.log(10**logN_HI_p1) + np.log(10**log_hdens_p1) + np.log(10**logN_HI_p2) + np.log(10**log_hdens_p2) + np.log(10**logN_HI_p3) + np.log(10**log_hdens_p3)
                            else:
                                return -np.inf
                        else:
                            return -np.inf
                    else:
                        return -np.inf
                else:
                    return -np.inf
            else:
                return -np.inf
        else:
            return -np.inf
    else:
        return -np.inf

def log_likelihood_three_phase(params, logN_dict, species_logN_interp):

    '''
    Likelihood function for comparing CLOUDY predicted column densities to the observed values from VP fit.
    If only some of the parameters need to be fit, the log_likelihood function can be overridden with a lambda function
    which calls this likelihood function with only some parameters being varied.

    params: parameters needed to generate CLOUDY predicted column densities
    logN_dict: dictionary of measured column densities from VP fit
    species_logN_interp: interpolated CLOUDY grid
    '''
    
    # Grid parameters being varied
    logN_HI_p1, log_hdens_p1, log_metals_p1, non_solar_dict_p1, logN_HI_p2, log_hdens_p2, log_metals_p2, non_solar_dict_p2, logN_HI_p3, log_hdens_p3, log_metals_p3, non_solar_dict_p3 = params

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
        if ion != 'HI':
            y_bar_1 = species_logN_interp[s]([logN_HI_p1, log_hdens_p1, log_metals_p1])[0]
            y_bar_2 = species_logN_interp[s]([logN_HI_p2, log_hdens_p2, log_metals_p2])[0]
            y_bar_3 = species_logN_interp[s]([logN_HI_p3, log_hdens_p3, log_metals_p3])[0]
        else:
            y_bar_1 = logN_HI_p1
            y_bar_2 = logN_HI_p2
            y_bar_3 = logN_HI_p3

        # If there is departure from solar abundance, shift the predicted column density accordingly
        if s.split('+')[0] in non_solar_dict_p1:
            y_bar_1 += non_solar_dict_p1[s.split('+')[0]]

        if s.split('+')[0] in non_solar_dict_p2:
            y_bar_2 += non_solar_dict_p2[s.split('+')[0]]

        if s.split('+')[0] in non_solar_dict_p3:
            y_bar_3 += non_solar_dict_p3[s.split('+')[0]]

        y_bar = np.log10(10**y_bar_1 + 10**y_bar_2 + 10**y_bar_3)

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
            
            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_min = -10 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y_range_min, y+y_range_step, y_range_step)

            # CDF, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))
            
        # Lower limit
        # NOTE: not implemented yet
        elif logN_str[0] == '>':

            logN_arr = np.array(logN_str[1:].split(','), dtype=float)

            # Isolate the lower limit and sigma
            y = logN_arr[0]
            sig_y = 0.14

            y_range_max = 21.5 # Should extend to infinity, ideally
            y_range_step = 0.05

            y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

            # "Q"-function, marginalize over reported values
            ll += np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/sig_y**2)))

    # Return log likelihood for MCMC
    return ll

###########################################
#### Utilities to interpret posteriors ####
###########################################

def get_quantiles(dist):

    med = np.median(dist)
    sig_lo = np.percentile(dist, 100*(1-cdf_1sig))-med
    sig_hi = np.percentile(dist, 100*cdf_1sig)-med

    print(np.round(med, 2), np.round(sig_lo, 2), np.round(sig_hi,2))

def log_to_linear_PDF(flatchain, bins=250):

    '''
    Function to process the marginalized MCMC distribution of a log variable
    The distribution of walkers actually mimics the PDF for the quantity, not its log
    '''

    # First get the "incorrect" PDF - this is essentially getting the envelope of the walker distribution
    pdf, bin_edges = np.histogram(flatchain, bins=bins, density=True)
    # Get linear version for the points of the envelope derived above
    X = 10**(.5*(bin_edges[1:]+bin_edges[:-1]))
    # Now viewing the PDF on the linear scale (i.e. with X as the independent variable, not logX), 
    pdf = pdf/integrate.trapz(x=X, y=pdf)
    # numerically perform cumulative integration to get the CDF
    cdf = integrate.cumtrapz(x=X, y=pdf, initial=0)/integrate.trapz(x=X, y=pdf)
    # Also invert the CDF and get its interpolation for computing confidence levels
    cdf_inv_interp = interpolate.interp1d(x=cdf, y=X)

    return X, pdf, cdf, cdf_inv_interp

