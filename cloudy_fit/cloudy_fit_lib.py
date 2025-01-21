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
            'MgI': 0.1, 
            'AlII': 5.986,
            'MgII': 7.646,
            'FeII': 7.87,
            'SiII': 8.151,
            'SII': 10.36,
            'CII': 11.26,
            'OII': 13.618,
            'NII': 14.534,
            'SiIII': 16.345,
            'AlIII': 18.826,
            'SIII': 23.33,
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
                    'MgI': 'Mg',
                    'AlII': 'Al+',
                    'MgII': 'Mg+',
                    'FeII': 'Fe+',
                    'SiII': 'Si+',
                    'SII': 'S+',
                    'CII': 'C+',
                    'OII': 'O+',
                    'NII': 'N+',
                    'SiIII': 'Si+2',
                    'AlIII': 'Al+2',
                    'SIII': 'S+2',
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

# Full list of ionic species
species_names_ions = ['#column density H', 'H+', 
                    'He', 'He+', 'He+2', 
                    'Li', 'Li+', 'Li+2', 'Li+3', 
                    'Be', 'Be+', 'Be+2', 'Be+3', 'Be+4', 
                    'B', 'B+', 'B+2', 'B+3', 'B+4', 'B+5', 
                    'C', 'C+', 'C+2', 'C+3', 'C+4', 'C+5', 'C+6', 
                    'N', 'N+', 'N+2', 'N+3', 'N+4', 'N+5', 'N+6', 'N+7',  
                    'O', 'O+', 'O+2', 'O+3', 'O+4', 'O+5', 'O+6', 'O+7', 'O+8',
                    'F', 'F+', 'F+2', 'F+3', 'F+4', 'F+5', 'F+6', 'F+7', 'F+8', 'F+9',
                    'Ne', 'Ne+', 'Ne+10', 'Ne+2', 'Ne+3', 'Ne+4', 'Ne+5', 'Ne+6', 'Ne+7', 'Ne+8', 'Ne+9', 
                    'Na', 'Na+', 'Na+10', 'Na+11', 'Na+2', 'Na+3', 'Na+4', 'Na+5', 'Na+6', 'Na+7', 'Na+8', 'Na+9', 
                    'Mg', 'Mg+', 'Mg+10', 'Mg+11', 'Mg+12', 'Mg+2', 'Mg+3', 'Mg+4', 'Mg+5', 'Mg+6', 'Mg+7', 'Mg+8', 'Mg+9', 
                    'Al', 'Al+', 'Al+10', 'Al+11', 'Al+12', 'Al+13', 'Al+2', 'Al+3', 'Al+4', 'Al+5', 'Al+6', 'Al+7', 'Al+8', 'Al+9', 
                    'Si', 'Si+', 'Si+10', 'Si+11', 'Si+12', 'Si+13', 'Si+14', 'Si+2', 'Si+3', 'Si+4', 'Si+5', 'Si+6', 'Si+7', 'Si+8', 'Si+9', 
                    'P', 'P+', 'P+10', 'P+11', 'P+12', 'P+13', 'P+14', 'P+15', 'P+2', 'P+3', 'P+4', 'P+5', 'P+6', 'P+7', 'P+8', 'P+9', 
                    'S', 'S+', 'S+10', 'S+11', 'S+12', 'S+13', 'S+14', 'S+15', 'S+16', 'S+2', 'S+3', 'S+4', 'S+5', 'S+6', 'S+7', 'S+8', 'S+9', 
                    'Cl', 'Cl+', 'Cl+10', 'Cl+11', 'Cl+12', 'Cl+13', 'Cl+14', 'Cl+15', 'Cl+16', 'Cl+17', 'Cl+2', 'Cl+3', 'Cl+4', 'Cl+5', 'Cl+6', 'Cl+7', 'Cl+8', 'Cl+9', 
                    'Ar', 'Ar+', 'Ar+10', 'Ar+11', 'Ar+12', 'Ar+13', 'Ar+14', 'Ar+15', 'Ar+16', 'Ar+17', 'Ar+18', 'Ar+2', 'Ar+3', 'Ar+4', 'Ar+5', 'Ar+6', 'Ar+7', 'Ar+8', 'Ar+9', 
                    'K', 'K+', 'K+10', 'K+11', 'K+12', 'K+13', 'K+14', 'K+15', 'K+16', 'K+17', 'K+18', 'K+19', 'K+2', 'K+3', 'K+4', 'K+5', 'K+6', 'K+7', 'K+8', 'K+9', 
                    'Ca', 'Ca+', 'Ca+10', 'Ca+11', 'Ca+12', 'Ca+13', 'Ca+14', 'Ca+15', 'Ca+16', 'Ca+17', 'Ca+18', 'Ca+19', 'Ca+2', 'Ca+20', 'Ca+3', 'Ca+4', 'Ca+5', 'Ca+6', 'Ca+7', 'Ca+8', 'Ca+9', 
                    'Sc', 'Sc+', 'Sc+10', 'Sc+11', 'Sc+12', 'Sc+13', 'Sc+14', 'Sc+15', 'Sc+16', 'Sc+17', 'Sc+18', 'Sc+19', 'Sc+2', 'Sc+20', 'Sc+21', 'Sc+3', 'Sc+4', 'Sc+5', 'Sc+6', 'Sc+7', 'Sc+8', 'Sc+9', 
                    'Ti', 'Ti+', 'Ti+10', 'Ti+11', 'Ti+12', 'Ti+13', 'Ti+14', 'Ti+15', 'Ti+16', 'Ti+17', 'Ti+18', 'Ti+19', 'Ti+2', 'Ti+20', 'Ti+21', 'Ti+22', 'Ti+3', 'Ti+4', 'Ti+5', 'Ti+6', 'Ti+7', 'Ti+8', 'Ti+9', 
                    'V', 'V+', 'V+10', 'V+11', 'V+12', 'V+13', 'V+14', 'V+15', 'V+16', 'V+17', 'V+18', 'V+19', 'V+2', 'V+20', 'V+21', 'V+22', 'V+23', 'V+3', 'V+4', 'V+5', 'V+6', 'V+7', 'V+8', 'V+9', 
                    'Cr', 'Cr+', 'Cr+10', 'Cr+11', 'Cr+12', 'Cr+13', 'Cr+14', 'Cr+15', 'Cr+16', 'Cr+17', 'Cr+18', 'Cr+19', 'Cr+2', 'Cr+20', 'Cr+21', 'Cr+22', 'Cr+23', 'Cr+24', 'Cr+3', 'Cr+4', 'Cr+5', 'Cr+6', 'Cr+7', 'Cr+8', 'Cr+9', 
                    'Mn', 'Mn+', 'Mn+10', 'Mn+11', 'Mn+12', 'Mn+13', 'Mn+14', 'Mn+15', 'Mn+16', 'Mn+17', 'Mn+18', 'Mn+19', 'Mn+2', 'Mn+20', 'Mn+21', 'Mn+22', 'Mn+23', 'Mn+24', 'Mn+25', 'Mn+3', 'Mn+4', 'Mn+5', 'Mn+6', 'Mn+7', 'Mn+8', 'Mn+9', 
                    'Fe', 'Fe+', 'Fe+10', 'Fe+11', 'Fe+12', 'Fe+13', 'Fe+14', 'Fe+15', 'Fe+16', 'Fe+17', 'Fe+18', 'Fe+19', 'Fe+2', 'Fe+20', 'Fe+21', 'Fe+22', 'Fe+23', 'Fe+24', 'Fe+25', 'Fe+26', 'Fe+3', 'Fe+4', 'Fe+5', 'Fe+6', 'Fe+7', 'Fe+8', 'Fe+9', 
                    'Co', 'Co+', 'Co+10', 'Co+11', 'Co+12', 'Co+13', 'Co+14', 'Co+15', 'Co+16', 'Co+17', 'Co+18', 'Co+19', 'Co+2', 'Co+20', 'Co+21', 'Co+22', 'Co+23', 'Co+24', 'Co+25', 'Co+26', 'Co+27', 'Co+3', 'Co+4', 'Co+5', 'Co+6', 'Co+7', 'Co+8', 'Co+9', 
                    'Ni', 'Ni+', 'Ni+10', 'Ni+11', 'Ni+12', 'Ni+13', 'Ni+14', 'Ni+15', 'Ni+16', 'Ni+17', 'Ni+18', 'Ni+19', 'Ni+2', 'Ni+20', 'Ni+21', 'Ni+22', 'Ni+23', 'Ni+24', 'Ni+25', 'Ni+26', 'Ni+27', 'Ni+28', 'Ni+3', 'Ni+4', 'Ni+5', 'Ni+6', 'Ni+7', 'Ni+8', 'Ni+9', 
                    'Cu', 'Cu+', 'Cu+10', 'Cu+11', 'Cu+12', 'Cu+13', 'Cu+14', 'Cu+15', 'Cu+16', 'Cu+17', 'Cu+18', 'Cu+19', 'Cu+2', 'Cu+20', 'Cu+21', 'Cu+22', 'Cu+23', 'Cu+24', 'Cu+25', 'Cu+26', 'Cu+27', 'Cu+28', 'Cu+29', 'Cu+3', 'Cu+4', 'Cu+5', 'Cu+6', 'Cu+7', 'Cu+8', 'Cu+9', 
                    'Zn', 'Zn+', 'Zn+10', 'Zn+11', 'Zn+12', 'Zn+13', 'Zn+14', 'Zn+15', 'Zn+16', 'Zn+17', 'Zn+18', 'Zn+19', 'Zn+2', 'Zn+20', 'Zn+21', 'Zn+22', 'Zn+23', 'Zn+24', 'Zn+25', 'Zn+26', 'Zn+27', 'Zn+28', 'Zn+29', 'Zn+3', 'Zn+30', 'Zn+4', 'Zn+5', 'Zn+6', 'Zn+7', 'Zn+8', 'Zn+9']

# Number densities of various elements in the Sun relative to hydrogen
solar_rel_dens_dict = {'H': 1.0,
                    'He': 0.1,
                    'Li': 2.04e-09,
                    'Be': 2.63e-11,
                    'Bo': 6.17e-10,
                    'C': 0.000245,
                    'N': 8.51e-05,
                    'O': 0.00049,
                    'F': 3.02e-08,
                    'Ne': 0.0001,
                    'Na': 2.14e-06,
                    'Mg': 3.47e-05,
                    'Al': 2.95e-06,
                    'Si': 3.47e-05,
                    'P': 3.2e-07,
                    'S': 1.84e-05,
                    'Cl': 1.91e-07,
                    'Ar': 2.51e-06,
                    'K': 1.32e-07,
                    'Ca': 2.29e-06,
                    'Sc': 1.48e-09,
                    'Ti': 1.05e-07,
                    'V': 1e-08,
                    'Cr': 4.68e-07,
                    'Mn': 2.88e-07,
                    'Fe': 2.82e-05,
                    'Co': 8.32e-08,
                    'Ni': 1.78e-06,
                    'Cu': 1.62e-08,
                    'Zn': 3.98e-08}

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

################################################################
#### Utilities for plotting column densities and linewidths ####
################################################################

def plot_linewidth_obs(b_dict, fig = None, ax = None, label_ions = True, gray_out = [], 
                       scatter_out = [], scatter_pos = 0.5,
                       fs=16):

    '''
    Method to plot observed linewidths from VP fit

    b_dict: Dictionary of column densities to be plotted, contains significant detections as well as non-detections (upper/ lower limits). 
    '''

    create_fig_ax = False
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, figsize=(4,4))
        create_fig_ax = True

    # Order the ions according to their ionization potential
    # NOTE: the assumption is that the dictionary of ionization potentials is already sorted. If not, sort it
    ions_ordered = [s for s in list(IP_dict.keys()) if s in list(b_dict.keys())]

    for i in range(len(ions_ordered)):
        
        ion = ions_ordered[i]

        if ion in gray_out:
            c='darkgray'
        else:
            c='black'
        
        b_str = b_dict[ion]
        
        # Detection
        if b_str[0] != '<' and b_str[0] != '>':
            b_arr = np.array(b_str.split(','), dtype=float)

            if ion in scatter_out:
                ax.scatter(i, b_arr[0], s=10, color=c, facecolor='none')
                if label_ions == True:
                    ax.text(x=i, y=b_arr[0]+scatter_pos, s=ion, color=c, 
                    horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
            else:
                ax.scatter(i, b_arr[0], s=3, color=c)
                ax.errorbar(x=i, y=b_arr[0], yerr=[[-b_arr[1]],[b_arr[2]]], color=c, linestyle='None',
                        fmt='o', markersize=3, capsize=4)
                if label_ions == True:
                    ax.text(x=i, y=b_arr[0]+b_arr[2], s=ion, color=c, 
                    horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
        
        # Upper limit
        elif b_str[0] == '<':
            b_lim = float(b_str[1:])
            ax.errorbar(x=i, y=b_lim, yerr=3, uplims=True, color=c, fmt='o', markersize=3)
            if label_ions == True:
                ax.text(x=i, y=b_lim, s=ion, color=c, 
                horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
        
        # Lower limit
        elif b_str[0] == '>':
            b_arr = np.array(b_str[1:].split(','), dtype=float)
            ax.errorbar(x=i, y=b_arr[0], yerr=3, lolims=True, color=c, fmt='o', markersize=3)
            if label_ions == True:
                ax.text(x=i, y=b_arr[0], s=ion, color=c, 
                horizontalalignment='center', verticalalignment='top', fontsize=fs)

    # Turn off ticks and label axes
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_ylabel(r'$b_{\mathrm{ion}}/\mathrm{km \ s}^{-1}$')
    ax.set_xlabel(r'Increasing Ionization Potential $\rightarrow$')

    # Set a limit on the y-axis
    ax.set_ylim(0,40)

    # Return figure object if one was created
    if create_fig_ax == True:
        return fig, ax

def plot_column_densities_obs(logN_dict, fig = None, ax = None, gray_out = [], label_ions=True, fs=16, dy=0.2, c_dy=1.5):

    '''
    Method to plot observed column densities of species from VP fit

    logN_dict: Dictionary of column densities to be plotted, contains significant detections as well as non-detections (upper/ lower limits).
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
            ax.errorbar(x=i, y=logN_arr[0], yerr=[[-logN_arr[1]],[logN_arr[2]]], color=c, linestyle='None',
                    fmt='o', markersize=3, capsize=4)
            if label_ions==True:
                ax.text(x=i, y=logN_arr[0]+logN_arr[2], s=ion, color=c, 
                horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
        
        # Upper limit
        elif logN_str[0] == '<':
            logN_lim = float(logN_str[1:])
            _, caps, _ = ax.errorbar(x=i, y=logN_lim, yerr=dy, uplims=True, color=c, fmt='o', markersize=3, elinewidth=0, markerfacecolor='None') # Make marker and spear
            caps[0].set_fillstyle('none')
            ax.errorbar(x=i, y=logN_lim, xerr=None, yerr=[[c_dy*dy],[0]], color=c, fmt='o', markersize=0, capsize=0) # Connect them
            if label_ions==True:
                ax.text(x=i, y=logN_lim, s=ion, color=c, 
                horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
        
        # Lower limit
        # NOT implemented yet
        elif logN_str[0] == '>':
            logN_lim = float(logN_str[1:])
            _, caps, _ = ax.errorbar(x=i, y=logN_lim, yerr=dy, lolims=True, color=c, fmt='o', markersize=3, elinewidth=0, markerfacecolor='None') # Make marker and spear
            caps[0].set_fillstyle('none')
            ax.errorbar(x=i, y=logN_lim, xerr=None, yerr=[[0],[c_dy*dy]], color=c, fmt='o', markersize=0, capsize=0) # Connect them
            if label_ions==True:
                ax.text(x=i, y=logN_arr[0], s=ion, color=c, 
                        horizontalalignment='center', verticalalignment='top', fontsize=fs)

    # Turn off ticks and label axes
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_ylabel(r'$\log (N_{\mathrm{ion}}/\mathrm{cm}^{-2})$')
    ax.set_xlabel(r'Increasing Ionization Potential $\rightarrow$')

    # Set a limit on the y-axis
    ax.set_ylim(10,18)

    # Return figure object if one was created
    if create_fig_ax == True:
        return fig, ax

def predict_col_dens(logN_dict, logN_HI, log_hdens, log_metals, species_logN_interp, X_alpha_dict = {}):

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
        logN_s = species_logN_interp[s]((logN_HI, log_hdens, log_metals))

        # If there is departure from solar abundance, shift the predicted column density accordingly
        # s.split('+')[0] is supposed to be the element name. This won't work for hydrogen, or helium, but they're not metals anyway
        if s.split('+')[0] in X_alpha_dict:
            logN_s += X_alpha_dict[s.split('+')[0]]

        # Get interpolated column density from CLOUDY grid
        logN_species_test.append(logN_s)

    return np.array(logN_species_test)

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
    err_lo_logN_ratio = -np.sqrt(err_lo_logN1**2 + err_hi_logN2**2) # Put negative sign for consistency w/ plotting functions
    err_hi_logN_ratio = np.sqrt(err_hi_logN1**2 + err_lo_logN2**2)  
    
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
                    log_hdens_min = -5, log_hdens_max = 1, log_hdens_grid_size = 5000,
                    colors = ['black', 'darkgray', 'firebrick', 'tan', 'deepskyblue', 
                            'cyan', 'green', 'yellowgreen', 'darkmagenta', 'orchid',
                            'darkorange', 'orangered', 'goldenrod', 'indigo', 'blueviolet',
                            'crimson'],
                    linestyles = ['-', '-', '-', '-', '-', 
                            '-', '-', '-', '-', '-',
                            '-', '-', '-', '-', '-',
                            '-'], label=True):

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
    

    # Plot the predicted column density ratios
    for i in range(len(logN_ratio_pred)):
        
        ion_pair_str = list(logN_ratio_pred.keys())[i]
        
        # Plot model predictions
        ax.plot(log_hdens_grid, logN_ratio_pred[ion_pair_str], label=ion_pair_str, c=colors[i], ls=linestyles[i])
        
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
            
        ax.plot(log_hdens_grid*mask, logN_ratio_pred[ion_pair_str]*mask, lw=4, c=colors[i], ls=linestyles[i])
    
    ax.set_xlabel(r'$\log (n_{\mathrm{H}}/\mathrm{cm}^{-3})$')
    ax.set_ylabel(r'$\log (\mathrm{Column Density Ratio})$')

    if label==True:
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
        return 0.
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

            # Use avg. of lower and upper error for defining Gaussian distribution of column density
            sig_y = .5*(-logN_ratio_arr[1]+logN_ratio_arr[2]) # max(-logN_ratio_arr[1], logN_ratio_arr[2])

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

def get_cloud_size(logN_HI, log_hdens, species_logN_interp, log_metals):

    logN_HII = species_logN_interp['H+']((logN_HI, log_hdens, log_metals))
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
            return 0.
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

            # Use avg. of lower and upper error for defining Gaussian distribution of column density
            sig_y = .5*(-logN_arr[1]+logN_arr[2])  #max(-logN_arr[1], logN_arr[2])

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
                        # Cloud sizes
                        l_p1 = get_cloud_size(logN_HI_p1, log_hdens_p1, species_logN_interp, log_metals_p1)
                        l_p2 = get_cloud_size(logN_HI_p2, log_hdens_p2, species_logN_interp, log_metals_p2)
                        # Cloud size limits for p2
                        N_H_sonic = 1e17
                        l_p2_S = 0.03*(10**log_hdens_p2/1e-3)**-1 # in kpc
                        l_p2_J = 40*(10**log_hdens_p2/1e-3)**-0.5 # also in kpc
                        if l_p1<l_p2 and l_p2_S<l_p2<l_p2_J: # Denser phase more compact, diffuse phase stable
                            return np.log(10**logN_HI_p1) + np.log(10**logN_HI_p2) # Convert log10 to linear, then take natural log
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

            # Use avg of lower and upper error for defining Gaussian distribution of column density
            sig_y = .5*(-logN_arr[1]+logN_arr[2]) # max(-logN_arr[1], logN_arr[2])

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
                                return np.log(10**logN_HI_p1) + np.log(10**logN_HI_p2) + np.log(10**logN_HI_p3)
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

            # Use avg. of lower and upper error for defining Gaussian distribution of column density
            sig_y = .5*(-logN_arr[1] + logN_arr[2]) # max(-logN_arr[1], logN_arr[2])

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

def get_logN_residuals(logN_dict, logN_species_med):

    '''
    Function to get residuals of best fit ionization model relative to measurements

    logN_dict: dictionary of measurements
    logN_species_med: median prediction of best-fit model
    '''

    # Order measurement dictionary ions by ionization potential
    ions_ordered = [s for s in list(IP_dict.keys()) if s in list(logN_dict.keys())]

    # Dictionary for residuals
    logN_res_dict = {}

    for i in range(len(ions_ordered)):
        
        # Get measurement for ion
        ion = ions_ordered[i]    
        logN_str = logN_dict[ion]
        
        # In case of upper limit
        if logN_str[0] == '<':
            logN_up = float(logN_str[1:]) # Get the upper limit
            # Just subtract model prediction from upper limit
            # Round to one decimal place
            logN_res_dict[ion] = '<' + str(np.round(logN_up-logN_species_med[i],1))

        # In case of lower limit
        elif logN_str[0] == '>':
            logN_lo = float(logN_str[1:]) # Get the upper limit
            # Just subtract model prediction from upper limit
            # Round to one decimal place
            logN_res_dict[ion] = '>' + str(np.round(logN_lo-logN_species_med[i],1))
            
        # Detection
        else:

            # Get the measurement value, and errorbars
            logN_str_split = logN_str.split(',')
            logN_med = float(logN_str_split[0])
            dlogN_lo = -float(logN_str_split[1]) # Account for negative sign
            dlogN_hi = float(logN_str_split[2])
            
            # Define median of residual
            logN_res = logN_med-logN_species_med[i]
                
            # Build string for residual
            logN_res_str = '{}, -{}, {}'.format(np.round(logN_res,2), np.round(dlogN_lo,2), np.round(dlogN_hi,2))
            # Append to the dictionary
            logN_res_dict[ion] = logN_res_str

    return logN_res_dict

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

#################################
#### Utilities for TDP fitting ##
#################################

def scat_logN_ratio(ax, p_x, p_y, logN_ratio_dict):
    
    '''
    Function to make a scatter plot of measured column density ratios

    ax: axes object onto which scatter the column density ratios
    p_x: column density ratio for the x-axis
    p_y: column density ratio for the y-axis
    logN_ratio_dict: dictionary to column density ratios
    '''

    # Isolate ion pairs from dictionary
    s_x = logN_ratio_dict[p_x].split(',')
    x, dx_lo, dx_hi = float(s_x[0]), -float(s_x[1]), float(s_x[2])
    
    s_y = logN_ratio_dict[p_y].split(',')
    y, dy_lo, dy_hi = float(s_y[0]), -float(s_y[1]), float(s_y[2])
    
    # Plot with asymmetric error-bars
    ax.errorbar(x, y, xerr=[[dx_lo],[dx_hi]],yerr=[[dy_lo],[dy_hi]],
                     marker='*', markersize=15, markerfacecolor='goldenrod', capsize=3, alpha=.5)
    

def plot_logN_ratio_track(ax, ion1, ion2, ion3, ion4, logX_dict_TDP_interp,
                       log_metals_plot, log_hdens_plot,
                       logT_plot_min, logT_plot_max, dlogT_plot,
                       logT_mark_min, logT_mark_max, dlogT_mark,
                       xmin, xmax, ymin, ymax,
                       ls, horz_al='left', vert_al='top', 
                       logT_special=None):
    

    '''
    Function to plot TDP model grids to compare with measurement

    ion1: denominator ion for x-axis
    ion2: numerator ion for x-axis
    ion3: denominator ion for y-axis
    ion4: numerator ion for y-axis
    logX_dict_TDP_interp: interpolated grid of TDP ion fractions
    log_metals_plot: metallicity for plotting
    log_hdens_plot: density for plotting
    logT_plot_min: minimum temperature for plotting the track
    logT_plot_max: maximum temperature for plotting the track
    dlogT_plot: step size in temperature for plotting track
    logT_mark_min: minimum temperature to mark along track
    logT_mark_max: maximum temperature to mark along track
    dlogT_mark: step size in temperature when marking track
    xmin: minimum x-value for plot
    xmax: maximum x-value for plot
    ymin: minimum y-value for plot
    ymax: maxmimum y-value for plot
    ls: linestyle for track
    horz_al: horizontal alignment selection for track markers in temperature
    vert_al: vertical alignment selection for track markers in temperature
    '''
    
    # Set of temperature for plotting the track
    logT_plot_arr = np.arange(logT_plot_min,logT_plot_max,dlogT_plot)

    # Model access key
    k = (log_metals_plot,log_hdens_plot,logT_plot_arr)
    
    # Access ion fractions at grid points
    x1 = logX_dict_TDP_interp[ion1](k)
    x2 = logX_dict_TDP_interp[ion2](k)
    
    y1 = logX_dict_TDP_interp[ion3](k)
    y2 = logX_dict_TDP_interp[ion4](k)
    
    # Check if all ion fractions have not fallen off
    # -5 is a numerical tolerance I found by plotting the ion fractions
    idx = ((x1>-5)&(x2>-5)&(y1>-5)&(y2>-5))
    
    # Get elements
    if ion1=='HI':
        ex1 = 'H'
    else: 
        ex1 = ion_species_dict[ion1].split('+')[0]
    
    if ion2=='HI':
        ex2 = 'H'
    else:
        ex2 = ion_species_dict[ion2].split('+')[0]

    if ion3 == 'HI':
        ey1 = 'H'
    else:
        ey1 = ion_species_dict[ion3].split('+')[0]

    if ion4 == 'HI':
        ey2 = 'H'
    else:
        ey2 = ion_species_dict[ion4].split('+')[0]

    # Plot the track, label its density
    #print(ey1, ey2, np.log10(solar_rel_dens_dict[ey2]/solar_rel_dens_dict[ey1]))

    dx21 = x2[idx]-x1[idx]+np.log10(solar_rel_dens_dict[ex2]/solar_rel_dens_dict[ex1])
    dy21 = y2[idx]-y1[idx]+np.log10(solar_rel_dens_dict[ey2]/solar_rel_dens_dict[ey1])

    ax.plot(dx21, 
            dy21, color='black', linestyle=ls,
            label=r'$\log (n_\mathrm{{H}}/\mathrm{{cm}}^{{-3}}) = {0:.1f}$'.format(log_hdens_plot))
    
    # Label track with temperature
    for logT in np.arange(logT_mark_min,logT_mark_max,dlogT_mark):
        
        # Access specific grid point
        k0 = (log_metals_plot,log_hdens_plot,logT)

        x10 = logX_dict_TDP_interp[ion1](k0)
        x20 = logX_dict_TDP_interp[ion2](k0)
        dx210 = (x20-x10)+np.log10(solar_rel_dens_dict[ex2]/solar_rel_dens_dict[ex1])

        y10 = logX_dict_TDP_interp[ion3](k0)
        y20 = logX_dict_TDP_interp[ion4](k0)
        dy210 = (y20-y10)+np.log10(solar_rel_dens_dict[ey2]/solar_rel_dens_dict[ey1])
        
        s = str(np.round(logT,1))
        
        # Check if all ion fractions are valid and within bounds for the plot
        if x10>-5 and x20>-5 and y10>-5 and y20>-5 and xmin<dx210<xmax and ymin<dy210<ymax:
    
            if logT == logT_special:
                c = 'red'
            else:
                c = 'black'
            ax.scatter(dx210, dy210, color=c, facecolor='none')
            ax.text(dx210, dy210, s, fontsize=10, color=c, horizontalalignment=horz_al, verticalalignment=vert_al)

    # Set plot bounds
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

def get_logN_HI(ion, logN_ion, log_metals, log_hdens, logT_arr, logX_dict_TDP_interp):

    '''
    Function to predict HI column density given an oxygen ion column density

    ion: name of oxygen ion
    logN_ion: column density of oxygen ion
    log_metals: metallicity
    log_hdens: density
    logT_arr: temperature grid
    logX_dict_TDP_interp: interpolated grid of TDP ion fractions
    '''

    # Access key for models
    k = (log_metals,log_hdens,logT_arr)
    # Filter out numerically zero ion fractions
    idx = logX_dict_TDP_interp[ion](k)>-5

    # Generate logN_HI prediction at valid grid points
    logN_HI = logN_ion-log_metals-np.log10(solar_rel_dens_dict['O'])-logX_dict_TDP_interp[ion](k)[idx]+logX_dict_TDP_interp['HI'](k)[idx]

    # Return filtered temperatures and logN_HI
    return logT_arr[idx], logN_HI

def predict_col_dens_TDP(ion, log_metals, log_hdens, logT, logN_HI, logX_dict_TDP_interp):
    
    '''
    Function to produce ionic column density given HI column density

    ion: metal ion of interest
    log_metals: metallicity
    log_hdens: density
    logT: temperature
    logN_HI: HI column density
    logX_dict_TDP_interp: interpolated TDP ion fraction grid
    '''

    if ion == 'HI':
        return logN_HI
    else:
        # Model access key
        k = (log_metals,log_hdens,logT)
        # Element of ion
        elem = ion_species_dict[ion].split('+')[0]

        logN = logN_HI-logX_dict_TDP_interp['HI'](k)+np.log10(solar_rel_dens_dict[elem])+logX_dict_TDP_interp[ion](k)

        # Ionic column density for metals
        if elem != 'He':
            logN += log_metals
        
        return logN

def predict_col_dens_model_TDP(logN_dict, log_metals, log_hdens, logT, logN_HI, logX_dict_TDP_interp, C_O=0, N_O=0):

    '''
    Function to generate column density predictions for ions with measurements

    logN_dict: dictionary of column density measurements
    log_metals: metallicity
    log_hdens: density
    logT: temperature
    logN_HI: HI column density
    logX_dict_TDP_interp: interpolated TDP grid
    C_O: carbon relative abundance
    N_O: nitrogen relative abundance
    '''
    # Order ions according to ionization potential
    ions_ordered = [s for s in list(IP_dict.keys()) if s in list(logN_dict.keys())]

    logN_model_arr = np.zeros(len(ions_ordered))

    for i in range(len(ions_ordered)):

        # Ion and corresponding element
        ion = ions_ordered[i]
        elem = ion_species_dict[ion].split('+')[0]
        
        logN = predict_col_dens_TDP(ion,log_metals,log_hdens,logT,logN_HI,logX_dict_TDP_interp)        
        if elem == 'C':
            logN += C_O
        if elem == 'N':
            logN += N_O

        logN_model_arr[i] = logN

    return logN_model_arr

def compute_ll(logN_str, y_bar):

    '''
    Function to compute log likelihood for a data point

    logN_str: string representation of datapoint
    y_bar: model prediction
    ''' 

    if logN_str[0]=='<': # Upper limit
        y = float(logN_str[1:])
        dy = 1/(3*np.log(10)) # 3-sigma upper limit
        
        y_range_min = -10 # Should extend to negative infinity, ideally
        y_range_step = 0.05

        y_range = np.arange(y_range_min, y+y_range_step, y_range_step)

        # CDF, marginalize over reported values
        ll = np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/dy**2)))
        
    elif logN_str[0]=='>': # Lower limit
        y = float(logN_str[1:])
        dy = 1/(3*np.log(10)) # 3-sigma lower limit
        
        y_range_max = 21.5 # Should extend to infinity, ideally
        y_range_step = 0.05

        y_range = np.arange(y, y_range_max+y_range_step, y_range_step)

        # "Q"-function, marginalize over reported values
        ll = np.log(integrate.simpson(x=y_range, y=np.exp(-.5*(y_range-y_bar)**2/dy**2)))

    else: # Measurement
        y = float(logN_str.split(',')[0])
        dy = max(-float(logN_str.split(',')[1]), float(logN_str.split(',')[2]))             
        ll = -0.5*((y-y_bar)**2/dy**2) # Simple chi-sq for this situation

    return ll

def get_logl_TDP(log_metals, log_hdens, logT, logN_HI, logX_dict_TDP_interp):

    '''
    Function to return cloud sizes

    log_metals: metallicity
    log_hdens: density
    logT: temperature
    logN_HI: HI column density
    logX_dict_TDP_interp: interpolated TDP ion fraction grid
    '''
    
    # Model access key
    k = (log_metals, log_hdens, logT)
    # kpc correction
    logl = logN_HI-log_hdens-logX_dict_TDP_interp['HI'](k)+np.log10(3.24078e-19*1e-3)
    
    return logl