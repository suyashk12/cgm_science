import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from astropy import constants
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
from pathlib import Path
import scipy.optimize as opt
from scipy.interpolate import interp1d
import scipy.stats as st

rootdir = '/Users/thepoetoftwilight/Documents/CUBS/Data/CONTACT/STIS/'

# Reference FITS file for saving stuff
fits_ref = fits.open(rootdir+'PG1522+101_E230M_new.fits')

######################
# Poisson statistics #
######################

# Poisson distribution
def poisson(n, lam):
    
    '''
    Returns the Poisson discrete PDF
    '''

    return lam**n*np.exp(-lam)/np.math.factorial(n)

# Equation for upper bound
def lam_u_eqn(lam_u, n, C):
    
    '''
    Equation whose roots return upper bound for n observations at a 
    confidence interval C (accounting for small number statistics). 
    Taken from Gehrels 1986.
    '''

    # Generate array of x values
    x_arr = np.arange(0,n+1)
    
    poisson_x = np.array([poisson(x, lam_u) for x in x_arr])
    
    return np.sum(poisson_x)-(1-C)

# Equation for lower bound
def lam_l_eqn(lam_l, n, C):
    
    '''
    Equation whose roots return lower bound for n observations at a 
    confidence interval C (accounting for small number statistics). 
    Taken from Gehrels 1986.    
    '''

    # Generate array of x values
    x_arr = np.arange(0,n)
    
    poisson_x = np.array([poisson(x, lam_l) for x in x_arr])
    
    return np.sum(poisson_x)-C

# Approximate solutions for seeding
def lam_u_approx(n, C):
    
    '''
    Approximate solution to upper bound equation, used for seeding exact solution.
    Taken from Gehrels 1986.
    '''

    # First compute number of Gaussian sigma
    S = st.norm.ppf(C)
    
    # Now use the formula
    return n + S*np.sqrt(n+1) + ((S**2+2)/3)

def lam_l_approx(n, C):
    
    '''
    Approximate solution to lower bound equation, used for seeding exact solution.
    Taken from Gehrels 1986.
    '''

    if(n==0):
        return 0
    
    # First compute number of Gaussian sigma
    S = st.norm.ppf(C)
    
    # Now use the formula
    return n - S*np.sqrt(n) + ((S**2-1)/3)

# Solve upper bound
def lam_u_eqn_solve(n, C):

    '''
    Root of the upper bound equation, found numerically with approximate solution
    as seed.
    '''
    
    lam_u = opt.fsolve(lam_u_eqn, x0=lam_u_approx(n, C), args=(n, C))[0]
        
    return lam_u

# Solve lower bound
def lam_l_eqn_solve(n, C):
    
    '''
    Root of the lower bound equation, found numerically with approximate solution
    as seed.
    '''

    if(n==0):
        return 0
    
    lam_l = opt.fsolve(lam_l_eqn, x0=lam_l_approx(n, C), args=(n, C))[0]
    
    return lam_l

# Pre-compute errors and interpolate for non-integer counts
n_arr = np.arange(0, 100)
lam_u_exact_arr = np.array([lam_u_eqn_solve(n, 0.8413) for n in n_arr])
lam_l_exact_arr = np.array([lam_l_eqn_solve(n, 0.8413) for n in n_arr])
lam_l_interp = interp1d(n_arr, lam_l_exact_arr, fill_value='extrapolate')
lam_u_interp = interp1d(n_arr, lam_u_exact_arr, fill_value='extrapolate')

#######################
# Co-adding utilities #
#######################

def load_fits(qso_name, exp_names, es=4):

    '''
    Load fits files for a specific QSO, for a specified list of exposures. 
    The extraction size (extrsize parameter in CALSTIS) is taken to be 4 unless specified previously.
    '''
    # Get list of FITS files
    fits_list = [fits.open(rootdir+'{}/unstitched_es={}/{}_x1d.fits'.format(qso_name, es, e))[1] for e in exp_names]
    
    # Save grid shape
    n_exps = len(exp_names)
    n_orders = fits_list[0].data['wavelength'].shape[0]
    n_pixels = fits_list[0].data['wavelength'].shape[1]
    
    # Get exposure times
    exp_times = [f.header['exptime'] for f in fits_list]

    return fits_list, n_exps, n_orders, n_pixels, exp_times


def parse_fits_list(fits_list, n_exps, n_orders, n_pixels, exp_times):

    '''
    Function to parse a list of FITS files, and store wavelength-dependent counts, data quality, and fluxes.
    These quantities are saved across all exposures and orders
    '''

    # From all exposures, all orders, all pixels, load in wavelengths, gross counts, background counts, exposure times, data quality flags, and flux
    spec_data = np.zeros((n_exps, n_orders, n_pixels, 6))

    for i in range(n_exps):
        for j in range(n_orders):
            # Load orders in reverse because that means increasing wavelengths
            spec_data[i,j,:,0] = fits_list[i].data[n_orders-j-1]['wavelength']
            spec_data[i,j,:,1] = fits_list[i].data[n_orders-j-1]['gross']*exp_times[i] # Convert to counts
            spec_data[i,j,:,2] = fits_list[i].data[n_orders-j-1]['background']*exp_times[i] # To give net counts
            spec_data[i,j,:,3] = exp_times[i] # For weighting multiple exposures
            spec_data[i,j,:,4] = fits_list[i].data[n_orders-j-1]['dq']
            spec_data[i,j,:,5] = fits_list[i].data[n_orders-j-1]['flux'] # For flux calibration later

    return spec_data

def wav_grid_rectify(spec_data):

    '''
    Function to obtain a properly defined wavelength grid across multiple orders.
    This grid will be used to re-sample various exposures and orders to co-add counts 
    for each wavelength.
    '''

    # Some orders have overlapping wavelengths, so take care of that by modifying the wavelength grid
    wav_order_grid = np.zeros((spec_data.shape[1], spec_data.shape[2]))
    # Lowest wavelength order is fine, keep as is
    wav_order_grid[0,:] = spec_data[0,0,:,0]

    # For each subsequent order
    for i in range(spec_data.shape[1]-1):

        # Wavelengths of order 1
        wav_order_1 = spec_data[0,i,:,0]
        # Wavelengths of order 2
        wav_order_2 = spec_data[0,i+1,:,0]

        # Compute differences of order 1 wavelengths from the 0th element of order 2
        del_wav_arr = wav_order_1-wav_order_2[0]

        # This operation gives the index of the element of order 1 whose wavelength is right before the 0th element of order 2
        idx = list(del_wav_arr).index(max(del_wav_arr[del_wav_arr<0]))

        # Number of pixels in first order that are offset from the second order
        n_rem_order_1 = len(wav_order_1[idx:])
        # Copy over wavelengths to the second order
        wav_order_2[0:n_rem_order_1] = wav_order_1[idx:].copy()

        # Save the modified second order in the wavelength grid
        wav_order_grid[i+1,:] = wav_order_2.copy()

    return wav_order_grid

def spec_data_resample(spec_data, wav_order_grid):

    '''
    Function to interpolate resample counts and fluxes over a fixed wavelength grid
    '''

    # Interpolate fluxes over the rectified wavelength grid
    spec_data_interp = spec_data.copy()

    for i in range(spec_data.shape[0]):
        for j in range(spec_data.shape[1]):

            # The shifted wavelength grid for an order
            wav_order = wav_order_grid[j,:]

            # Original grid
            wav_order_og = spec_data[i,j,:,0]
            gross_order_og = spec_data[i,j,:,1]
            bg_order_og = spec_data[i,j,:,2]
            flux_order_og = spec_data[i,j,:,5]

            # Interpolate flux and variance
            gross_order = np.interp(wav_order, wav_order_og, gross_order_og)
            bg_order = np.interp(wav_order, wav_order_og, bg_order_og)
            flux_order = np.interp(wav_order, wav_order_og, flux_order_og)

            # Store in the new full grid
            spec_data_interp[i,j,:,0] = wav_order
            spec_data_interp[i,j,:,1] = gross_order
            spec_data_interp[i,j,:,2] = bg_order
            spec_data_interp[i,j,:,5] = flux_order  

    # Stitch wavelengths
    wav_stitch = np.unique(wav_order_grid)     

    return wav_stitch, spec_data_interp

def wav_map_construct(wav_stitch, spec_data_interp):

    '''
    Function to construct a dictionary of counts corresponding to a certain wavelength.
    Helpful for co-adding counts later.
    '''

    # Map fluxes and errors for each wavelength across all exposures
    wav_dict = {w:[] for w in wav_stitch}

    for i in range(spec_data_interp.shape[0]):
        for j in range(spec_data_interp.shape[1]):

            wav_order = spec_data_interp[i,j,:,0]

            for k in range(spec_data_interp.shape[2]):

                w = spec_data_interp[i,j,k,0] # Wavelength
                g = spec_data_interp[i,j,k,1] # Gross
                bg = spec_data_interp[i,j,k,2] # Background
                et = spec_data_interp[i,j,k,3] # Exposure time
                dq = spec_data_interp[i,j,k,4] # Data quality
                f = spec_data_interp[i,j,k,5] # Flux

                # Trim edges of each order
                if wav_order[0]+2<w<wav_order[-1]-5:            
                    wav_dict[w].append([g,bg,et,dq,f])    

    return wav_dict

def counts_coadd(wav_stitch, wav_dict):

    '''
    Function to co-add counts and convert them to count rates using exposure time array.
    '''

    # Store co-added gross and net count rates
    gross_cts_stitch = np.zeros(len(wav_stitch))
    net_cts_stitch = np.zeros(len(wav_stitch))
    err_u_cts_stitch = np.zeros(len(wav_stitch))
    err_d_cts_stitch = np.zeros(len(wav_stitch))
    err_cts_stitch = np.zeros(len(wav_stitch))
    exp_time_stitch = np.zeros(len(wav_stitch))

    # For flux calibration later
    net_ct_rate_dict = {}
    flux_dict = {}

    for i in range(len(wav_stitch)):

        # Get wavelength, and quantities (flux, error) to be combined
        w = wav_stitch[i]
        spec_arr = np.array(wav_dict[w])

        if len(spec_arr)!=0:
            gross_arr = spec_arr[:,0]
            bg_arr = spec_arr[:,1]
            et_arr = spec_arr[:,2]
            dq_arr = spec_arr[:,3]
            flux_arr = spec_arr[:,4]

            # Reject counts that have bad data quality
            idx = (dq_arr!=0)

            # Counts and exposure times after rejecting
            gross_arr = gross_arr[~idx]
            bg_arr = bg_arr[~idx]
            et_arr = et_arr[~idx]
            dq_arr = dq_arr[~idx]
            flux_arr = flux_arr[~idx]

            # Save net count rate and fluxes first
            net_ct_rate_dict[w] = (gross_arr-bg_arr)/et_arr
            flux_dict[w] = flux_arr

            # Record the co-added fluxes and errors
            gross_counts = np.sum(gross_arr)
            # Obtain errors using gross counts
            err_lo = gross_counts-lam_l_interp(gross_counts)
            err_hi = lam_u_interp(gross_counts)-gross_counts

            gross_cts_stitch[i] = gross_counts
            net_cts_stitch[i] = gross_counts-np.sum(bg_arr)
            err_u_cts_stitch[i] = err_hi
            err_d_cts_stitch[i] = err_lo
            err_cts_stitch[i] = .5*(err_lo + err_hi) # Take the average 
            exp_time_stitch[i] = np.sum(et_arr)

    # Convert counts to count rates to flux calibrate
    net_ct_rates_stitch = net_cts_stitch/exp_time_stitch
    err_u_ct_rates_stitch = err_u_cts_stitch/exp_time_stitch
    err_d_ct_rates_stitch = err_d_cts_stitch/exp_time_stitch
    err_ct_rates_stitch = err_cts_stitch/exp_time_stitch

    return gross_cts_stitch, net_cts_stitch, exp_time_stitch, net_ct_rates_stitch, err_u_ct_rates_stitch, err_d_ct_rates_stitch, err_ct_rates_stitch, net_ct_rate_dict, flux_dict

def flux_calibrate(wav_stitch, net_ct_rates_stitch, err_u_ct_rates_stitch, err_d_ct_rates_stitch, err_ct_rates_stitch, net_ct_rate_dict, flux_dict):

    '''
    Function to flux calibrate 1D spectrum from count rates to flux
    '''

    # Create sensitivity function
    # Flatten wavelength array to account for scatter in relation
    wav_flat = []
    sens_flat = []
    for w in wav_stitch:

        if w in list(flux_dict.keys()):
            for i in range(len(flux_dict[w])):
                wav_flat.append(w)
                sens_flat.append(flux_dict[w][i]/net_ct_rate_dict[w][i])
        else:
            wav_flat.append(w)
            sens_flat.append(np.nan)
            
    # Median relation
    sens_med = np.zeros(len(wav_stitch))

    for i in range(len(wav_stitch)):
        w = wav_stitch[i]
        if w in list(flux_dict.keys()):
            sens_med[i] = np.median(flux_dict[w]/net_ct_rate_dict[w])
        else:
            sens_med[i] = 0
     
    # Flux calibrated
    flux_stitch = net_ct_rates_stitch*sens_med
    err_u_stitch = err_u_ct_rates_stitch*sens_med
    err_d_stitch = err_d_ct_rates_stitch*sens_med
    err_stitch = err_ct_rates_stitch*sens_med
    
    return wav_stitch, flux_stitch, err_u_stitch, err_d_stitch, err_stitch

def coadd_1dspec(qso_name, exp_names, es=4):

    '''
    Function to combine individual pipeline steps and return the co-added 1D, flux calibrated spectrum
    across various exposures
    '''

    # First, load FITS files
    fits_list, n_exps, n_orders, n_pixels, exp_times = load_fits(qso_name, exp_names, es=es)

    # Next, parse the FITS files
    spec_data = parse_fits_list(fits_list, n_exps, n_orders, n_pixels, exp_times)

    # Obtain uniform wavelength grid for resampling
    wav_order_grid = wav_grid_rectify(spec_data)

    # Now resample counts
    wav_stitch, spec_data_interp = spec_data_resample(spec_data, wav_order_grid)

    # Form a wave map
    wav_dict = wav_map_construct(wav_stitch, spec_data_interp)

    # Co-add counts
    gross_cts_stitch, net_cts_stitch, exp_time_stitch, net_ct_rates_stitch, err_u_ct_rates_stitch, err_d_ct_rates_stitch, err_ct_rates_stitch, net_ct_rate_dict, flux_dict = counts_coadd(wav_stitch, 
                                                                                                                                                                                wav_dict)

    # Flux calibrate
    wav_stitch, flux_stitch, err_u_stitch, err_d_stitch, err_stitch = flux_calibrate(wav_stitch, net_ct_rates_stitch, err_u_ct_rates_stitch, 
                                                                                     err_d_ct_rates_stitch, err_ct_rates_stitch, net_ct_rate_dict, flux_dict)


    # Pack into a dictionary
    d = {'wave': wav_stitch,
         'flux': flux_stitch,
         'error': err_stitch,
         'error_u': err_u_stitch,
         'error_d': err_d_stitch,
         'counts_total': gross_cts_stitch,
         'counts_net': net_cts_stitch,
         'exptime': exp_time_stitch}

    return d

def save_spec1d_fits(savedir, fname, d):

    '''
    Function to save FITS for 1D spectrum in a format friendly with plotabs
    to help future absorption line search. Input should be packed into a dict.
    '''

    # Load in a reference FITS file for header

    # Define data columns
    c1 = fits.Column(name='wave    ', array=d['wave'], format='D')
    c2 = fits.Column(name='flux    ', array=d['flux'], format='D')
    c3 = fits.Column(name='error   ', array=d['error'], format='D')
    c4 = fits.Column(name='error_u ', array=d['error_u'], format='D')
    c5 = fits.Column(name='error_d ', array=d['error_d'], format='D')
    c6 = fits.Column(name='counts_total', array=d['counts_total'], format='D')
    c7 = fits.Column(name='counts_net', array=d['counts_net'], format='D')
    c8 = fits.Column(name='npix    ', array=np.zeros(len(d['wave'])), format='D')
    c9 = fits.Column(name='exptime ', array=d['exptime'], format='D')
    c10 = fits.Column(name='mask    ', array=np.ones(len(d['wave'])), format='K')
    c11 = fits.Column(name='continuum', array=np.ones(len(d['wave'])), format='D')

    # Make HDU table
    table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, 
                                               c4, c5, c6, 
                                               c7, c8, c9, 
                                               c10, c11])
    
    # Make HDU
    hdu = fits.HDUList([fits_ref[0], table_hdu])  


    # Check if path exists, otherwise make it
    Path(savedir).mkdir(parents=True, exist_ok=True)
    # Write HDU
    hdu.writeto(savedir + '{}.fits'.format(fname), overwrite=True)

def load_spec1d_fits(loaddir, fname):

    '''
    Function to load saved spectrum. FITS file assumed to be stored in a plotabs friendly format.
    '''

    f = fits.open(loaddir+fname+'.fits')
    wav = np.array(f[1].data['wave'])
    flux = np.array(f[1].data['flux'])
    err = np.array(f[1].data['error'])

    return wav, flux, err
