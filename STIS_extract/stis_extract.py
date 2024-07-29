import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from astropy import constants
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits

rootdir = '/Users/thepoetoftwilight/Documents/CUBS/Data/CONTACT/'

class spec_1d:

    '''
    Class for extracting 1D spectra using stistools. Assuming that the stenv environment is active.
    
    '''
    def __init__(self, qso_name, es=None):
 
        # Save QSO name and extraction size
        self.qso_name = qso_name
        self.es = es

        # For specifying directory given extraction size
        if es == None:
            self.es_str = ''
        else:
            self.es_str = '_es={}'.format(es)


    def load_fits(self):

        '''
        Function to load 1D FITS files extracted from the MAST 2D products using the stistools routine
        '''
        # Load up 1D files obtained from stistools
        self.file_list = glob.glob(rootdir + '{}/unstitched{}/*x1d.fits'.format(self.qso_name, self.es_str))

        # Open the FITS files for individual exposures
        self.fits_list = [fits.open(f) for f in self.file_list]

        # Number of exposures
        self.n_exps = len(self.fits_list)
        # Number of wavelength orders in each exposure
        self.n_orders = self.fits_list[0][1].data.shape[0]
        # Number of wavelength pixels in each wavelength order
        self.n_pixels = self.fits_list[0][1].data[0][2].shape[0]

    def save_spec_unstitched(self):

        '''
        Function to extract fluxes for all wavelength orders across all exposures
        '''

        # From all exposures, all orders, all pixels, load in wavelengths, fluxes, and errors
        spec_data = np.zeros((self.n_exps, self.n_orders, self.n_pixels, 3))

        for i in range(self.n_exps):
            for j in range(self.n_orders):
                # Load orders in reverse because that means increasing wavelengths
                spec_data[i,j,:,0] = self.fits_list[i][1].data[self.n_orders-j-1][2]
                spec_data[i,j,:,1] = self.fits_list[i][1].data[self.n_orders-j-1][6]
                spec_data[i,j,:,2] = self.fits_list[i][1].data[self.n_orders-j-1][7]

        # Some orders have overlapping wavelengths, so take care of that by modifying the wavelength grid
        wav_order_grid = np.zeros((self.n_orders, self.n_pixels))
        # Lowest wavelength order is fine, keep as is
        wav_order_grid[0,:] = spec_data[0,0,:,0]

        # For each subsequent order
        for i in range(self.n_orders-1):

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

        # Interpolate fluxes over the rectified wavelength grid
        spec_data_interp = spec_data.copy()

        for i in range(self.n_exps):
            for j in range(self.n_orders):
                
                # The shifted wavelength grid for an order
                wav_order = wav_order_grid[j,:]
            
                # Original grid
                wav_order_og = spec_data[i,j,:,0]
                flux_order_og = spec_data[i,j,:,1]
                var_order_og = spec_data[i,j,:,2]**2
                
                # Interpolate flux and variance
                flux_order = np.interp(wav_order, wav_order_og, flux_order_og)
                var_order = np.interp(wav_order, wav_order_og, var_order_og)
                
                # Store in the new full grid
                spec_data_interp[i,j,:,0] = wav_order
                spec_data_interp[i,j,:,1] = flux_order
                spec_data_interp[i,j,:,2] = np.sqrt(var_order)

        # Store the renewed data grid
        self.wav_order_grid = wav_order_grid
        self.spec_data_interp = spec_data_interp


    def stitch_orders(self, SN_thresh=-3, sig_thresh=3):

        '''
        Function for stitching 1D orders across all exposures. The first rejection criterion is points with negative flux,
        the second is for sigma-clipping.
        '''

        # Unique wavelengths, for final 1D stitched spectrum
        wav_stitch = np.unique(self.wav_order_grid)

        # For each unique wavelength, record fluxes and errors
        # Each wavelength has overlapping measurements because of multiple exposures, and also overlapping orders

        wav_dict = {w:[] for w in wav_stitch}

        for i in range(self.n_exps):
            for j in range(self.n_orders):
                for k in range(self.n_pixels):
                    
                    w = self.spec_data_interp[i,j,k,0]
                    f = self.spec_data_interp[i,j,k,1]
                    s = self.spec_data_interp[i,j,k,2]
                    
                    wav_dict[w].append([f,s])

        # Store co-added fluxes and errors
        flux_stitch = np.zeros(len(wav_stitch))
        err_stitch = np.zeros(len(wav_stitch))

        # Also keep track of fraction of rejected measurements from S/N threshold and sigma-clipping 
        reject_stitch_1 = np.zeros(len(wav_stitch))
        reject_stitch_2 = np.zeros(len(wav_stitch))

        for i in range(len(wav_stitch)):
            
            # Get wavelength, and quantities (flux, error) to be combined
            w = wav_stitch[i]
            spec_arr = np.array(wav_dict[w])
            flux_arr = spec_arr[:,0]
            err_arr = spec_arr[:,1]
            
            # Number of measurements, from multiple exposures and overlapping orders
            obs_count = len(flux_arr)
            
            # Reject fluxes that are "too" negative, record fraction of rejections
            idx_1 = flux_arr/err_arr < SN_thresh
            reject_stitch_1[i] = np.sum(idx_1)/obs_count
            
            # Fluxes and errors after rejecting
            flux_arr = flux_arr[~idx_1]
            err_arr = err_arr[~idx_1]
            
            # Next, after rejecting these points, perform median filtering
            mu = np.median(flux_arr)
            sigma = np.sqrt(np.mean((flux_arr-mu)**2))

            # Reject points, record number of rejections
            idx_2 = np.abs(flux_arr-mu)>sig_thresh*sigma
            reject_stitch_2[i] = np.sum(idx_2)/obs_count
            
            # Fluxes and errors that are good to go
            flux_arr = flux_arr[~idx_2]
            err_arr = err_arr[~idx_2]  
            
            # Decide weights for each measurement, inverse varance weighting
            # This minimizes the variance
            wts_arr = 1/err_arr**2
            # Ensure weights are normalized to one
            wts_arr /= np.sum(wts_arr)
            
            # Record the co-added fluxes and errors
            flux_stitch[i] = np.sum(wts_arr*flux_arr)
            # Apply square of weights to variance, then add, then 
            err_stitch[i] = np.sqrt(np.sum(wts_arr**2*err_arr**2))

            # Save the spectrum and rejection parameters
            self.reject_stitch_1 = reject_stitch_1
            self.reject_stitch_2 = reject_stitch_2
            self.wav_stitch = wav_stitch
            self.flux_stitch = flux_stitch
            self.err_stitch = err_stitch

    def save_spec_stitched(self):

        '''
        Function to save stitched spectrum
        '''
        spec_final_grid = np.array([self.wav_stitch, self.flux_stitch, self.err_stitch]).T
        np.savetxt(rootdir+'{}/{}_x1d{}.dat'.format(self.qso_name, self.qso_name, self.es_str), spec_final_grid)
