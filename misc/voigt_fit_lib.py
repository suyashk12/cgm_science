from astropy.io import fits
from astropy import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.special import voigt_profile
from lmfit import Model, Parameters, minimize
from lmfit.models import PolynomialModel
from astropy.convolution import convolve
import emcee
import corner
import pickle
import os
from IPython.display import display, Latex
import operator
from scipy import integrate
from scipy import interpolate

# Use Zhijie's plotting style
plt.style.use('/Users/thepoetoftwilight/Documents/Astro/Code/PG1522+101/science.mplstyle')

# List of atmoic numbers
A_dict = {'H' : 1.00797 ,
'He' : 4.00260 ,
'Li' : 6.941 ,
'Be' : 9.01218 ,
'B' : 10.81 ,
'C' : 12.011 ,
'N' : 14.0067 ,
'O' : 15.994 ,
'F' : 18.9994 ,
'Ne' : 20.179 ,
'Na' : 22.98977 ,
'Mg' : 24.305 ,
'Al' : 26.98154 ,
'Si' : 28.0855 ,
'P' : 30.97376 ,
'S' : 32.06 ,
'Cl' : 35.453 ,
'A' : 39.948 ,
'K' : 39.0983 ,
'Ca' : 40.08 ,
'Sc' : 44.9559 ,
'Ti' : 47.90 ,
'V' : 50.9415 ,
'Te' : 51.996 ,
'Mn' : 54.9380 ,
'Fe' : 55.847 ,
'Co' : 58.70 ,
'Ni' : 58.9332 ,
'Cu' : 63.546 ,
'Zn' : 65.38,
}

# Significance levels
# Sigma values taken from Gehrels 1986
cdf_1sig = 0.8413
cdf_2sig = 0.9772
cdf_3sig = 0.9987

# List of instruments

# Base reference: HIRES MW Ca H&K
# FUV: delta_v of 6 km/s from uncalibrated wavelength using MW CII*1335 and delta_v of 7 km/s from NI1200 in COS FUV compared to HIRES MW H&K
# STIS: delta_v of 2 km/s from uncalibrated wavelength using MW FeII2344 and FeII2586 in COS STIS compared to HIRES Ca H&K
# NUV: delta_v of 2 km/s from uncalibrated wavelength using z=1.16591 HI 918 and collective fitting of HI (previous lines from calibrated STIS) and z=1.09454 CIII977 in STIS and NUV
instruments_dict = {'FUV': 0, 'NUV': 1, 'STIS': 2, 'HIRES': 3}
#delta_wav_list = [0, 0, 0, 0]
delta_wav_list = [-0.016379252766527918, 0, 0, 0]
#delta_wav_list = [0.027361350561960006, 0.020175157458413, 0.016434071712905403, 0.016764412856649667]


# Speed of light in km/s
c = 2.99792458e+5
# Boltzmann constant, adjusted to use km instead of m
k_B = 1.380649e-29
# Mass of hydrogen atom
amu = 1.66054e-27

# Classical electron cross-section
sigma_0 = 2.654E-2

# Table of transitions
atomic_data_table = np.loadtxt('/Users/thepoetoftwilight/Documents/Astro/pabs0/data/linelists/atom.dat', dtype=str)

# Assuming that we won't run into more than four components
colors = ['indigo', 'forestgreen', 'darkgoldenrod', 'olive', 'darkgray', 'darkmagenta']    

class ion_transition:

    '''
    This class is supposed to capture the continuum normalized flux of an ionic transition in terms of velocity.
    The methods defined for the class allow fitting a Voigt profile to the absorption feature. 
    The uncertainties on physical parameters are not completely robust at the moment, there is a scope to implement MCMC in the future.
    '''

    def __init__(self, ion_name, wav0_rest, el, z, instrument):
        '''
        Initiate an ionic transition object

        name: Name of the ion
        wav0_rest: Rest wavelength of the transition
        el: Name of the element
        sys_idx: System ID as seen in the plotabs software
        instrument: The instrument that the spectrum was captured using
        '''
        self.ion_name = ion_name
        self.wav0_rest = wav0_rest

        # Create the transition name
        if self.wav0_rest%1 == 0:
            self.name = self.ion_name + str(np.round(self.wav0_rest))
        else:
            self.name = self.ion_name + str(np.round(self.wav0_rest, 1))

        # Figure out which element the ionic transition belongs to with some list comprehension
        self.A = A_dict[el]
        self.z = z
        self.instrument = instrument

    def grab_ion_transition_info(self, delta_v=0):

        '''
        Method to grab universal information about the ionic transition

        delta_v: Wavelength calibration parameter.
                 Intuitively, this is the amount by which your desired zero velocity is offset from the current zero.
                 If delta_v > 0, the spectrum will be shifted to the left, and vice-versa for delta_v < 0.
        '''

        # Set the velocity calibration parameter
        self.delta_v = delta_v
    
        # Locate ion in atomic data table using the rest wavelength
        # First, get subset of table with the appropriate ion
        atomic_data_table_sub = atomic_data_table[atomic_data_table[:,0]==self.ion_name] 
        table_idx = np.argmin(np.abs(np.float_(atomic_data_table_sub[:,1])-self.wav0_rest))

        # Reset the rest wavelength to a more accurate value
        wav0_rest_init = self.wav0_rest
        self.wav0_rest = float(atomic_data_table_sub[table_idx][1])

        # Reassign transition name
        if wav0_rest_init%1 == 0:
            self.name = self.ion_name + str(int(np.floor(self.wav0_rest)))
        else:
            self.name = self.ion_name + str(np.round(self.wav0_rest, 1))

        # Calculate observed frame wavelength
        self.wav0_obs = self.wav0_rest*(1+self.z)

        # Get the oscillator strength and damping factor
        self.f = float(atomic_data_table_sub[table_idx][2])

        # Load gamma in units of frequency
        gamma_nu = float(atomic_data_table_sub[table_idx][3])
        # Convert it to velocity
        self.gamma = (gamma_nu*(self.wav0_rest * 1e-10))*1e-3 # First convert wavelength to meters, then velocity to km/s

    def grab_ion_transition_spec(self, spec_fits_list, v_range=[-300,300], masks = [], delta_wav = None):

        '''
        Method to grab spectrum of the ionic transition

        spec_fits_list: list of all spectral FITS files as used by plotabs
        v_range: the velocity range within which the spectrum must be extracted
        masks: range of velocities to be masked during fitting etc.
        '''
        
        # Isolate the appropriate spectrum file
        spec_fits = spec_fits_list[instruments_dict[self.instrument]]

        # Convert it to an array
        spec_data = pd.DataFrame(spec_fits[1].data).to_numpy()

        # Isolate the wavelength, flux, and error
        wav = spec_data[:,0]

        # Add wavelength calibration parameter
        if delta_wav is not None:
            wav = wav - delta_wav
        else:
            wav = wav - delta_wav_list[instruments_dict[self.instrument]]

        flux = spec_data[:,1]
        err = spec_data[:,2] 

        # Get LOS velocities
        # REMEMBER: This puts us in the rest frame of the ion
        v = c*(wav-self.wav0_obs)/self.wav0_obs

        # Apply the wavelength calibration parameter
        v = v-self.delta_v

        # Slice velocities within given bounds
        slice_idx = (v>v_range[0])&(v<v_range[1])
        wav_obs = wav[slice_idx]
        v_obs = v[slice_idx]

        # Isolate the flux and error
        flux_obs = flux[slice_idx]
        err_obs = err[slice_idx]

        # Finally set the velocity, normalized flux and error for the ion object. Also store the wavelengths just in case
        self.wav = wav_obs
        self.v = v_obs
        self.flux = flux_obs
        self.err = err_obs

        # Save the masks
        self.masks = masks

    def cont_norm_flux(self, renorm = True, v_abs_range = [-50,50], degree = 1):

        '''
        Method to continuum normalize the flux

        nodes_fits_list: list of all continuum nodes FITS files as used by plotabs
        renorm: whether or not to locally renormalize the continuum
        v_abs_range: range of the absorption feature, to be avoided while performing local continuum renormalization, in addition to masks
        '''

        # Save the continuum limits for plotting
        self.v_abs_min = v_abs_range[0]
        self.v_abs_max = v_abs_range[1]

        # Store the final normalized flux, modify if renormalization has been requested
        self.renorm = renorm
        self.flux_norm = np.copy(self.flux)
        self.err_norm = np.copy(self.err)
        self.cont_params = None
        self.cont_flux = np.ones(len(self.flux_norm))

        # If there is a specification to further renormalize, do so
        if(self.renorm == True):

            # Get a slice of the continuum points, first avoiding the absorption feature
            abs_mask = ((self.v<v_abs_range[0])|(self.v>v_abs_range[1]))
            v_cont = self.v[abs_mask]
            flux_norm_cont = self.flux[abs_mask]
            err_norm_cont = self.err[abs_mask]

            # Further exclude all the masks during continuum renormalization
            # This will be done here iteratively, is there a better way?

            # For each mask
            for i in range(len(self.masks)):

                v_mask = self.masks[i]

                # Get boolean indices corresponding to mask
                mask = ((v_cont<v_mask[0])|(v_cont>v_mask[1]))

                # Update velocities, fluxes, and errors
                v_cont = v_cont[mask]
                flux_norm_cont = flux_norm_cont[mask]
                err_norm_cont = err_norm_cont[mask]
                
            # Fit a polynomial model
            params = Parameters()
            params.add("c0", value=1)
            
            for i in range(1, degree+1):
                params.add("c{}".format(i), value=0)
            
            cont_model = PolynomialModel(degree=degree)
                
            result = cont_model.fit(data=flux_norm_cont, params=params, x=v_cont,
                                    weights=1/err_norm_cont)
            
            # Regenerate the continuum using the best fit parameters
            cont_params = list(result.best_values.values())
            cont_flux = np.zeros(len(self.v))

            for i in range(degree+1):
                cont_flux += cont_params[i]*self.v**i
            
            # Renormalize the normalized flux and error
            self.cont_params = cont_params
            self.cont_flux = cont_flux
            self.flux_norm = self.flux_norm/cont_flux
            self.err_norm = self.err_norm/cont_flux

    def plot_ion_transition_spec_cont(self, fig = None, axes = None, draw_masks = True, draw_cont_bounds = True, label_axes=True):

        '''
        Method to draw the continuum normalized spectrum of the transition

        fig: Figure object to make the plot
        axes: Axes object to make the plot
        draw_masks: Whether or not to draw masked regions
        draw_cont_bounds: Whether or not to draw the bounds of the continuum
        label_axes: Whether or not to label axes
        '''

        # Plot the spectrum before and after renormalization
        create_fig_ax = False
        if fig == None and axes == None:
            create_fig_ax = True
            fig, axes = plt.subplots(1, 2, figsize=(14,4), sharex=True)

        # First plot the spectrum before renormalization
        axes[0].step(self.v, self.flux, color='black', where='mid', lw=1.5)
        axes[0].step(self.v, self.err, color='cyan', where='mid', lw=1.5)


        # Draw the masked regions
        if(draw_masks == True):
            for i in range(len(self.masks)):
                v_mask = self.masks[i]
                axes[0].axvspan(v_mask[0], v_mask[1], color='darkgrey')

        # Draw the continuum boundaries
        if(draw_cont_bounds == True):
            axes[0].axvline(self.v_abs_min, color='blue', linestyle=':')
            axes[0].axvline(self.v_abs_max, color='red', linestyle=':')

        # Plot the best fit for the continuum
        axes[0].plot(self.v, self.cont_flux, color='salmon')

        # Plot reference lines
        axes[0].axvline(0, color='brown', linestyle=':')

        # Repeat the same process, but for the renormalized spectrum
        axes[1].step(self.v, self.flux_norm, color='black', where='mid', lw=1.5)
        axes[1].step(self.v, self.err_norm, color='cyan', where='mid', lw=1.5)   

        # Plot reference lines
        axes[1].axhline(0, color='red', linestyle=':')
        axes[1].axhline(1, color='green', linestyle=':')
        axes[1].axvline(0, color='brown', linestyle=':')

        # Label the line
        axes[1].text(self.v[0]+30, np.max(self.err_norm)+.1, self.name, fontsize=12)

        if label_axes == True:
            # Answer 2 on https://stackoverflow.com/questions/42372509/how-to-add-a-shared-x-label-and-y-label-to-a-plot-created-with-pandas-plot
            # Create a big subplot
            ax_label = fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            ax_label.set_xticks([])
            ax_label.set_yticks([])

            ax_label.set_xlabel('Velocity (km/s)', labelpad=15) # Use argument `labelpad` to move label downwards.
            ax_label.set_ylabel('Flux (continuum normalized)', labelpad=25)

        #plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        if create_fig_ax == True:
            return fig, axes

    def plot_ion_transition_spec(self, fig = None, ax = None, draw_masks = True, draw_cont_bounds = True, label_axes=True):

        '''
        Method to draw the continuum normalized spectrum of the transition

        fig: Figure object to make the plot
        axes: Axes object to make the plot
        draw_masks: Whether or not to draw masked regions
        draw_cont_bounds: Whether or not to draw the bounds of the continuum
        label_axes: Whether or not to label axes
        '''

        # Plot the spectrum before and after renormalization
        create_fig_ax = False
        if fig == None and ax == None:
            create_fig_ax = True
            fig, ax = plt.subplots(1, figsize=(7,4))

        # Repeat the same process, but for the renormalized spectrum
        ax.step(self.v, self.flux_norm, color='black', where='mid', lw=1.5)
        ax.step(self.v, self.err_norm, color='cyan', where='mid', lw=1.5)   

        # Draw the masked regions
        if(draw_masks == True):
            for i in range(len(self.masks)):
                v_mask = self.masks[i]
                ax.axvspan(v_mask[0], v_mask[1], color='darkgrey')    

        # Plot reference lines
        ax.axhline(0, color='red', linestyle=':')
        ax.axhline(1, color='green', linestyle=':')
        ax.axvline(0, color='brown', linestyle=':')

        if(label_axes == True):
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Flux (continuum normalized)')

        ax.text(self.v[0]+30, np.max(self.err_norm)+.1, self.name+ ', ' + self.instrument, fontsize=12)
        ax.text(self.v[-1]-120, np.max(self.err_norm)+.1, '$\lambda = $' + str(np.round(self.wav0_obs,2)) + 'Å', fontsize=12)

        if create_fig_ax == True:
            return fig, ax

    def grab_ion_transition_lsf(self, lsf_fits_list):
        
        '''
        Method to grab relevant LSF for transition

        lsf_fits_list: List of LSFs for all instruments
        '''

        # First grab the LSF FITS file
        lsf_fits = lsf_fits_list[instruments_dict[self.instrument]]

        # Convert it to an array
        lsf_data = np.array(lsf_fits[1].data)
        
        # For each wavelength, isolate LSFs
        lsf_wavs = []

        for i in range(len(lsf_data)):
            lsf_wavs.append(lsf_data[i][0])

        lsf_wavs = np.array(lsf_wavs)

        # Isolate the closest LSF
        lsf_idx = np.argmin(np.abs(self.wav0_obs-lsf_wavs))
        lsf = lsf_data[lsf_idx][2]

        # Also generate the velocity scale corresponding to the LSF
        delta_v = lsf_data[lsf_idx][1]
        lsf_central_idx = np.argmax(lsf)
        lsf_pix_rel = np.arange(0, len(lsf))-lsf_central_idx
        lsf_pix_v = delta_v*lsf_pix_rel

        # Finally store the LSF velocities and the profile itself
        self.lsf = lsf
        self.v_lsf = lsf_pix_v

    def plot_ion_transition_lsf(self, fig=None, ax=None):

        '''
        Method to draw the relevant LSF of the ionic transition

        ax: The axes object upon which the LSF should be drawn
        '''

        create_fig_ax = False

        if fig == None and ax == None:
            create_fig_ax = True
            fig, ax = plt.subplots(1, figsize=(7,4))

        ax.plot(self.v_lsf, self.lsf)

        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('LSF')

        ax.text(np.min(self.v_lsf)+30, np.min(self.lsf) + .1*(np.max(self.lsf)-np.min(self.lsf)), self.name, fontsize=12)

        if create_fig_ax == True:
            return fig, ax

    def gen_b_dist(self, ion_suite, logT_name, b_NT_name):

        '''
        Method to get distribution of b-values given (log) temperature and non-thermal motion distributions (from Voigt profile fitting of an ionic suite)
        '''

        logT_dist = ion_suite.result_emcee.flatchain[logT_name]
        b_NT_dist = ion_suite.result_emcee.flatchain[b_NT_name]
        b_T_dist = np.sqrt(2*k_B*10**logT_dist/(self.A*amu))
        b_dist = np.sqrt(2*k_B*10**logT_dist/(self.A*amu) + b_NT_dist**2)

        self.b_T_dist = b_T_dist
        self.b_dist = b_dist

        print('b_T, 1sig: {:.1f}, {:.1f}, +{:.1f}'.format(np.median(self.b_T_dist), 
                                                       np.percentile(self.b_T_dist, 100*(1-cdf_1sig))-np.median(self.b_T_dist), 
                                                       np.percentile(self.b_T_dist, 100*(cdf_1sig))-np.median(self.b_T_dist)))
        
        print('b_T, 3sig: <{:.1f}'.format(np.percentile(self.b_T_dist, 100*cdf_3sig)))

        print('b, 1sig: {:.1f}, {:.1f}, +{:.1f}'.format(np.median(self.b_dist), 
                                                       np.percentile(self.b_dist, 100*(1-cdf_1sig))-np.median(self.b_dist), 
                                                       np.percentile(self.b_dist, 100*(cdf_1sig))-np.median(self.b_dist)))

    def gen_conv_fwhm(self, b, logN_ref):

        # Generate a finely sampled convolved profile
        v_mod = np.linspace(self.v[0], self.v[-1], 10*len(self.v))

        model_flux_conv = comp_model_spec_gen(v_mod, np.array([[logN_ref, b, 0]]), 
                                        self.wav0_rest, self.f, self.gamma, self.A,
                                        True,
                                        self.lsf, self.v_lsf)[1]
        
        # Convert it back to optical depth
        model_tau_conv = -np.log(model_flux_conv)

        # Get tau for FWHM
        tau_fwhm_conv = 0.5*np.max(model_tau_conv)

        # Compute FWHM
        tau_2_v = interpolate.interp1d(x=model_tau_conv[v_mod>0], y=v_mod[v_mod>0], fill_value='extrapolate')
        fwhm_conv = 2*tau_2_v([tau_fwhm_conv])[0]

        return fwhm_conv
    
    def gen_inv_curve_of_growth(self, b, logN_min = 9, logN_max = 18, logN_step=0.01):

        '''
        Generates (inverse of) curve of growth for the ion transition to generate column density given an equivalent width
        '''

        # Create a grid of logN
        logN_grid = np.arange(logN_min, logN_max+logN_step, logN_step)

        # Get the wavelength pixel size in milli-Angstrom
        delta_lambda = np.mean(((self.v[1:]-self.v[:-1])*self.wav0_rest/3e+5)*1e+3) # in mA

        # Generate a grid of models, and for each of them evaluate the equivalent width
        EW_grid = np.zeros(len(logN_grid))

        for i in range(len(logN_grid)):

            # Generate a convolved model profile
            model = comp_model_spec_gen(self.v, np.array([[logN_grid[i], b, 0]]), 
                                        self.wav0_rest, self.f, self.gamma, self.A,
                                        True,
                                        self.lsf, self.v_lsf)[1]
            EW_grid[i] = np.sum((1-model[:-1])*delta_lambda)

        EW_2_logN = interpolate.interp1d(x=EW_grid, y=logN_grid, fill_value='extrapolate')
    
        return EW_2_logN

    def get_EW(self, b, v_c, logN_ref=13.):

        '''
        Method to get EW, 1-sigma upper limit on EW
        '''

        fwhm = self.gen_conv_fwhm(b, logN_ref)

        idx = (self.v>v_c-fwhm)&(self.v<v_c+fwhm)
        
        v_abs = self.v[idx]
        flux_abs = self.flux_norm[idx][:-1]
        err_abs = self.err_norm[idx][:-1]

        # This is a safeguard against a 2-element v_abs
        delta_lambda = np.mean(((v_abs[1:]-v_abs[:-1])*self.wav0_rest/3e+5)*1e+3) # in mA

        EW = np.sum((1-flux_abs)*delta_lambda)
        EW_1sig = np.sqrt(np.sum((err_abs*delta_lambda)**2))

        self.EW = EW
        self.EW_1sig = EW_1sig

        # Convert to 1-sigma and 3-sigma errors in N
        #self.N_1sig = (EW_1sig*1e-3/self.wav0_rest)*(3e+8/(self.wav0_rest*1e-10))*(2.654e-2*self.f)**-1
        #self.logN_1sig = np.log10(self.N_1sig)
        
        # Compute 3-sigma limit of logN using inverse curve of growth
        self.COG = self.gen_inv_curve_of_growth(b)
        self.logN_3sig = self.COG([3*self.EW_1sig])[0]

        # Print EW, 1-sigma error, 3-sigma error, and 3-sigma upper limit in logN
        print('Integration window: ' + '[{}, {}]'.format(int(np.round(v_c-fwhm)), int(np.round(v_c+fwhm))))
        print('EW, 1sig: {}, {}'.format(int(np.round(self.EW)), int(np.round(self.EW_1sig))))
        print('EW-3sig: {}'.format(int(np.round(3*self.EW_1sig))))
        print('logN-3sig: {:.1f}'.format(np.round(self.logN_3sig,1)))

    def get_EW_total(self, v_min, v_max):

        idx = (self.v>v_min)&(self.v<v_max)

        v_abs = self.v[idx]
        flux_abs = self.flux_norm[idx][:-1]
        err_abs = self.err_norm[idx][:-1]

        delta_lambda = np.mean(((v_abs[1:]-v_abs[:-1])*self.wav0_rest/3e+5)*1e+3) # in mA

        EW = np.sum((1-flux_abs)*delta_lambda)
        EW_1sig = np.sqrt(np.sum((err_abs*delta_lambda)**2))

        print('Integration window: ' + '[{}, {}]'.format(int(np.round(v_min)), int(np.round(v_max))))
        print('EW, 1sig: {}, {}'.format(int(np.round(EW)), int(np.round(EW_1sig))))
        print('EW-3sig: {}'.format(int(np.round(3*EW_1sig))))

    def init_ion_transition(self, init_values, lsf_convolve = True):

        '''
        Method to initiate a multi-component Voigt profile for the ionic transition

        init_values: 2D array of initial values to use for generating a multi-component Voigt profile
        lsf_convolve: Whether or not to convolve the model spectrum with the LSF
        '''

        # Get number of components
        n_components = len(init_values)

        # Generate fluxes for each component
        init_comp_fluxes, init_total_flux = comp_model_spec_gen(self.v, init_values, 
                                                            self.wav0_rest, self.f, self.gamma, self.A,
                                                            lsf_convolve, self.lsf, self.v_lsf)

        # Store the initial profile in the ionic transition object
        self.n_components = n_components
        self.init_values = init_values
        self.lsf_convolve = lsf_convolve
        self.init_comp_fluxes = init_comp_fluxes
        self.init_total_flux = init_total_flux

    def plot_ion_transition_init_fit(self, fig=None, ax=None, draw_masks=True, legend=True, label_axes=True):

        '''
        Method to draw the initial Voigt profile for the ionic transition

        fig: A figure object that is provided to make a plot
        ax: An axis object that is provided to make a plot
        draw_masks: Whether or not to shade masked regions
        legend: Whether or not to draw the legend, carrying the parameter values for each component
        '''

        create_fig_ax = False

        if fig == None and ax == None:
            create_fig_ax = True
            fig, ax = plt.subplots(1, figsize=(7, 4))

        # First draw the background spectrum

        ax.step(self.v, self.flux_norm, color='black', where='mid', lw=1.5)
        ax.step(self.v, self.err_norm, color='cyan', where='mid', lw=1.5)

        # Shade masked regions, if indicated
        if(draw_masks == True):
            
            for i in range(len(self.masks)):
                v_mask = self.masks[i]
                ax.axvspan(v_mask[0], v_mask[1], color='darkgrey')

        # Then draw all the components, with relevant labelling
        # Can this code be condensed further?

            # If there are multiple components
            for i in range(self.n_components):

                # If constraining b-value directly
            
                if len(self.init_values[i]) == 3:

                    ax.plot(self.v, self.init_comp_fluxes[i], 
                            label='log$N$ = ' + str(np.round(self.init_values[i][0], 2)) + '\n' +
                            '$b$ = ' + str(np.round(self.init_values[i][1], 2)) + ' km/s' + '\n' +
                            'd$v_c$ = ' + str(np.round(self.init_values[i][2], 2)) + ' km/s',
                            lw=1, color=colors[i])

                    # Also indicate the velocity centroid
                    ax.axvline(self.init_values[i][2], color=colors[i], linestyle=':')

                # If constraining logT and b_NT

                elif len(self.init_values[i]) == 4:

                    ax.plot(self.v, self.init_comp_fluxes[i], 
                            label='log$N$ = ' + str(np.round(self.init_values[i][0], 2)) + '\n' +
                            'log$T$ = ' + str(np.round(self.init_values[i][1], 2)) + '\n' +
                            '$b_{NT}$ = ' + str(np.round(self.init_values[i][2], 2)) + ' km/s' + '\n'
                            'd$v_c$ = ' + str(np.round(self.init_values[i][3], 2)) + ' km/s',
                            lw=1, color=colors[i])

                    # Also indicate the velocity centroid
                    ax.axvline(self.init_values[i][3], color=colors[i], linestyle=':')

            # Plot the combined flux
            ax.plot(self.v, self.init_total_flux,
                    lw=1.5, color='red')

        # Draw reference lines
        ax.axhline(0, color='red', linestyle=':')
        ax.axhline(1, color='green', linestyle=':')
        ax.axvline(0, color='brown', linestyle=':')

        # Add axes labels
        if label_axes == True:
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Continuum normalized flux')

        #  Indicate the transition properly in the plot
        ax.text(self.v[0]+30, np.max(self.err_norm)+.1, self.name, fontsize=12)

        # Indicate legend if requested
        if(legend == True):
            ax.legend(loc='lower right')
        if create_fig_ax == True:
            return fig, ax

    def fit_ion_transition(self, 
                           logN_min = -np.inf, logN_max = 25., 
                           b_min = 0., b_max = 100., 
                           logT_min = 4., logT_max = 6.,
                           b_NT_min = 0., b_NT_max = 50.,    
                           dv_c_min = -250., dv_c_max = 250.,
                           tie_params_list=[], 
                           fix_params_list=[], 
                           lower_bounds_dict={}, 
                           upper_bounds_dict={},
                           exclude_models=None):

        '''
        Method to fit the multi-component Voigt profile to the ionic transition

        logN_min: default minimum of column density
        logN_max: default maximum of column density

        b_min: default minimum of Doppler width
        b_max: default maximum of Dopper width

        dv_c_min: default minimum of centroid shift
        dv_c_max: default maximum of centroid shift

        tie_params_list: List of parameters to be tied together. Each entry in the list is pair of strings corresponding to parameters to be tied
        fix_params_list: List of parameters to not vary 
        lower_bounds_dict: Dictionary of lower bounds on parameters
        fix_indices_list: Dictionary of upper bounds on parameters

        exclude_models: String representations for models to exclude from the prior (will be fed into the objective function which can handle it)
        '''

        # Create an LMFIT parameters object
        params = Parameters()

        # First, set the initial values of all parameters

        # For each component
        for j in range(self.n_components):
            # Create parameters corresponding to the column density, Doppler width, and velocity centroid
            # If only constraining b-value
            if len(self.init_values[j]) == 3:
                params.add(name = "it1c{}_logN".format(j+1), value = self.init_values[j][0], min=logN_min, max=logN_max)
                params.add(name = "it1c{}_b".format(j+1), value = self.init_values[j][1], min=b_min, max=b_max)
                params.add(name = "it1c{}_dv_c".format(j+1), value = self.init_values[j][2], min=dv_c_min, max=dv_c_max)

            elif len(self.init_values[j]) == 4:
                params.add(name = "it1c{}_logN".format(j+1), value = self.init_values[j][0], min=logN_min, max=logN_max)
                params.add(name = "it1c{}_logT".format(j+1), value = self.init_values[j][0], min=logT_min, max=logT_max)
                params.add(name = "it1c{}_b_NT".format(j+1), value = self.init_values[j][1], min=b_NT_min, max=b_NT_max)
                params.add(name = "it1c{}_dv_c".format(j+1), value = self.init_values[j][2], min=dv_c_min, max=dv_c_max)

        # Tie all indicated parameter pairs together
        # The tying should occur first so that the list of bounds can be minimal
        for pair in tie_params_list:
            params[pair[1]].set(expr=pair[0])

        # For indicated parameters, fix their values
        for p in fix_params_list:
            params[p].set(vary = False)
        
        # For indicated parameters, set lower bounds for prior
        for p in list(lower_bounds_dict.keys()):
            params[p].set(min = lower_bounds_dict[p])

        # For indicated parameters, set upper bounds for prior
        for p in list(upper_bounds_dict.keys()):
            params[p].set(max = upper_bounds_dict[p])        

        # Generate the masked observations that must be fit
        # Store these for later use in fitting
        self.v_mask = np.copy(self.v)
        self.flux_norm_mask = np.copy(self.flux_norm)
        self.err_norm_mask = np.copy(self.err_norm)

        # This will be done iteratively, as during continuum normalization
        for i in range(len(self.masks)):
            mask = ((self.v_mask<self.masks[i][0])|(self.v_mask>self.masks[i][1]))
            self.v_mask = self.v_mask[mask]
            self.flux_norm_mask = self.flux_norm_mask[mask]
            self.err_norm_mask = self.err_norm_mask[mask]
        
        # Now, optimize the objective function
        self.exclude_models = exclude_models
        result = minimize(weighted_residuals, params, args=([self], 'lmfit'), kws={'exclude_models':self.exclude_models})
        self.result = result

        # Record the best-fit parameters
        self.best_values = []
        self.best_errs = []

        for i in range(self.n_components):

            best_values_comp = []
            best_errs_comp = []

            # Isolate best values and errors based on whether only b is being constrained or logT and b_NT

            if len(self.init_values[i]) == 3:
                best_values_comp = [self.result.params['it1c{}_logN'.format(i+1)].value, 
                                    self.result.params['it1c{}_b'.format(i+1)].value, 
                                    self.result.params['it1c{}_dv_c'.format(i+1)].value]
                
                best_errs_comp = [self.result.params['it1c{}_logN'.format(i+1)].stderr, 
                                    self.result.params['it1c{}_b'.format(i+1)].stderr, 
                                    self.result.params['it1c{}_dv_c'.format(i+1)].stderr]

            if len(self.init_values[i]) == 4:
                best_values_comp = [self.result.params['it1c{}_logN'.format(i+1)].value, 
                                    self.result.params['it1c{}_logT'.format(i+1)].value,
                                    self.result.params['it1c{}_b_NT'.format(i+1)].value, 
                                    self.result.params['it1c{}_dv_c'.format(i+1)].value]
                
                best_errs_comp = [self.result.params['it1c{}_logN'.format(i+1)].stderr, 
                                  self.result.params['it1c{}_logT'.format(i+1)].value,
                                    self.result.params['it1c{}_b_NT'.format(i+1)].stderr, 
                                    self.result.params['it1c{}_dv_c'.format(i+1)].stderr]
                
            self.best_values.append(best_values_comp)
            self.best_errs.append(best_errs_comp)
            
        # Generate fluxes for each component
        best_comp_fluxes, best_total_flux = comp_model_spec_gen(self.v, self.best_values, 
                                                            self.wav0_rest, self.f, self.gamma, self.A,
                                                            self.lsf_convolve, self.lsf, self.v_lsf)

        # Set all these properties for the ion object
        self.best_comp_fluxes = best_comp_fluxes
        self.best_total_flux = best_total_flux

    def plot_ion_transition_best_fit(self, fig = None, ax = None, draw_masks = True, draw_cont_bounds = True, label_axes=True, legend=True):

        '''
        Method to plot the best fit profile

        fig: Figure object to make the plot on
        ax: Axis object to make the plot on
        draw_masks: Whether or not to shade masked regions
        legend: Whether or not to draw the legend
        '''

        create_fig_ax = False
        if fig == None and ax == None:
            create_fig_ax = True
            fig, ax = plt.subplots(1, figsize=(7, 4))

        ax.step(self.v, self.flux_norm, color='black', where='mid', lw=1.5)
        ax.step(self.v, self.err_norm, color='cyan', where='mid', lw=1.5)

        # Shade masked regions, if indicated
        if(draw_masks == True):
            
            for i in range(len(self.masks)):
                v_mask = self.masks[i]
                ax.axvspan(v_mask[0], v_mask[1], color='darkgrey')


        for i in range(self.n_components):

            if len(self.best_values[i]) == 3:
                
                logN_str = str(np.round(self.best_values[i][0], 2))
                b_str = str(np.round(self.best_values[i][1], 2))
                dv_c_str = str(np.round(self.best_values[i][2], 2))

                if self.best_errs[i][0] == None:
                    logN_err_str = ''
                else:
                    logN_err_str = '±' + str(np.round(self.best_errs[i][0], 2))

                if self.best_errs[i][1] == None:
                    b_err_str = ''
                else:
                    b_err_str = '±' + str(np.round(self.best_errs[i][1], 2))

                if self.best_errs[i][2] == None:
                    dv_c_err_str = ''
                else:
                    dv_c_err_str = '±' + str(np.round(self.best_errs[i][2], 2))

                ax.plot(self.v, self.best_comp_fluxes[i], 
                        label='log$N$ = ' + logN_str + logN_err_str + '\n' +
                        '$b$ = ' + b_str + b_err_str + ' km/s' + '\n' +
                        'd$v_c$ = ' + dv_c_str + dv_c_err_str + ' km/s',
                        lw=1, color=colors[i])

                ax.axvline(self.best_values[i][2], color=colors[i], linestyle=':')

            # If constraining logT and b_NT
            if len(self.best_values[i]) == 4:

                logN_str = str(np.round(self.best_values[i][0], 2))
                logT_str = str(np.round(self.best_values[i][1], 2))
                b_NT_str = str(np.round(self.best_values[i][2], 2))
                dv_c_str = str(np.round(self.best_values[i][3], 2))

                if self.best_errs[i][0] == None:
                    logN_err_str = ''
                else:
                    logN_err_str = '±' + str(np.round(self.best_errs[i][0], 2))

                if self.best_errs[i][1] == None:
                    logT_err_str = ''
                else:
                    logT_err_str = '±' + str(np.round(self.best_errs[i][1], 2))

                if self.best_errs[i][2] == None:
                    b_NT_err_str = ''
                else:
                    b_NT_err_str = '±' + str(np.round(self.best_errs[i][2], 2))

                if self.best_errs[i][3] == None:
                    dv_c_err_str = ''
                else:
                    dv_c_err_str = '±' + str(np.round(self.best_errs[i][3], 2))

                ax.plot(self.v, self.best_comp_fluxes[i], 
                        label='log$N$ = ' + logN_str + logN_err_str + '\n' +
                        'log$T$ = ' + logT_str + logT_err_str + '\n' +
                        '$b_{NT}$ = ' + b_NT_str + b_NT_err_str + ' km/s' + '\n' +
                        'd$v_c$ = ' + dv_c_str + dv_c_err_str + ' km/s',
                        lw=1, color=colors[i])

                ax.axvline(self.best_values[i][3], color=colors[i], linestyle=':')       

        ax.plot(self.v, self.best_total_flux,
                lw=1.5, color='red')

        if label_axes == True:
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Flux (continuum normalized)')

        ax.axhline(0, color='red', linestyle=':')
        ax.axhline(1, color='green', linestyle=':')
        ax.axvline(0, color='brown', linestyle=':')

        ax.text(self.v[0]+30, np.max(self.err_norm)+.1, self.name, fontsize=12)

        if(legend == True):
            ax.legend(loc='lower right')

        if create_fig_ax == True:
            return fig, ax   

# Build a class to deal with a suite of ions as a whole

class ion(ion_transition):

    '''
    This class combines multiple transitions for an ion and tries to fit them simultaneously. It inherits from the ion_transition class
    '''

    def __init__(self, z, name, ion_transitions_list):

        '''
        Construct the ion object

        ion_name: the name of the ionic species being dealt with
        ion_transitions_list: the list of ionic transitions that are available for the given ion
        '''

        # Store the ion name
        self.z = z
        self.name = name

        self.ion_transitions_list = ion_transitions_list
        # Store the number of transitions available for the ion
        self.n_ion_transitions = len(ion_transitions_list)

        # Store the ionic transitions as a list - copy stuff over so we can modify things without affecting original objects
        self.ion_transitions_name_list = [ion_transition.name for ion_transition in self.ion_transitions_list]

    def plot_ion(self, fig=None, axes=None, draw_masks=True, label_axes=True, n_cols=2):

        '''
        Method to draw the available transitions for the ion

        axes: The multi-dimensional axes object upon which the transitions will be drawn
        draw_masks: Whether or not to shade the masked regions
        draw_cont_bounds: Whether or not to draw the continuum bounds
        label_axes: Whether or not to label axes
        '''

        create_fig_ax = False

        # For a single transition
        if self.n_ion_transitions == 1:

            ion_transition = self.ion_transitions_list[0]
            fig, axes = ion_transition.plot_ion_transition_spec(draw_masks=draw_masks, label_axes=label_axes)

            if fig is not None and axes is not None:
                create_fig_ax = True

        # For plotting the spectra of the ions in the suite in a panel
        else:


            n_rows = int(np.ceil(self.n_ion_transitions/n_cols))

            if fig == None and axes == None:
                create_fig_ax = True
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,4*n_rows), sharex=True, sharey=True)

            for i in range(self.n_ion_transitions):

                # Access the particular axis
                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i%n_rows, i//n_rows]

                ion_transition = self.ion_transitions_list[i]
                ion_transition.plot_ion_transition_spec(draw_masks=draw_masks, label_axes=False, fig=fig, ax=ax)

            if label_axes == True:
                # Answer 2 on https://stackoverflow.com/questions/42372509/how-to-add-a-shared-x-label-and-y-label-to-a-plot-created-with-pandas-plot
                # Create a big subplot
                ax_label = fig.add_subplot(111, frameon=False)
                # hide tick and tick label of the big axes
                ax_label.set_xticks([])
                ax_label.set_yticks([])
                ax_label.set_xlabel('Velocity (km/s)', labelpad=15) # Use argument `labelpad` to move label downwards.
                ax_label.set_ylabel('Flux (continuum normalized)', labelpad=25)

            plt.subplots_adjust(wspace=0, hspace=0)

        if create_fig_ax == True:
            return fig, axes

    def init_ion(self, init_values_list):

        '''
        Method to redefine the initial profile of the ion

        init_values_list: list of initial parameters for all Voigt components for all transitions. The first axis is
                          is the transition, the second is the component, and the last is logN, b, dv_c
        '''
        
        # Store initial guesses
        self.init_values_list = init_values_list

        for i in range(self.n_ion_transitions):

            ion_transition = self.ion_transitions_list[i]

            init_values = self.init_values_list[i]

            # Once again, using the inherited function
            ion_transition.init_ion_transition(init_values, lsf_convolve=ion_transition.lsf_convolve)


    def plot_ion_init_fit(self, draw_masks=True, legend=True, label_axes=True, n_cols=2):

        '''
        Method to plot the initial model for the ionic transitions

        axes: The axes object onto which the plot is drawn
        draw_masks: Whether or not to indicate masked regions
        legend: Whether or not to include a legend
        n_cols: Number of columns in axes object
        '''

        # For single transition
        if len(self.ion_transitions_list) == 1:

            ion_transition = self.ion_transitions_list[0]
            fig, axes = ion_transition.plot_ion_transition_init_fit(draw_masks=draw_masks, legend=legend, label_axes=label_axes)

        # For multiple transitions
        else:

            n_rows = int(np.ceil(self.n_ion_transitions/n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,4*n_rows), sharex=True, sharey=True)

            for i in range(self.n_ion_transitions):

                # Access the particular axis
                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i%n_rows, i//n_rows]

                ion_transition = self.ion_transitions_list[i]
                ion_transition.plot_ion_transition_init_fit(draw_masks=draw_masks, legend=legend, fig=fig, ax=ax, label_axes=False)

            if label_axes == True:
                ax_label = fig.add_subplot(111, frameon=False)
                ax_label.set_xticks([])
                ax_label.set_yticks([])
                ax_label.set_xlabel('Velocity (km/s)', labelpad=15)
                ax_label.set_ylabel('Flux (continuum normalized)', labelpad=25)

            plt.subplots_adjust(wspace=0, hspace=0)
 
        return fig, axes
    
    def get_upper_limit(self, b, v_c, load=False, loaddir = '', logN_min = -10, logN_max = 18.5, N_trials = 1000, N_samples=1000, cdf_ul=.975):

        '''
        Method to get a 2-sigma upper limit for a single component of an ion 

        b: the fixed Doppler parameter component of the ionic species (hopefully determined from joint Voigt profile fitting of detected ions)
        v_c: the centroid for the component (also determined from Voigt profile analysis)
        load: whether or not to load an already stored result
        loaddir: where to load stuff from
        logN_min: the lower bound of the uniform distribution for logN from which to draw samples for the Monte Carlo experiment
        logN_max: the upper bound of the uniform distribution for logN from which to draw samples for the Monte Carlo experiment
        N_trials: the number of trials to run for the Monte Carlo experiment
        N_samples: the number of samples of logN to draw from a uniform distribution of logN=[logN_min, logN_max]
        '''    

        if load == True:

            logN_MC_arr = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_MC_arr.npy')
            PDF_MC_arr = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/PDF_MC_arr.npy')
            CDF_MC_arr = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/CDF_MC_arr.npy')
            logN_CDF_median_grid = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_CDF_median_grid.npy')
            CDF_median = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/CDF_median.npy')
            #logN_PDF_samples = np.load(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_PDF_samples.npy')

        else:
            # First calculate the FWHM corresponding to the b-value
            # NOTE: this should ideally be computed numerically, but we'll just assume the profile to be effectively Gaussian

            fwhm = 2*np.sqrt(np.log(2))*b # Formula for Gaussian FWHM

            # Specify data strucutres to store column densities, and probabilistic quantities across various trials

            # For storing sampled column densities across trials
            logN_MC_arr = np.zeros((N_trials, N_samples))

            # For storing calculated chi-square across all trials
            chi_sq_MC_arr = np.zeros((N_trials, N_samples))

            # For storing PDF obtained from chi-sq for all trials
            # NOTE: this is a PDF of N, NOT logN, despite the samples being drawn from logN!
            # This is because logN is only a convenient way of sampling a large dynamic range in linear space
            # If we interpret the PDF to be defined over logN, it will not be normalizable 
            # Consider logN from -infinity to 0 to see why - the probability will stay flat, making the PDF not normalizable
            # If it is defined over linear space instead, it will be normalizable and can be used for subsequence CDF calculations
            PDF_MC_arr = np.zeros((N_trials, N_samples))

            # For storing numerically evaluated CDFs for the PDFs across various trials
            CDF_MC_arr = np.zeros((N_trials, N_samples))

            # List of interpolated CDFs across all trials, will help in median stacking all the CDFs later
            CDF_MC_interp_arr = []

            # Begin trials of the Monte Carlo experiment

            for i in range(N_trials):

                # Draw samples of logN from a uniform distribution
                logN_samples = np.sort(np.random.uniform(low=logN_min, high=logN_max, size=N_samples))

                # Now consider a particular value of logN
                for j in range(N_samples):
                    
                    # Finally, iterate through all the transitions present in the current ion
                    for k in range(self.n_ion_transitions):

                        # Isolate the ion transition
                        ion_transition = self.ion_transitions_list[k]

                        # Initialize the ion transition with the current value of logN, fixed b and v_c

                        # Up next, isolate the part of the spectrum within 2*fwhm of the centroid
                        # NOTE: use masked velocities! You will have need to run fit_ion_transition() for the masked velocities to be created
                        idx = (ion_transition.v_mask>v_c-fwhm) & (ion_transition.v_mask<v_c+fwhm)

                        # Calculate the chi-sq using the masked region
                        # Just add on the value you compile across transitions
                        model_flux = comp_model_spec_gen(ion_transition.v_mask, np.array([[logN_samples[j], b, v_c]]), 
                                                        ion_transition.wav0_rest, ion_transition.f,ion_transition.gamma,ion_transition.A,
                                                        ion_transition.lsf_convolve,
                                                        ion_transition.lsf, ion_transition.v_lsf)[1]
                        chi_sq_MC_arr[i,j] += np.sum(((model_flux[idx]-ion_transition.flux_norm_mask[idx])/ion_transition.err_norm_mask[idx])**2)

                # Store the samples from logN drawn for this trial
                logN_MC_arr[i,:] = logN_samples
                # Construct the PDF for the chi-sq evaluated for these samples
                PDF_MC_arr[i,:] = np.exp(-.5*chi_sq_MC_arr[i,:])
                # Normalize the PDF - REMEMBER, this is a PDF for N, not logN!
                PDF_MC_arr[i,:] /= integrate.trapz(x=10**logN_samples, y=PDF_MC_arr[i,:])
                # Construct the CDF
                CDF_MC_arr[i,:] = integrate.cumtrapz(x=10**logN_samples, y=PDF_MC_arr[i,:], initial=0)
                # Interpolate the CDF and store it
                CDF_MC_interp_arr.append(interpolate.interp1d(x=10**logN_samples, y=CDF_MC_arr[i,:], fill_value='extrapolate'))

            # Then, median stack the CDFs
            # Begin by generating a uniform grid for logN
            logN_CDF_median_grid = np.linspace(logN_min, logN_max, N_samples)

            # For storing the median CDF
            CDF_median = np.zeros(N_samples)

            for j in range(N_samples):
                # Evaluate the interpolated CDF for each trial (denoted by i) at each grid point (denoted by j)
                CDF_median[j] = np.median([CDF_MC_interp_arr[i](10**logN_CDF_median_grid[j]) for i in range(N_trials)])


            # Save everything
            if not os.path.exists(loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)):
                os.makedirs(loaddir+'Ions/z={0}/{1}'.format(self.z, self.name))
            np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_MC_arr.npy', arr=logN_MC_arr)
            np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/PDF_MC_arr.npy', arr=PDF_MC_arr)
            np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/CDF_MC_arr.npy', arr=CDF_MC_arr)
            np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_CDF_median_grid.npy', arr=logN_CDF_median_grid)
            np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/CDF_median.npy', arr=CDF_median)

        # Interpolate the inverse of the median stacked CDF
        CDF_median_inv_interp = interpolate.interp1d(x=CDF_median, y=10**logN_CDF_median_grid, fill_value='extrapolate')

        # Generate samples of using the median CDF
        CDF_samples = np.random.uniform(size=N_samples)

        # Evaluate the inverses for all these CDF samples
        # NOTE: this won't work well for upper limits, because a HUGE range of column densities have similar values for the CDF
        logN_PDF_samples = np.array([np.log10(CDF_median_inv_interp(y)) for y in CDF_samples])
        # Save this for CLOUDY modeling later
        np.save(file=loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)+'/logN_PDF_samples.npy', arr=logN_PDF_samples)

        # Evaluate the interpolated inverse median-stacked CDF at the required level of confidence to get the upper limit
        logN_ul = np.log10(CDF_median_inv_interp(cdf_ul))

        # Set the logN samples, normalized PDFs, and numerically evaluated CDFs from the Monte Carlo run
        # Also returned the uniform grid of logN and the median stacked CDF evaluated across this uniform grid 
        # Finally, report the calculated upper limit
        self.logN_MC_arr = logN_MC_arr
        self.PDF_MC_arr = PDF_MC_arr
        self.CDF_MC_arr = CDF_MC_arr 
        self.logN_CDF_median_grid = logN_CDF_median_grid
        self.CDF_median = CDF_median
        self.logN_PDF_samples = logN_PDF_samples
        self.logN_ul = logN_ul

    def fit_ion(self, logN_min = -np.inf, logN_max = 25., 
                      b_min = 0., b_max = 100., 
                      logT_min = 4., logT_max = 6.,
                      b_NT_min = 0., b_NT_max = 50.,
                      dv_c_min = -250., dv_c_max = 250.,
                      tie_params_list=[], 
                      fix_params_list=[], 
                      lower_bounds_dict={}, upper_bounds_dict={},
                      exclude_models=None):

        '''
        Method to simultaneously fit available transitions of the ion

        logN_min: default minimum of column density
        logN_max: default maximum of column density

        b_min: default minimum of Doppler width
        b_max: default maximum of Dopper width

        logT_min:
        logT_max:

        b_NT_min:
        b_NT_max:

        dv_c_min: default minimum of centroid shift
        dv_c_max: default maximum of centroid shift

        tie_params_list:
        fix_params_list:
        lower_bounds_dict:

        exclude_models: String representations for models to exclude from the prior (will be fed into the objective function which can handle it)
        '''

        # Create an LMFIT parameters object
        params = Parameters()

        # First, set the initial values of all parameters

        # For all ion transitions
        for i in range(self.n_ion_transitions):
            ion_transition = self.ion_transitions_list[i]
            # For each component
            for j in range(ion_transition.n_components):
                # If constraining only b-value
                if len(ion_transition.init_values[j]) == 3:
                    # Create parameters corresponding to the column density, Doppler width, and velocity centroid
                    params.add(name = "it{}c{}_logN".format(i+1,j+1), value = ion_transition.init_values[j][0], min=logN_min, max=logN_max)
                    params.add(name = "it{}c{}_b".format(i+1,j+1), value = ion_transition.init_values[j][1], min=b_min, max=b_max)
                    params.add(name = "it{}c{}_dv_c".format(i+1,j+1), value = ion_transition.init_values[j][2], min=dv_c_min, max=dv_c_max)
                # If constraining logT and b_NT
                if len(ion_transition.init_values[j]) == 4:
                    # Create parameters corresponding to the column density, Doppler width, and velocity centroid
                    params.add(name = "it{}c{}_logN".format(i+1,j+1), value = ion_transition.init_values[j][0], min=logN_min, max=logN_max)
                    params.add(name = "it{}c{}_logT".format(i+1,j+1), value = ion_transition.init_values[j][1], min=logT_min, max=logT_max)
                    params.add(name = "it{}c{}_b_NT".format(i+1,j+1), value = ion_transition.init_values[j][2], min=b_NT_min, max=b_NT_max)
                    params.add(name = "it{}c{}_dv_c".format(i+1,j+1), value = ion_transition.init_values[j][3], min=dv_c_min, max=dv_c_max)

        # Tie all indicated parameter pairs together
        for pair in tie_params_list:
            params[pair[1]].set(expr=pair[0])

        # For indicated parameters, fix their values
        for p in fix_params_list:
            params[p].set(vary = False)
        
        # For indicated parameters, set lower bounds for prior
        for p in list(lower_bounds_dict.keys()):
            params[p].set(min = lower_bounds_dict[p])

        # For indicated parameters, set upper bounds for prior
        for p in list(upper_bounds_dict.keys()):
            params[p].set(max = upper_bounds_dict[p])        

        # Now, optimize the objective function
        # NaN policy of propagate for excluding models, is this safe?
        self.exclude_models = exclude_models
        result = minimize(weighted_residuals, params, args=(self.ion_transitions_list, 'lmfit', self.exclude_models), nan_policy='propagate')

        self.result = result 

        # Isolate the best values and errors for the shared components
        self.best_values_list = []
        self.best_errs_list = []

        for i in range(self.n_ion_transitions):
            ion_transition = self.ion_transitions_list[i]

            # Get list of parameters corresponding to this ion transition
            # Including c after {} helps break degeneracy between cases like it1 and it10
            param_names = [l for l in list(params.valuesdict().keys()) if 'it{}c'.format(i+1) in l]

            # Get best values corresponding to this transition
            best_values = [self.result.params[p].value for p in param_names]
            best_errs = [self.result.params[p].stderr for p in param_names]

            self.best_values_list.append(best_values)
            self.best_errs_list.append(best_errs)

            ion_transition.best_values = []
            ion_transition.best_errs = []

            for j in range(ion_transition.n_components):
                # If constraining only b-value
                if 'it{}c{}_b'.format(i+1, j+1) in param_names:
                    ion_transition.best_values.append([best_values[param_names.index('it{}c{}_logN'.format(i+1, j+1))],
                                                       best_values[param_names.index('it{}c{}_b'.format(i+1, j+1))],
                                                       best_values[param_names.index('it{}c{}_dv_c'.format(i+1, j+1))]])
                    
                    ion_transition.best_errs.append([best_errs[param_names.index('it{}c{}_logN'.format(i+1, j+1))],
                                                     best_errs[param_names.index('it{}c{}_b'.format(i+1, j+1))],
                                                     best_errs[param_names.index('it{}c{}_dv_c'.format(i+1, j+1))]])
                    
                # If constraining logT and b_NT
                else:
                    ion_transition.best_values.append([best_values[param_names.index('it{}c{}_logN'.format(i+1, j+1))],
                                                       best_values[param_names.index('it{}c{}_logT'.format(i+1, j+1))],
                                                       best_values[param_names.index('it{}c{}_b_NT'.format(i+1, j+1))],
                                                       best_values[param_names.index('it{}c{}_dv_c'.format(i+1, j+1))]])
                    
                    ion_transition.best_errs.append([best_errs[param_names.index('it{}c{}_logN'.format(i+1, j+1))],
                                                       best_errs[param_names.index('it{}c{}_logT'.format(i+1, j+1))],
                                                       best_errs[param_names.index('it{}c{}_b_NT'.format(i+1, j+1))],
                                                       best_errs[param_names.index('it{}c{}_dv_c'.format(i+1, j+1))]])

        # Generate the component fluxes

        for i in range(self.n_ion_transitions):

            ion_transition = self.ion_transitions_list[i]

            # Generate fluxes for each component
            best_comp_fluxes, best_total_flux = comp_model_spec_gen(ion_transition.v, ion_transition.best_values, 
                                                                    ion_transition.wav0_rest, 
                                                                    ion_transition.f, ion_transition.gamma, ion_transition.A,
                                                                    ion_transition.lsf_convolve, 
                                                                    ion_transition.lsf, ion_transition.v_lsf)

            # Set all these properties for the ion object
            ion_transition.best_comp_fluxes = best_comp_fluxes
            ion_transition.best_total_flux = best_total_flux

    def plot_ion_best_fit(self, draw_masks=True, legend=True, label_axes=True, n_cols=2):

        '''
        Method to plot the best fit profiles for all transitions of the ion

        draw_masks: Whether or not to indicate masked regions
        legend: Whether or not to include a legend
        '''

        # If there is a single transition

        if len(self.ion_transitions_list) == 1:

            ion_transition = self.ion_transitions_list[0]
            fig, axes = ion_transition.plot_ion_transition_best_fit(draw_masks=draw_masks, label_axes=label_axes, legend=legend)           

        # If there are multiple transitions
        else:

            n_rows = int(np.ceil(self.n_ion_transitions/n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,4*n_rows), sharex=True, sharey=True)

            for i in range(self.n_ion_transitions):

                # Access the particular axis
                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i%n_rows, i//n_rows]

                ion_transition = self.ion_transitions_list[i]
                ion_transition.plot_ion_transition_best_fit(fig=fig, ax=ax, draw_masks=draw_masks, label_axes=False, legend=legend)

            if label_axes == True:
                ax_label = fig.add_subplot(111, frameon=False)
                ax_label.set_xticks([])
                ax_label.set_yticks([])
                ax_label.set_xlabel('Velocity (km/s)', labelpad=15)
                ax_label.set_ylabel('Flux (continuum normalized)', labelpad=25)

            plt.subplots_adjust(wspace=0, hspace=0)                

        return fig, axes

    def fit_ion_emcee(self, load=False, loaddir='', n_walkers=100, n_steps=8000, n_burn=500, n_thin=1, scale_covar=1):

        '''
        Run an MCMC chain on the parameter space using the LMFIT solution (with covariance) as an initializing region

        load: Whether or not to load chain
        loaddir: Where to load the chain from, if at all
        n_walkers: Number of walkers to deploy
        n_steps: Number of steps to be taken by each walker
        scale_covar: The factor by which to scale the covariance matrix
        exclude_models:
        '''

        if load == True:
            with open(loaddir+'Ions/z={0}/{1}/params.pkl'.format(self.z, self.name), 'rb') as f:
                self.result_emcee = pickle.load(f)

        else:
            # Draw samples from a multivariate normal distribution centered at the LMFIT solution
            mean = []

            for k in self.result.var_names:
                mean.append(self.result.params[k].value)
            
            mean = np.array(mean)

            if(str(type(self.result.covar)) == "<class 'NoneType'>"):
                covar = 1e-4 * np.random.randn(len(mean), len(mean))
            else:
                covar = self.result.covar

            covar = scale_covar*covar

            init_pos = np.random.multivariate_normal(mean, covar, size=n_walkers)            

            # Run the MCMC chain
            # Again NaN policy is simply for handling excluded models
            result_emcee = minimize(weighted_residuals, self.result.params, method='emcee', args=(self.ion_transitions_list, 'emcee', self.exclude_models),
                                    float_behavior='posterior', nan_policy='propagate',
                                    nwalkers=n_walkers, steps=n_steps, burn=n_burn, thin=n_thin, pos=init_pos, is_weighted=True, progress=True)

            self.result_emcee = result_emcee

            if not os.path.exists(loaddir+'Ions/z={0}/{1}'.format(self.z, self.name)):
                os.makedirs(loaddir+'Ions/z={0}/{1}'.format(self.z, self.name))
            with open(loaddir+'Ions/z={0}/{1}/params.pkl'.format(self.z, self.name), 'wb') as f:
                pickle.dump(self.result_emcee, f)

        # Store the median, 16 percent, and 84 percent errors, and 5% and 95% values
        self.param_errs_lo = []
        self.param_medians = []
        self.param_errs_hi = []
        self.param_lower_lims = []
        self.param_upper_lims = []

        for i in range(len(self.result_emcee.init_vals)):
            
            flat_sample = self.result_emcee.flatchain[self.result_emcee.var_names[i]]
            self.param_medians.append(np.median(flat_sample))
            self.param_errs_lo.append(self.param_medians[i]-np.percentile(flat_sample, 100*(1-cdf_1sig)))
            self.param_errs_hi.append(np.percentile(flat_sample, 100*cdf_1sig)-self.param_medians[i])
            self.param_lower_lims.append(np.percentile(flat_sample, 100*(1-cdf_3sig)))
            self.param_upper_lims.append(np.percentile(flat_sample, 100*cdf_3sig))
        
        # Also get maximum likelihood estimate
        step_max, walker_max = np.unravel_index(self.result_emcee.lnprob.argmax(), self.result_emcee.lnprob.shape)
        self.params_mle = self.result_emcee.chain[step_max, walker_max, :].tolist()

    def plot_chains(self, fig=None, axes=None, n_cols=2, label_axes=True):

        '''
        Method to plot emcee chains
        '''

        create_fig_ax = False

        n_steps = self.result_emcee.chain.shape[0]
        n_walkers = self.result_emcee.chain.shape[1]
        n_params = self.result_emcee.chain.shape[2]

        n_rows = int(np.ceil(n_params/n_cols))

        if fig==None and axes==None:
            create_fig_ax = True
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_rows, 4*n_cols))

        for i in range(n_params):
            for j in range(n_walkers):
                chain = self.result_emcee.chain[:,j,i]

                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i%n_rows, i//n_rows]

                ax.plot(chain)
            
            ax.set_ylabel(self.result_emcee.var_names[i])

        if label_axes == True:
            ax_label = fig.add_subplot(111, frameon=False)
            ax_label.set_xticks([])
            ax_label.set_yticks([])
            ax_label.set_xlabel('Steps', labelpad=15)

        plt.tight_layout()      

        if create_fig_ax == True:
            return fig, axes

    def plot_samples(self, fig=None, axes=None, n_samples=100, draw_masks=True, label_axes=True, n_cols=2):

        '''
        Method to plot randomly chosen samples from the posterior explored by MCMC walkers
        '''

        create_fig_ax = False

        # Isolate number of steps and walkers
        n_steps = self.result_emcee.chain.shape[0]
        n_walkers = self.result_emcee.chain.shape[1]

        # For each sample
        # Select a walker
        walker_selections = np.random.randint(n_walkers, size=n_samples)
        # Select a corresponding step
        step_selections = np.random.randint(n_steps, size=n_samples)

        # Generate an ensemble of samples lists
        sample_lists_ensemble = []

        for i in range(n_samples+2):

            # Isolate the MCMC sample
            if i < n_samples:
                # Isolate walker
                walker = walker_selections[i]
                # Isolate step
                step = step_selections[i]
                sample_emcee = self.result_emcee.chain[step, walker, :].tolist()
            elif i == n_samples:
                # Get median
                sample_emcee = self.param_medians
            else:
                # Get MLE
                sample_emcee = self.params_mle

            # Map sample components to variable names
            sample_emcee_dict = {k:v for (k,v) in zip(self.result_emcee.var_names, sample_emcee)}

            # Construct the full list of sample values
            sample_values_list = []

            for j in range(self.n_ion_transitions):

                ion_transition = self.ion_transitions_list[j]
                sample_values = []

                # Get list of parameters corresponding to this ion transition
                param_names = [l for l in list(self.result_emcee.params.valuesdict().keys()) if 'it{}c'.format(j+1) in l]

                # Get values corresponding to this transition
                for p in param_names:
                    # If parameter was varied, get its value from the MCMC sample
                    if self.result_emcee.params[p].vary == True:
                        sample_values.append(sample_emcee_dict[p])
                    # If the parameter is not varied
                    else:
                        # See if the parameter has an expression
                        if self.result_emcee.params[p].expr is not None:
                            # Get the tied parameter name
                            p_tie = self.result_emcee.params[p].expr
                            # If the tied parameter is varied, pull its value from the sample
                            if self.result_emcee.params[p_tie].vary == True:
                                sample_values.append(sample_emcee_dict[p_tie])
                            # If the tied parameter is NOT varied, get its fixed value
                            # It is assumed here that the tied parameter is NOT tied to yet another parameter
                            # Not respecting this assumption can lead to problems - possible solution, implement recursive solution
                            else:
                                sample_values.append(self.result_emcee.params[p].value)
                        # If the parameter does not have an expression
                        else:                                
                            # Get the fixed value
                            sample_values.append(self.result_emcee.params[p].value)

                sample_values_list.append(sample_values)

            sample_lists_ensemble.append(sample_values_list)
       
        # Generate the full samples for the median and MLE parameters
        
        self.param_medians_reshape = []
        self.params_mle_reshape = []

        # If there are multiple transitions

        if fig == None and axes == None:

            create_fig_ax = True

            if len(self.ion_transitions_list)==1:
                n_cols=1

            n_rows = int(np.ceil(self.n_ion_transitions/n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,4*n_rows), sharex=True, sharey=True)

            if len(self.ion_transitions_list)==1:
                axes = [axes]

        for i in range(self.n_ion_transitions):

            if n_rows == 1:
                ax = axes[i]
            else:
                ax = axes[i%n_rows, i//n_rows]

            ion_transition = self.ion_transitions_list[i]
            param_names = [l for l in list(self.result_emcee.params.valuesdict().keys()) if 'it{}c'.format(i+1) in l]

            ion_transition.plot_ion_transition_spec(fig=fig, ax=ax, label_axes=False)
            
            for j in range(n_samples+2):
                
                sample_values_list = sample_lists_ensemble[j]
                sample_values = sample_values_list[i]
                sample_values_reshape = []

                for k in range(ion_transition.n_components):

                    if 'it{}c{}_b'.format(i+1, k+1) in param_names:
                        sample_values_reshape.append([sample_values[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                            sample_values[param_names.index('it{}c{}_b'.format(i+1, k+1))],
                                            sample_values[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])
                        
                    else:
                        sample_values_reshape.append([sample_values[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                            sample_values[param_names.index('it{}c{}_logT'.format(i+1, k+1))],
                                            sample_values[param_names.index('it{}c{}_b_NT'.format(i+1, k+1))],
                                            sample_values[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])


                sample_comp_fluxes, sample_total_flux = comp_model_spec_gen(ion_transition.v, sample_values_reshape, 
                                                                    ion_transition.wav0_rest, ion_transition.f, ion_transition.gamma, ion_transition.A,
                                                                    ion_transition.lsf_convolve, ion_transition.lsf, ion_transition.v_lsf)  


                if j < n_samples:
                    # For MCMC samples
                    c = 'lightgray'
                    alpha=.6
                    lw=.5
                elif j == n_samples:
                    # For median
                    self.param_medians_reshape.append(sample_values_reshape)
                    c = 'red'
                    alpha=1
                    lw=1.5
                    
                    for q in range(len(sample_comp_fluxes)):
                        ax.vlines(x=sample_values_reshape[q][-1], ymin=1.1, ymax=1.3, color=colors[q], lw=2)
                        ax.plot(ion_transition.v, sample_comp_fluxes[q], color=colors[q], lw=.8)
                else:
                    # For MLE
                    self.params_mle_reshape.append(sample_values_reshape)
                    c = 'indigo'
                    alpha=0
                    lw=0

                ax.plot(ion_transition.v, sample_total_flux, color=c, alpha=alpha, lw=lw)

            if label_axes == True:
                ax_label = fig.add_subplot(111, frameon=False)
                ax_label.set_xticks([])
                ax_label.set_yticks([])
                ax_label.set_xlabel('Velocity (km/s)', labelpad=15)
                ax_label.set_ylabel('Flux (continuum normalized)', labelpad=25)

            plt.subplots_adjust(wspace=0, hspace=0)      

        if create_fig_ax == True:
            return fig, axes

    def plot_corner(self, loaddir='', quantiles=(1-cdf_1sig, 0.5, cdf_1sig)):

        self.corner_plot = corner.corner(self.result_emcee.flatchain, labels=self.result_emcee.var_names,quantiles=quantiles,
                                        show_titles=True, plot_density=True, levels=[2*cdf_1sig-1, 2*cdf_2sig-1], contour_kwargs={'colors':'red'})
        
        if loaddir != '':
            plt.savefig(loaddir+'Ions/z={0}/{1}/corner.pdf'.format(self.z, self.name), dpi=300)

class ion_suite(ion):

    def __init__(self, z, name, ion_list):

        # Save list of ions
        self.z = z
        self.name = name
        self.ion_list = ion_list

        # Get total number of transitions
        self.ion_transitions_list = []
        self.n_ion_transitions = 0

        for ion in self.ion_list:
            
            self.n_ion_transitions = self.n_ion_transitions + ion.n_ion_transitions

            # Append all ion transitions
            for ion_transition in ion.ion_transitions_list:
                self.ion_transitions_list.append(ion_transition)

        # Also store names of ions and ion transitions
        self.ion_name_list = [ion.name for ion in self.ion_list]
        self.ion_transitions_name_list = [ion_transition.name for ion_transition in self.ion_transitions_list]

    def plot_ion_suite(self, fig=None, axes=None, draw_masks=True, label_axes=True, n_cols=2):

        '''
        Method to draw the available transitions for the ion

        fig: Figure object to make the plot
        axes: Axes object to make the plot
        draw_masks: Whether or not to shade the masked regions
        draw_cont_bounds: Whether or not to draw the continuum bounds
        '''

        fig, axes = super().plot_ion(fig=fig, axes=axes, draw_masks=draw_masks, label_axes=label_axes, n_cols=n_cols)

        if fig is not None and axes is not None:
            return fig, axes

    def init_ion_suite(self, init_values_list):

        '''
        Method to redefine the initial profile of the ion

        init_values_list: list of initial parameters for all Voigt components for all transitions. The first axis is
                          is the transition, the second is the component, and the last is logN, b, dv_c
        '''
        
        super().init_ion(init_values_list)

    def plot_ion_suite_init_fit(self, draw_masks=True, legend=True, n_cols=2):

        '''
        Method to plot the initial model for the ionic transitions

        axes: The axes object onto which the plot is drawn
        draw_masks: Whether or not to indicate masked regions
        legend: Whether or not to include a legend
        '''

        fig, axes = super().plot_ion_init_fit(draw_masks=draw_masks, legend=legend, n_cols=n_cols)

        if fig is not None and axes is not None:
            return fig, axes        

    def fit_ion_suite(self, logN_min = -np.inf, logN_max = 25., 
                      b_min = 0., b_max = 100., 
                      logT_min = 4., logT_max = 6.,
                      b_NT_min = 0., b_NT_max = 50.,
                      dv_c_min = -250., dv_c_max = 250.,
                      tie_params_list=[], fix_params_list=[], 
                      lower_bounds_dict={}, upper_bounds_dict={},
                      exclude_models=None):

        '''
        Method to simultaneously fit available transitions of the ion

        logN_min: default minimum of column density
        logN_max: default maximum of column density

        b_min: default minimum of Doppler width
        b_max: default maximum of Dopper width

        dv_c_min: default minimum of centroid shift
        dv_c_max: default maximum of centroid shift

        tie_params_list:
        fix_params_list:
        lower_bounds_dict:

        '''

        super().fit_ion(logN_min=logN_min, logN_max=logN_max, 
                        b_min=b_min, b_max=b_max, 
                        logT_min=logT_min, logT_max=logT_max,
                        b_NT_min=b_NT_min, b_NT_max=b_NT_max,
                        dv_c_min = dv_c_min, dv_c_max=dv_c_max,
                        tie_params_list=tie_params_list, fix_params_list=fix_params_list, 
                        lower_bounds_dict=lower_bounds_dict, upper_bounds_dict=upper_bounds_dict, exclude_models=exclude_models)  

    def plot_ion_suite_best_fit(self, draw_masks=True, legend=True, n_cols=2):

        '''
        Method to plot the best fit profiles for all transitions of the ion

        draw_masks: Whether or not to indicate masked regions
        legend: Whether or not to include a legend
        '''

        fig, axes = super().plot_ion_best_fit(draw_masks=draw_masks, legend=legend, n_cols=n_cols)

        if fig is not None and axes is not None:
            return fig, axes           

    def fit_ion_suite_emcee(self, load=False, loaddir='', n_walkers=100, n_steps=8000, n_burn=500, n_thin=1, scale_covar=1):

        '''
        Run an MCMC chain on the parameter space using the LMFIT solution (with covariance) as an initializing region

        load: Whether or not to load chain
        loaddir: Where to load the chain from, if at all
        n_walkers: Number of walkers to deploy
        n_steps: Number of steps to be taken by each walker
        scale_covar: The factor by which to scale the covariance matrix
        '''

        super().fit_ion_emcee(load=load, loaddir=loaddir,
                              n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn, n_thin=n_thin, scale_covar=scale_covar)

    def plot_chains(self, fig=None, axes=None, n_cols=2):

        '''
        Method to plot emcee chains
        '''

        fig, axes = super().plot_chains(fig=fig, axes=axes, n_cols=n_cols)

        if fig is not None and axes is not None:
            return fig, axes

    def plot_samples(self, fig=None, axes=None, n_samples=100, draw_masks=True, label_axes=True, n_cols=2):

        '''
        Method to plot randomly chosen samples from the posterior explored by MCMC walkers
        '''

        fig, axes = super().plot_samples(fig=fig, axes=axes, n_samples=n_samples, draw_masks=draw_masks, label_axes=label_axes, n_cols=n_cols)

        if fig is not None and axes is not None:
            return fig, axes

    def plot_corner(self, loaddir='',  quantiles=(1-cdf_1sig, 0.5, cdf_1sig)):

        super().plot_corner(loaddir=loaddir, quantiles=quantiles)

class ion_summary(ion_suite):

    def __init__(self, ion_suite_list):

        # Save list of ions
        self.ion_suite_list = ion_suite_list

        # Get total number of transitions
        self.ion_transitions_list = []
        self.n_ion_transitions = 0

        for ion_suite in self.ion_suite_list:
            
            self.n_ion_transitions = self.n_ion_transitions + ion_suite.n_ion_transitions

            # Append all ion transitions
            for ion_transition in ion_suite.ion_transitions_list:
                self.ion_transitions_list.append(ion_transition)     

        # Store names of ions and ion transitions
        self.ion_suite_name_list = [ion_suite.name for ion_suite in self.ion_suite_list]
        self.ion_transitions_name_list = [ion_transition.name for ion_transition in self.ion_transitions_list]

    def plot_samples(self, fig=None, axes=None, n_samples=100, 
                     draw_masks=True, label_axes=True, label_axes_pad_x = 15, label_axes_pad_y = 25,
                     n_cols=2):

        '''
        Method to plot randomly chosen samples from the posterior explored by MCMC walkers
        '''

        create_fig_ax = False

        # Get number of rows by combining total number of transitions across all ions
        n_rows = int(np.ceil(self.n_ion_transitions/n_cols))

        if fig == None and axes == None:
            create_fig_ax = True
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4*n_rows), sharex=True, sharey=True)

        # Create a counter for ion transitions
        ion_transition_ctr = 0

        # For each ion suite
        for ion_suite in self.ion_suite_list:

            ion_suite.param_medians_reshape = []
            # While at the medians reshape the errors
            ion_suite.param_errs_lo_reshape = []
            ion_suite.param_errs_hi_reshape = []
            # Reshape the MLE too
            ion_suite.params_mle_reshape = []

            # Construct the full MCMC sample for the ion suite

            # Isolate number of steps and walkers
            n_steps = ion_suite.result_emcee.chain.shape[0]
            n_walkers = ion_suite.result_emcee.chain.shape[1]

            # For each sample
            # Select a walker
            walker_selections = np.random.randint(n_walkers, size=n_samples)
            # Select a corresponding step
            step_selections = np.random.randint(n_steps, size=n_samples)

            # Generate an ensemble of samples lists
            sample_lists_ensemble = []

            for i in range(n_samples+2):

                # Isolate the MCMC sample
                if i < n_samples:
                    # Isolate walker
                    walker = walker_selections[i]
                    # Isolate step
                    step = step_selections[i]
                    sample_emcee = ion_suite.result_emcee.chain[step, walker, :].tolist()
                elif i == n_samples:
                    # Get median
                    sample_emcee = ion_suite.param_medians
                    # Also reshape errors while at the median
                    sample_emcee_errs_lo = ion_suite.param_errs_lo
                    sample_emcee_errs_hi = ion_suite.param_errs_hi
                else:
                    # Get MLE
                    sample_emcee = ion_suite.params_mle

                # Map sample components to variable names
                sample_emcee_dict = {k:v for (k,v) in zip(ion_suite.result_emcee.var_names, sample_emcee)}

                # Also reshape errors while at the median
                if i == n_samples:
                    sample_emcee_errs_lo_dict = {k:v for (k,v) in zip(ion_suite.result_emcee.var_names, sample_emcee_errs_lo)}
                    sample_emcee_errs_hi_dict = {k:v for (k,v) in zip(ion_suite.result_emcee.var_names, sample_emcee_errs_hi)}

                # Construct the full list of sample values
                sample_values_list = []

                # Reshape errors while at the median
                if i == n_samples:
                    sample_values_errs_lo_list = []
                    sample_values_errs_hi_list = []

                for j in range(ion_suite.n_ion_transitions):

                    ion_transition = ion_suite.ion_transitions_list[j]
                    sample_values = []
                    # Reshape errors while at the median
                    if i == n_samples:
                        sample_values_errs_lo = []
                        sample_values_errs_hi = []

                    # Get list of parameters corresponding to this ion transition
                    param_names = [l for l in list(ion_suite.result_emcee.params.valuesdict().keys()) if 'it{}c'.format(j+1) in l]

                    # Get values corresponding to this transition
                    for p in param_names:
                        # If parameter was varied, get its value from the MCMC sample
                        if ion_suite.result_emcee.params[p].vary == True:
                            sample_values.append(sample_emcee_dict[p])

                            # Append errors while at the median
                            if i == n_samples:
                                sample_values_errs_lo.append(sample_emcee_errs_lo_dict[p])
                                sample_values_errs_hi.append(sample_emcee_errs_hi_dict[p])

                        # If the parameter is not varied
                        else:
                            # See if the parameter has an expression
                            if ion_suite.result_emcee.params[p].expr is not None:
                                # Get the tied parameter name
                                p_tie = ion_suite.result_emcee.params[p].expr
                                # If the tied parameter is varied, pull its value from the sample
                                if ion_suite.result_emcee.params[p_tie].vary == True:
                                    sample_values.append(sample_emcee_dict[p_tie])

                                    # Append tied errors when at the median
                                    if i == n_samples:
                                        sample_values_errs_lo.append(sample_emcee_errs_lo_dict[p_tie])
                                        sample_values_errs_hi.append(sample_emcee_errs_hi_dict[p_tie])
                                    
                                # If the tied parameter is NOT varied, get its fixed value
                                # It is assumed here that the tied parameter is NOT tied to yet another parameter
                                # Not respecting this assumption can lead to problems - possible solution, implement recursive solution
                                else:
                                    sample_values.append(ion_suite.result_emcee.params[p_tie].value)
                                    
                                    # Append zero errors for fixed median
                                    if i == n_samples:
                                        sample_values_errs_lo.append(0)
                                        sample_values_errs_hi.append(0)

                            else:
                                # Get the fixed value
                                sample_values.append(ion_suite.result_emcee.params[p].value)

                                # Append zero errors for fixed median
                                if i == n_samples:
                                    sample_values_errs_lo.append(0)
                                    sample_values_errs_hi.append(0)

                    sample_values_list.append(sample_values)
                    if i == n_samples:
                        sample_values_errs_lo_list.append(sample_values_errs_lo)
                        sample_values_errs_hi_list.append(sample_values_errs_hi)                    

                sample_lists_ensemble.append(sample_values_list)

            for i in range(ion_suite.n_ion_transitions):

                ax = axes[ion_transition_ctr%n_rows, ion_transition_ctr//n_rows]

                ion_transition = ion_suite.ion_transitions_list[i]
                param_names = [l for l in list(ion_suite.result_emcee.params.valuesdict().keys()) if 'it{}c'.format(i+1) in l]
                
                ion_transition.plot_ion_transition_spec(fig=fig, ax=ax, label_axes=False)
                
                for j in range(n_samples+2):
                
                    sample_values_list = sample_lists_ensemble[j]
                    sample_values = sample_values_list[i]
                    sample_values_reshape = []   

                    if j == n_samples:
                        sample_values_errs_lo = sample_values_errs_lo_list[i]
                        sample_values_errs_hi = sample_values_errs_hi_list[i]
                        sample_values_errs_lo_reshape = []
                        sample_values_errs_hi_reshape = []        

                    for k in range(ion_transition.n_components):

                        if 'it{}c{}_b'.format(i+1, k+1) in param_names:
                            sample_values_reshape.append([sample_values[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                sample_values[param_names.index('it{}c{}_b'.format(i+1, k+1))],
                                                sample_values[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])
                            
                            if j == n_samples:

                                sample_values_errs_lo_reshape.append([sample_values_errs_lo[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    sample_values_errs_lo[param_names.index('it{}c{}_b'.format(i+1, k+1))],
                                                    sample_values_errs_lo[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])

                                sample_values_errs_hi_reshape.append([sample_values_errs_hi[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    sample_values_errs_hi[param_names.index('it{}c{}_b'.format(i+1, k+1))],
                                                    sample_values_errs_hi[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])   
                                                            
                        else:
                            if j != n_samples:
                                sample_values_reshape.append([sample_values[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    sample_values[param_names.index('it{}c{}_logT'.format(i+1, k+1))],
                                                    sample_values[param_names.index('it{}c{}_b_NT'.format(i+1, k+1))],
                                                    sample_values[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])
                            
                            if j == n_samples:

                                if ion_suite.result_emcee.params['it{}c{}_logT'.format(i+1, k+1)].vary == True:
                                    logT_dist = ion_suite.result_emcee.flatchain['it{}c{}_logT'.format(i+1, k+1)]
                                    b_NT_dist = ion_suite.result_emcee.flatchain['it{}c{}_b_NT'.format(i+1, k+1)]
                                else:
                                    logT_dist = ion_suite.result_emcee.flatchain[ion_suite.result_emcee.params['it{}c{}_logT'.format(i+1, k+1)].expr]
                                    b_NT_dist = ion_suite.result_emcee.flatchain[ion_suite.result_emcee.params['it{}c{}_b_NT'.format(i+1, k+1)].expr]
                                
                                b_dist = np.sqrt(2*k_B*10**logT_dist/(ion_transition.A*amu) + b_NT_dist**2)

                                b_mid = np.median(b_dist)
                                b_err_lo = b_mid - np.percentile(b_dist, 100*(1-cdf_1sig))
                                b_err_hi = np.percentile(b_dist, 100*cdf_1sig) - b_mid

                                sample_values_reshape.append([sample_values[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    b_mid,
                                                    sample_values[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])

                                sample_values_errs_lo_reshape.append([sample_values_errs_lo[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    b_err_lo,
                                                    sample_values_errs_lo[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])
                                
                                sample_values_errs_hi_reshape.append([sample_values_errs_hi[param_names.index('it{}c{}_logN'.format(i+1, k+1))],
                                                    b_err_hi,
                                                    sample_values_errs_hi[param_names.index('it{}c{}_dv_c'.format(i+1, k+1))]])

                    sample_comp_fluxes, sample_total_flux = comp_model_spec_gen(ion_transition.v, sample_values_reshape, 
                                                                        ion_transition.wav0_rest, ion_transition.f, ion_transition.gamma, ion_transition.A,
                                                                        ion_transition.lsf_convolve, ion_transition.lsf, ion_transition.v_lsf)  

                    if j < n_samples:
                        # For MCMC samples
                        c = 'lightgray'
                        alpha=.6
                        lw=.5

                    elif j == n_samples:
                        # For median
                        ion_suite.param_medians_reshape.append(sample_values_reshape)
                        ion_suite.param_errs_lo_reshape.append(sample_values_errs_lo_reshape)
                        ion_suite.param_errs_hi_reshape.append(sample_values_errs_hi_reshape)
                        c = 'red'
                        alpha=1
                        lw=1.5

                        # Plot velocity centroids from the median samples
                        for q in range(ion_transition.n_components):
                            ax.plot(ion_transition.v, sample_comp_fluxes[q], color=colors[q], lw=.8)
                            ax.vlines(x=sample_values_reshape[q][-1], ymin=1.1, ymax=1.3, color=colors[q], lw=2)
                    
                    else:
                        # For MLE
                        ion_suite.params_mle_reshape.append(sample_values_reshape)
                        c = 'indigo'
                        alpha=0
                        lw=0

                    ax.plot(ion_transition.v, sample_total_flux, color=c, alpha=alpha, lw=lw)

                ion_transition_ctr = ion_transition_ctr + 1

        if label_axes == True:
            ax_label = fig.add_subplot(111, frameon=False)
            ax_label.set_xticks([])
            ax_label.set_yticks([])
            #plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            #plt.minorticks_off()

            ax_label.set_xlabel('Velocity (km/s)', labelpad=label_axes_pad_x)
            ax_label.set_ylabel('Flux (continuum normalized)', labelpad=label_axes_pad_y)

        plt.subplots_adjust(wspace=0, hspace=0)      

        if create_fig_ax == True:
            return fig, axes

    def print_summary(self):

        # For each ion suite
        for ion_suite in self.ion_suite_list:

            for i in range(len(ion_suite.ion_transitions_name_list)):
                
                ion_transition_name = ion_suite.ion_transitions_name_list[i]
                
                print(ion_transition_name)
                
                ion_transition_param_medians = ion_suite.param_medians_reshape[i]
                ion_transition_param_errs_lo = ion_suite.param_errs_lo_reshape[i]
                ion_transition_param_errs_hi = ion_suite.param_errs_hi_reshape[i]
                
                for j in range(len(ion_transition_param_medians)):
                    
                    logN = np.round(ion_transition_param_medians[j][0], 2)
                    b = np.round(ion_transition_param_medians[j][1], 1)
                    dv_c = np.round(ion_transition_param_medians[j][2], 1)
                    
                    logN_err_lo = np.round(ion_transition_param_errs_lo[j][0], 2)
                    b_err_lo = np.round(ion_transition_param_errs_lo[j][1], 1)
                    dv_c_err_lo = np.round(ion_transition_param_errs_lo[j][2], 1)

                    logN_err_hi = np.round(ion_transition_param_errs_hi[j][0], 2)
                    b_err_hi = np.round(ion_transition_param_errs_hi[j][1], 1)
                    dv_c_err_hi = np.round(ion_transition_param_errs_hi[j][2], 1)
                    
                    if np.abs(logN_err_hi-logN_err_lo)<0.1:
                        logN_str = '{:.2f} \pm {:.2f}'.format(logN, np.max([logN_err_lo, logN_err_hi]))
                    else:
                        logN_str = '{:.2f}_{{-{:.2f}}}^{{+{:.2f}}}'.format(logN, logN_err_lo, logN_err_hi)
            
                    if np.abs(b_err_hi-b_err_lo)<0.1:
                        b_str = '{:.1f} \pm {:.1f}'.format(b, np.max([b_err_lo, b_err_hi]))
                    else:
                        b_str = '{:.1f}_{{-{:.1f}}}^{{+{:.1f}}}'.format(b, b_err_lo, b_err_hi)
                        
                    if np.abs(dv_c_err_hi-dv_c_err_lo)<0.1:
                        dv_c_str = '{:.1f} \pm {:.1f}'.format(dv_c, np.max([dv_c_err_lo, dv_c_err_hi]))
                    else:
                        dv_c_str = '{:.1f}_{{-{:.1f}}}^{{+{:.1f}}}'.format(dv_c, dv_c_err_lo, dv_c_err_hi)
                        
                    print('dv_c = ' + dv_c_str + ', logN = ' + logN_str + ', b = ' + b_str)

def gen_flux_def(v_mod, logN, b, wav0_rest, f, gamma):

    '''
    Function to generate flux deficit

    v_mod: Relevant velocities
    logN: log of column density
    b: Doppler parameter (km/s)
    wav0_rest: Rest wavelength of transition
    gamma: Lorentzian braodening (converted to km/s using gamma_v = gamma_nu * wav0_rest)
    '''
    
    # Get column density
    N = 10**logN

    # Get velocity dispersion from Doppler parameter
    sigma_v = b/np.sqrt(2)

    # Generate the Voigt profile

    # Rough sketch for the normalization factor
    # -----------------------------------------
    # The Voigt function, as it is called, is phi_v, and should come with a velocity normalization to make tau unitless
    # To obtain the normalization, we can use the normalization of 1 Hz for phi_nu
    # One can establish by using the marginalization of phi_nu and delta_nu = nu_jk * v/c 
    # That phi_v = (nu_jk/c)*phi_nu, obviously the velocity and frequency need to be matched
    # Now phi_v dv = phi_nu d nu by conservation of energy, so by plugging in the relation between
    # phi_v and phi_nu, we can say that the velocity normalization will be (c/nu_jk)*(1 Hz)
    # A.K.A., the normalization would be lambda_jk*1Hz
    # We have the rest wavelength in Ang, so first converting it to meters we get a factor of 1e-10
    # Then converting velocity to km/s (since that's what v_mod, sigma_v, gamma are in), we tack on another 1e-3

    tau = N*sigma_0*f*voigt_profile(v_mod, sigma_v, gamma/(4*np.pi))*wav0_rest*1e-13
    
    # Return the flux deficit
    return np.exp(-tau)

def model_spec_gen(v_obs, params, 
                   wav0_rest, f, gamma, A,
                   lsf_convolve, lsf, v_lsf):

    '''
    Model to generate LSF convolved flux deficits
    '''

    # For constraining b value directly
    if len(params) == 3:
        logN = params[0]
        b = params[1]
        dv_c = params[2]
    # For constraining temperature and non-thermal effects simultaneously
    elif len(params) == 4:
        logN = params[0]
        logT = params[1]
        b_NT = params[2]
        dv_c = params[3]

        # Generate b value
        # Useful shortcut to compute temperature: 2*k_B/amu = 0.016629
        # Multiplying with T in kelvin and A in amu will yield b-value in km/s
        b = np.sqrt(2*k_B*10**logT/(A*amu) + b_NT**2)

    # Redifine shifted velocity
    v_shift = v_obs - dv_c

    # Extend the velocity range beyond given for purposes of convolution

    # First get LSF velocity pixel
    delta_v_lsf = np.mean(v_lsf[1:]-v_lsf[:-1])

    # Then, decide by how much you want to extend in each side
    delta_v_extend = len(lsf)*delta_v_lsf

    # Resample the wavelengths with some padding
    #print(v_shift, delta_v_extend, delta_v_lsf)
    v_mod = np.arange(v_shift[0]-delta_v_extend, v_shift[-1]+delta_v_extend, delta_v_lsf)
    
    # Generate model spectrum
    f_v_norm_mod = gen_flux_def(v_mod, logN, b, wav0_rest, f, gamma)

    if lsf_convolve == True:

        # First perform the convolution
        f_v_mod_conv = np.convolve(f_v_norm_mod, lsf, mode='valid')

        # Get the velocity pixels for the convolution
        v_mod_conv = v_mod[np.argmax(np.flip(lsf)):np.argmax(np.flip(lsf))+len(f_v_mod_conv)]

        # Interpolate the convoluted model
        f_v_mod_interp = np.interp(v_obs, v_mod_conv+dv_c, f_v_mod_conv)

    else:
        f_v_mod_interp = np.interp(v_obs, v_mod+dv_c, f_v_norm_mod)
    
    return f_v_mod_interp


def comp_model_spec_gen(v_obs, params_list, 
                   wav0_rest, f, gamma, A,
                   lsf_convolve, lsf, v_lsf):

    # Get number of components
    n_components = len(params_list)

    # Generate fluxes for each component
    component_fluxes = np.zeros((n_components, len(v_obs)))

    for i in range(n_components):

        params = params_list[i]

        component_fluxes[i,:] = model_spec_gen(v_obs, params, 
                                               wav0_rest, f, gamma, A,
                                               lsf_convolve, lsf, v_lsf)
        

    # Generate the best fit model itself - it's the product of all components, NOT the sum
    # This is because optical depths can add, that corresponds to multiplying flux deficits
    best_fit_flux = np.prod(component_fluxes, axis=0)

    return component_fluxes, best_fit_flux

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

def weighted_residuals(params, ion_transitions_list, method, exclude_models):

    # Get number of ion transitions
    n_ion_transitions = len(ion_transitions_list)

    # Begin a list of residuals
    resid_list = []

    # Define an objective function that can take in multiple datasets and generate a flattened residual array

    for i in range(n_ion_transitions):

        ion_transition = ion_transitions_list[i]

        # Get list of parameters corresponding to this ion transition
        param_names = [l for l in list(params.valuesdict().keys()) if 'it{}c'.format(i+1) in l]
        # Construct the parameters array
        params_list = [params[p].value for p in param_names]
        # Create a dictionary
        params_dict = {k:v for (k,v) in zip(param_names, params_list)}

        # Get number of components
        # Get number of column density entries
        n_components = len([l for l in param_names if 'it{}'.format(i+1) in l and 'logN' in l])

        # For each component

        # Create list of reshaped parameters
        params_list_reshape = []

        for j in range(n_components):

            params_comp = []

            # If constraining b-value only
            if 'it{}c{}_b'.format(i+1, j+1) in param_names:
                params_comp = [params_dict['it{}c{}_logN'.format(i+1,j+1)], params_dict['it{}c{}_b'.format(i+1,j+1)], params_dict['it{}c{}_dv_c'.format(i+1,j+1)]]
            else:
                params_comp = [params_dict['it{}c{}_logN'.format(i+1,j+1)], params_dict['it{}c{}_logT'.format(i+1,j+1)],
                               params_dict['it{}c{}_b_NT'.format(i+1,j+1)], params_dict['it{}c{}_dv_c'.format(i+1,j+1)]]
                
            params_list_reshape.append(params_comp)


        resid = (ion_transition.flux_norm_mask - comp_model_spec_gen(ion_transition.v_mask, params_list_reshape, 
                                                    ion_transition.wav0_rest, ion_transition.f,ion_transition.gamma,ion_transition.A,
                                                    ion_transition.lsf_convolve,
                                                    ion_transition.lsf, ion_transition.v_lsf)[1])/ion_transition.err_norm_mask

        resid_list.append(resid)

    resid_flat = np.array([item for sublist in resid_list for item in sublist])

    # Check if this model is supposed to be excluded
    # Granted, it would be more efficient to check for this BEFORE constructing the residual array itself, 
    # BUT for the sake of clarity, we'll perform the check at this point

    # For now, we'll only check conditions that compare relative values of two given parameters
    # If some conditions are specified
    if exclude_models is not None:
        # Assume that if present, exclude_models is a list of strings
        # Each string is a condition, which if obeyed, must mean that the current model should be excluded

        operator_dict = {'<':operator.lt, '>':operator.gt, '<=':operator.le, '>=':operator.ge}

        for s in exclude_models:
            # We assume the string to be of the format p1 operator p2
            # Spaces are NECESSARY, since that is how we split the string
            s_split = s.split(' ')
            p1 = s_split[0]
            op  = operator_dict[s_split[1]]
            p2 = s_split[2]
            # If the condition is satisfied
            if op(params[p1].value, params[p2].value):
                #print('test')
                # Render the model improbable
                resid_flat *= np.inf
    
    # If method is LMFIT, flatten the output and convert to a numpy array to pass to lmfit
    if method=='lmfit':
        return resid_flat
    # For emcee, compute the squared sum of the flattened residual array to obtain chi-sq, 
    # multiply by -0.5 to get log likelihood probability
    elif method=='emcee':
        return -0.5*np.sum(resid_flat**2)
    