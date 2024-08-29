import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

plt.style.use('/Users/thepoetoftwilight/Documents/Astro/Code/PG1522+101/science.mplstyle')

gal_lines = np.loadtxt('redshiftLines.dat', skiprows=1, dtype=str)

def load_ldss_spec_2d(loaddir, mask_num, slit_num):

    spec_2d_fits = fits.open(loaddir + 'Spectra_2D/m{}/{}sum.fits'.format(mask_num, slit_num))

    spec_2d = spec_2d_fits[0].data

    wav0 = spec_2d_fits[0].header['CRVAL1']
    delta_wav = spec_2d_fits[0].header['CDELT1']
    
    wav = np.arange(wav0, wav0+spec_2d.shape[1]*delta_wav, delta_wav)

    return wav, spec_2d

def load_ldss_spec_1d(loaddir, mask_num, slit_num):

    # Load the flat-fielded spectrum 

    spec_1d_fits = fits.open(loaddir + 'Spectra_1D/m{}/{}_1dspec.fits'.format(mask_num, slit_num))
    
    wav0 = spec_1d_fits[0].header['CRVAL1']
    delta_wav = spec_1d_fits[0].header['CDELT1']
    
    flux = spec_1d_fits[0].data[4,:]
    err = spec_1d_fits[0].data[5,:]
    
    wav = np.arange(wav0, wav0+len(flux)*delta_wav, delta_wav)

    # Load the response function
    resp_df = pd.read_csv(loaddir + 'ldss_vph_red_resp.csv')

    wav_resp = resp_df['Wavelength']
    flux_resp = resp_df['Response']

    flux_resp_interp = np.interp(wav, wav_resp, flux_resp)

    # Divide out by the response function
    flux /= flux_resp_interp
    err /= flux_resp_interp
    
    # Save the 1D spectrum
    np.savetxt(loaddir + 'Spectra_1D/m{}/{}.dat'.format(mask_num, slit_num), 
               np.vstack((wav, flux, err)).T,
              delimiter = '\t')

    return wav, flux, err


def load_eigenspec(loaddir='/Users/thepoetoftwilight/Documents/Astro/Data/Eigenspectra/eigen_galaxy_Bolton2012.csv'):

    'Function to read SDSS based Eigenspectra to be used for redshift fitting'

    eigenspec_df = pd.read_csv(loaddir)

    wav_eigen = eigenspec_df['wav']
    c1 = eigenspec_df['c1']
    c2 = eigenspec_df['c2']
    c3 = eigenspec_df['c3']
    c4 = eigenspec_df['c4']

    return wav_eigen, c1, c2, c3, c4

def z_shift_eigen(z, wav_arr):

    '''
    Function to redshift eigenspectra and interpolate them at the observed wavelengths
    '''
    
    wav_eigen, c1, c2, c3, c4 = load_eigenspec()

    wav_eigen_shift = wav_eigen*(1+z)
    
    gal_ind = (wav_eigen_shift>=wav_arr[0])&(wav_eigen_shift<=wav_arr[-1])
    wav_eigen_shift_gal = wav_eigen_shift[gal_ind]
    
    c1_gal = c1[gal_ind]
    c1_gal_interp = np.interp(wav_arr, wav_eigen_shift_gal, c1_gal)

    c2_gal = c2[gal_ind]
    c2_gal_interp = np.interp(wav_arr, wav_eigen_shift_gal, c2_gal)
    
    c3_gal = c3[gal_ind]
    c3_gal_interp = np.interp(wav_arr, wav_eigen_shift_gal, c3_gal)

    c4_gal = c4[gal_ind]
    c4_gal_interp = np.interp(wav_arr, wav_eigen_shift_gal, c4_gal)
    
    return c1_gal_interp, c2_gal_interp, c3_gal_interp, c4_gal_interp

def best_model(c1, c2, c3, c4, y, y_err):

    '''
    Best linear combination of template spectra given an observed spectrum
    '''

    A11 = np.sum((c1/y_err)**2)
    A12 = np.sum(c1*c2/y_err**2)
    A13 = np.sum(c1*c3/y_err**2)
    A14 = np.sum(c1*c4/y_err**2)
    
    A21 = np.sum(c2*c1/y_err**2)
    A22 = np.sum((c2/y_err)**2)
    A23 = np.sum(c2*c3/y_err**2)
    A24 = np.sum(c2*c4/y_err**2)
    
    A31 = np.sum(c3*c1/y_err**2)
    A32 = np.sum(c3*c2/y_err**2)
    A33 = np.sum((c3/y_err)**2)
    A34 = np.sum(c3*c4/y_err**2)
    
    A41 = np.sum(c4*c1/y_err**2)
    A42 = np.sum(c4*c2/y_err**2)
    A43 = np.sum(c4*c3/y_err**2)
    A44 = np.sum((c4/y_err)**2)
    
    b1 = np.sum(y*c1/y_err**2)
    b2 = np.sum(y*c2/y_err**2)
    b3 = np.sum(y*c3/y_err**2)
    b4 = np.sum(y*c4/y_err**2)
    
    A_mat = np.array([[A11, A12, A13, A14],[A21, A22, A23, A24],[A31, A32, A33, A34],[A41, A42, A43, A44]])
    b_vec = np.array([b1, b2, b3, b4])
    
    x_vec = np.linalg.inv(A_mat)@b_vec
    
    a = x_vec[0]
    b = x_vec[1]
    c = x_vec[2]
    d = x_vec[3]

    return a, b, c, d

def eval_red_chi_sq(y_hat, y, y_err, dof):
    
    '''
    Evaluation of reduced chi-square for a 
    '''
    return np.sum(((y-y_hat)/y_err)**2)/(len(y)-dof)


def eval_spec_z(loaddir, mask_num, slit_num,
                z_min=0, z_max=1.4, z_step=1e-4, wav_min=5000, wav_max=9800):

    '''
    Evaluate best linear combination of template spectra in a coarse grid of redshifts
    '''

    wav, flux, err = load_ldss_spec_1d(loaddir, mask_num, slit_num)

    z_arr = np.arange(z_min, z_max+z_step, z_step)

    red_chi_sq_arr = np.zeros(len(z_arr))
    model_params = np.zeros((len(z_arr), 4))

    flux_idx = ~np.isnan(flux)
    err_idx = err!=0

    wav_idx = (wav>=wav_min) & (wav<=wav_max)

    A_band_idx = (wav<=7588) | (wav>=7684)

    idx = np.bool_(flux_idx*err_idx*wav_idx*A_band_idx)

    for i in range(len(z_arr)):
        
        z = z_arr[i]
        
        c1_shift, c2_shift, c3_shift, c4_shift = z_shift_eigen(z, wav[idx])
        
        a, b, c, d = best_model(c1_shift, c2_shift, c3_shift, c4_shift, 
                                flux[idx], err[idx])
        
        model_params[i,0] = a
        model_params[i,1] = b
        model_params[i,2] = c
        model_params[i,3] = d 
        
        red_chi_sq = eval_red_chi_sq(a*c1_shift+b*c2_shift+c*c3_shift+d*c4_shift, 
                                            flux[idx], err[idx], model_params.shape[1])
                
        red_chi_sq_arr[i] = red_chi_sq

    np.savetxt(loaddir + 'Redshifts/m{}/{}.dat'.format(mask_num, slit_num), 
               np.vstack((z_arr, model_params[:,0], model_params[:,1], model_params[:,2], model_params[:,3], red_chi_sq_arr)).T,
              delimiter = '\t')

def get_local_min(x_arr, y_arr, x_min, x_max):

    idx = (x_arr>x_min)&(x_arr<x_max)
    x_slice = x_arr[idx]
    y_slice = y_arr[idx]
    return x_slice[np.argmin(y_slice)]


def plot_chi_sq_z(ax, loaddir, slit_num, z_min = 0, z_max= 1.4):

    model_arr = np.loadtxt(loaddir+'Redshifts/{}.dat'.format(slit_num), delimiter='\t')

    z_arr = model_arr[:,0]

    red_chi_sq_arr = model_arr[:,5]

    z_best = get_local_min(z_arr, red_chi_sq_arr, z_min, z_max)

    ax.plot(z_arr, red_chi_sq_arr)
    ax.axvline(z_best, linestyle=':')

    ax.set_title('SLIT {}, '.format(slit_num) + r'$z_{\mathrm{best}} = $'+' {}'.format(np.round(z_best, 4)))
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'Reduced $\chi^2$')

    return z_best

def gen_composite_spec(z, wav, a, b, c, d):

    '''
    Generate a specific linear combination of template spectra at a given redshift
    '''

    c1_shift, c2_shift, c3_shift, c4_shift = z_shift_eigen(z, wav)

    return a*c1_shift+b*c2_shift+c*c3_shift+d*c4_shift


def plot_model_spec(ax, z_spec, wav, loaddir, slit_num):

    model_arr = np.loadtxt(loaddir+'Redshifts/{}.dat'.format(slit_num), delimiter='\t')

    z_arr = model_arr[:,0]
    a_arr = model_arr[:,1]
    b_arr = model_arr[:,2]
    c_arr = model_arr[:,3]
    d_arr = model_arr[:,4]

    idx = np.argmin(np.abs(z_spec-z_arr))

    ax.plot(wav, gen_composite_spec(z_arr[idx], wav, a_arr[idx], b_arr[idx], c_arr[idx], d_arr[idx]), color='red', lw=1.2)

def smooth_func(x_arr, y_arr, dx):

    y_smooth = np.zeros(len(x_arr))

    for i in range(len(x_arr)):
        
        x = x_arr[i]
        idx = (x_arr>=x-dx/2)&(x_arr<=x+dx/2)
        y_smooth[i] = np.median(y_arr[idx])

    return y_smooth

def plot_gal_lines(ax, z, lw, y_pos, plot_list, wav_min, wav_max):

    for i in range(len(plot_list)):
        l = plot_list[i]
        indices = [idx for idx,e in enumerate(list(gal_lines[:,1])) if e==l]
        for idx in indices:
            wav_pos = float(gal_lines[idx,0])*(1+z)
            if wav_min <= wav_pos <= wav_max:
                ax.axvline(wav_pos, linestyle=':', lw=lw)
                ax.text(x=wav_pos, y=y_pos, rotation=270, s=l)

def plot_spec_2d(ax, loaddir, mask_num, slit_num, aspect='auto', 
                wav_min = 6000, wav_max = 9800,
                y_min = 5, y_max = 22, 
                vmin=0, vmax=500, cmap='gist_yarg', interpolation='antialiased',
                plot_lines=False, z_gal=0, lw=0.7, y_pos=800, 
                plot_list=['Ha', 'Hb', 'Hg', 'Hd', '[OII]', '[OIII]', 'CaIIH', 'CaIIK', 'MgI', 'NaI', 'G-band']):

    wav, spec_2d = load_ldss_spec_2d(loaddir, mask_num, slit_num)
    y = np.arange(0, spec_2d.shape[0])

    wav_min_idx = np.argmin(np.abs(wav-wav_min))
    wav_max_idx = np.argmin(np.abs(wav-wav_max))

    y_min_idx = np.argmin(np.abs(y-y_min))
    y_max_idx = np.argmin(np.abs(y-y_max))

    spec_2d_slice = spec_2d[y_min_idx:y_max_idx+1, wav_min_idx:wav_max_idx+1]
 
    ax.imshow(spec_2d_slice, origin='lower', aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, 
         extent=[wav_min, wav_max, max(0, y_min), min(y_max,spec_2d.shape[0]-1)], interpolation=interpolation)
    
    if plot_lines == True:
        plot_gal_lines(ax, z_gal, lw, y_pos, plot_list, wav_min, wav_max)

    ax.set_xlim(wav_min, wav_max)
    #ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Slit Pixel')

def plot_spec_1d(ax, loaddir, mask_num, slit_num, smooth = False, dlam = 0,
                wav_min = 6000, wav_max = 9800,
                plot_lines=False, z_gal=0, lw=0.7, y_pos=800, 
                plot_list=['Ha', 'Hb', 'Hg', 'Hd', '[OII]', '[OIII]', 'CaIIH', 'CaIIK', 'MgI', 'NaI', 'G-band'],
                filt_windows = [[6850, 6950], [7588, 7684]],
                plot_model=False):

    wav, flux, err = load_ldss_spec_1d(loaddir, mask_num, slit_num)
    
    if smooth==False:
        ax.step(wav, flux, where='mid', color='black')
    else:
        ax.step(wav, smooth_func(wav, flux, dlam))

    for i in range(len(filt_windows)):
        ax.axvspan(xmin=filt_windows[i][0], xmax=filt_windows[i][1], color='grey', alpha=.5)

    #ax.axvspan(xmin=6850, xmax=6950, color='grey', alpha=.5)
    #ax.axvspan(xmin=7588, xmax=7684, color='grey', alpha=.5)

    ax.step(wav, err, where='mid', color='cyan')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel(r'$f_{\lambda}$' + ' (erg/cm${}^2$/s/Å)')
    
    if plot_lines == True:
        plot_gal_lines(ax, z_gal, lw, y_pos, plot_list, wav_min, wav_max)

    if plot_model == True:
        plot_model_spec(ax, z_gal, wav, loaddir, slit_num)
        
    ax.set_xlim(wav_min, wav_max)

