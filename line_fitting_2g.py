# -*- coding: utf-8 -*-

'''
    Routines to fit single spectral lines
    
    '''

import numpy as np
import matplotlib.pyplot as plt
from function_fit import *
from scipy.optimize import curve_fit

def fit_2g(nomefile, user, z):
    
    wavelength, flux = spectrum_open(nomefile)
    line = linesel(user,z)
    
    
    flux, wavelength, rms_cont = continuum_sub(line, wavelength, flux)
    
    index = np.where((wavelength > line.x0-50) & (wavelength < line.x0+50))
    
    '''
        Here i'm restricting the wavelength range considered by the fit.
        In such a way i'm avoiding the problems caused by having multiple
        lines in the range used for the continuum subtraction.
        '''
    lam = wavelength[index]
    f_flux = flux[index]
    
    
    guess = [5e-17, line.x0, 5.,1e-17, line.x0+2, 25.]                                                               #first guess for the fit
    
    popt, popv = curve_fit(two_gaussians, lam, f_flux, guess, maxfev=10000)
    result_1 = montecarlo(100, two_gaussians, rms_cont, lam, f_flux, popt, limit=10000)                           #montecarlo simulations for sigma on measures
    
    
    line_flux_1 = []
    for n in range(len(result_1)):
        line_flux_1.append(np.sqrt(2*np.pi)*(result_1[n][0]*np.abs(result_1[n][2])))                      #flux for each simulation for component 1
    line_flux_2 = []
    for n in range(len(result_1)):
        line_flux_2.append(np.sqrt(2*np.pi)*(result_1[n][3]*np.abs(result_1[n][5])))                      #flux for each simulation for component 2

    mean_1 = np.abs(np.mean(line_flux_1))
    std_1 = np.std(line_flux_1)
    mean_2 = np.abs(np.mean(line_flux_2))
    std_2 = np.std(line_flux_2)


    result_mean = np.abs(np.mean(result_1, axis=0))                                                        #averaging results of the fit to obtain the parameters for the plot
    result_std = np.std(result_1, axis=0)                                                        #averaging results of the fit to obtain the parameters for the plot


    print('############# C1 ################')
    print('rms:  {:1.3e}' .format(rms_cont))
    print('I0:   {:1.3e} +- {:1.3e}' .format(result_mean[0], result_std[0]))
    print('x0:   {:6.2f} +- {:.4f}' .format(result_mean[1], result_std[1]))
    print('sig0: {:4.2f} +- {:.4f}' .format(result_mean[2], result_std[2]))
    print('flux: {:1.3e} +- {:1.3e}' .format(mean_1, std_1))
    print('#################################')
    print('############# C2 ################')
    print('rms:  {:1.3e}' .format(rms_cont))
    print('I0:   {:1.3e} +- {:1.3e}' .format(result_mean[3], result_std[3]))
    print('x0:   {:6.2f} +- {:.4f}' .format(result_mean[4], result_std[4]))
    print('sig0: {:4.2f} +- {:.4f}' .format(result_mean[5], result_std[5]))
    print('flux: {:1.3e} +- {:1.3e}' .format(mean_2, std_2))
    print('#################################')

    f = open('fitting_log2.txt', 'a')
    print('filename, line, rms, I0, I0_e, x0, x0_e, sig0, sig0_e, flux, flux_e', file = f )
    print(nomefile, line.name+'c1','{:1.3e} {:1.3e} {:1.3e} {:6.2f} {:.4f} {:4.2f} {:.4f} {:1.3e} {:1.3e}' .format(rms_cont, result_mean[0], result_std[0],result_mean[1], result_std[1],result_mean[2], result_std[2], mean_1, std_1), file = f)
    print(nomefile, line.name+'c2','{:1.3e} {:1.3e} {:1.3e} {:6.2f} {:.4f} {:4.2f} {:.4f} {:1.3e} {:1.3e}' .format(rms_cont, result_mean[3], result_std[3],result_mean[4], result_std[4],result_mean[5], result_std[5], mean_2, std_2), file = f)
    f.close()

    plt.plot(wavelength,flux, c='black', lw =1)                                                        #plot
    plt.plot(lam,two_gaussians(lam,*result_mean[0:6]), c='red', lw = 0.8)
    plt.plot(lam,gaussian(lam,*result_mean[0:3]), c='blue', lw = 0.8)
    plt.plot(lam,gaussian(lam,*result_mean[3:6]), c='blue', lw = 0.8)
    plt.show()



if __name__ == '__main__':
    
    nomefile = '../PA90/mrk783_PA90_r147.der.fits'
    z = 0.0672
    user = 'Hb'
    
    fit_2g(nomefile, user, z)

#GIF = GraphicalInteractiveFit_v002(wavelength[f_wave], flux[f_wave], wavelength[f_wave], Hb_model, par_init, dpar_init)
