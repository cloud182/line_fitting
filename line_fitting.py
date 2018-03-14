# -*- coding: utf-8 -*-

'''
Routines to fit single spectral lines

'''

import numpy as np
import matplotlib.pyplot as plt
from function_fit import *
from scipy.optimize import curve_fit

def fit_1g(nomefile, user, z):

    wavelength, flux = spectrum_open(nomefile) 
    line = linesel(user,z)


    flux, wavelength, rms_cont = continuum_sub(line, wavelength, flux)
    
    index = np.where((wavelength > line.x0-50) & (wavelength < line.x0+30))

    '''
    Here i'm restricting the wavelength range considered by the fit. 
    In such a way i'm avoiding the problems caused by having multiple
    lines in the range used for the continuum subtraction.
    '''
    lam = wavelength[index]
    f_flux = flux[index]


    guess = [1e-17, line.x0, 5.]                                                               #first guess for the fit

    popt, popv = curve_fit(gaussian, lam, f_flux, guess, maxfev=10000)
    result_1 = montecarlo(100, gaussian, rms_cont, lam, f_flux, popt, limit=10000)                           #montecarlo simulations for sigma on measures


    line_flux = []
    for n in range(len(result_1)):
        line_flux.append(np.sqrt(2*np.pi)*(result_1[n][0]*np.abs(result_1[n][2])))                      #flux for each simulation
            
    mean = np.abs(np.mean(line_flux))
    std = np.std(line_flux)

    result_mean = np.abs(np.mean(result_1, axis=0))                                                        #averaging results of the fit to obtain the parameters for the plot
    result_std = np.std(result_1, axis=0)                                                        #averaging results of the fit to obtain the parameters for the plot

    print('#################################')
    print('rms:  {:1.3e}' .format(rms_cont))
    print('I0:   {:1.3e} +- {:1.3e}' .format(result_mean[0], result_std[0]))
    print('x0:   {:6.2f} +- {:.4f}' .format(result_mean[1], result_std[1]))
    print('sig0: {:4.2f} +- {:.4f}' .format(result_mean[2], result_std[2]))
    print('flux: {:1.3e} +- {:1.3e}' .format(mean, std))
    print('#################################')

    f = open('fitting_log.txt', 'a')
    print('filename, line, rms, I0, I0_e, x0, x0_e, sig0, sig0_e, flux, flux_e', file = f )
    print(nomefile, line.name,'{:1.3e} {:1.3e} {:1.3e} {:6.2f} {:.4f} {:4.2f} {:.4f} {:1.3e} {:1.3e}' .format(rms_cont, result_mean[0], result_std[0],result_mean[1], result_std[1],result_mean[2], result_std[2], mean, std), file = f)
    f.close()

    plt.plot(wavelength,flux, c='black', lw =1)                                                        #plot
    plt.plot(lam,gaussian(lam,*result_mean[0:3]), c='red', lw = 0.8)
    plt.show()



if __name__ == '__main__':

    nomefile = '../PA90/mrk783_PA90_r232.der.fits'
    z = 0.0672
    user = 'o3'
    
    fit_1g(nomefile, user, z)

#GIF = GraphicalInteractiveFit_v002(wavelength[f_wave], flux[f_wave], wavelength[f_wave], Hb_model, par_init, dpar_init)






