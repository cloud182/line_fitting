# -*- coding: utf-8 -*-

'''
    Routines to fit single spectral lines
    
    '''

import numpy as np
import matplotlib.pyplot as plt
from function_fit import *
from scipy.optimize import curve_fit
from widgets_prototypes import *
import os
import sys


def guessing(function, lam, flux, par_init, dpar_init, debug = False):
    
    GIF = GraphicalInteractiveFit_v002(lam, flux, lam, function, par_init, dpar_init)
    lala = input('Continue?')
    
    return GIF.best_fit_par

def s2(x,h1,c1,w1,h2,c2):
    return two_gaussians(x,h1,c1,w1,h2,c2,w1)

def fit_ha(nomefile, user, z):
    
    wavelength, flux = spectrum_open(nomefile)
    line = linesel(user,z)
    
    
    flux, wavelength, rms_cont = continuum_sub(line, wavelength, flux)
    
    index = np.where((wavelength > line.x0-60) & (wavelength < line.x0+60))
    
    '''
        Here i'm restricting the wavelength range considered by the fit.
        In such a way i'm avoiding the problems caused by having multiple
        lines in the range used for the continuum subtraction.
        '''
    lam = wavelength[index]
    f_flux = flux[index]
    
    
    par_init = [1e-17, line.x0, 5.,1e-17, line.x0+15.]                                                               #first guess for the fit
    dpar_init = [1e-17, 20., 10.,1e-17, 20.]
    
    guess = guessing(s2, lam, f_flux, par_init, dpar_init, debug = False)
    
    popt, popv = curve_fit(s2, lam, f_flux, guess, maxfev=10000)
    result_1 = montecarlo(100, s2, rms_cont, lam, f_flux, popt, limit=10000)                           #montecarlo simulations for sigma on measures
    
    
    line_flux_1 = []                                #6717
    line_flux_2 = []                                #6731

    for n in range(len(result_1)):
        line_flux_1.append(np.sqrt(2*np.pi)*(result_1[n][0]*np.abs(result_1[n][2])))                      #flux for each simulation
        line_flux_2.append(np.sqrt(2*np.pi)*(result_1[n][3]*np.abs(result_1[n][2])))                      #flux for each simulation

    mean_1 = np.abs(np.mean(line_flux_1))       #Ha_1
    std_1 = np.std(line_flux_1)
    mean_2 = np.abs(np.mean(line_flux_2))       #Ha_2
    std_2 = np.std(line_flux_2)


    result_mean = np.abs(np.mean(result_1, axis=0))                                                        #averaging results of the fit to obtain the parameters for the plot
    result_std = np.std(result_1, axis=0)                                                        #averaging results of the fit to obtain the parameters for the plot

    print('##############6717###############')
    print('rms:  {:1.3e}' .format(rms_cont))
    print('I0:   {:1.3e} +- {:1.3e}' .format(result_mean[0], result_std[0]))
    print('x0:   {:6.2f} +- {:.4f}' .format(result_mean[1], result_std[1]))
    print('sig0: {:4.2f} +- {:.4f}' .format(result_mean[2], result_std[2]))
    print('flux: {:1.3e} +- {:1.3e}' .format(mean_1, std_1))
    print('##############6731###############')
    print('rms:  {:1.3e}' .format(rms_cont))
    print('I0:   {:1.3e} +- {:1.3e}' .format(result_mean[3], result_std[3]))
    print('x0:   {:6.2f} +- {:.4f}' .format(result_mean[4], result_std[4]))
    print('sig0: {:4.2f} +- {:.4f}' .format(result_mean[2], result_std[2]))
    print('flux: {:1.3e} +- {:1.3e}' .format(mean_2, std_2))
    print('#################################')

    f = open('fitting_log.txt', 'a')
    print('filename, line, rms, I0, I0_e, x0, x0_e, sig0, sig0_e, flux, flux_e', file = f )
    print(nomefile, line.name+'c1','{:1.3e} {:1.3e} {:1.3e} {:6.2f} {:.4f} {:4.2f} {:.4f} {:1.3e} {:1.3e}' .format(rms_cont, result_mean[0], result_std[0],result_mean[1], result_std[1],result_mean[2], result_std[2], mean_1, std_1), file = f)
    print(nomefile, line.name+'c2','{:1.3e} {:1.3e} {:1.3e} {:6.2f} {:.4f} {:4.2f} {:.4f} {:1.3e} {:1.3e}' .format(rms_cont, result_mean[3], result_std[3],result_mean[4], result_std[4],result_mean[2], result_std[2], mean_2, std_2), file = f)
    f.close()


    plt.plot(wavelength,flux, c='black', lw =1)                                                        #plot
    plt.plot(lam,s2(lam,*result_mean), c='red', lw = 0.8)
    plt.plot(lam,gaussian(lam,*result_mean[0:3]), c='blue', lw = 0.8)
    plt.plot(lam,gaussian(lam,result_mean[3],result_mean[4],result_mean[2]), c='green', lw = 0.8)
    plt.show()



if __name__ == '__main__':
    
    nomefile = '../PA90/mrk783_PA90_r152.der.fits'
    z = 0.0672
    user = 's2'
    
    fit_ha(nomefile, user, z)
