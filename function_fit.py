# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
#import scipy.stats
from sys import exit
from scipy import optimize
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.wcs import WCS

############################## SCELTA RIGHE PER IL FIT #################################################

def linesel(user, z):
    if user == 'o3':
        line = emission_line('[OIII]_07', 5006.84, [4525.,4575.], [5080.,5120.],z)
    elif user == 'o3_1':
        line = emission_line('[OIII]_59', 4958.91, [4525.,4575.], [5080.,5120.],z)
    elif user == 'Hb':
        line = emission_line('Hb', 4861.332, [4525.,4575.], [5080.,5120.],z)
    elif user == 'o1':
        line = emission_line('[OI]_00', 6300.33, [6200.,6250.], [6850.,6900.],z)
    elif user == 'Ha':
        line = emission_line('Ha', 6562.817, [6200.,6250.], [6850.,6900.],z)
    elif user == 'n2':
        line = emission_line('[NII]', 6583.6, [6200.,6250.], [6850.,6900.],z)
    elif user == 'he':
        line = emission_line('HeI', 6678.149, [6200.,6250.], [6850.,6900.],z)
    elif user == 's2':
        line = emission_line('[SII]', 6716.47, [6200.,6250.], [6850.,6900.],z)
    elif user == 'o2':
        line = emission_line('[OII]', 3727, [3650.,3700,], [3750.,3800.],z)
    return line

############################## LETTURA SPETTRO .FITS #################################################
'''
Classe in cui salvare dettagli utili per il fit per una singola riga di emissione 
'''

class emission_line:

    '''
    emission_line class:
    it contains all the information about an emission_line.
    name = name of the line
    x0 = central wavelength (rest)
    lcont = interval to measure the continuum
    ucont = interval to measure the continuum
    ngaus = # of gaussian to be use later in the fit
    '''

    def __init__(self,name, x0, lcont, ucont,z):
        self.z = z                                  #redshift
        self.name = name                            #name of the line
        self.x0 = x0*(1+z)                          #central wavelength
        self.lcont = np.array(lcont)*(1+z)          #lower continuum window
        self.ucont = np.array(ucont)*(1+z)          #upper continuum window
    

############################## LETTURA SPETTRO .FITS #################################################

def spectrum_open(nomefile):
    hdu = fits.open(nomefile)                       #opening spectrum
    flux = hdu[0].data                              #selecting the data
    header = hdu[0].header                          #selecting the header
    wcs = WCS(header)                               #extracting wavelengths (code from astropy tutorial)
    index = np.arange(header['NAXIS1'])
    wavelength = wcs.wcs_pix2world(index[:,np.newaxis],0)
    wavelength = wavelength.flatten()
    
    return wavelength, flux

############################## DEFINIZIONE FUNZIONE GAUSSIANA #################################################


def gaussian(x, height, center, width):				
    return height*np.exp(-(x - center)**2/(2*width**2))


############################## DEFINIZIONE FUNZIONE 3 GAUSSIANE ###############################################

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3):	
    return (gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2) +
        gaussian(x, h3, c3, w3))

############################## DEFINIZIONE 2 GAUSSIANE ########################################################

def two_gaussians(x, h1, c1, w1, h2, c2, w2):		#definisce funzione composta da due gaussiane
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0,0,1)

############################## DEFINIZIONE DOPPIETTO S2 #######################################################

def s2_doublet(x, h, c, w, h1):
	return two_gaussians(x, h, c, w, h1, c+14.3, w)

def s2_doublet_2(x, h, c, w, h1, c1, w1, h2, h3):
	return s2_doublet(x, h, c, w, h2)+s2_doublet(x, h1, c1, w1, h3)
############################## DEFINIZIONE Ha+N2 #######################################################

def ha_n2(x, h, c, w, h1, c1, w1, h2, h3, c3, w3):
	return (three_gaussians(x, h, c, w, h1, c1, w1, h2, c1-35.5, w1) +gaussian(x, h3, c3, w3))

############################## DEFINIZIONE Ha+N2 #######################################################

def n2_doublet(x, h, c, w):
	return two_gaussians(x, h, c, w, h/3, c-35.5, w)

############################## DEFINIZIONE RETTA ##############################################################

def sline(x,a,b):									
	return a*x+b

############################## SIMULAZIONI MONTECARLO #########################################################
'''
La funzione prende il numero di iterazioni da effettuare, la funzione da usare per il fit, l'rms del continuo
un array di lunghezze d'onda, uno di flussi, e un array con le guess iniziali.
Ad ogni iterazione il programma prende il flusso e ci aggiunge un rumore gaussiano centrato in 0, con sigma 
pari all'rms del continuo.
A questo punto fa il fit usando come guess iniziali i parametri forniti e per ogni ciclo salva i risultati.
Finiti gli niter cicli di fitting restituisce un array con i risultati di ogni fit.
'''

def montecarlo(niter, function, rms, lam, flux, guess, limit=4000):	#montecarlo per stima errori
	i=0
	print('niter =', niter)
	result=[]
	while i<niter:
		nflux = flux+np.random.normal(0, rms, len(flux))
		optim_line, success_line = optimize.curve_fit(function, lam, nflux, guess[:], maxfev=limit)
		result.append(optim_line)
		i=i+1
	return result
# Con limiti invece che maxfev

def montecarlo2(niter, function, rms, lam, flux, guess, limits):	#montecarlo per stima errori
	i=0
	print('niter =', niter)
	result=[]
	while i<niter:
		nflux = flux+np.random.normal(0, rms, len(flux))
		optim_line, success_line = optimize.curve_fit(function, lam, nflux, guess[:], bounds=limits)
		result.append(optim_line)
		i=i+1
	return result

############################### CLIPPING CONTINUO ##############################################################
#'''
#La funzione prende due array, uno di lunghezze d'onda e uno di flusso. 
#Calcola l'rms del flusso e la lunghezza dell'array. Ora per ogni punto controlla se si distanza dalla media per 
#più di n sigma. Se si, memorizza l'indice in un array che viene usato in seguito per togliere i relativi elementi 
#sia dai flussi che dalle lunghezze d'onda. Il processo viene rifatto fino a che nessun elemento viene rimosso 
#dalla lista iniziale, che vengono infine restituite. 
#'''

#def clipping(x, y, n=5, debug = False):									
#	i=1
#	while i > 0:
#		rms = np.std(y)								#calcolo l'rms del continuo
#		if debug:
#			print(rms)
#		index=[]
#		l = len(y)
#		if debug:
#			print(l)
#		for j in range(len(y)):							#per ogni punto controllo se è più alto di x sigma e in caso memorizzo l'indice
#			if np.abs(y[j])-n*rms>0:
#				index.append(j)
#		if debug:
#			print(index)
#		y = np.delete(y,index, axis=0)						#rimuovo tutto quello che è più alto di x sigma
#		x = np.delete(x,index, axis=0)
#		i = l-len(y)								#continuo fino a quando non tolgo niente
#		print(i)
#	return x,y

############################### FIT DEL CONTINUO ###############################################################
#'''
#La funzione richiede un array di lunghezze d'onda, uno di flussi, un limite minimo e un limite massimo.
#Per prima cosa seleziona due regioni [-r_max,-r_min] e [r_min, r_max] da considerare per fare un fit del continuo.
#Le regioni devono essere possibilmente al di fuori della riga. A questo punto usa la funzione di clipping per 
#eliminare dalle regioni eventuali picchi o avvallamenti troppo grandi, spesso possibili altre righe spettrali.
#Un algoritmo di ottimizzazione dei parametri viene usato per fittare il continuo con una retta e subito dopo 
#viene misurato l'rms dello spettro rispetto al fit. Per esaminare il fit viene plottato lo spettro con sovraimposto 
#il continuo fittato che viene infine sottratto dallo spettro. Un plot permette di esaminare il risultato.
#L'output della funzione è quindi lo spettro senza continuo e l'rms dello stesso.

#'''
#				
#def continuum(lam, flux, l_min, l_max, line, n=5, s2 = False, o3 = False, debug = False):									#fit del continuo


#	
#	if s2:
#		cont_b = []															#scelgo gli intervalli per fittare il continuo
#		cont_r = np.where((lam>line+l_min)&(lam<line+l_max+200))
#		if debug:
#			print(cont_r)
#	elif o3:
#		cont_b = []															#scelgo gli intervalli per fittare il continuo
#		cont_r = np.where((lam>line+l_min+20)&(lam<line+l_max+100))
#		if debug:
#			print(cont_r)
#	else:
#		cont_b = np.where((lam>line-l_max)&(lam<line-l_min))				#scelgo gli intervalli per fittare il continuo
#		cont_r = np.where((lam>line+l_min)&(lam<line+l_max))

#	cont=np.concatenate((lam[cont_b],lam[cont_r]))

#	cont_f = np.concatenate((flux[cont_b],flux[cont_r]))

#	cont, cont_f = clipping(cont, cont_f, n, True)						#applico clipping

#	optim_cont, success_cont = optimize.curve_fit(sline, cont, cont_f)
#	
#	rms_cont = np.sqrt(np.sum((cont_f - sline(cont,*optim_cont))**2)/(len(cont_f)-1))

#	if debug:
#		print(optim_cont)
# 
#	print(rms_cont)				

#	plt.plot(lam,flux, c='black')
#	plt.plot(lam,sline(lam,*optim_cont), c='blue')
#	plt.show()

#	flux = flux - sline(lam,*optim_cont)
#	if s2:
#		aaa = np.mean(cont_f - sline(cont,*optim_cont))
#		for i in range(len(lam)):
#			if lam[i]<6690:
#				flux[i] = aaa
#	elif o3:
#		aaa = np.mean(cont_f - sline(cont,*optim_cont))
#	
#	plt.plot(lam,flux, c='black')
#	plt.show()

#	return flux, rms_cont

############################## FIT DEL CONTINUO 2 ###############################################################

'''
Versione avanzata rispetto al precedente. Usa Astropy per il fit, calcola il continuo su intervalli prefisati 
stabiliti a priori.

'''
def continuum_sub(line, wavelength, flux):
    lcont = line.lcont                                                      
    ucont = line.ucont
    int1 = np.where((wavelength > lcont[0]) & (wavelength < lcont[1]))
    int2 = np.where((wavelength > ucont[0]) & (wavelength < ucont[1]))
    f_wave = wavelength[int1]                                               #wavelengths for fit 
    f_wave = np.append(f_wave,wavelength[int2])
    f_flux = flux[int1]                                                     #fluxes
    f_flux = np.append(f_flux,flux[int2])
    m_cont = models.Linear1D(0,0)                                           #performing the fit
    fitter = fitting.LinearLSQFitter()
    gg_fit = fitter(m_cont, f_wave,f_flux)
    
    rms_cont = np.std(f_flux -gg_fit(f_wave))  #rms_cont for montecarlo simulations (flux - fit)
    
    plt.plot(wavelength[int1[0][0]: int2[0][-1]],flux[int1[0][0]: int2[0][-1]])
    plt.plot(wavelength[int1[0][0]: int2[0][-1]], gg_fit(wavelength[int1[0][0]: int2[0][-1]]))
    plt.show(block = 1)
    
    out_flux = flux[int1[0][0]: int2[0][-1]] - gg_fit(wavelength[int1[0][0]: int2[0][-1]]) #flux in the interesting interval corrected for the continuum
    out_wave = wavelength[int1[0][0]: int2[0][-1]]
    return out_flux, out_wave, rms_cont

############################## DATA SELECTION #################################################################

def selection(data, x, y, line, r, debug = False):
	if debug:
		print(x, y, line, r)
		exit()

	aa = np.where((data[:,x]>line-r)&(data[:,x]<line+r))		#seleziono la riga
	lam = data[aa,x][0]
	flux = data[aa,y][0]
	return lam, flux 

