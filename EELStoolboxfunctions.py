# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:24:38 2020

@author: Sophia
"""

import numpy as np

# scientific python
import scipy as spy
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
# needed to plot colormap
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pandas as pd
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)


def ZLPalignment(eelsSI, dispersion, ZLPmax, zlpSI=None, startLowLoss=None):
    if zlpSI is None and startLowLoss is not None:
        return print('Make sure to provide both an array containing the ZLP spectra and the onset of the LowLoss region!')
    elif zlpSI is not None and zlpSI is None:
        return print('Make sure to provide both an array containing the ZLP spectra and the onset of the LowLoss region!')
    elif zlpSI is None and startLowLoss is None:
        ZLP_max = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        ZLP_max_energy = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        spectrum = np.zeros([np.shape(eelsSI)[2], 2])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                spectrum[:, 0] = np.arange(np.shape(eelsSI)[2]) * dispersion
                spectrum[:, 1] = eelsSI[i, j, :]
                if sum(spectrum[:, 1]) != 0:
                    S1B = hs.signals.EELSSpectrum(spectrum[:, 1])
                    maxPeak = S1B.find_peaks1D_ohaver(amp_thresh=ZLPmax, peakgroup=50, maxpeakn=1)
                    ZLP_max[i, j] = maxPeak[0][0][0]
                    ZLP_max_energy[i, j] = spectrum[int(maxPeak[0][0][0]), 0]
                else:
                    ZLP_max[i, j] = 0
                    ZLP_max_energy[i, j] = 0
        eelsSI_aligned = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1], np.shape(eelsSI)[2] + 100])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                for k in range(np.shape(eelsSI)[2]):
                    eelsSI_aligned[i, j, (k + 200 - int(np.round(ZLP_max[i, j])))] = eelsSI[i, j, k]
                    if int(np.round(ZLP_max[i, j])) < 200:
                        eelsSI_aligned[i, j, (k + 200-int(np.round(ZLP_max[i, j])))] = eelsSI[i, j, k]
                    elif int(np.round(ZLP_max[i, j])) > 200 and k + int(np.round(ZLP_max[i, j])) - 200 < np.shape(eelsSI)[2]:
                        eelsSI_aligned[i, j, k] = eelsSI[i, j, (k + int(np.round(ZLP_max[i, j])) - 200)]
        energyZLP = np.arange(np.shape(eelsSI)[2])*dispersion - 200*dispersion
        return eelsSI_aligned, energyZLP, ZLP_max
    else:
        ZLP_max = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        ZLP_max_energy = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        spectrum = np.zeros([np.shape(eelsSI)[2], 2])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                spectrum[:, 0] = np.arange(np.shape(eelsSI)[2]) * dispersion
                spectrum[:, 1] = zlpSI[i, j, :]
                if sum(spectrum[:, 1]) != 0:
                    S1B = hs.signals.EELSSpectrum(spectrum[:, 1])
                    maxPeak = S1B.find_peaks1D_ohaver(amp_thresh=ZLPmax, peakgroup=50, maxpeakn=1)
                    ZLP_max[i, j] = maxPeak[0][0][0]
                    ZLP_max_energy[i, j] = spectrum[int(maxPeak[0][0][0]), 0]
                else:
                    ZLP_max[i, j] = 0
                    ZLP_max_energy[i, j] = 0
        eelsSI_aligned = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1], np.shape(eelsSI)[2] + 200])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                for k in range(np.shape(eelsSI)[2]):
                    if int(np.round(ZLP_max[i, j])) < 200:
                        eelsSI_aligned[i, j, k + 200 - int(np.round(ZLP_max[i, j]))] = eelsSI[i, j, k]
                    if int(np.round(ZLP_max[i, j])) > 200 and k + int(np.round(ZLP_max[i, j])) - 200 < np.shape(eelsSI)[2]:
                        eelsSI_aligned[i, j, k] = eelsSI[i, j, (k + int(np.round(ZLP_max[i, j])) - 200)]
        energyZLP = np.arange(np.shape(eelsSI)[2]+200)*dispersion - 200*dispersion
        energyLowLoss = np.arange(np.shape(eelsSI)[2]+200)*dispersion + startLowLoss
        return eelsSI_aligned, energyZLP, ZLP_max, energyLowLoss


def Bkg_fit(eelsSI_data, energy, startBkg, endBkg, BkgModel, fitpara=None, fitbounds=None):
    # ZLP correction
    def powerlaw(x, A, r):
        return A*np.power(x, -r)

    def polyfitfunc(x, A, m, r, b):
        return A*np.power((x + m), -r) + b

    def polyfitfunc2(x, A, m, r, c):
        return A*np.power((x + m), (-r-(c*x)))
    
    def linearfunc(x, A, t):
        return A*x + t

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    start_Bkg2 = np.where(energy == find_nearest(energy, startBkg))
    end_Bkg2 = np.where(energy == find_nearest(energy, endBkg))
        
    if np.shape(eelsSI_data.shape)[0] == 2:
        PowerLawFit = np.zeros([np.shape(eelsSI_data)[0], 2])
        Linear = np.zeros([np.shape(eelsSI_data)[0], 2])
        PolyFit = np.zeros([np.shape(eelsSI_data)[0], 4])
        Poly2Fit = np.zeros([np.shape(eelsSI_data)[0], 4])
        for i in range(np.shape(eelsSI_data)[0]):
            if sum(eelsSI_data[i]) != 0:
                if BkgModel == 'PL':
                    try:
                        if fitbounds is not None:
                            poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        elif fitbounds is None:
                            poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                        PowerLawFit[i, :] = poptPL[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        continue
                elif BkgModel == 'Poly1':
                    try:
                        poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        PolyFit[i, :] = poptPoly[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        continue
                elif BkgModel == 'Poly2':
                    try:
                        poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        Poly2Fit[i, :] = poptPoly2[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        continue
                elif BkgModel == 'Linear':
                    try:
                        poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        Linear[i, :] = poptLinear[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        
        eelsSI_woBG = np.zeros([np.shape(eelsSI_data)[0], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_energy = np.zeros([np.shape(energy)[0]-start_Bkg2[0][0], 1])
    
        for i in range(np.shape(eelsSI_data)[0]):
            k = 0
            if sum(eelsSI_data[i]) != 0:
                for j in range(start_Bkg2[0][0], np.shape(energy)[0]):
                    if BkgModel == 'PL':
                        eelsSI_woBG[i, k, 0] = eelsSI_data[i, j] - powerlaw(energy[j], PowerLawFit[i, 0], PowerLawFit[i, 1])
                    elif BkgModel == 'Poly1':
                        eelsSI_woBG[i, k, 0] = eelsSI_data[i, j] - polyfitfunc(energy[j], PolyFit[i, 0], PolyFit[i, 1], PolyFit[i, 2], PolyFit[i, 3])
                    elif BkgModel == 'Poly2':
                        eelsSI_woBG[i, k, 0] = eelsSI_data[i, j] - polyfitfunc2(energy[j], Poly2Fit[i, 0], Poly2Fit[i, 1], Poly2Fit[i, 2], Poly2Fit[i, 3])
                    elif BkgModel == 'Linear':
                        eelsSI_woBG[i, k, 0] = eelsSI_data[i, j] - linearfunc(energy[j], Linear[i, 0], Linear[i, 1])
                    eelsSI_woBG_energy[k, 0] = energy[j]
                    k = k + 1 
                    
        eelsSI_woBG_zero = np.zeros([np.shape(eelsSI_data)[0], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_smooth = np.zeros([np.shape(eelsSI_data)[0], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_smooth_zero = np.zeros([np.shape(eelsSI_data)[0], np.shape(energy)[0]-start_Bkg2[0][0], 1])

        for i in range(np.shape(eelsSI_data)[0]):
            eelsSI_woBG_zero[i, :, 0] = eelsSI_woBG[i, :, 0] - np.min(eelsSI_woBG[i, 0:end_Bkg2[0][0]-start_Bkg2[0][0], 0])
            eelsSI_woBG_smooth[i, :, 0] = savgol_filter(eelsSI_woBG[i, :, 0], 9, 2)
            eelsSI_woBG_smooth_zero[i, :, 0] = eelsSI_woBG_smooth[i, :, 0] - np.min(eelsSI_woBG_smooth[i, 0:end_Bkg2[0][0]-start_Bkg2[0][0], 0])
     
    elif np.shape(eelsSI_data.shape)[0] == 3:  
        PowerLawFit = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], 2])
        Linear = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], 2])
        PolyFit = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], 4])
        Poly2Fit = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], 4])
        for i in range(np.shape(eelsSI_data)[0]):
            for j in range(np.shape(eelsSI_data)[1]):
                if sum(eelsSI_data[i, j, :]) != 0:
                    if BkgModel == 'PL':
                        try:
                            if fitbounds is not None:
                                poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            elif fitbounds is None:
                                poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                            PowerLawFit[i, j, :] = poptPL[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i))
                            continue
                    elif BkgModel == 'Poly1':
                        try:
                            poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            PolyFit[i, j, :] = poptPoly[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i))
                            continue
                    elif BkgModel == 'Poly2':
                        try:
                            poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            Poly2Fit[i, j, :] = poptPoly2[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i))
                            continue
                    elif BkgModel == 'Linear':
                        try:
                            poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            Linear[i, j, :] = poptLinear[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i))  

        eelsSI_woBG = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_energy = np.zeros([np.shape(energy)[0]-start_Bkg2[0][0], 1])
    
        for i in range(np.shape(eelsSI_data)[0]):
            for j in range(np.shape(eelsSI_data)[1]):
                k = 0
                if sum(eelsSI_data[i, j, :]) != 0:
                    for h in range(start_Bkg2[0][0], np.shape(energy)[0]):
                        if BkgModel == 'PL':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - powerlaw(energy[h], PowerLawFit[i, j,  0], PowerLawFit[i, j, 1])
                        elif BkgModel == 'Poly1':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - polyfitfunc(energy[h], PolyFit[i, j, 0], PolyFit[i, 1], PolyFit[i, j, 2], PolyFit[i, j, 3])
                        elif BkgModel == 'Poly2':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - polyfitfunc2(energy[h], Poly2Fit[i, j, 0], Poly2Fit[i, j, 1], Poly2Fit[i, j, 2], Poly2Fit[i, j, 3])
                        elif BkgModel == 'Linear':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - linearfunc(energy[h], Linear[i, j, 0], Linear[i, j, 1])
                        eelsSI_woBG_energy[k, 0] = energy[h]
                        k = k + 1 
                
        eelsSI_woBG_zero = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_smooth = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_smooth_zero = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], np.shape(energy)[0]-start_Bkg2[0][0], 1])

        for i in range(np.shape(eelsSI_data)[0]):
            for j in range(np.shape(eelsSI_data)[1]):
                eelsSI_woBG_zero[i, j, :] = eelsSI_woBG[i, j, :] - np.min(eelsSI_woBG[i, j, 0:end_Bkg2[0][0]-start_Bkg2[0][0]])
                eelsSI_woBG_smooth[i, j, :, 0] = savgol_filter(eelsSI_woBG[i, j, :, 0], 9, 2)
                eelsSI_woBG_smooth_zero[i, j, :] = eelsSI_woBG_smooth[i, j, :] - np.min(eelsSI_woBG_smooth[i, j, 0:end_Bkg2[0][0]-start_Bkg2[0][0]])
    
    return eelsSI_woBG, eelsSI_woBG_smooth, eelsSI_woBG_energy, eelsSI_woBG_zero, eelsSI_woBG_smooth_zero

def SI_average(eelsSI, windowSize):
    eelsSI_averaged = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1], np.shape(eelsSI)[2]])

    for i in range(np.shape(eelsSI)[0]-windowSize):
        for j in range(np.shape(eelsSI)[1]-windowSize):
            eelsSI_averaged[i, j, :] = sum(sum(eelsSI[i : i+windowSize, j : j+windowSize, :]))
    
    return eelsSI_averaged


def peak_deconvolution_flexible(eels_SI_woBG, energy, SIregion1, SIregion2, startFit, endFit, peak_number, fitParaStart=None, fitParaBounds=None):

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    start_Fit2 = np.where(energy == find_nearest(energy, startFit))
    end_Fit2 = np.where(energy == find_nearest(energy, endFit))

    def _1gaussian(x, amp1,cen1,sigma1):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

    def _fit4G(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, amp4, cen4, sigma4):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen3)/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen4)/sigma4)**2))) 

    def _fit5G(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, amp4, cen4, sigma4, amp5, cen5, sigma5):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen3)/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen4)/sigma4)**2))) + \
                amp5*(1/(sigma5*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen5)/sigma5)**2)))

    pars_1 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
    pars_2 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
    pars_3 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
    pars_4 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
    pars_5 = np.zeros([np.shape(eels_SI_woBG)[0], 3])

    gauss_peak = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0], 5])
    sumspec = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0]])

    if peak_number == 4:
        parameters = np.zeros([np.shape(eels_SI_woBG)[0], 12])
        for i in range(SIregion1, SIregion2):
            if sum(eels_SI_woBG[i, :]) != 0:
                if fitParaBounds is not None:
                    try:
                        popt_fit, pcov_fit = spy.optimize.curve_fit(_fit4G, energy[start_Fit2[0][0]:end_Fit2[0][0], 0], eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0], 0], p0=fitParaStart, bounds=fitParaBounds, maxfev=8000)
                    except RuntimeError:
                        print('Fit did not converge for spectrum ' + str(i))
                        popt_fit = np.zeros(12)
                        pcov_fit = None
                else:
                    try:
                        popt_fit, pcov_fit = spy.optimize.curve_fit(_fit4G, energy[start_Fit2[0][0]:end_Fit2[0][0], 0], eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0], 0], p0=fitParaStart, maxfev=8000)
                    except RuntimeError:
                        print('Fit did not converge for spectrum ' + str(i))
                        popt_fit = np.zeros(12)
                        pcov_fit = None
                pars_1[i] = popt_fit[0:3]
                pars_2[i] = popt_fit[3:6]
                pars_3[i] = popt_fit[6:9]
                pars_4[i] = popt_fit[9:12]
                parameters[i, :] = popt_fit
                gauss_peak[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_1[i])
                gauss_peak[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_2[i])
                gauss_peak[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_3[i])
                gauss_peak[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_4[i])
                sumspec[i, :] = gauss_peak[i, :, 0] + gauss_peak[i, :, 1] + gauss_peak[i, :, 2] + gauss_peak[i, :, 3]
                del popt_fit, pcov_fit
                #except ValueError:
                 #   print('Deconvolution was not successful for spectrum ' + str(i))
    if peak_number == 5:
        parameters = np.zeros([np.shape(eels_SI_woBG)[0], 15])
        for i in range(SIregion1, SIregion2):
            if sum(eels_SI_woBG[i, :]) != 0:
                if fitParaBounds is not None:
                    try:
                        popt_fit, pcov_fit = spy.optimize.curve_fit(_fit5G, energy[start_Fit2[0][0]:end_Fit2[0][0], 0], eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0], 0], p0=fitParaStart, bounds=fitParaBounds, maxfev=8000)
                    except RuntimeError:
                        print('Fit did not converge for spectrum ' + str(i))
                        popt_fit = np.zeros(15)
                        pcov_fit = None
                else:
                    try:
                        popt_fit, pcov_fit = spy.optimize.curve_fit(_fit5G, energy[start_Fit2[0][0]:end_Fit2[0][0], 0], eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0], 0], p0=fitParaStart, maxfev=8000)
                    except RuntimeError:
                        print('Fit did not converge for spectrum ' + str(i))
                        popt_fit = np.zeros(15)
                        pcov_fit = None
                pars_1[i] = popt_fit[0:3]
                pars_2[i] = popt_fit[3:6]
                pars_3[i] = popt_fit[6:9]
                pars_4[i] = popt_fit[9:12]
                pars_5[i] = popt_fit[12:15]
                parameters[i, :] = popt_fit
                gauss_peak[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_1[i])
                gauss_peak[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_2[i])
                gauss_peak[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_3[i])
                gauss_peak[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_4[i])
                gauss_peak[i, :, 4] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0], 0], *pars_5[i])
                sumspec[i, :] = gauss_peak[i, :, 0] + gauss_peak[i, :, 1] + gauss_peak[i, :, 2] + gauss_peak[i, :, 3] + gauss_peak[i, :, 4]
                del popt_fit, pcov_fit

    return sumspec, parameters, gauss_peak


def signal_intensity(eelsSI, startEnergy, endEnergy):

    if np.shape(eelsSI)[0] == 3:
        eelsIntensity = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                eelsIntensity[i, j] = sum(eelsSI[i, j])
    elif np.shape(eelsSI)[0] == 2:
        eelsIntensity = np.zeros([np.shape(eelsSI)[0], 1])
        for i in range(np.shape(eelsSI)[0]):
            eelsIntensity[i] = sum(eelsSI[i])

    return eelsIntensity


def roll_av(data, windowsize):
    data_rollav = np.zeros(np.shape(data)[0])
    for i in range(windowsize, np.shape(data)[0]):
        data_rollav[i-int(windowsize/2)] = sum(data[i-windowsize:i])/windowsize
    for i in range(int(windowsize/2)+1):
        data_rollav[i] = sum(data[0:i + int(windowsize/2)])/(int(windowsize/2) + i)
    k = 0
    for i in range(np.shape(data)[0]-int(windowsize/2), np.shape(data)[0]):
        data_rollav[i] = sum(data[i - int(windowsize/2):np.shape(data)[0]])/(int(windowsize/2) + np.shape(data)[0] - i)
        k = k + 1
    return data_rollav