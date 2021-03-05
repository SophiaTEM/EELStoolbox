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
from scipy.signal import find_peaks
# needed to plot colormap
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pandas as pd
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
from lmfit import Model


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
                        if fitbounds is not None:
                            poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        elif fitbounds is None:
                            poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                        PolyFit[i, :] = poptPoly[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        continue
                elif BkgModel == 'Poly2':
                    try:
                        if fitbounds is not None:
                            poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        elif fitbounds is None:
                            poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                        Poly2Fit[i, :] = poptPoly2[:]
                    except RuntimeError:
                        print("Error - curve_fit failed for datapoint " + str(i))
                        continue
                elif BkgModel == 'Linear':
                    try:
                        if fitbounds is not None:
                            poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                        elif fitbounds is None:
                            poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
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
        
        if BkgModel == 'Poly1':
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), 3]))
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), 3]))
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), 3]))    
        elif BkgModel == 'Poly2':
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), 3]))
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), 3]))
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), 3]))
        elif BkgModel == 'PL':
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/2), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/2), 1]))
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/4), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/4), 1]))
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/1.5), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/1.5), 1]))
        elif BkgModel == 'Linear':
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/2), 0], Linear[int(np.shape(eelsSI_woBG)[0]/2), 1]))
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/4), 0], Linear[int(np.shape(eelsSI_woBG)[0]/4), 1]))
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/1.5), 0], Linear[int(np.shape(eelsSI_woBG)[0]/1.5), 1]))      

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
                            if fitbounds is not None:
                                poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            elif fitbounds is None:
                                poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                            PolyFit[i, j, :] = poptPoly[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i) + '-' + str(j))
                            continue
                    elif BkgModel == 'Poly2':
                        try:
                            if fitbounds is not None:
                                poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            elif fitbounds is None:
                                poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                            Poly2Fit[i, j, :] = poptPoly2[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i) + '-' + str(j))
                            continue
                    elif BkgModel == 'Linear':
                        try:
                            if fitbounds is not None:
                                poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                            elif fitbounds is None:
                                poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[i, j, start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                            Linear[i, j, :] = poptLinear[:]
                        except RuntimeError:
                            print("Error - curve_fit failed for datapoint " + str(i) + '-' + str(j))  

        eelsSI_woBG = np.zeros([np.shape(eelsSI_data)[0], np.shape(eelsSI_data)[1], np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_energy = np.zeros([np.shape(energy)[0]-start_Bkg2[0][0], 1])

        for i in range(np.shape(eelsSI_data)[0]):
            for j in range(np.shape(eelsSI_data)[1]):
                if sum(eelsSI_data[i, j, :]) != 0:
                    k = 0
                    for h in range(start_Bkg2[0][0], np.shape(energy)[0]):
                        if BkgModel == 'PL':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - powerlaw(energy[h], PowerLawFit[i, j,  0], PowerLawFit[i, j, 1])
                        elif BkgModel == 'Poly1':
                            eelsSI_woBG[i, j, k] = eelsSI_data[i, j, h] - polyfitfunc(energy[h], PolyFit[i, j, 0], PolyFit[i, j, 1], PolyFit[i, j, 2], PolyFit[i, j, 3])
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

        if BkgModel == 'Poly1':
            Parameters = PolyFit
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 3]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 3]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), :])
            plt.plot(energy[:], polyfitfunc(energy[:], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 0], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 1], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 2], PolyFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 3]))    
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
        elif BkgModel == 'Poly2':
            Parameters = Poly2Fit
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 3]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 3]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), :])
            plt.plot(energy[:], polyfitfunc2(energy[:], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 0], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 1], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 2], Poly2Fit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 3]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
        elif BkgModel == 'PL':
            Parameters = PowerLawFit
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), :])
            plt.plot(energy[:], powerlaw(energy[:], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 0], PowerLawFit[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
        elif BkgModel == 'Linear':
            Parameters = Linear
            plt.figure(1)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 0], Linear[int(np.shape(eelsSI_woBG)[0]/2), int(np.shape(eelsSI_woBG)[1]/2), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(2)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 0], Linear[int(np.shape(eelsSI_woBG)[0]/4), int(np.shape(eelsSI_woBG)[1]/4), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
            plt.figure(3)
            plt.plot(energy[:], eelsSI_data[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), :])
            plt.plot(energy[:], linearfunc(energy[:], Linear[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 0], Linear[int(np.shape(eelsSI_woBG)[0]/1.5), int(np.shape(eelsSI_woBG)[1]/1.5), 1]))
            plt.xlim([start_Bkg2[0][0], end_Bkg2[0][0]+200])
    
    else:
        PowerLawFit = np.zeros(2)
        Linear = np.zeros(2)
        PolyFit = np.zeros(4)
        Poly2Fit = np.zeros(4)
        if sum(eelsSI_data) != 0:
            if BkgModel == 'PL':
                try:
                    if fitbounds is not None:
                        poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                    elif fitbounds is None:
                        poptPL, pcov = curve_fit(powerlaw, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                    PowerLawFit[:] = poptPL[:]
                except RuntimeError:
                    print("Error - curve_fit failed for spectrum.")
            elif BkgModel == 'Poly1':
                try:
                    if fitbounds is not None:
                        poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                    elif fitbounds is None:
                        poptPoly, pcov = curve_fit(polyfitfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                    PolyFit[:] = poptPoly[:]
                except RuntimeError:
                    print("Error - curve_fit failed for spectrum.")
            elif BkgModel == 'Poly2':
                try:
                    if fitbounds is not None:
                        poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                    elif fitbounds is None:
                        poptPoly2, pcov = curve_fit(polyfitfunc2, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                    Poly2Fit[:] = poptPoly2[:]
                except RuntimeError:
                    print("Error - curve_fit failed for spectrum.")
            elif BkgModel == 'Linear':
                try:
                    if fitbounds is not None:
                        poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, bounds=fitbounds, maxfev=8000)
                    elif fitbounds is None:
                        poptLinear, pcov = curve_fit(linearfunc, energy[start_Bkg2[0][0]:end_Bkg2[0][0]], eelsSI_data[start_Bkg2[0][0]:end_Bkg2[0][0]], p0=fitpara, maxfev=8000)
                    Linear[:] = poptLinear[:]
                except RuntimeError:
                    print("Error - curve_fit failed for spectrum.")

        eelsSI_woBG = np.zeros([np.shape(energy)[0]-start_Bkg2[0][0], 1])
        eelsSI_woBG_energy = np.zeros([np.shape(energy)[0]-start_Bkg2[0][0], 1])
    
        k = 0
        if sum(eelsSI_data) != 0:
            for j in range(start_Bkg2[0][0], np.shape(energy)[0]):
                if BkgModel == 'PL':
                    eelsSI_woBG[k, 0] = eelsSI_data[j] - powerlaw(energy[j], PowerLawFit[0], PowerLawFit[1])
                    Parameters = PowerLawFit
                elif BkgModel == 'Poly1':
                    eelsSI_woBG[k, 0] = eelsSI_data[j] - polyfitfunc(energy[j], PolyFit[0], PolyFit[1], PolyFit[2], PolyFit[3])
                    Parameters = PolyFit
                elif BkgModel == 'Poly2':
                    Parameters = Poly2Fit
                    eelsSI_woBG[k, 0] = eelsSI_data[j] - polyfitfunc2(energy[j], Poly2Fit[0], Poly2Fit[1], Poly2Fit[2], Poly2Fit[3])
                elif BkgModel == 'Linear':
                    Parameters = Linear
                    eelsSI_woBG[k, 0] = eelsSI_data[j] - linearfunc(energy[j], Linear[0], Linear[1])
                eelsSI_woBG_energy[k, 0] = energy[j]
                k = k + 1 

        eelsSI_woBG_zero = eelsSI_woBG[:, 0] - np.min(eelsSI_woBG[0:end_Bkg2[0][0]-start_Bkg2[0][0]])
        eelsSI_woBG_smooth = savgol_filter(eelsSI_woBG[:, 0], 9, 2)
        eelsSI_woBG_smooth_zero = eelsSI_woBG_smooth[:] - np.min(eelsSI_woBG_smooth[0:end_Bkg2[0][0]-start_Bkg2[0][0]])

            
    return eelsSI_woBG, eelsSI_woBG_energy, Parameters#, eelsSI_woBG_smooth, eelsSI_woBG_zero, eelsSI_woBG_smooth_zero


def SI_average(eelsSI, windowSizeX, windowSizeY):
    if np.shape(eelsSI)[0]-windowSizeX == 0:
        eelsSI_averaged = np.zeros([np.shape(eelsSI)[1]-windowSizeY, np.shape(eelsSI)[2]])
        for j in range(np.shape(eelsSI)[1]-windowSizeY):
            eelsSI_averaged[j, :] = sum(sum(eelsSI[:, j:j + windowSizeY, :]))
    elif np.shape(eelsSI)[1]-windowSizeY == 0:
        eelsSI_averaged = np.zeros([np.shape(eelsSI)[0]-windowSizeX, np.shape(eelsSI)[2]])
        for i in range(np.shape(eelsSI)[0]-windowSizeX):
            eelsSI_averaged[i, :] = sum(sum(eelsSI[i:i + windowSizeX, :, :]))
    else:
        eelsSI_averaged = np.zeros([np.shape(eelsSI)[0]-windowSizeX, np.shape(eelsSI)[1]-windowSizeY, np.shape(eelsSI)[2]])
        for i in range(np.shape(eelsSI)[0]-windowSizeX):
            for j in range(np.shape(eelsSI)[1]-windowSizeY):
                eelsSI_averaged[i, j, :] = sum(sum(eelsSI[i:i + windowSizeX, j:j + windowSizeY, :]))

    return eelsSI_averaged


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def peak_deconvolution_flexible(eels_SI_woBG, energy, SIregion1, SIregion2, startFit, endFit, peak_number, fitParaStart=None, fitParaBounds=None):

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

    if len(np.shape(eels_SI_woBG)) == 3:
        if peak_number == 4:
            pars_1 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_2 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_3 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_4 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
        
            gauss_peak = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0], 4])
            sumspec = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0]])
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
            pars_1 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_2 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_3 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_4 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_5 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
        
            gauss_peak = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0], 5])
            sumspec = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0]])
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
    if len(np.shape(eels_SI_woBG)) == 2:
        if peak_number == 4:
            pars_1 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_2 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_3 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_4 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
        
            gauss_peak = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0], 4])
            sumspec = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0]])
            parameters = np.zeros([np.shape(eels_SI_woBG)[0], 12])
            for i in range(SIregion1, SIregion2):
                if sum(eels_SI_woBG[i, :]) != 0:
                    if fitParaBounds is not None:
                        eels_SI_norm = (eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]-np.min(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]))/(np.max(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]])-np.min(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]))
                        try:
                            popt_fit, pcov_fit = spy.optimize.curve_fit(_fit4G, energy[start_Fit2[0][0]:end_Fit2[0][0]], eels_SI_norm, p0=fitParaStart, bounds=fitParaBounds, maxfev=8000)
                        except RuntimeError:
                            print('Fit did not converge for spectrum ' + str(i))
                            popt_fit = np.zeros(12)
                            pcov_fit = None
                    else:
                        try:
                            popt_fit, pcov_fit = spy.optimize.curve_fit(_fit4G, energy[start_Fit2[0][0]:end_Fit2[0][0]], eels_SI_norm, p0=fitParaStart, maxfev=8000)
                        except RuntimeError:
                            print('Fit did not converge for spectrum ' + str(i))
                            popt_fit = np.zeros(12)
                            pcov_fit = None
                    pars_1[i] = popt_fit[0:3]
                    pars_2[i] = popt_fit[3:6]
                    pars_3[i] = popt_fit[6:9]
                    pars_4[i] = popt_fit[9:12]
                    parameters[i, :] = popt_fit
                    gauss_peak[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_1[i])
                    gauss_peak[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_2[i])
                    gauss_peak[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_3[i])
                    gauss_peak[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_4[i])
                    sumspec[i, :] = gauss_peak[i, :, 0] + gauss_peak[i, :, 1] + gauss_peak[i, :, 2] + gauss_peak[i, :, 3]
                    del popt_fit, pcov_fit
                    #except ValueError:
                     #   print('Deconvolution was not successful for spectrum ' + str(i))
        if peak_number == 5:
            pars_1 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_2 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_3 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_4 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
            pars_5 = np.zeros([np.shape(eels_SI_woBG)[0], 3])
        
            gauss_peak = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0], 5])
            sumspec = np.zeros([np.shape(eels_SI_woBG)[0], end_Fit2[0][0] - start_Fit2[0][0]])
            parameters = np.zeros([np.shape(eels_SI_woBG)[0], 15])
            for i in range(SIregion1, SIregion2):
                if sum(eels_SI_woBG[i, :]) != 0:
                    if fitParaBounds is not None:
                        eels_SI_norm = (eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]-np.min(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]))/(np.max(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]])-np.min(eels_SI_woBG[i, start_Fit2[0][0]:end_Fit2[0][0]]))
                        try:
                            popt_fit, pcov_fit = spy.optimize.curve_fit(_fit5G, energy[start_Fit2[0][0]:end_Fit2[0][0]], eels_SI_norm, p0=fitParaStart, bounds=fitParaBounds, maxfev=8000)
                        except RuntimeError:
                            print('Fit did not converge for spectrum ' + str(i))
                            popt_fit = np.zeros(15)
                            pcov_fit = None
                    else:
                        try:
                            popt_fit, pcov_fit = spy.optimize.curve_fit(_fit5G, energy[start_Fit2[0][0]:end_Fit2[0][0]], eels_SI_norm, p0=fitParaStart, maxfev=8000)
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
                    gauss_peak[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_1[i])
                    gauss_peak[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_2[i])
                    gauss_peak[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_3[i])
                    gauss_peak[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_4[i])
                    gauss_peak[i, :, 4] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *pars_5[i])
                    sumspec[i, :] = gauss_peak[i, :, 0] + gauss_peak[i, :, 1] + gauss_peak[i, :, 2] + gauss_peak[i, :, 3] + gauss_peak[i, :, 4]
                    del popt_fit, pcov_fit

    return sumspec, parameters, gauss_peak

def peak_deconvolution_fixed(TimeSeries, energy, TSregion1, TSregion2, startFit, endFit, peak_number, PksPos, fit_guess, fit_bounds):
 
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    if np.shape(PksPos)[0] != peak_number or np.shape(fit_guess)[0] != peak_number*2+1 or np.shape(fit_bounds)[1] != peak_number*2+1:
        print('Make sure that you provide enough values for the peak position, guess of the fit parameters and bounds of the fit parameters!')
        return None
    
    start_Fit2 = np.where(energy == find_nearest(energy, startFit))
    end_Fit2 = np.where(energy == find_nearest(energy, endFit))
    GaussFit = np.zeros([np.shape(TimeSeries)[0], end_Fit2[0][0]-start_Fit2[0][0], peak_number])
    Parameter = np.zeros([np.shape(TimeSeries)[0], peak_number*3])
    SumSpec = np.zeros([np.shape(TimeSeries)[0], end_Fit2[0][0]-start_Fit2[0][0]])

    def _1gaussian(x, cen1, amp1, sigma1):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

    def _fit2G(x, PksPos1, PksPos2, cen, amp1, sigma1, amp2, sigma2):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos1))/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos2))/sigma2)**2))) 
        
    def _fit3G(x, PksPos1, PksPos2, PksPos3, cen, amp1, sigma1, amp2, sigma2, amp3, sigma3):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos1))/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos2))/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos3))/sigma3)**2))) 

    def _fit4G(x, PksPos1, PksPos2, PksPos3, PksPos4, cen, amp1, sigma1, amp2, sigma2, amp3, sigma3, amp4, sigma4):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos1))/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos2))/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos3))/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos4))/sigma4)**2)))

    def _fit5G(x, PksPos1, PksPos2, PksPos3, PksPos4, PksPos5, cen, amp1, sigma1, amp2, sigma2, amp3, sigma3, amp4, sigma4, amp5, sigma5):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos1))/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos2))/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos3))/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos4))/sigma4)**2))) + \
                amp5*(1/(sigma5*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos5))/sigma5)**2)))

    def _fit6G(x, PksPos1, PksPos2, PksPos3, PksPos4, PksPos5, PksPos6, cen, amp1, sigma1, amp2, sigma2, amp3, sigma3, amp4, sigma4, amp5, sigma5, amp6, sigma6):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos1))/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos2))/sigma2)**2))) + \
                amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos3))/sigma3)**2))) + \
                amp4*(1/(sigma4*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos4))/sigma4)**2))) + \
                amp5*(1/(sigma5*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos5))/sigma5)**2))) + \
                amp6*(1/(sigma6*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-(cen+PksPos6))/sigma6)**2))) 

    if peak_number == 2:
        for i in range(TSregion1, TSregion2):
            if sum(TimeSeries[i, :]) != 0:
                try:
                    fmodel = Model(_fit2G)
                    params = fmodel.make_params(PksPos1=PksPos[0], PksPos2=PksPos[1], PksPos3=PksPos[2], PksPos4=PksPos[3],
                             cen=fit_guess[0],
                             amp1=fit_guess[1], amp2=fit_guess[2], amp3=fit_guess[3], amp4=fit_guess[4],
                             sigma1=fit_guess[5], sigma2=fit_guess[6], sigma3=fit_guess[7], sigma4=fit_guess[8])
                    params['PksPos1'].vary = False
                    params['PksPos2'].vary = False
                    params['cen'].min = fit_bounds[0][0]
                    params['cen'].max = fit_bounds[1][0]
                    params['amp1'].min = fit_bounds[0][1]
                    params['amp1'].max = fit_bounds[1][1]
                    params['amp2'].min = fit_bounds[0][2]
                    params['amp2'].max = fit_bounds[1][1]
                    params['sigma1'].min = fit_bounds[0][5]
                    params['sigma1'].max = fit_bounds[1][5]
                    params['sigma2'].min = fit_bounds[0][6]
                    params['sigma2'].max = fit_bounds[1][6]

                    result = fmodel.fit(TimeSeries[i, start_Fit2[0][0]:end_Fit2[0][0]], params, x=energy[start_Fit2[0][0]:end_Fit2[0][0]])

                    Parameter[i, 0] = result.params['cen'].value + PksPos[0]
                    Parameter[i, 1] = result.params['amp1'].value
                    Parameter[i, 2] = result.params['sigma1'].value
                    Parameter[i, 3] = result.params['cen'].value + PksPos[1]
                    Parameter[i, 4] = result.params['amp2'].value
                    Parameter[i, 5] = result.params['sigma2'].value

                    GaussFit[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 0:3])
                    GaussFit[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 3:6])

                    SumSpec[i, :] = GaussFit[i, :, 0] + GaussFit[i, :, 1]
                    del result
                except RuntimeError:
                    print("Error - deconvolution failed")
                    continue

    if peak_number == 3:
        for i in range(TSregion1, TSregion2):
            if sum(TimeSeries[i, :]) != 0:
                try:
                    fmodel = Model(_fit3G)
                    params = fmodel.make_params(PksPos1=PksPos[0], PksPos2=PksPos[1], PksPos3=PksPos[2],
                             cen=fit_guess[0],
                             amp1=fit_guess[1], amp2=fit_guess[2], amp3=fit_guess[3],
                             sigma1=fit_guess[4], sigma2=fit_guess[5], sigma3=fit_guess[6])
                    params['PksPos1'].vary = False
                    params['PksPos2'].vary = False
                    params['PksPos3'].vary = False
                    params['cen'].min = fit_bounds[0][0]
                    params['cen'].max = fit_bounds[1][0]
                    params['amp1'].min = fit_bounds[0][1]
                    params['amp1'].max = fit_bounds[1][1]
                    params['amp2'].min = fit_bounds[0][2]
                    params['amp2'].max = fit_bounds[1][1]
                    params['amp3'].min = fit_bounds[0][3]
                    params['amp3'].max = fit_bounds[1][3]
                    params['sigma1'].min = fit_bounds[0][4]
                    params['sigma1'].max = fit_bounds[1][4]
                    params['sigma2'].min = fit_bounds[0][5]
                    params['sigma2'].max = fit_bounds[1][5]
                    params['sigma3'].min = fit_bounds[0][6]
                    params['sigma3'].max = fit_bounds[1][6]

                    result = fmodel.fit(TimeSeries[i, start_Fit2[0][0]:end_Fit2[0][0]], params, x=energy[start_Fit2[0][0]:end_Fit2[0][0]])

                    Parameter[i, 0] = result.params['cen'].value + PksPos[0]
                    Parameter[i, 1] = result.params['amp1'].value
                    Parameter[i, 2] = result.params['sigma1'].value
                    Parameter[i, 3] = result.params['cen'].value + PksPos[1]
                    Parameter[i, 4] = result.params['amp2'].value
                    Parameter[i, 5] = result.params['sigma2'].value
                    Parameter[i, 6] = result.params['cen'].value + PksPos[2]
                    Parameter[i, 7] = result.params['amp3'].value
                    Parameter[i, 8] = result.params['sigma3'].value

                    GaussFit[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 0:3])
                    GaussFit[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 3:6])
                    GaussFit[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 6:9])

                    SumSpec[i, :] = GaussFit[i, :, 0] + GaussFit[i, :, 1] + GaussFit[i, :, 2]
                    del result
                except RuntimeError:
                    print("Error - deconvolution failed")
                    continue   

    if peak_number == 4:
        for i in range(TSregion1, TSregion2):
            if sum(TimeSeries[i, :]) != 0:
                try:
                    fmodel = Model(_fit4G)
                    params = fmodel.make_params(PksPos1=PksPos[0], PksPos2=PksPos[1], PksPos3=PksPos[2], PksPos4=PksPos[3],
                             cen=fit_guess[0],
                             amp1=fit_guess[1], amp2=fit_guess[2], amp3=fit_guess[3], amp4=fit_guess[4],
                             sigma1=fit_guess[5], sigma2=fit_guess[6], sigma3=fit_guess[7], sigma4=fit_guess[8])
                    params['PksPos1'].vary = False
                    params['PksPos2'].vary = False
                    params['PksPos3'].vary = False
                    params['PksPos4'].vary = False
                    params['cen'].min = fit_bounds[0][0]
                    params['cen'].max = fit_bounds[1][0]
                    params['amp1'].min = fit_bounds[0][1]
                    params['amp1'].max = fit_bounds[1][1]
                    params['amp2'].min = fit_bounds[0][2]
                    params['amp2'].max = fit_bounds[1][1]
                    params['amp3'].min = fit_bounds[0][3]
                    params['amp3'].max = fit_bounds[1][3]
                    params['amp4'].min = fit_bounds[0][4]
                    params['amp4'].max = fit_bounds[1][4]
                    params['sigma1'].min = fit_bounds[0][5]
                    params['sigma1'].max = fit_bounds[1][5]
                    params['sigma2'].min = fit_bounds[0][6]
                    params['sigma2'].max = fit_bounds[1][6]
                    params['sigma3'].min = fit_bounds[0][7]
                    params['sigma3'].max = fit_bounds[1][7]
                    params['sigma4'].min = fit_bounds[0][8]
                    params['sigma4'].max = fit_bounds[1][8]

                    result = fmodel.fit(TimeSeries[i, start_Fit2[0][0]:end_Fit2[0][0]], params, x=energy[start_Fit2[0][0]:end_Fit2[0][0]])

                    Parameter[i, 0] = result.params['cen'].value + PksPos[0]
                    Parameter[i, 1] = result.params['amp1'].value
                    Parameter[i, 2] = result.params['sigma1'].value
                    Parameter[i, 3] = result.params['cen'].value + PksPos[1]
                    Parameter[i, 4] = result.params['amp2'].value
                    Parameter[i, 5] = result.params['sigma2'].value
                    Parameter[i, 6] = result.params['cen'].value + PksPos[2]
                    Parameter[i, 7] = result.params['amp3'].value
                    Parameter[i, 8] = result.params['sigma3'].value
                    Parameter[i, 9] = result.params['cen'].value + PksPos[3]
                    Parameter[i, 10] = result.params['amp4'].value
                    Parameter[i, 11] = result.params['sigma4'].value

                    GaussFit[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 0:3])
                    GaussFit[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 3:6])
                    GaussFit[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 6:9])
                    GaussFit[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 9:12])

                    SumSpec[i, :] = GaussFit[i, :, 0] + GaussFit[i, :, 1] + GaussFit[i, :, 2] + GaussFit[i, :, 3]
                    del result
                except RuntimeError:
                    print("Error - deconvolution failed")
                    continue    

    if peak_number == 5:
        for i in range(TSregion1, TSregion2):
            if sum(TimeSeries[i, :]) != 0:
                try:
                    fmodel = Model(_fit5G)
                    params = fmodel.make_params(PksPos1=PksPos[0], PksPos2=PksPos[1], PksPos3=PksPos[2], PksPos4=PksPos[3], PksPos5=PksPos[4],
                             cen=fit_guess[0],
                             amp1=fit_guess[1], amp2=fit_guess[2], amp3=fit_guess[3], amp4=fit_guess[4], amp5=fit_guess[5],
                             sigma1=fit_guess[6], sigma2=fit_guess[7], sigma3=fit_guess[8], sigma4=fit_guess[9], sigma5=fit_guess[10])
                    params['PksPos1'].vary = False
                    params['PksPos2'].vary = False
                    params['PksPos3'].vary = False
                    params['PksPos4'].vary = False
                    params['PksPos5'].vary = False
                    params['cen'].min = fit_bounds[0][0]
                    params['cen'].max = fit_bounds[1][0]
                    params['amp1'].min = fit_bounds[0][1]
                    params['amp1'].max = fit_bounds[1][1]
                    params['amp2'].min = fit_bounds[0][2]
                    params['amp2'].max = fit_bounds[1][1]
                    params['amp3'].min = fit_bounds[0][3]
                    params['amp3'].max = fit_bounds[1][3]
                    params['amp4'].min = fit_bounds[0][4]
                    params['amp4'].max = fit_bounds[1][4]
                    params['amp5'].min = fit_bounds[0][5]
                    params['amp5'].max = fit_bounds[1][5]
                    params['sigma1'].min = fit_bounds[0][6]
                    params['sigma1'].max = fit_bounds[1][6]
                    params['sigma2'].min = fit_bounds[0][7]
                    params['sigma2'].max = fit_bounds[1][7]
                    params['sigma3'].min = fit_bounds[0][8]
                    params['sigma3'].max = fit_bounds[1][8]
                    params['sigma4'].min = fit_bounds[0][9]
                    params['sigma4'].max = fit_bounds[1][9]
                    params['sigma5'].min = fit_bounds[0][10]
                    params['sigma5'].max = fit_bounds[1][10]

                    result = fmodel.fit(TimeSeries[i, start_Fit2[0][0]:end_Fit2[0][0]], params, x=energy[start_Fit2[0][0]:end_Fit2[0][0]])

                    Parameter[i, 0] = result.params['cen'].value + PksPos[0]
                    Parameter[i, 1] = result.params['amp1'].value
                    Parameter[i, 2] = result.params['sigma1'].value
                    Parameter[i, 3] = result.params['cen'].value + PksPos[1]
                    Parameter[i, 4] = result.params['amp2'].value
                    Parameter[i, 5] = result.params['sigma2'].value
                    Parameter[i, 6] = result.params['cen'].value + PksPos[2]
                    Parameter[i, 7] = result.params['amp3'].value
                    Parameter[i, 8] = result.params['sigma3'].value
                    Parameter[i, 9] = result.params['cen'].value + PksPos[3]
                    Parameter[i, 10] = result.params['amp4'].value
                    Parameter[i, 11] = result.params['sigma4'].value
                    Parameter[i, 12] = result.params['cen'].value + PksPos[4]
                    Parameter[i, 13] = result.params['amp5'].value
                    Parameter[i, 14] = result.params['sigma5'].value

                    GaussFit[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 0:3])
                    GaussFit[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 3:6])
                    GaussFit[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 6:9])
                    GaussFit[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 9:12])
                    GaussFit[i, :, 4] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 12:15])

                    SumSpec[i, :] = GaussFit[i, :, 0] + GaussFit[i, :, 1] + GaussFit[i, :, 2] + GaussFit[i, :, 3] + GaussFit[i, :, 4]
                    del result
                except RuntimeError:
                    print("Error - deconvolution failed")
                    continue

    if peak_number == 6: 
        for i in range(TSregion1, TSregion2):
            if sum(TimeSeries[i, :]) != 0:
                try:
                    fmodel = Model(_fit6G)
                    params = fmodel.make_params(PksPos1=PksPos[0], PksPos2=PksPos[1], PksPos3=PksPos[2], PksPos4=PksPos[3], PksPos5=PksPos[4], PksPos6=PksPos[5],
                             cen=fit_guess[0],
                             amp1=fit_guess[1], amp2=fit_guess[2], amp3=fit_guess[3], amp4=fit_guess[4], amp5=fit_guess[5], amp6=fit_guess[6],
                             sigma1=fit_guess[7], sigma2=fit_guess[8], sigma3=fit_guess[9], sigma4=fit_guess[10], sigma5=fit_guess[11], sigma6=fit_guess[12])
                    params['PksPos1'].vary = False
                    params['PksPos2'].vary = False
                    params['PksPos3'].vary = False
                    params['PksPos4'].vary = False
                    params['PksPos5'].vary = False
                    params['PksPos6'].vary = False
                    params['cen'].min = fit_bounds[0][0]
                    params['cen'].max = fit_bounds[1][0]
                    params['amp1'].min = fit_bounds[0][1]
                    params['amp1'].max = fit_bounds[1][1]
                    params['amp2'].min = fit_bounds[0][2]
                    params['amp2'].max = fit_bounds[1][1]
                    params['amp3'].min = fit_bounds[0][3]
                    params['amp3'].max = fit_bounds[1][3]
                    params['amp4'].min = fit_bounds[0][4]
                    params['amp4'].max = fit_bounds[1][4]
                    params['amp5'].min = fit_bounds[0][5]
                    params['amp5'].max = fit_bounds[1][5]
                    params['amp6'].min = fit_bounds[0][6]
                    params['amp6'].max = fit_bounds[1][6]
                    params['sigma1'].min = fit_bounds[0][7]
                    params['sigma1'].max = fit_bounds[1][7]
                    params['sigma2'].min = fit_bounds[0][8]
                    params['sigma2'].max = fit_bounds[1][8]
                    params['sigma3'].min = fit_bounds[0][9]
                    params['sigma3'].max = fit_bounds[1][9]
                    params['sigma4'].min = fit_bounds[0][10]
                    params['sigma4'].max = fit_bounds[1][10]
                    params['sigma5'].min = fit_bounds[0][11]
                    params['sigma5'].max = fit_bounds[1][11]
                    params['sigma6'].min = fit_bounds[0][12]
                    params['sigma6'].max = fit_bounds[1][12]

                    result = fmodel.fit(TimeSeries[i, start_Fit2[0][0]:end_Fit2[0][0]], params, x=energy[start_Fit2[0][0]:end_Fit2[0][0]])

                    Parameter[i, 0] = result.params['cen'].value + PksPos[0]
                    Parameter[i, 1] = result.params['amp1'].value
                    Parameter[i, 2] = result.params['sigma1'].value
                    Parameter[i, 3] = result.params['cen'].value + PksPos[1]
                    Parameter[i, 4] = result.params['amp2'].value
                    Parameter[i, 5] = result.params['sigma2'].value
                    Parameter[i, 6] = result.params['cen'].value + PksPos[2]
                    Parameter[i, 7] = result.params['amp3'].value
                    Parameter[i, 8] = result.params['sigma3'].value
                    Parameter[i, 9] = result.params['cen'].value + PksPos[3]
                    Parameter[i, 10] = result.params['amp4'].value
                    Parameter[i, 11] = result.params['sigma4'].value
                    Parameter[i, 12] = result.params['cen'].value + PksPos[4]
                    Parameter[i, 13] = result.params['amp5'].value
                    Parameter[i, 14] = result.params['sigma5'].value
                    Parameter[i, 15] = result.params['cen'].value + PksPos[5]
                    Parameter[i, 16] = result.params['amp6'].value
                    Parameter[i, 17] = result.params['sigma6'].value

                    GaussFit[i, :, 0] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 0:3])
                    GaussFit[i, :, 1] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 3:6])
                    GaussFit[i, :, 2] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 6:9])
                    GaussFit[i, :, 3] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 9:12])
                    GaussFit[i, :, 4] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 12:15])
                    GaussFit[i, :, 5] = _1gaussian(energy[start_Fit2[0][0]:end_Fit2[0][0]], *Parameter[i, 15:18])

                    SumSpec[i, :] = GaussFit[i, :, 0] + GaussFit[i, :, 1] + GaussFit[i, :, 2] + GaussFit[i, :, 3] + GaussFit[i, :, 4] + GaussFit[i, :, 5]
                    del result
                except RuntimeError:
                    print('Error - deconvolution failed')
                    continue

    return SumSpec, Parameter, GaussFit


def signal_intensity(eelsSI, energy, startEnergy, endEnergy):
    start_Integral = np.where(energy == find_nearest(energy, startEnergy))
    end_Integral = np.where(energy == find_nearest(energy, endEnergy))
    if np.shape(eelsSI.shape)[0] >= 3:
        eelsIntensity = np.zeros([np.shape(eelsSI)[0], np.shape(eelsSI)[1]])
        for i in range(np.shape(eelsSI)[0]):
            for j in range(np.shape(eelsSI)[1]):
                eelsIntensity[i, j] = sum(eelsSI[i, j, start_Integral[0][0]:end_Integral[0][0]])[0]
    elif np.shape(eelsSI)[0] == 2:
        eelsIntensity = np.zeros([np.shape(eelsSI)[0], 1])
        for i in range(np.shape(eelsSI)[0]):
            eelsIntensity[i] = sum(eelsSI[i, start_Integral[0][0]:end_Integral[0][0]])[0]

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

def thickness_analysis(data, energy, E_resolution, E_plasmon):
    def gauss(x, a, x0, sigma):
        y = a*np.exp(-(x-x0)**2/(2*sigma**2))
        return y
    def lorentzian(x, x0, a, gam):
        return a * gam**2 / ( gam**2 + ( x - x0 )**2)
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    shift = 0 - energy[np.where(data[:] == np.max(data[:]))[0][0]]
    spec_aligned = np.zeros([np.shape(data[:])[0], 2])
    spec_aligned[:, 0] = energy + shift
    spec_aligned[:, 1] = data[:]
    
    popt, pcov = curve_fit(gauss, spec_aligned[:, 0], spec_aligned[:, 1], method='dogbox', bounds=(-0.1, [np.max(spec_aligned[:, 1]), 0.1, E_resolution]))
    plasmon_limit = np.where(spec_aligned[:, 0]==find_nearest(spec_aligned[:, 0], E_plasmon))[0][0]
    ZLP_limit = np.where(spec_aligned[:, 0]==find_nearest(spec_aligned[:, 0], 5))[0][0]
    Int_ZLP2 = sum(gauss(spec_aligned[0:np.where(spec_aligned[:, 0]==find_nearest(spec_aligned[:, 0], 5))[0][0], 0], *popt))
    Int_ZLP = sum(spec_aligned[0:ZLP_limit, 1])
    Int_Plasmon = sum(spec_aligned[0:plasmon_limit, 1])
    thickness = -np.log(Int_ZLP/Int_Plasmon)
    thickness2 = -np.log(Int_ZLP2/Int_Plasmon)
    
    return spec_aligned, thickness, thickness2
