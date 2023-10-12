#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# https://www.youtube.com/watch?v=Hp4-C4Incgpw&ab_channel=CurioRes

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from csvconverter import * 
from plotting import * 
from numpy import std
from features import * 
import numpy as np
# import the required modules
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

import statsmodels.api as sm
import pandas as pd
from scipy.fft import rfft, rfftfreq

from transformData import *

from discover import *


# This file uses makes plots using matplot lib. 
# The plots are made per feature to view how the pipeline steps (downsampling, lowpassfiltering, osciallating around zero, feature extraction) influence the (axis) data
# To view a plot, call the specific show function

def showButterWorth():

    # plt.show()
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    ax = mat[:,0]
    ay = mat[:,1]
    az = mat[:,2]
    gx = mat[:,3]
    gy = mat[:,4]
    gz = mat[:,5]

    # sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    sosl = signal.butter(5, 20, 'lp', fs=20, output='sos')
    sosh = signal.butter(5, 20, 'hp', fs=20, output='sos')
    filteredl = signal.sosfilt(sosl, ax)
    filteredh = signal.sosfilt(sosh, ax)

    filteredlax = signal.sosfilt(sosl, ax)
    filteredlay = signal.sosfilt(sosl, ay)
    filteredlaz = signal.sosfilt(sosl, az)

    meanx = np.mean(ax)
    meany = np.mean(ay)
    meanz = np.mean(az)

    meanax = ax - meanx
    meanay = ay - meany
    meanaz = az - meanz
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    sdaz = np.std(az)
    normalized_value = (meanaz) / sdaz
    print(sdaz)
    print(meanaz)


    plot_readings(filteredlaz, meanazfiltered, normalized_value, az, meanaz, normalized_value, timeSeconds=(8*60))

def showButterWorth2():
    mat = readCSVFile("./1106/20hz/data/1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/NB1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/data/2tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    duration = 3 * 60
    nrWindows = (duration // 30)
    windowSize = len(mat) // nrWindows
    print("windowsize: ", windowSize)
    
    # col = mat[:windowSize*6,2]
    col = mat[2*windowSize:3*windowSize, 4] #2
    data = mat[:, 4]

    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    transparency=0.8
    plt.subplot(8,1,1)
    plt.title('')
    plt.plot(data,label='downsample 20hz',alpha=transparency)
    plt.legend()

    transparency=0.8
    plt.subplot(8,1,2)
    plt.title('raw col')
    plt.plot(col,label='raw partly',alpha=transparency)
    plt.legend()

    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.subplot(8,1,3)
    plt.plot(meanazfiltered,label='20-1000',alpha=transparency)
    plt.legend()

    plt.subplot(8,1,4)
    sosl = signal.butter(5, 9, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.plot(meanazfiltered,label='9-20',alpha=transparency)
    plt.legend()

    plt.subplot(8,1,5)
    sosl = signal.butter(5, 5, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.plot(meanazfiltered,label='5-20',alpha=transparency)
    plt.legend()

    plt.subplot(8,1,6)
    sosl = signal.butter(5, 2, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.plot(meanazfiltered,label='2-20',alpha=transparency)
    plt.legend()

    plt.subplot(8,1,7)
    sosl = signal.butter(5, 3, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.plot(meanazfiltered,label='3-20',alpha=transparency)
    plt.legend()

    plt.subplot(8,1,8)
    sosl = signal.butter(5, 4, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    plt.plot(meanazfiltered,label='butterworth',alpha=transparency)
    plt.legend()

    plt.show()


def showNormalization():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    col = mat[:,2]
    mean = np.mean(col)
    res = col - mean

    transparency = 0.8
    plt.plot(res,label='normalized',alpha=transparency)
    plt.plot(col,label='raw',alpha=transparency)
    plt.legend()
    plt.title('Acceleration col 2')
    plt.show()

def showZeroCrossings():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    col = mat[:,2]
    mean = np.mean(col)
    res = col - mean

    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)

    meanz = np.mean(col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    zerocrossings = getZeroCrossings(meanazfiltered) #res)
   

    transparency = 0.8
    print(zerocrossings)
    plt.plot(zerocrossings, meanazfiltered[zerocrossings], "x")
    plt.plot(res,label='normalized',alpha=transparency)
    plt.plot(col,label='raw',alpha=transparency)
    plt.legend()
    plt.title('Acceleration col 2')
    plt.show()

def showVarZeroCrossings():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    duration = 8*60
    varianceZC = varZeroCrossings(meanazfiltered, duration)
    print("=> VAR ZERO-CROSS: ", varianceZC)
    print(np.var([1, 2, 3, 4, 5]))

def showAutoCorrelation():
    #mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./0206/1tosti0206.txt", (0, 1, 2, 3, 4, 5))
    mat = readCSVFile("./1106/20hz/data/1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/NB1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/data/2tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    duration = 3 * 60
    nrWindows = (duration // 30)
    windowSize = len(mat) // nrWindows
    print("windowsize: ", windowSize)
    
    # col = mat[:windowSize*6,2]
    col = mat[2*windowSize:3*windowSize, 4] #2
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    # data = [3, 16, 156, 47, 246, 176, 233, 140, 130, 
    #     101, 166, 201, 200, 116, 118, 247, 
    #     209, 52, 153, 232, 128, 27, 192, 16, 208, 
    #     187, 228, 86, 30, 151, 18, 254, 
    #     76, 112, 67, 244, 179, 150, 89, 49, 83, 147, 90, 
    #     33, 6, 158, 80, 35, 186, 127]
    
    # x = [22, 24, 25, 25, 28, 29, 34, 37, 40, 44, 51, 48, 47, 50, 51]

    # # Read in the data
    # data = pd.read_csv('AirPassengers.csv')
    # data = data['#Passengers']
    # data = col
    data = meanazfiltered

    # data = x

    print(len(data))
    print(data)

    lags = range(10)
    acNP = getAutocorrelationNP(data)
    acFFT = getAutocorrelationFFT(data)
    acPython = getAutocorrelationPython(data, lags)
    acStats = getAutocorrelationStats(data, lags)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    transparency = 0.8


    print(acorr)

    #print(zerocrossings)
    #plt.plot(zerocrossings, meanazfiltered[zerocrossings], "x")
    # plt.subplot(6,1,1)
    # plt.title('Acceleration col 2')
    # plt.plot(np.linspace(0, len(acStats), len(acNP)), acNP,label='acNP',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acFFT)), acFFT,label='acFFT',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acPython)), acPython,label='acPython',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acorr)), acorr,label='acorr',alpha=transparency)
    # plt.plot(acStats,label='acStats',alpha=transparency)
    # plt.legend()
    

    # lags = range(50)
    # acNP = getAutocorrelationNP(data)
    # acFFT = getAutocorrelationFFT(data)
    # acPython = getAutocorrelationPython(data, lags)
    # acStats = getAutocorrelationStats(data, lags)
    # acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    # plt.subplot(8,1,2)
    # plt.title('Acceleration col 2 lags diff')
    # plt.plot(np.linspace(0, len(acStats), len(acNP)), acNP,label='acNP',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acFFT)), acFFT,label='acFFT',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acPython)), acPython,label='acNPython',alpha=transparency)
    # plt.plot(np.linspace(0, len(acStats), len(acorr)), acorr,label='acorr',alpha=transparency)
    # plt.plot(acStats,label='acStats',alpha=transparency)
    # plt.legend()
    
    plt.subplot(7,1,1)
    plt.title('raw')
    plt.plot(col,label='raw',alpha=transparency)
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")
    plt.legend()

    plt.subplot(7,1,2)
    plt.title('raw buttered')
    plt.plot(data,label='raw',alpha=transparency)
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")
    plt.legend()

    ## --- 
    plt.subplot(7,1,3)
    lags = range(10)
    acorr2 = sm.tsa.acf(data, nlags = len(lags)-1)
    peaks = getPeakIndx(acorr2)
    plt.plot(peaks, acorr2[peaks], "x")
    plt.plot(acorr2,label='10',alpha=transparency)
    plt.legend

    
    plt.subplot(7,1,4)
    lags = range(50)
    acorr3 = sm.tsa.acf(data, nlags = len(lags)-1)
    peaks = getPeakIndx(acorr3)
    plt.plot(peaks, acorr3[peaks], "x")
    plt.plot(acorr3,label='50',alpha=transparency)
    plt.legend()

    # data = mat[:, 4]
    plt.subplot(7,1,5)
    lags = range(100)
    acorr4 = sm.tsa.acf(data, nlags = len(lags)-1)
    peaks = getPeakIndx(acorr4)
    plt.plot(peaks, acorr4[peaks], "x")
    plt.plot(acorr4,label='100',alpha=transparency)
    plt.legend()

    plt.subplot(7,1,6)
    lags = range(250)
    acorr5 = sm.tsa.acf(data, nlags = len(lags)-1)
    plt.plot(acorr5,label='250',alpha=transparency)
    peaks = getPeakIndx(acorr5)
    plt.plot(peaks, acorr5[peaks], "x")
    plt.legend()

    plt.subplot(7,1,7)
    lags = range(500)
    acorr5 = sm.tsa.acf(data, nlags = len(lags)-1)
    plt.plot(acorr5,label='500',alpha=transparency)
    peaks = getPeakIndx(acorr5)
    plt.plot(peaks, acorr5[peaks], "x")
    plt.legend()

    ## --- USE PLOT_ACF STANDARD FUNCTION
    # plt.subplot(4,1,4)
    # # Plot autocorrelation
    # plt.rc("figure", figsize=(11,5))
    # plot_acf(data, lags=48)
    # plt.ylim(0,1)
    # plt.xlabel('Lags', fontsize=18)
    # plt.ylabel('Correlation', fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.title('Autocorrelation Plot', fontsize=20)
    # plt.tight_layout()
    # plt.show()

    plt.show()
    #https://scicoding.com/4-ways-of-calculating-autocorrelation-in-python/
    #https://www.alpharithms.com/autocorrelation-time-series-python-432909/

def showPeaks():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered

    # data = pd.read_csv('AirPassengers.csv')
    # data = data['#Passengers']

    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)
    
    transparency = 0.8
    plt.subplot(3,1,1)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")
    plt.legend()


    plt.subplot(3,1,2)
    plt.title('')
    plt.plot(data,label='normalized data',alpha=transparency)
    peaks = getPeakIndx(data)
    plt.plot(peaks, data[peaks], "x")
    plt.legend()
    
    data = col

    plt.subplot(3,1,3)
    plt.title('')
    plt.plot(data,label='data',alpha=transparency)
    peaks = getPeakIndx(data)
    plt.plot(peaks, data[peaks], "x")
    plt.legend()
    plt.show()

def showMaxPeak():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered

    data = pd.read_csv('AirPassengers.csv')
    data = data['#Passengers']

    lags = range(50)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    
    transparency = 0.8
    plt.subplot(3,1,1)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    peaks = getPeakIndx(acorr)
    maxPeak = np.max(acorr[peaks])
    print(maxPeak)
    print(acorr[peaks])

    plt.axhline(y=maxPeak, color='r', linestyle='--', label='Horizontal Line')
    plt.legend()
    plt.show()

def showProminentPeaks():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered

    # data = pd.read_csv('AirPassengers.csv')
    # data = data['#Passengers']

    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)
    

    transparency = 0.8
    plt.subplot(3,1,1)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    #prominences = peak_prominences(data, peaks)[0] #need a threshold of 0.25??
    threshold = 0.15
    prompeaks, _ = find_peaks(acorr, prominence=threshold)
    #nrProminentPeaks = getNrProminentPeaks(prominences, 0.25)
    plt.plot(prompeaks, acorr[prompeaks], ".")
    plt.legend()
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")

    plt.subplot(3,1,2)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    prominences = peak_prominences(data, peaks)[0] #need a threshold of 0.25??
    threshold = 0.15
    prompeaks = np.where(prominences > 5)[0] #find_peaks(prominences, prominence=threshold)
    #nrProminentPeaks = getNrProminentPeaks(prominences, 0.25)
    plt.plot(prompeaks, acorr[prompeaks], ".")
    print(prompeaks)
    print(prominences)
    plt.legend()
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")




    # Find peaks in the signal 
    peaks, _ = find_peaks(acorr)

    #Iterate over the peaks and count the ones that satisfy the condition
    count = 0
    threshold = 0.0001
    thresholdedPeakIndx = []
    for peak in peaks:
        left_neighbor = acorr[peak - 1]
        right_neighbor = acorr[peak + 1]
        peak_value = acorr[peak]
        print(acorr[peak])
        
        if (peak_value > left_neighbor + threshold) and (peak_value > right_neighbor + threshold):
            count += 1
            thresholdedPeakIndx.append(peak)
    
    transparency = 0.8
    plt.subplot(3,1,3)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    peaks = getPeakIndx(acorr)
    plt.plot(thresholdedPeakIndx, acorr[thresholdedPeakIndx], ".")

    plt.plot(peaks, acorr[peaks], "x")
    plt.legend()
    print(prominences)
    print(thresholdedPeakIndx)


    

    plt.show()

def showWeakPeaks():
    mat = readCSVFile("./1106/20hz/data/1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[600:1200:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered

    # data = pd.read_csv('AirPassengers.csv')
    # data = data['#Passengers']

    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)
    
    transparency = 0.8
    plt.subplot(2,1,1)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    #prominences = peak_prominences(data, peaks)[0] #need a threshold of 0.25??
    threshold = 0.01
    prompeaks, _ = find_peaks(acorr, prominence=threshold)
    #nrProminentPeaks = getNrProminentPeaks(prominences, 0.25)
    plt.plot(prompeaks, acorr[prompeaks], ".")
    plt.legend()
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")
    
    print(peaks)
        # Find peaks in the signal 
    peaks, _ = find_peaks(acorr)
    
    print(peaks)
    print(acorr[peaks])

    #Iterate over the peaks and count the ones that satisfy the condition
    count = 0
    threshold = 0.0005
    thresholdedPeakIndxSmaller = []
    thresholdedPeakIndxBigger = []
    for peakIndex in range(len(peaks)):
        left_neighbor = None
        if peakIndex - 1 >= 0:
            left_neighbor = acorr[ peaks[peakIndex -1] ]
        right_neighbor = None
        if peakIndex + 1 < len(peaks):
            right_neighbor = acorr[peaks[peakIndex +1]]
        peak_value = acorr[peaks[peakIndex]]
        
        if (left_neighbor is not None and right_neighbor is not None):
            if (peak_value < left_neighbor + threshold) and (peak_value < right_neighbor + threshold):
                count += 1
                thresholdedPeakIndxSmaller.append(peaks[peakIndex])
            
            elif (peak_value > left_neighbor + threshold) and (peak_value > right_neighbor + threshold):
                count += 1
                print((peak_value > left_neighbor + threshold), peak_value, " ",  left_neighbor + threshold, " ",right_neighbor + threshold  )
                print((peak_value > right_neighbor + threshold))
                print("\n")
                thresholdedPeakIndxBigger.append(peaks[peakIndex])
    
    transparency = 0.8
    plt.subplot(2,1,2)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    peaks = getPeakIndx(acorr)
    plt.axhline(y = 0.7603950422625558, color = 'r', linestyle = '-')
    plt.axhline(y =  0.53218983, color = 'b', linestyle = '-')
    plt.axhline(y = 0.3370236, color = 'g', linestyle = '-')
    plt.plot(thresholdedPeakIndxSmaller, acorr[thresholdedPeakIndxSmaller], ".")
    plt.plot(thresholdedPeakIndxBigger, acorr[thresholdedPeakIndxBigger], "x")

    # plt.plot(peaks, acorr[peaks], "x")
    plt.legend()
    # print(thresholdedPeakIndx)
    print(len(thresholdedPeakIndxSmaller))
    print(len(thresholdedPeakIndxBigger))
    print(len(peaks))
    print((acorr[peaks]))
    


    plt.show()

def showFirstPeakAfterZC():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered
    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    firstZCIndex = getFirstZeroCrossing(acorr)
    peaks = getPeakIndx(acorr)
    peakIndex = firstIndexAfterIndex(peaks, firstZCIndex)
    print(firstZCIndex)
    print(peakIndex)

    transparency = 0.8
    plt.subplot(1,1,1)
    plt.title('col 2')
    plt.plot(acorr,label='acorr',alpha=transparency)
    plt.legend()
    peaks = getPeakIndx(acorr)
    plt.plot(peaks, acorr[peaks], "x")
    plt.axvline(x=firstZCIndex, color='r', linestyle='--', label='Horizontal Line')
    plt.axhline(y=acorr[peakIndex], color='g', linestyle='--', label='Horizontal Line')
    plt.show()

def showRMS():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered
    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    rms1 = RMSarr(acorr)
    rms2 = RMS(acorr)
    print(rms1)
    print(rms2)

    exampleData = [1, 2, 3, 4]
    print("example data 1 (2.74): ", RMSarr(exampleData))
    print("example data 2: ", RMS(exampleData))

def showPeakFrequency_PeakPower():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2] 
    #col = mat[:(len(mat) * 30)//(8*60):,2]
    # print(len(col))
    # print(len(mat))
    # print((len(mat) * 30)/(8*60))
    # print((len(mat) * 30)//(8*60))
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanaz = col - np.mean(col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered#col #meanazfiltered
    normalized_tone = data 
    # Number of samples in normalized_tone
    N = len(data) #SAMPLE_RATE * DURATION
    SAMPLE_RATE = N / (8*60)#samples/duration 
    ## ----- FFT with gathered column data
    # Note the extra 'r' at the front
    yf = rfft(normalized_tone)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    plt.subplot(4,1,1)
    plt.title("FFT")
    plt.plot(xf, np.abs(yf))

    # print dominant frequency
    peak_coefficient = np.argmax(np.abs(yf))
    freqs = np.fft.fftfreq(len(yf))
    peak_freq = freqs[peak_coefficient]
    
    print("peak power col : ",np.max(np.abs(yf)))
    print("peak freuquency col: ", abs(peak_freq * SAMPLE_RATE) )

    ## --- Example: https://realpython.com/python-scipy-fft/
    SAMPLE_RATE = 44100  # Hertz
    DURATION = 5  # Seconds

    def generate_sine_wave(freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return x, y

    # Generate a 2 hertz sine wave that lasts for 5 seconds
    x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

    _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
    noise_tone = noise_tone * 0.3

    mixed_tone = nice_tone + noise_tone
    normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)
    N = SAMPLE_RATE * DURATION

    yf = fft(normalized_tone)
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    plt.subplot(4,1,2)
    plt.plot(xf, np.abs(yf))

    #print dominant frequency
    peak_coefficient = np.argmax(np.abs(yf))
    freqs = np.fft.fftfreq(len(yf))
    peak_freq = freqs[peak_coefficient]
    
    print("peak power example : ",np.max(np.abs(yf)))
    print("peak freuquency example: ", abs(peak_freq * SAMPLE_RATE) )

    ## --- Regular meanazfiltered
    plt.subplot(4,1,3)
    transparency = 0.8
    plt.title('col 2')
    plt.plot(meanazfiltered,label='meanaxfiltered',alpha=transparency)
    plt.legend()

    ## --- Raw
    plt.subplot(4,1,4)
    transparency = 0.8
    plt.title('col 2')
    # plt.plot(col,label='raw',alpha=transparency)
    plt.plot(normalized_tone,label='raw',alpha=transparency)
    plt.legend()

    plt.show()

    ## --- dominant frequency
 
def showVariance():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered
    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    rms1 = RMSarr(acorr)
    var1 = rms1 * rms1
    var2 = variance(acorr)
    print(var1)
    print(var2)

    exampleData = [1, 2, 3, 4]
    print("example data 1 variance function: ", variance(exampleData))
    print("example data 2 rms^2: ", RMS(exampleData)*RMS(exampleData) )
    print("ask about what is written in the paper? aboit variace being the square of rms")

def showSd():
    mat = readCSVFile("./2305/1bread2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered
    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    sd = np.std(acorr)
    print("standard deviation: ", sd)
    print("standard deviation example (1.118): ", np.std([1, 2, 3, 4]))

def showMean():
    mat = readCSVFile("./2305/1movie2305_20hz.txt", (0, 1, 2, 3, 4, 5))
    #mat = readCSVFile("./0206/1drinking0206.txt", (1, 2, 3, 4, 5, 6))
    col = mat[:,2]
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)

    data = meanazfiltered
    lags = range(50000)
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)

    mean = np.mean(acorr)
    print("smean: ", mean)
    print("standard deviation example (2.5): ", np.mean([1, 2, 3, 4]))

# ------

def showAutoCorrelation(mat, title):
    lags = range(250)
    # lags = range(10)
    # lags = range(500)
    # acorr = sm.tsa.acf(axisData, nlags = len(lags)-1)
                       
    transparency=0.8
    plt.subplot(7,1,1)
    plt.title(title)
    plt.plot(mat[:,0],label='raw',alpha=transparency)
    plt.plot(mat[:,1],label='raw',alpha=transparency)
    plt.plot(mat[:,2],label='raw',alpha=transparency)
    plt.plot(mat[:,3],label='raw',alpha=transparency)
    plt.plot(mat[:,4],label='raw',alpha=transparency)
    plt.plot(mat[:,5],label='raw',alpha=transparency)
    plt.legend()

    ax = mat[:,0]
    ay = mat[:,1]
    az = mat[:,2]
    gx = mat[:,3]
    gy = mat[:,4]
    gz = mat[:,5]
    acorrax = sm.tsa.acf(ax, nlags = len(lags)-1)
    acorray = sm.tsa.acf(ay, nlags = len(lags)-1)
    acorraz = sm.tsa.acf(az, nlags = len(lags)-1)
    acorrgx = sm.tsa.acf(gx, nlags = len(lags)-1)
    acorrgy = sm.tsa.acf(gy, nlags = len(lags)-1)
    acorrgz = sm.tsa.acf(gz, nlags = len(lags)-1)

    acpeaks = getPeakIndx(acorrax)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorray)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorraz)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgx)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgy)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgz)
    print(len(acpeaks))

    plt.subplot(7,1,2)
    plt.plot(acorrax,label='ax',alpha=transparency)
    peaks = getPeakIndx(acorrax)
    plt.plot(peaks, acorrax[peaks], "x")
    plt.legend()

    plt.subplot(7,1,3)
    plt.plot(acorray,label='ay',alpha=transparency)
    peaks = getPeakIndx(acorray)
    plt.plot(peaks, acorray[peaks], "x")
    plt.legend()

    plt.subplot(7,1,4)
    plt.plot(acorraz,label='az',alpha=transparency)
    peaks = getPeakIndx(acorraz)
    plt.plot(peaks, acorraz[peaks], "x")
    plt.legend()

    plt.subplot(7,1,5)
    plt.plot(acorrgx,label='gx',alpha=transparency)
    peaks = getPeakIndx(acorrgx)
    plt.plot(peaks, acorrgx[peaks], "x")
    plt.legend()

    plt.subplot(7,1,6)
    plt.plot(acorrgy,label='gy',alpha=transparency)
    peaks = getPeakIndx(acorrgy)
    plt.plot(peaks, acorrgy[peaks], "x")
    plt.legend()

    plt.subplot(7,1,7)
    plt.plot(acorrgz,label='gz',alpha=transparency)
    peaks = getPeakIndx(acorrgz)
    plt.plot(peaks, acorrgz[peaks], "x")
    plt.legend()

    plt.show()

def showWeak_MaxACPeaks(mat, title):
    lags = range(250)
    threshold = 0.15
    
    # lags = range(10)
    # lags = range(500)
    # acorr = sm.tsa.acf(axisData, nlags = len(lags)-1)
                       
    transparency=0.8
    plt.subplot(7,1,1)
    plt.title(title)
    plt.plot(mat[:,0],label='raw',alpha=transparency)
    plt.plot(mat[:,1],label='raw',alpha=transparency)
    plt.plot(mat[:,2],label='raw',alpha=transparency)
    plt.plot(mat[:,3],label='raw',alpha=transparency)
    plt.plot(mat[:,4],label='raw',alpha=transparency)
    plt.plot(mat[:,5],label='raw',alpha=transparency)
    plt.legend()

    ax = mat[:,0]
    ay = mat[:,1]
    az = mat[:,2]
    gx = mat[:,3]
    gy = mat[:,4]
    gz = mat[:,5]
    acorrax = sm.tsa.acf(ax, nlags = len(lags)-1)
    acorray = sm.tsa.acf(ay, nlags = len(lags)-1)
    acorraz = sm.tsa.acf(az, nlags = len(lags)-1)
    acorrgx = sm.tsa.acf(gx, nlags = len(lags)-1)
    acorrgy = sm.tsa.acf(gy, nlags = len(lags)-1)
    acorrgz = sm.tsa.acf(gz, nlags = len(lags)-1)

    acpeaks = getPeakIndx(acorrax)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorray)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorraz)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgx)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgy)
    print(len(acpeaks))
    acpeaks = getPeakIndx(acorrgz)
    print(len(acpeaks))


    plt.subplot(7,1,2)
    plt.plot(acorrax,label='ax',alpha=transparency)
    peaks = getPeakIndx(acorrax)
    plt.plot(peaks, acorrax[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks, acorrax, threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks, acorrax,threshold)

    plt.plot(thresholdedPeakIndxBigger, acorrax[thresholdedPeakIndxBigger], "+", color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorrax[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.subplot(7,1,3)
    plt.plot(acorray,label='ay',alpha=transparency)
    peaks = getPeakIndx(acorray)
    print(acorray[peaks])
    plt.plot(peaks, acorray[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks,acorray, threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks,acorray, threshold)
    plt.plot(thresholdedPeakIndxBigger, acorray[thresholdedPeakIndxBigger], "+", color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorray[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.subplot(7,1,4)
    plt.plot(acorraz,label='az',alpha=transparency)
    peaks = getPeakIndx(acorraz)
    plt.plot(peaks, acorraz[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks,acorraz, threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks,acorraz, threshold)
    plt.plot(thresholdedPeakIndxBigger, acorraz[thresholdedPeakIndxBigger], "+", color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorraz[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.subplot(7,1,5)
    plt.plot(acorrgx,label='gx',alpha=transparency)
    peaks = getPeakIndx(acorrgx)
    plt.plot(peaks, acorrgx[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks, acorrgx,threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks, acorrgx,threshold)
    plt.plot(thresholdedPeakIndxBigger, acorrgx[thresholdedPeakIndxBigger], "+", color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorrgx[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.subplot(7,1,6)
    plt.plot(acorrgy,label='gy',alpha=transparency)
    peaks = getPeakIndx(acorrgy)
    plt.plot(peaks, acorrgy[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks, acorrgy,threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks, acorrgy,threshold)
    plt.plot(thresholdedPeakIndxBigger, acorrgy[thresholdedPeakIndxBigger], "+", color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorrgy[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.subplot(7,1,7)
    plt.plot(acorrgz,label='gz',alpha=transparency)
    peaks = getPeakIndx(acorrgz)
    plt.plot(peaks, acorrgz[peaks], "x")
    thresholdedPeakIndxSmaller = getWeakPeaks(peaks, acorrgz,threshold)
    thresholdedPeakIndxBigger = getPromPeaks(peaks, acorrgz,threshold)
    plt.plot(thresholdedPeakIndxBigger, acorrgz[thresholdedPeakIndxBigger], "+",  color='g')
    plt.plot(thresholdedPeakIndxSmaller, acorrgz[thresholdedPeakIndxSmaller], ".", color='r')
    plt.legend()

    plt.show()


def showButterWorth3Paper():
    # mat = readCSVFile("./1106/20hz/data/1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/NB1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/data/2tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    mat = readCSVFile("./1106/20hz/data/1cookies1106_20hz.txt", range(0, 6))

    duration = 3 * 60
    nrWindows = (duration // 30)
    windowSize = len(mat) // nrWindows
    print("windowsize: ", windowSize)
    
    # col = mat[:windowSize*6,2]
    # col = mat[2*windowSize:3*windowSize, 4] #2
    col = mat[1200:1800,3]
    data = mat[:, 2]

    transparency=0.8
    plt.subplot(1,1,1)
    plt.title('1 - downsampled to 20Hz')
    plt.plot(col,label='',alpha=transparency)
    plt.xlabel('window samples')
    # plt.legend()

    sosl = signal.butter(5, 4, 'lp', fs=20, output='sos')
    filteredlaz = signal.sosfilt(sosl, col)
    meanazfiltered = filteredlaz - np.mean(filteredlaz)
    # plt.subplot(4,1,2)
    # plt.title('2 - low - pass filter and oscillate around 0')
    # plt.plot(meanazfiltered,label='',alpha=transparency)
    # plt.legend()

    # plt.subplot(4,1,3)
    # lags = range(250)
    # acorr = sm.tsa.acf(meanazfiltered, nlags = len(lags)-1)
    # plt.title("3 - autocorrelation (250 lag)")
    # plt.plot(acorr,label='',alpha=transparency)
    # plt.legend()

    # plt.subplot(4,1,4)
    # peaks = getPeakIndx(acorr)
    # plt.plot(peaks, acorr[peaks], "x")
    # plt.title("4 - Show peaks after autocorrelation (250 lag)")
    # plt.plot(acorr,label='',alpha=transparency)
    # plt.legend()

    
   
    # plt.subplots_adjust(
    #                 bottom=0.1,
    #                 )
    plt.show()




showButterWorth3Paper()