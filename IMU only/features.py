import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyaC
from statistics import variance 
from fractions import Fraction as fr
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from scipy.stats import entropy
import scipy.signal
import statsmodels.api as sm
from scipy.signal import find_peaks, peak_prominences
from scipy.fft import fft, fftfreq
from csvconverter import readCSVFile,makeCSVFile
from scipy import signal


# a - GET ZEROCROSSING
def getZeroCrossings(data) :
    return np.where(np.diff(np.sign(data)))[0]

def nrZeroCrossings(data) :
    zero_crossings = getZeroCrossings()
    return len(zero_crossings)

# b - GET VARIANCE OF ZEROCROSSING
def varZeroCrossings(data, timeframe, zc=None):
    zerocrossings = []
    if zc is None:
        zerocrossings= getZeroCrossings(data)
    else:
        zerocrossings = zc

    if len(zerocrossings) < 3:
        return 0
    ratio = zc / len(data)
    timestamps = ratio * timeframe
    deltas = np.diff(timestamps)
    return variance(deltas) #gives "/n-1", needed for sample mean, np.var(x) give divide by n


# c - GET PEAK FREQUENCY with frequency transformation? dominant frequency

# d - NUMBER OF AUTO-CORRELATION PEAKS
# https://scicoding.com/4-ways-of-calculating-autocorrelation-in-python/
# https://stackoverflow.com/questions/54919679/get-peaks-in-plot-python

def getPeakIndx(data):
    return scipy.signal.find_peaks(data)[0]
    
def nrPeaks(data):
    peaks = scipy.signal.find_peaks(data)[0]#find_peaks_cwt(data, widths=np.ones(data.shape)*2)-1
    return len(peaks)

# e - prominent peaks
def getNrProminentPeaks(data, threshold):
    peaks, _ = find_peaks(data, prominence=threshold)
    return len(peaks)


# a - ROOT MEAN SQUARE
def RMSarr(arr):
    n = len(arr)
    square = 0
    mean = 0.0
    root = 0.0
    #Calculate square
    for i in range(0,n):
        square += (arr[i]*arr[i])
     
    #Calculate Mean
    mean = (square / (float)(n))
     
    #Calculate Root
    root = math.sqrt(mean)
    return root

def RMS(arr): 
    return np.sqrt(np.mean(np.power(arr, 2)))


#   CHOSSE APPROPIATE LAG
#https://scicoding.com/4-ways-of-calculating-autocorrelation-in-python/
def getAutocorrelationFFT(data):
# Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2*len(data) - 1)).astype('int')

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - np.mean(data)

    # Compute the FFT
    fft = np.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real  / var / len(data)

    return acorr

def getAutocorrelationNP(data):
# Nearest size with power of 2

    # Mean
    mean = np.mean(data)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - mean

    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
    acorr = acorr / var / len(ndata)

    return acorr

def getAutocorrelationStats(data, lags):
    acorr = sm.tsa.acf(data, nlags = len(lags)-1)
    return acorr
    
def getAutocorrelationPython(data, lags):
    # Pre-allocate autocorrelation table
    acorr = len(lags) * [0]

    # Mean
    mean = sum(data) / len(data) 

    # Variance
    var = sum([(x - mean)**2 for x in data]) / len(data) 

    # Normalized data
    ndata = [x - mean for x in data]


    # Go through lag components one-by-one
    for l in lags:
        c = 1 # Self correlation
        
        if (l > 0):
            tmp = [ndata[l:][i] * ndata[:-l][i] 
                for i in range(len(data) - l)]
            
            c = sum(tmp) / len(data) / var
            
        acorr[l] = c
    return acorr


def getFirstZeroCrossing(data):
    zcs = getZeroCrossings(data)
    if(len(zcs) != 0): 
        return zcs[0]
    else:
        return -1

def firstIndexAfterIndex(peakIndx, zcIndex):
    if zcIndex < 0: 
        return -1
    for i in peakIndx:
        if i > zcIndex:
            return i
    return -1



def getWeakPeaks(peaks, axis,  threshold):
    thresholdedPeakIndxSmaller = []
    # print(axis[peaks])
    for peakIndex in range(len(peaks)):
        left_neighbor = None
        if peakIndex - 1 >= 0:
            left_neighbor = axis[ peaks[peakIndex -1] ]
        right_neighbor = None
        if peakIndex + 1 < len(peaks):
            right_neighbor = axis[peaks[peakIndex +1]]
        peak_value = axis[peaks[peakIndex]]
        
        if (left_neighbor is not None and right_neighbor is not None):
            if (peak_value < left_neighbor - threshold) and (peak_value < right_neighbor - threshold):
                thresholdedPeakIndxSmaller.append(peaks[peakIndex])
        elif (left_neighbor is not None):
            if (peak_value < left_neighbor - threshold):
                thresholdedPeakIndxSmaller.append(peaks[peakIndex])
        elif (right_neighbor is not None):
            if (peak_value < right_neighbor - threshold):
                thresholdedPeakIndxSmaller.append(peaks[peakIndex])
            
    return thresholdedPeakIndxSmaller

def getNumWeakPeaks(peaks, axis,  threshold):
    # thresholdedPeakIndxSmaller = []
    count = 0
    for peakIndex in range(len(peaks)):
        left_neighbor = None
        if peakIndex - 1 >= 0:
            left_neighbor = axis[ peaks[peakIndex -1] ]
        right_neighbor = None
        if peakIndex + 1 < len(peaks):
            right_neighbor = axis[peaks[peakIndex +1]]
        peak_value = axis[peaks[peakIndex]]
        
        if (left_neighbor is not None and right_neighbor is not None):
            if (peak_value < left_neighbor - threshold) and (peak_value < right_neighbor - threshold):
                count += 1
                # thresholdedPeakIndxSmaller.append(peaks[peakIndex])
        elif (left_neighbor is not None):
            if (peak_value < left_neighbor - threshold):
                count += 1
                # thresholdedPeakIndxSmaller.append(peaks[peakIndex])
        elif (right_neighbor is not None):
            if (peak_value < right_neighbor - threshold):
                count += 1
                # thresholdedPeakIndxSmaller.append(peaks[peakIndex])
            
    return count

def getPromPeaks(peaks, axis, threshold):
    thresholdedPeakIndxBigger = []
    # print(axis[peaks])
    count = 0
    for peakIndex in range(len(peaks)):
        left_neighbor = None
        if peakIndex - 1 >= 0:
            left_neighbor = axis[ peaks[peakIndex -1] ]
        right_neighbor = None
        if peakIndex + 1 < len(peaks):
            right_neighbor = axis[peaks[peakIndex +1]]
        peak_value = axis[peaks[peakIndex]]
        
        if (left_neighbor is not None and right_neighbor is not None):
            if (peak_value > left_neighbor + threshold) and (peak_value > right_neighbor + threshold):
                count += 1
                thresholdedPeakIndxBigger.append(peaks[peakIndex])
        elif (left_neighbor is not None):
            if (peak_value > left_neighbor + threshold):
                count += 1
                thresholdedPeakIndxBigger.append(peaks[peakIndex])
        elif (right_neighbor is not None):
            if (peak_value > right_neighbor + threshold):
                count += 1
                thresholdedPeakIndxBigger.append(peaks[peakIndex])

    return thresholdedPeakIndxBigger

def getNumPromPeaks(peaks, axis, threshold):
    # thresholdedPeakIndxBigger = []
    count = 0
    for peakIndex in range(len(peaks)):
        left_neighbor = None
        if peakIndex - 1 >= 0:
            left_neighbor = axis[ peaks[peakIndex -1] ]
        right_neighbor = None
        if peakIndex + 1 < len(peaks):
            right_neighbor = axis[peaks[peakIndex +1]]
        peak_value = axis[peaks[peakIndex]]
        
        if (left_neighbor is not None and right_neighbor is not None):
            if (peak_value > left_neighbor + threshold) and (peak_value > right_neighbor + threshold):
                count += 1
                # thresholdedPeakIndxBigger.append(peaks[peakIndex])        
        elif (left_neighbor is not None):
            if (peak_value > left_neighbor + threshold):
                count += 1
                # thresholdedPeakIndxBigger.append(peaks[peakIndex])
        elif (right_neighbor is not None):
            if (peak_value > right_neighbor + threshold):
                count += 1
                # thresholdedPeakIndxBigger.append(peaks[peakIndex])
    return count




def normalize(col):
    mean = np.mean(col)
    return col - mean