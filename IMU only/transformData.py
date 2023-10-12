from csvconverter import readCSVFile
import numpy as np
from features import *
from scipy.stats import entropy
from scipy.signal import find_peaks, peak_prominences
from scipy.fft import rfft, rfftfreq
from statistics import mode


#0
MEAN = "mean"
#1
SD = "sd"
#2
VAR = "var"
#3
RANDMEANSQR = "randommeansquare"
#4
PEAKPOWER = "peakpower"
#5
NRZC = "numberzerocrossings"
#6
VARZC = "variancezerocrossings"
#7
PEAKFREQ = "peakfrequency"
#8
NRAUTOCORPEAKS = "numeberautocorrelationpeaks"
#9
WEAKPEAKS = "weakpeaksautocorrelation"
#10
PROMINENTPEAKS = "prominentpeaksautocorrelation"
#11
MAXACVALUE = "maximumautocorrelationvalue"
#12
FIRSTPEAKZC = "firstpeakafterzerocrossingautocorrelation"


# features = [MEAN, SD, VAR, RANDMEANSQR, PEAKPOWER, NRZC, VARZC, PEAKFREQ, NRAUTOCORPEAKS, WEAKPEAKS, PROMINENTPEAKS, MAXACVALUE, FIRSTPEAKZC]
# print(len(features))
#12*6 = features for each axis

def getFeatures(window, windowDur, featureIndx):
    res = []
    featMat = [[], [], [], [], [], []]
    for index in featureIndx:
        mod = (index ) % 13 # feature
        div = (index ) // 13 # axis
        # print(mod, " " , div)
        featMat[div].append(mod)

    for axisIndex in range(0, 6):
        # print("axisIndex ", axisIndex)
        featAxis = featMat[axisIndex]
        axisData = window[:, axisIndex]

        # prin t(featAxis)

        if 0 in featAxis : # - mean
            mean = np.mean(axisData)
            res.append(mean)
        if 1 in featAxis : # - standard deviation 
            sd = np.std(axisData)
            res.append(sd)
        if 2 in featAxis : # - variance
            var = variance(axisData)
            res.append(var)
        if 3 in featAxis : # - random mean square
            rms = RMS(axisData)
            res.append(rms)
        # ZEROCROSSINGS
        if 5 in featAxis or 6 in featAxis :
            zero_crossings = getZeroCrossings(axisData)
            # print(zero_crossings)
            if 5 in featAxis : # - number zero crossings
                res.append(len(zero_crossings))
            if 6 in featAxis : # - variance zerocrossings
                vzc = varZeroCrossings(axisData, windowDur, zero_crossings)
                res.append(vzc)


        # FAST FOURIER TRANSFORM
        if(7 in featAxis or 4 in featAxis ) :
            normalized_tone = axisData # -- normalize data?????
            N = len(normalized_tone) #SAMPLE_RATE * DURATION
            SAMPLE_RATE = N / windowDur
            yf = rfft(normalized_tone)
            xf = rfftfreq(N, 1 / SAMPLE_RATE)
            if 7 in featAxis : # - peak frequency (FFT)
                    # print dominant frequency
                peak_coefficient = np.argmax(np.abs(yf))
                freqs = np.fft.fftfreq(len(yf))
                peak_freq = freqs[peak_coefficient]
                peakFreq = abs(peak_freq * SAMPLE_RATE) 
                res.append(peakFreq)
            if 4 in featAxis : # - peakpower (FFT)
                peakPow = np.max(np.abs(yf))
                res.append(peakPow)

        #AUTOCORRELATIOON
        if(8 in featAxis or 9 in featAxis or 10 in featAxis or 11 in featAxis or 12 in featAxis) :
            lags = range(250) # - to tune parameter!!
            acorr = sm.tsa.acf(axisData, nlags = len(lags)-1)
            threshold = 0.15 # - ????
            acpeaks = getPeakIndx(acorr)
            if 8 in featAxis : # - number peaks autocorrelations
                res.append(len(acpeaks))
            if 9 in featAxis : # - weak peaks autocorrelation
                res.append(getNumWeakPeaks(acpeaks, acorr, threshold))
            if 10 in featAxis : # - prominent peaks autocorrelation
                res.append(getNumPromPeaks(acpeaks, acorr, threshold))
            if 11 in featAxis : # - max value autocorrelation
                # print(acpeaks)
                if len(acpeaks) == 0:
                    res.append(0) #no peaks in ac
                else:
                    maxPeak = np.max(acorr[acpeaks])
                    res.append(maxPeak)
            if 12 in featAxis : # - first peak after zerocrossing autocorrelation
                firstZCIndex = getFirstZeroCrossing(acorr)
                firstPeakIndex = firstIndexAfterIndex(acpeaks, firstZCIndex)
                if firstPeakIndex < 0: 
                    res.append(0)
                else:
                    res.append(acorr[firstPeakIndex])


    return np.asarray(res)




#.. ALSO COMNINE NAMES
# also combine labels
def getFeatDataFile(features, datafile, newFileNameRes, labelfile, newFileNameLabel, samplenamefile, newFileNameSN, windowDur, totalDur, colnames="" ):
    resultMatrix = None
    newLabelArray = []
    newSamplenameArray = []
    print("in tranformDataFile")
    data = readCSVFile(datafile  , (0,1,2,3,4,5))
    labels = readCSVFile(labelfile  , (0))
    samplenames = readCSVFile(samplenamefile , (0), 'str')
    windowIndex = -1
    
    windowSize = math.ceil( (len(data) // totalDur) * windowDur)
    # print(len(data))
    # print(windowSize)

    for sampleIndex in range(0, len(data), windowSize): 
        print(sampleIndex)
        print(len(data))
        windowIndex = windowIndex + 1
 
        window = data[sampleIndex:sampleIndex + windowSize, :]
        print("windowIndex: ", windowIndex)
        # print(window)
        windowFeatures = getFeatures(window, 30, features)
        # print(windowFeatures)

        if resultMatrix is None:
            resultMatrix = np.append([features], [windowFeatures], axis=0)
            # print("added to resultmatrix: ", len(resultMatrix))
        else:
            resultMatrix = np.append(resultMatrix, [windowFeatures], axis=0)

        #labelsWindow
        labelWindow = labels[sampleIndex:sampleIndex + windowSize]
        newLabelArray.append(mode(labelWindow))

        #labelsWindow
        samplenameWindow = samplenames[sampleIndex:sampleIndex + windowSize]
        newSamplenameArray.append(mode(samplenameWindow))
            
        
    resultMatrix = resultMatrix[1:, :]
    #print("without 1-9:\n", len(resultMatrix))
    headerText = "from " + str(datafile) + " - features: " + str(features) + " - windowduration: " +  str(windowDur) + " - totalDur: " + str(totalDur)
    makeCSVFile(resultMatrix, newFileNameRes, headerText)
    makeCSVFile(newLabelArray, newFileNameLabel, headerText)
    makeCSVFile(newSamplenameArray, newFileNameSN, headerText, '%s')
    print("done with file: ", datafile, "\n")



def getFeatMultFiles(files, freq, date, windowDur, featLoc):
    for file in files:
        filename = file[0]
        totalDur = file[1]

        features = range(6*13)
        datafile = r"./usedData/butteredAndNorm/BN" + filename + date + "_" + str(freq) + "hz.txt" 
        newFileNameRes = r"./usedData/features/FeatBN" + filename + date + "_" + str(freq) + "hz.txt" 
        labelfile =  r"./usedData/labels/LabelBN" + filename + date + "_" + str(freq) + "hz.txt"  
        newFileNameLabel = r"./usedData/features/FLabelBN" + filename + date + "_" + str(freq) + "hz.txt" 
        samplenamefile =  r"./usedData/samplenames/NameBN" + filename + date + "_" + str(freq) + "hz.txt"
        newFileNameSN = r"./usedData/features/FNameBN" + filename + date + "_" + str(freq) + "hz.txt" 



        getFeatDataFile(features, datafile, newFileNameRes, labelfile, newFileNameLabel, samplenamefile, newFileNameSN, windowDur, totalDur)




