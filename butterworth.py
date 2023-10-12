#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# https://www.youtube.com/watch?v=Hp4-C4Incgpw&ab_channel=CurioRes

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from csvconverter import * 
from plotting import * 
from numpy import std

#shows how the lowpass filter changes the axis
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
    sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
    sosh = signal.butter(5, 20, 'hp', fs=1000, output='sos')
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

#applies lowpass filter to axis. Then computes the mean and substracts it, to bring the axis arounf the 0
def butterWorthNormalize(files, fileLocation, storeLocation, date):
    for fileTuple in files:
        filename = fileLocation + fileTuple[0] + date +  "_" + str(freq) + "hz.txt"
        data = readCSVFile(filename, (0, 1, 2, 3, 4, 5))
        # sosl = signal.butter(5, 20, 'lp', fs=1000, output='sos')
        sosl = signal.butter(5, 4, 'lp', fs=20, output='sos')
        for col in range(0, 6):
            axis = data[:, col]
            filtered = signal.sosfilt(sosl, axis)
            mean = np.mean(axis)
            meanfiltered = filtered - mean
            data[:, col] = meanfiltered
        newFileName = storeLocation + "BN" + fileTuple[0] + date + "_" + str(freq) + "hz.txt"
        makeCSVFile(data, newFileName, "normalized and butterworthfiltered" + newFileName )

