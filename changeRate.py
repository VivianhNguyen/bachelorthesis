from csvconverter import readCSVFile, makeCSVFile
from matplotlib import pyplot as plt
from plotting import plot_readings
import numpy as np

# --- change rate AND add samplename
def changeSampleRate(data, durationSeconds, newSampleRate):
    neededSamples = durationSeconds * newSampleRate
    sampleReduceWindow = round(len(data) / neededSamples)
    #print(sampleReduceWindow, " ", len(data) / neededSamples)
    reduceMat = data[sampleReduceWindow::sampleReduceWindow]
    return reduceMat

#change the rate of multiple files input is an iterable of iterables
def changeRateOfFiles(files, location, date, rate):

    for file in files:
        filename = location + file[0] + date + ".txt"
        newFilename = file[0] + date + "_" + str(rate) + "hz.txt"

        data = readCSVFile(filename, (1, 2, 3, 4, 5, 6))
        headerText = newFilename + " - duration: " + str(file[1])
        changedData = changeSampleRate(data, file[1], rate)
        newLocation = r"./usedData/20hz/"
        makeCSVFile(changedData, newLocation + newFilename, headerText)
        print((len(changedData)//20)/60) #gives the duration in seconds

