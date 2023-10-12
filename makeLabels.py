from csvconverter import readCSVFile, makeCSVFile
import numpy as np

from csvconverter import readCSVFile, makeCSVFile
from matplotlib import pyplot as plt
from plotting import plot_readings

def makeLabels(files, location, date, rate):
     for file in files:
        print(file)
        filename = "BN" + file[0] + date + "_" + str(rate) + "hz.txt"
        data = readCSVFile(location + filename, (0)) #can just take 1 column
        print(len(data))

        labelclass = file[2]
        labels = []
        if labelclass == 0:
            labels = np.zeros(len(data), dtype=np.int8)
        elif labelclass == 1:
            labels = np.ones(len(data), dtype=np.int8)

        print(len(labels))
        print(labels[0])
        newLocationLabel = r"./usedData/labels/"
        newFilename = "Label" + filename 
        fileHeader = newFilename
        makeCSVFile(labels, newLocationLabel + newFilename, fileHeader)

        nameArray = np.full(len(data), file[0] + date + "_" + str(rate) + "hz")
        newLocationName = r"./usedData/samplenames/"
        newFilenameName = "Name" + filename 
        fileHeader = fileHeader + " name of sample"
        makeCSVFile(nameArray, newLocationName + newFilenameName, fileHeader, fmt="%s")







