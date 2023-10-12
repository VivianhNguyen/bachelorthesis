import numpy as np
from numpy import genfromtxt

# example array
arr = np.asarray([ 
    [1, 150, 10.5], 
    [2, 220, 3.1], 
    [3, 121, 10.1],
    [4, 300, 3.2], 
    [5, 541, 6.7], 
    [6, 321, 9.99],
])

def makeCSVFile(arr, filename, headerText, fmt='%.18e'):
    np.savetxt(filename, arr, delimiter=' ', fmt=fmt, header=headerText)
    print("made: ", filename)
    # np.savetxt(filename, arr, delimiter=' ', header=headerText) #store column names somewhere?

def readCSVFile(filename, usecolsTuple, dtype=None):
    with open(filename) as f:
        extra = (line for line in f if line.startswith('#'))
        #print(extra)
        lines = (line for line in f if not line.startswith('#'))
        FH = None
        if dtype == None:
            FH = np.loadtxt(lines, delimiter=' ', skiprows=0, usecols=usecolsTuple)
        else:
            FH = np.loadtxt(lines, delimiter=' ', skiprows=0, usecols=usecolsTuple, dtype='str')
        # print("read " + filename)
    return FH

date = "1106"
filename = "1cookies"
freq = 20
samplenamefile =  r"./" + date + r"/" + str(20) + r"hz/sampleNames/Name" + filename + date + "_" + str(freq) + "hz.txt"
