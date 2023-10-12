from csvconverter import *
from features import *

#print out the features and axis by giving the indexes

def discoverPerAxis(windowFeatures):
    features = ["MEAN", "SD", "VAR", "RANDMEANSQR", 
                "NRZC", "VARZC",
                "PEAKFREQ", "PEAKPOWER", 
                "NRAUTOCORPEAKS", 
                "WEAKPEAKS", "PROMINENTPEAKS",
                "MAXACVALUE", "FIRSTPEAKZC"]
    for axis in range(6):
        for feature in range(13):
            index = (axis * 13 ) + feature
            print(axis, " - ", features[feature] ," :", windowFeatures[index])

def discoverPerFeature(windowFeatures):
    features = ["MEAN", "SD", "VAR", "RANDMEANSQR", 
                "NRZC", "VARZC",
                "PEAKFREQ", "PEAKPOWER", 
                "NRAUTOCORPEAKS", 
                "WEAKPEAKS", "PROMINENTPEAKS",
                "MAXACVALUE", "FIRSTPEAKZC"]
    for feature in range(13):
         for axis in range(6):
            index = (axis * 13 ) + feature
            print(axis, " - ", features[feature] ," :", windowFeatures[index])

def discoverPerFeatureNan(windowFeatures):
    features = ["MEAN", "SD", "VAR", "RANDMEANSQR", 
                "NRZC", "VARZC",
                "PEAKFREQ", "PEAKPOWER", 
                "NRAUTOCORPEAKS", 
                "WEAKPEAKS", "PROMINENTPEAKS",
                "MAXACVALUE", "FIRSTPEAKZC"]
    for feature in range(13):
         for axis in range(6):
            index = (axis * 13 ) + feature
            # print(windowFeatures[index])
            if(math.isnan(windowFeatures[index])):
            # if(windowFeatures[index] == 0):
                print(axis, " ", feature, " - ", features[feature] ," :", windowFeatures[index], " - i: ", index)





        
