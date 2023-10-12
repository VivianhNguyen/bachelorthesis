from csvconverter import *
from discover import *

filename = ""
X_test = readCSVFile("filename", range(0, 6*13))

for i in range(len(X_test)):
    windowdFeat = X_test[i]
    # print(windowdFeat)

    discoverPerFeatureNan(windowdFeat)

nans = X_test[:, 38]
print(nans)