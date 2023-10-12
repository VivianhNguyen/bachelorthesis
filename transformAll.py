from changeRate import *
from butterworth import *
from makeLabels import *
from transformData import *
#https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from csvconverter import *

from mlxtend.evaluate import PredefinedHoldoutSplit
from append import *
from sklearn.feature_selection import SequentialFeatureSelector
from selection import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# ---- 1106
files1106 = [("1cookies", 90, 1, False), 
        ("1icecream", 90, 1, False),
        # ("1tosti", 3*60, 1),
        # ("1video", 6*60, 0 ),
        # ("1walking", 8*60, 0 ),
        ("2icecream", 90, 1, False ),
        # ("2tosti", 3*60, 1),
        # ("3tosti", 2*60, 1),
        # ("4tosti", 1*60, 1)
        ]

## -----1206
files1206 = [("1cucumber", 3 * 60, 1, True),
         ("1ontbijtkoek", 1 * 60, 1, True),
         ("1phone", 4 * 60, 0, True),
        #  ("1sauzijs", 4 * 60, 1),
        #  ("1studying", 7* 60, 0),
         ("2ontbijtkoek", 1 * 60, 1, True)]

# ----- 1306

files1306 = [
        # ("1brushingteeth", 2 * 60, 0),
        #  ("1studying", 7 * 60, 0),
        #  ("1tosti", 2.5 * 30, 1),
         ("1reading", 4 * 60, 0, True),
         ("1walkinginroom", 3 * 60, 0, True),
         ("2reading", 3 * 60, 0, True)
         ]


# ---- 2006

files2006 = [("1acrackerp2", 2 * 60, 1, True),
        #      ("1applep3", 2 * 60, 1),
             ("1crackerp3", 2 * 60, 1, True),
             ("1crackerp2", 2 * 60, 1, True),
             ("1cucumberp2", 2* 60, 1, False),
             ("1pindap2", 2 * 60, 1, True),
             ("1rijstwafelp2", 1.5 * 60, 1,False),

             ("1phonep2", 5 * 60, 0, True),
             ("1readingp2", 4 * 60, 0, True),
             ("1talkingp3", 4 * 60, 0,  False),
             ("1videop3", 5 * 60, 0, True),
             ("1videop2", 7 * 60, 0, True),
             ("1walkingp3", 7 * 60, 0, True),
             ("1walkingp2", 5 * 60, 0, True),
             ("2talkingp2", 4 * 60, 0, False),
             ("2walkingp2", 4 * 60, 0, False)
             ]

# --- 2106
files2106 = [("1walking", 8 * 60, 0, False),
             ("1video", 6 * 60, 0, False),
             ("1studying", 7 * 60, 0, True),
             ("2studying", 4 * 60, 0, False),
             ("1talking", 4 * 60, 0, False),
             ("1mandarijnpellen", 3 * 60, 0, False),

             ("1melkbroodje", 3 * 60, 1, False),
             ("1mandarijn", 3 * 60, 1, False),
             ("2mandarijn", 2 * 60, 1, False),
             ("1rijstwafel", 3 * 60, 1, False),
             ("2melkbroodje", 2 * 60, 1, False),
             ("2rijstwafel", 2 * 60, 1, False),
        #      ("6?", 2 * 60, 1)
             ]

files2106_2 = [("1studyp4", 5 * 60, 0, False), 
               ("1sultanap4", 3 * 60, 1, True),
               ("1talkingp4", 3.5 * 60, 0, True),
               ("2sultanap4", 3 * 60, 1,False),
               ("1walkingp4", 4 * 60, 0, True),
               ]


totalFiles = [("1106", files1106),
         ("1206", files1206),
         ("1306", files1306),
         ("2006", files2006),
         ("2106", files2106),
         ("2106", files2106_2)]

# totalFiles = [("2106", [("1walkingp4", 4 * 60, 0, False)])]


# --- to 20 hz
# for fileList in totalFiles :
#     date = fileList[0]
#     files = fileList[1]
#     location = r"./usedData/" + date + r"/"
#     changeRateOfFiles(files, location, date, 20)

# --- butterworth and normalize
# for fileList in totalFiles :
#     date = fileList[0]
#     files = fileList[1]
#     fileLocation = r"./usedData/20hz/"
#     storeLocation = r"./usedData/butteredAndNorm/"
#     butterWorthNormalize(files, fileLocation, storeLocation, date)

# --- labels
# for fileList in totalFiles :
#     date = fileList[0]
#     files = fileList[1]
#     fileLocation = r"./usedData/butteredAndNorm/"
#     makeLabels(files, fileLocation, date, 20)


# --- features
# for fileList in totalFiles :
#     date = fileList[0]
#     files = fileList[1]
#     fileLocation = r"./usedData/butteredAndNorm/"
#     getFeatMultFiles(files, 20, date, 30, "")


def combineFilesTestTrain(tuples, freq, name):
        totalStringFilesTrain = ""
        totalStringFilesTest = ""
        totalDataTrain = None
        totalLabelsTrain = None
        totalSNTrain = None
        totalDataTest = None
        totalLabelsTest = None
        totalSNTest = None
        countTest = 0
        countTrain = 0
        for tuple in tuples:
                date = tuple[0]
                files = tuple[1]

                for file in files:
                        filename = file[0]
                        #def readCSVFile(filename, usecolsTuple, dtype=None):
                        data = readCSVFile(r"./usedData/features/FeatBN" + filename + date + "_" + str(freq) + "hz.txt"
                                        , range(6*13))
                        labels = readCSVFile(r"./usedData/features/FLabelBN" + filename + date + "_" + str(freq) + "hz.txt"
                                        , (0))
                        samplenames = readCSVFile(r"./usedData/features/FNameBN" + filename + date + "_" + str(freq) + "hz.txt"
                                        , (0), 'str')

                        if not file[3]: #Train
                                countTrain = countTrain + file[1]
                                totalStringFilesTrain = totalStringFilesTrain + "(" + filename + ", " + date + ")" + " "
                                if totalDataTrain is None:
                                        totalDataTrain = data
                                        # print(data)
                                else:
                                        totalDataTrain = np.append(totalDataTrain, data, axis=0)
                                if totalLabelsTrain is None:
                                        totalLabelsTrain = labels
                                else: 
                                        totalLabelsTrain = np.append(totalLabelsTrain, labels)
                                if totalSNTrain is None:
                                        totalSNTrain = samplenames
                                else:
                                        print(len(totalSNTrain))
                                        totalSNTrain = np.append(totalSNTrain, samplenames)
                                        print(len(totalSNTrain))
                        else: #Test
                                countTest = countTest + file[1]
                                totalStringFilesTest = totalStringFilesTest + "(" + filename + ", " + date + ")" + " "
                                if totalDataTest is None:
                                        totalDataTest = data
                                        # print(data)
                                else:
                                        totalDataTest = np.append(totalDataTest, data, axis=0)
                                if totalLabelsTest is None:
                                        totalLabelsTest = labels
                                else: 
                                        totalLabelsTest = np.append(totalLabelsTest, labels)
                                if totalSNTest is None:
                                        totalSNTest = samplenames
                                else:
                                        print(len(totalSNTest))
                                        totalSNTest = np.append(totalSNTest, samplenames)
                                        print(len(totalSNTest))
                        
        

        extension = "5.txt"
        headerText = str(countTrain) + " combined: " + totalStringFilesTrain 
        makeCSVFile(totalDataTrain, "dataFTrain" + extension, headerText)
        makeCSVFile(totalLabelsTrain, "dataLTrain" + extension, headerText)
        makeCSVFile(totalSNTrain, "dataSTrain" + extension, headerText, '%s')
        print(len(totalDataTrain), len(totalLabelsTrain), len(totalSNTrain))
        print(headerText)

        headerText = str(countTest)  + " combined: " + totalStringFilesTest
        makeCSVFile(totalDataTest, "dataFTest" + extension, headerText)
        makeCSVFile(totalLabelsTest, "dataLTest" + extension, headerText)
        makeCSVFile(totalSNTest, "dataSTest" + extension, headerText, '%s')
        print(len(totalDataTest), len(totalLabelsTest), len(totalSNTest))
        print(headerText)

# combineFilesTestTrain(totalFiles, 20, "")


def featureSelectionScikit(Xtotal, ytotal, split, nrfeat, clf):
        #rf = RandomForestClassifier()
        # rf = RandomForestClassifier()
        # dt = DecisionTreeClassifier()
        # svm = svm.SVC(random_state=0, class_weight={1: 3})


        sfs = SequentialFeatureSelector(clf, 
                                        n_features_to_select=nrfeat, 
                                        scoring="f1", 
                                        cv=split)
        sfs.fit(Xtotal, ytotal)

        # arr = range(6*13)
        print(
        "\nFeatures selected by forward sequential selection: "
        f"{sfs.get_support(indices=True)}"
        )
        
        return sfs.get_support(indices=True)
