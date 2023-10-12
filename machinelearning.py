# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from csvconverter import *
import math
from sklearn.tree import DecisionTreeClassifier
from transformAll import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def countAndPrintClassficiation(y_pred, y_test):
    incorrect = []
    correct = []
    countTrain = 0
    countTest = 0
    for i in range(len(y_pred)):
        if(int(y_pred[i]) is not int(y_test[i])):
            incorrect.append(samplenames_test[i])
        else:
            correct.append(samplenames_test[i])
        if (int(y_test[i]) == 1):
            countTest = countTest+1

    for i in range(len(y_train)):
        if (int(y_train[i]) == 1):
            countTrain = countTrain+1

    print("CORRECT")
    for item in correct:
        print(item)

    print("\nINCORRECT")
    for item in incorrect:
        print(item)

    print(countTrain/len(y_train), " ", countTrain, " ", len(y_train))
    print(countTest/len(y_test), " ", countTest, " ", len(y_test))

# done
knnparameters = [{
'n_neighbors': [3, 5, 8, 10, 15, 20]}]
knn = KNeighborsClassifier()
knnclf = GridSearchCV(knn
        , knnparameters, scoring='f1'
    )
#done
svmparameters = [{
'kernel':  ['linear'], #['rbf','sigmoid', 'poly'], #
'C': [1,2,3,300,500],
'max_iter': [1000,100000] # -1]
# 'verbose': [1]
}]
svc = svm.SVC()
# svmclf = svm.SVC(kernel='linear', max_iter=-1)
svmclf = GridSearchCV(svc
        , svmparameters, scoring='f1'
    )

#done
lrparameters = [{
'penalty': ['l1', 'l2', 'elastic', None],
'C':[0.001,.009,0.01,.09,1,5,10,25],
'class_weight': ['balanced', None],
'random_state': [0, None],
'solver': ['liblinear', 'lbfgs' ]
}]
lr = LogisticRegression()
lrclf = GridSearchCV(lr
        , lrparameters, scoring='f1'
    )
# lrclf1 = GridSearchCV(lr
#         , lrparameters, scoring='f1'
#     )
# lrclf2 = GridSearchCV(lr
#         , lrparameters, scoring='f1'
#     )
# lrclf3 = GridSearchCV(lr
#         , lrparameters, scoring='f1'
#     )
# lrclf4 = GridSearchCV(lr
#         , lrparameters, scoring='f1'
#     )
# lrclf5 = GridSearchCV(lr
#         , lrparameters, scoring='f1'
#     )
# lrcfs = [lrclf1, lrclf2,lrclf3,lrclf4,lrclf5]

# - done
dtparameters = [{
'criterion': ['gini', 'entropy'],
'max_depth': [None, 1, 2, 3, 10, 20],
'min_samples_leaf': [1, 2, 3, 5],
'min_samples_leaf': [1, 2, 3],
'class_weight': ['balanced', None],
}]
dt = DecisionTreeClassifier()
dtclf = GridSearchCV(dt
        , dtparameters, scoring='f1'
    )

rfparameters = [{
'n_estimators': [100, 20, 50], #removed 200
'criterion': ['entropy'], #removed gini
'max_depth': [None, 1, 2, 3, 10, 20],
'min_samples_leaf': [1, 2, 3, 5],
'min_samples_leaf': [1, 2, 3],
'class_weight': ['balanced'], # removed none
}]
rf = RandomForestClassifier()
rfclf = GridSearchCV(rf
        , rfparameters, scoring='f1'
    )

l1 = [11]
l2 = [2, 11]
l3 = [2, 5, 11]
l5 = [1, 2, 5, 6, 11]
l10 = [1, 2, 3, 5, 6, 8, 9, 11, 12, 12]
l20 = [1, 2, 3, 5, 5, 5, 6, 6, 6, 8, 9, 9, 9, 10, 11, 11, 11, 12, 12, 12]
l30 = [0, 1, 1, 2, 2, 2, 3, 3, 5, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12]

al1 = [2]
al2 = [1, 2]
al3 = [1, 0, 2]
al5 = [1, 1, 0, 0, 2]
al10 = [1, 1, 2, 0, 0, 5, 4, 2, 1, 4]
al20 = [1, 1, 2, 0, 4, 5, 0, 1, 4, 5, 1, 2, 4, 3, 1, 2, 4, 1, 3, 4]
al30 = [1, 1, 3, 1, 2, 4, 1, 2, 0, 4, 5, 0, 1, 2, 4, 1, 5, 0, 1, 2, 4, 2, 3, 5, 1, 2, 4, 1, 3, 4]


featList = [l1, l2, l3, l5, l10, l20, l30]
axisList = [al1, al2, al3, al5, al10, al20, al30]

for f in featList:
    print(len(f))
for a in axisList:
    print(len(a))

selectionTotalList = []
for sizeIndex in range(7):
    selectionList = []
    for i in range(len(featList[sizeIndex])):
        index = (axisList[sizeIndex][i] * 13 ) + featList[sizeIndex][i]
        selectionList.append(index)
    selectionTotalList.append(selectionList)

print(selectionTotalList)

clfs = [("knn", knnclf), ("svm", svmclf),("lr", lrclf), ("dt", dtclf), ("rf", rfclf)]
clfspart = [ ("svm", svmclf)]
resultString = ""

# -- change "i" in the second forloop, to change the traint-test split
# -- do sequential feature selection for the different feature set sizes

# for tuple in clfspart:
#     for size in size in enumerate([1, 2, 3, 5, 10, 20, 30]): #1, 2
#         i = 2
#         idx = 6
#         extension = str(i) + ".txt"
#         X_train = np.nan_to_num(readCSVFile("./usedData/split/dataFTrain" + extension, range(0, 6*13)))
#         y_train = np.nan_to_num(readCSVFile("./usedData/split/dataLTrain" + extension, (0)))
#         X_test = np.nan_to_num(readCSVFile("./usedData/split/dataFTest" + extension, range(0, 6*13)))
#         y_test = np.nan_to_num(readCSVFile("./usedData/split/dataLTest" + extension, (0)))
#         samplenames_test = readCSVFile("./usedData/split/dataSTest" + extension, (0), 'str')

#         totalX = appendAll([X_train, X_test], 0)
#         totalY = appendAll([y_train, y_test])

#         lengthTrain = len(X_train)
#         stopIndex = len(totalY)
#         val_indices = [*range(lengthTrain, stopIndex-1)]
#         train_indices = [*range(len(y_train))]

#         split = [(train_indices, val_indices)]
#         clf2 = tuple[1]
#         print("start selection")
#         selected = selectionTotalList[idx] #featureSelectionScikit(totalX, totalY, split, size, clf2)
#         X_train = X_train[:, selected]
#         X_test = X_test[:, selected]

#         clf = tuple[1]
#         print("before fitting:")
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         print(clf.best_params_)
#         # print(clf.get_params())
#         accuracy = accuracy_score(y_test, y_pred)
#         f1score = f1_score(y_test, y_pred)
#         print("- start predicting set " +  str(i) + " " + tuple[0] + " :")
#         print("setsize: " + str(size) + "Accuracy:", accuracy)
#         print("setsize: " + str(size) + "f1:", f1score)
#         featuresprint = printFeatureNamesByFeat(selected)
#         resultString = resultString + "\n\n- start predicting set " 
#         resultString = resultString +  str(i) + " " + tuple[0] + " :\n" + "setsize: " + str(size) 
#         resultString = resultString + "Accuracy:" +  str(accuracy) + "\n" + "setsize: " 
#         resultString = resultString + str(size) + "f1:" + str(f1score) + "\n" + featuresprint
#         resultString = resultString + str(clf.best_params_)
# print(resultString)


# --- with or without acc/fft
clfs = [("knn", knnclf), ("svm", svmclf),("lr", lrclf), ("dt", dtclf), ("rf", rfclf)]
acorrIndices = [8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47, 48, 49, 50, 51, 60, 61, 62, 63, 64, 73, 74, 75, 76, 77]
FFTIndices = [6, 7, 19, 20, 32, 33, 45, 46, 58, 59, 71, 72]
acorrFFTIndices = []
acorrFFTIndices.extend(FFTIndices)
acorrFFTIndices.extend(acorrIndices)
indices = [("all", []), ("noac", acorrIndices), ("no fft", FFTIndices), ("nofftac", acorrFFTIndices) ]
for tupleIdx in [("no fft", FFTIndices)]:#indices:
    for tupleClf in [("lr", lrclf)]: #[("knn", knnclf), ("dt", dtclf), ("rf", rfclf)]:
        i = 2
        extension = str(i) + ".txt"
        X_train = np.nan_to_num(readCSVFile("./usedData/split/dataFTrain" + extension, range(0, 6*13)))
        y_train = np.nan_to_num(readCSVFile("./usedData/split/dataLTrain" + extension, (0)))
        X_test = np.nan_to_num(readCSVFile("./usedData/split/dataFTest" + extension, range(0, 6*13)))
        y_test = np.nan_to_num(readCSVFile("./usedData/split/dataLTest" + extension, (0)))
        samplenames_test = readCSVFile("./usedData/split/dataSTest" + extension, (0), 'str')

        X_train = np.delete(X_train, tupleIdx[1], axis=1)
        X_test = np.delete(X_test, tupleIdx[1], axis=1) 
        print(len(X_train[0]))

        print("\nbefore fitting:")
        clf = tupleClf[1]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.best_params_)
        # print(clf.get_params())
        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        print(tupleIdx[0])
        print("- start predicting set " +  str(i) + " " + tupleClf[0] + " :")
        print("Accuracy:", accuracy)
        print("f1:", f1score)










