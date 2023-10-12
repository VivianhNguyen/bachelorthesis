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


# multiclass example of using mlextend sequential feature selection
def exampleMulticlass3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)


    sfs1 = SFS(knn, 
            k_features=3, 
            forward=True, 
            floating=True, #False, 
            verbose=2,
            scoring='accuracy', #accuracy',
            cv=0)
    df_X = pd.DataFrame(X, columns=["Sepal length", "Sepal width", "Petal length", "Petal width"])
    df_X.head()

    sfs1 = sfs1.fit(df_X, y)

    print('Best accuracy score: %.2f' % sfs1.k_score_)
    print('Best subset (indices):', sfs1.k_feature_idx_)
    print('Best subset (corresponding names):', sfs1.k_feature_names_)

# binaryclass example of using mlextend sequential feature selection
def exampleBinaryclass():
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    knn = KNeighborsClassifier(n_neighbors=4)


    sfs1 = SFS(knn, 
            k_features=3, 
            forward=True, 
            floating=True, #False, 
            verbose=2,
            scoring='f1', #accuracy',
            cv=0)

    sfs1 = sfs1.fit(X, y)

    print('Best accuracy score: %.2f' % sfs1.k_score_)
    print('Best subset (indices):', sfs1.k_feature_idx_)
    print('Best subset (corresponding names):', sfs1.k_feature_names_)

# mlextend sfs with determined test/train data. (gave an error)
def binaryclass(totalX, totalY, val_indices):

    validation_indices = val_indices
    piter = PredefinedHoldoutSplit(validation_indices)

    rf = RandomForestClassifier()

    sfs1 = SFS(rf, 
            k_features=3, 
            forward=True, 
            floating=True, #False, 
            verbose=2,
            scoring='accuracy', #f1
            cv = piter)
    sfs1 = sfs1.fit(totalX, totalY)

    print('Best accuracy score: %.2f' % sfs1.k_score_)
    print('Best subset (indices):', sfs1.k_feature_idx_)
    print('Best subset (corresponding names):', sfs1.k_feature_names_)

#print feature names by giving the index of the 78 features. Print sorted on feature
def printFeatureNamesByFeat(indexArr):
    features = ["MEAN", "SD", "VAR", "RANDMEANSQR", 
            "NRZC", "VARZC",
            "PEAKFREQ", "PEAKPOWER", 
            "NRAUTOCORPEAKS", 
            "WEAKPEAKS", "PROMINENTPEAKS",
            "MAXACVALUE", "FIRSTPEAKZC"]
    s = ""
    for feature in range(13):
         for axis in range(6):
            index = (axis * 13 ) + feature
            if(index in indexArr):
                print("axis: ", axis, " - feature: ", feature, " - ", features[feature] )
                s = s + "axis: " + str(axis) + " - feature: " + str(feature) + " - " + str(features[feature]) + "\n"
    return s

#print feature names by giving the index of the 78 features. Print sorted on axis
def printFeatureNamesByAxis(indexArr):
    features = ["MEAN", "SD", "VAR", "RANDMEANSQR", 
            "NRZC", "VARZC",
            "PEAKFREQ", "PEAKPOWER", 
            "NRAUTOCORPEAKS", 
            "WEAKPEAKS", "PROMINENTPEAKS",
            "MAXACVALUE", "FIRSTPEAKZC"]
    for axis in range(6):
        for feature in range(13):
            index = (axis * 13 ) + feature
            if(index in indexArr):
                print("axis: ", axis, " - feature: ", feature, " - ", features[feature] )

#the used sequential feature selecion by scikit. Split is an iterable that determine hwo the test and train data are split
def featureSelection(X, y, X_test, y_test, split):
    print("rf")
    rf = RandomForestClassifier()

    sfs = SequentialFeatureSelector(rf, 
                                    n_features_to_select=2, 
                                    scoring="f1", 
                                    cv=split)
    sfs.fit(X, y)

    # arr = range(6*13)
    print(
    "Features selected by forward sequential selection: "
    f"{sfs.get_support(indices=True)}"
    )
    transformedX = sfs.transform(X)
    print(len(X), " ", len(X[0]), " - ", len(transformedX), " ", len(transformedX[0]))

    rf.fit(X, y)
    y_pred = rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("f1:", f1_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred))

    return sfs.get_support(indices=True)





