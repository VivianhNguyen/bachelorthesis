# Support Vector Machines
# Naive Bayes
# Nearest Neighbor
# Decision Trees
# Logistic Regression
# Neural Networksfrom sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from csvconverter import readCSVFile
from sklearn import tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score 

accuracies = [65, 8, 60, 45, 74, 61]
algos = ["SVM - acc\n(all feat)", "SVM - f1\n(all feat)" , "DT - acc\n(all feat)", "DT - f1\n(all feat)", "RF - acc\n(all feat)", "RF - f1\n(all feat)"]

# Accuracy: 0.65
# f1: 0.4878048780487805
#accuracies.append(0)
def plotGraph(accuracies):
    
    # x-coordinates of left sides of bars 
    left = list(range(1, len(accuracies) + 1)) #

    # heights of bars
    height = accuracies #
    
    # labels for bars
    tick_label = algos#['decision tree', 'two', 'three', 'four', 'five']
    
    plt.rcParams.update({'font.size': 15})
    
    # plotting a bar chart
    plt.bar(left, height, tick_label = tick_label,
            width = 0.3, color = ['red', 'green'])
    
    # naming the x-axis
    plt.xlabel('METRIC AND FEATURE SET')
    # naming the y-axis
    plt.ylabel('performance %')
    # plot title
    plt.title('PERFORMANCE')
    plt.ylim(0, 100)
    # function to show the plot
    plt.show()

plotGraph(accuracies)


