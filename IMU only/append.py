import numpy as np

#append a list of lists 
def appendAll(listOfLists, axis=None):
    total = []
    for i in range(len(listOfLists)):
        if i == 0 :
            total = listOfLists[i]
        else:
            total = np.append(total, listOfLists[i], axis)
    
    return total