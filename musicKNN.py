import pandas as pd
import numpy as np
import operator
import read_preprocess_data as rpd
from sklearn.metrics import confusion_matrix

# module to find euclidean distance
def ED(x1, x2, length):
    # distance between points is calculated here
    dist = 0
    for x in range(length):
        dist += np.square(x1[x] - x2[x])
    # print(np.sqrt(distance))
    return np.sqrt(dist)

# module that performs knn algorithm
def knn(trainingset, testset, k):
    # store the distances
    distances = {}
    # find number of columns
    length = testset.shape[1]
    for x in range(len(trainingset)):
        dist = ED(testset, trainingset.iloc[x], length)
        distances[x] = dist[0]
    # sorted distances stored
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))

    # Put the index of col to sort with and find the k neighbours
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x])

    # poll and find the max votes
    votes = {}
    # most frequent class of rows
    for x in range(len(neighbors)):
        response = trainingset.iloc[neighbors[x]][-1]
        # To get the last column for corresponding index
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sortvotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortvotes[0][0], neighbors)

def KNNCLASSIFIER():
    # get data split as training and testing
    (x_train,y_train),(x_test, y_test) = rpd.getSplitData()
    dataSet = (x_train, y_train)
    data = pd.DataFrame(dataSet)

    testSet = (x_test,y_test)
    test = pd.DataFrame(testSet)
    k = 6
    k1 = 3
    # invoking the algorithm with 3 parameters - training data, test data and k value
    result, neigh = knn(data, test, k)
    result1, neigh1 = knn(data, test, k1)

    # confusion matrix used to determine accuracy
    res = confusion_matrix(result1, neigh1)
    print(res)

# invoke the utility
KNNCLASSIFIER()
