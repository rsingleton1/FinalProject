# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


def avoid0div(x):
    if x == 0:
        return 0.000000001
    else:
        return x


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def minkowski_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs((row1[i] - row2[i]) ** p)
    return distance ** (1 / float(p))


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, p):
    distances = list()
    for train_row in train:
        dist = minkowski_distance(test_row, train_row, p)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    neighborswdistances = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
        neighborswdistances.append(distances[i])
    return neighbors, neighborswdistances


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, inverse_distance, p):
    neighbors, neighborswdistances = get_neighbors(train, test_row, num_neighbors, p)
    if inverse_distance:
        class0 = [row[1] for row in neighborswdistances if row[0][4] == 0]
        class1 = [row[1] for row in neighborswdistances if row[0][4] == 1]
        class2 = [row[1] for row in neighborswdistances if row[0][4] == 2]
        classVotes = [class0, class1, class2]
        output_values = [0, 1, 2]
        prediction = max(output_values, key=lambda output_values: len(classVotes[output_values])
                                                                  * (1 / avoid0div(sum(classVotes[output_values]))))

    else:
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=(output_values.count))
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, inverse_distance, p):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, inverse_distance, p)
        predictions.append(output)
    return (predictions)


# Test the kNN on the Iris Flowers dataset
seed(1)
#filename = 'iris.csv'
#dataset_MinMax = np.load('Test_Images_Features.npy')
#dataset_MinMax = dataset_MinMax.tolist()

#dataset_Zscore = np.load('Test_Images_Features.npy')
#dataset_Zscore = dataset_Zscore.tolist()

dataset = np.load('Test_Images_Features.npy')
dataset = dataset.tolist()

n_folds = 5

datasets = [dataset]
# stuff for graphs

k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]


# Generate data for Manhattan distance p = 1, no weighting
#dataset_k_accuracy = []
#dataset_k_MinMax_accuracy = []
#dataset_k_Zscore_accuracy = []

#for data in datasets:
#    for num_neighbors in range(30):
#        num_neighbors += 1
#        scores = evaluate_algorithm(data, k_nearest_neighbors, n_folds, num_neighbors, 0, 1)
#        #print('Scores: %s' % scores)
#        #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

#        if data == datasets[0]:
#            dataset_k_accuracy.append((sum(scores) / float(len(scores))))
#        elif data == datasets[1]:
#            dataset_k_MinMax_accuracy.append((sum(scores) / float(len(scores))))
#        elif data == datasets[2]:
#            dataset_k_Zscore_accuracy.append((sum(scores) / float(len(scores))))

#plt.figure(1)
#plt.title("Majority Unweighted Voting w Manhattan Distance ")
#plt.xlabel('K-Values')
#plt.ylabel('Mean Accuracy (%)')
#plt.plot(k, dataset_k_accuracy, linestyle='--', marker='o', color='b', label='Non-Normalized')
#plt.plot(k, dataset_k_MinMax_accuracy, linestyle='--', marker='o', color='r', label='Min-max Normalization')
#plt.plot(k, dataset_k_Zscore_accuracy, linestyle='--', marker='o', color='g', label='Z-Score Normalization')
#plt.legend()

#print('Manhattan distance p = 4, no weighting \n \n')

#print("The Max Accuracy of the Non-Normalized was: " + str(max(dataset_k_accuracy)) + " at K value "
#      + str(max(k, key=lambda k: dataset_k_accuracy[k - 1])))

#print("The Max Accuracy of the MinMax-Normalized was: " + str(max(dataset_k_MinMax_accuracy)) + " at K value "
#      + str(max(k, key=lambda k: dataset_k_MinMax_accuracy[k - 1])))

#print("The Max Accuracy of the Zscore-Normalized was: " + str(max(dataset_k_Zscore_accuracy)) + " at K value "
#      + str(max(k, key=lambda k: dataset_k_Zscore_accuracy[k - 1])))



# Generate data for Manhattan distance p = 4, inverse distance weighting

"""

dataset_k_accuracy = []
dataset_k_MinMax_accuracy = []
dataset_k_Zscore_accuracy = []

for data in datasets:
    for num_neighbors in range(30):
        num_neighbors += 1
        scores = evaluate_algorithm(data, k_nearest_neighbors, n_folds, num_neighbors, 1, 1)
        #print('Scores: %s' % scores)
        #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        if data == datasets[0]:
            dataset_k_accuracy.append((sum(scores) / float(len(scores))))
        elif data == datasets[1]:
            dataset_k_MinMax_accuracy.append((sum(scores) / float(len(scores))))
        elif data == datasets[2]:
            dataset_k_Zscore_accuracy.append((sum(scores) / float(len(scores))))

plt.figure(2)
plt.title("Inverse Distance Voting w Manhattan Distance")
plt.xlabel('K-Values')
plt.ylabel('Mean Accuracy (%)')
plt.plot(k, dataset_k_accuracy, linestyle='--', marker='o', color='b', label='Non-Normalized')
plt.plot(k, dataset_k_MinMax_accuracy, linestyle='--', marker='o', color='r', label='Min-max Normalization')
plt.plot(k, dataset_k_Zscore_accuracy, linestyle='--', marker='o', color='g', label='Z-Score Normalization')
plt.legend()


print('Manhattan distance, Inverse distance Voting \n \n')

print("The Max Accuracy of the Non-Normalized was: " + str(max(dataset_k_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_accuracy[k - 1])))

print("The Max Accuracy of the MinMax-Normalized was: " + str(max(dataset_k_MinMax_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_MinMax_accuracy[k - 1])))

print("The Max Accuracy of the Zscore-Normalized was: " + str(max(dataset_k_Zscore_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_Zscore_accuracy[k - 1])))

"""

# Generate data for Euclidean distance, no weighting
dataset_k_accuracy = []

for data in datasets:
    for num_neighbors in range(30):
        num_neighbors += 1
        scores = evaluate_algorithm(data, k_nearest_neighbors, n_folds, num_neighbors, 0, 2)
        #print('Scores: %s' % scores)
        #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        if data == datasets[0]:
            dataset_k_accuracy.append((sum(scores) / float(len(scores))))
#        elif data == datasets[1]:
#            dataset_k_MinMax_accuracy.append((sum(scores) / float(len(scores))))
#        elif data == datasets[2]:
#            dataset_k_Zscore_accuracy.append((sum(scores) / float(len(scores))))

print(dataset_k_accuracy)
"""
plt.figure(3)
plt.title("Majority Unweighted Voting w Euclidean Distance")
plt.xlabel('K-Values')
plt.ylabel('Mean Accuracy (%)')
plt.plot(k, dataset_k_accuracy, linestyle='--', marker='o', color='b', label='Non-Normalized')
plt.plot(k, dataset_k_MinMax_accuracy, linestyle='--', marker='o', color='r', label='Min-max Normalization')
plt.plot(k, dataset_k_Zscore_accuracy, linestyle='--', marker='o', color='g', label='Z-Score Normalization')
plt.legend()

print('Euclidean, no weighting \n \n')

print("The Max Accuracy of the Non-Normalized was: " + str(max(dataset_k_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_accuracy[k - 1])))

print("The Max Accuracy of the MinMax-Normalized was: " + str(max(dataset_k_MinMax_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_MinMax_accuracy[k - 1])))

print("The Max Accuracy of the Zscore-Normalized was: " + str(max(dataset_k_Zscore_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_Zscore_accuracy[k - 1])))


# Generate data for Euclidean distance, Inverse Distance Weighting
dataset_k_accuracy = []
dataset_k_MinMax_accuracy = []
dataset_k_Zscore_accuracy = []

for data in datasets:
    for num_neighbors in range(30):
        num_neighbors += 1
        scores = evaluate_algorithm(data, k_nearest_neighbors, n_folds, num_neighbors, 1, 2)
        #print('Scores: %s' % scores)
        #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

        if data == datasets[0]:
            dataset_k_accuracy.append((sum(scores) / float(len(scores))))
        elif data == datasets[1]:
            dataset_k_MinMax_accuracy.append((sum(scores) / float(len(scores))))
        elif data == datasets[2]:
            dataset_k_Zscore_accuracy.append((sum(scores) / float(len(scores))))

plt.figure(4)
plt.title("Inverse Distance Voting w Euclidean Distance")
plt.xlabel('K-Values')
plt.ylabel('Mean Accuracy (%)')
plt.plot(k, dataset_k_accuracy, linestyle='--', marker='o', color='b', label='Non-Normalized')
plt.plot(k, dataset_k_MinMax_accuracy, linestyle='--', marker='o', color='r', label='Min-max Normalization')
plt.plot(k, dataset_k_Zscore_accuracy, linestyle='--', marker='o', color='g', label='Z-Score Normalization')
plt.legend()

print('Euclidean distance, Inverse Weighting \n \n')

print("The Max Accuracy of the Non-Normalized was: " + str(max(dataset_k_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_accuracy[k - 1])))

print("The Max Accuracy of the MinMax-Normalized was: " + str(max(dataset_k_MinMax_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_MinMax_accuracy[k - 1])))

print("The Max Accuracy of the Zscore-Normalized was: " + str(max(dataset_k_Zscore_accuracy)) + " at K value "
      + str(max(k, key=lambda k: dataset_k_Zscore_accuracy[k - 1])))



# data visualization
print(dataset)
dataset = np.array(dataset)
dataset_Zscore = np.array(dataset_Zscore)
dataset_MinMax = np.array(dataset_MinMax)

plt.figure(5)
plt.title('Data After Normalizations')
plt.plot(dataset[:, 0], dataset[:, 1], 'bo', label=' Non-Norm features 1 v 2')
plt.plot(dataset_MinMax[:,0], dataset_MinMax[:,  1], 'go', label='Min-Max Norm 1 v 2')
plt.plot(dataset_Zscore[:, 0], dataset_Zscore[:, 1], 'ro', label='Z-Norm 1 v 2')
plt.legend()







plt.show()
"""