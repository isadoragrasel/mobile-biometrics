
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
import numpy as np
import math
from sklearn.svm import SVC  # Import SVM classifier


# Load facial landmarks (5 or 68) - Caltech dataset in this case
X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

# Euclidean distance between landmarks
def compute_euclidean_features(X):
    features = []
    for k in range(num_identities):
        person_k = X[k]
        features_k = []
        for i in range(person_k.shape[0]):
            for j in range(person_k.shape[0]):
                p1 = person_k[i, :]
                p2 = person_k[j, :]
                euclidean_distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                features_k.append(euclidean_distance)
        features.append(features_k)
    return np.array(features)

# Manhattan distance between landmarks
def compute_manhattan_features(X):
    features = []
    for k in range(num_identities):
        person_k = X[k]
        features_k = []
        for i in range(person_k.shape[0]):
            for j in range(person_k.shape[0]):
                p1 = person_k[i, :]
                p2 = person_k[j, :]
                manhattan_distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                features_k.append(manhattan_distance)
        features.append(features_k)
    return np.array(features)

# Function to evaluate the classifier using leave-one-out strategy
def evaluate_classifier(features, y):
    clf = SVC(kernel='linear', class_weight='balanced')
    
    num_correct = 0
    num_incorrect = 0

    for i in range(0, len(y)):
        query_X = features[i, :]  # Take the i-th sample as the query
        query_y = y[i]  # True label for the query

        # Remove the i-th sample from the training data
        template_X = np.delete(features, i, 0)
        template_y = np.delete(y, i)

        # Train the SVM classifier on the remaining data
        clf.fit(template_X, template_y)

        # Predict the label of the query
        y_pred = clf.predict(query_X.reshape(1, -1))

        # Compare predicted label with the actual label
        if y_pred == query_y:
            num_correct += 1
        else:
            num_incorrect += 1

    # Return the accuracy and the number of correct and incorrect predictions
    return num_correct, num_incorrect, num_correct / (num_correct + num_incorrect)

# 1. Evaluate using Euclidean features
print("\nEvaluating with Euclidean Distance")
euclidean_features = compute_euclidean_features(X)
num_correct, num_incorrect, accuracy = evaluate_classifier(euclidean_features, y)
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, accuracy))

# 2. Evaluate using Manhattan features
print("\nEvaluating with Manhattan Distance")
manhattan_features = compute_manhattan_features(X)
num_correct, num_incorrect, accuracy = evaluate_classifier(manhattan_features, y)
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, accuracy))