
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
import numpy as np
import math
from sklearn.svm import SVC  # Import SVM classifier


# Load facial landmarks (5 or 68) - SoF dataset in this case
X = np.load("X-68-SoF.npy")
y = np.load("y-68-SoF.npy")
num_identities = y.shape[0]

# Transform landmarks into features (using Euclidean distance between points)
features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]      
            features_k.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    features.append(features_k)
features = np.array(features)

# Create an instance of the SVM classifier with RBF kernel and balanced class weight
clf = SVC(kernel='rbf', class_weight='balanced')

num_correct = 0
num_incorrect = 0

for i in range(0, len(y)):
    query_X = features[i, :]
    query_y = y[i]
    
    template_X = np.delete(features, i, 0)
    template_y = np.delete(y, i)
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_y))
    y_hat[template_y == query_y] = 1 
    y_hat[template_y != query_y] = 0
    
    # Train the SVM classifier
    clf.fit(template_X, y_hat) 
    
    # Predict the label of the query
    y_pred = clf.predict(query_X.reshape(1,-1)) 
    
    # Get results
    if y_pred == 1:
        num_correct += 1
    else:
        num_incorrect += 1

# Print final results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 

