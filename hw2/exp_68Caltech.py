import warnings
warnings.simplefilter("ignore")
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load facial landmarks (5 or 68)
X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

# 1. Transform landmarks into features (distance-based features between points)
features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i, :]
            p2 = person_k[j, :]
            features_k.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    features.append(features_k)
features = np.array(features)

# 2. Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Apply PCA for dimensionality reduction (keep 50 components)
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features_scaled)

# 4. Create SVM classifier instance
svm = SVC(kernel='linear', C=0.1, gamma=1)

num_correct = 0
num_incorrect = 0

for i in range(0, len(y)):
    # Use leave-one-out technique
    query_X = features_pca[i, :]  # Take the i-th sample as the query
    query_y = y[i]  # True label for the query
    
    # Remove the i-th sample from the training data
    template_X = np.delete(features_pca, i, 0)
    template_y = np.delete(y, i)
    
    # Train the SVM classifier on the rest of the data
    svm.fit(template_X, template_y)
    
    # Predict the label of the query
    y_pred = svm.predict(query_X.reshape(1, -1))
    
    # Get results
    if y_pred == query_y:
        num_correct += 1
    else:
        num_incorrect += 1

# Print final results
print("\nFinal Model Performance:")
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct + num_incorrect)))