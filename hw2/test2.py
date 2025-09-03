'''
    <START SET UP>
    Suppress warnings and import necessary libraries.
    Import code for loading data and extracting features.
'''

import warnings
warnings.simplefilter("ignore")
import numpy as np
import math 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load facial landmarks (5 or 68)
X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

# <END SET UP>

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

# 4. Create classifier instances
knn = KNeighborsClassifier(n_neighbors=2, weights='distance', metric='euclidean')  # KNN
svm = SVC(kernel='linear', C=0.1, gamma=1)  # SVM with RBF kernel
dt = DecisionTreeClassifier(max_depth=35)  # Decision Tree

# 5. Evaluate models with cross-validation
# Using k-fold cross-validation (e.g., k=5)
models = {'KNN': knn, 'SVM': svm, 'Decision Tree': dt}

for model_name, model in models.items():
    # Perform cross-validation on each model
    scores = cross_val_score(model, features_pca, y, cv=5)  # 5-fold CV
    print(f"{model_name} Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

# Train the best model (for example, using KNN here)
best_model = knn
best_model.fit(features_pca, y)

# Predict using the trained model for evaluation on the entire dataset (not cross-validated)
y_pred = best_model.predict(features_pca)
num_correct = np.sum(y_pred == y)
num_incorrect = np.sum(y_pred != y)

# Print final accuracy on the whole dataset
print("\nFinal Model Performance:")
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct + num_incorrect)))