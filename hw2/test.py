import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load facial landmarks (5 or 68) - Replace this with the correct dataset path
X = np.load("X-68-SoF.npy")
y = np.load("y-68-SoF.npy")

# Normalize features
scaler = StandardScaler()

# Feature Transformation - Euclidean Distance
def compute_euclidean_features(X):
    features = []
    for k in range(X.shape[0]):
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

# Feature Transformation - Manhattan Distance
def compute_manhattan_features(X):
    features = []
    for k in range(X.shape[0]):
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

# Classifier Testing Function
def test_classifiers(features, y):
    # Scale features
    features_scaled = scaler.fit_transform(features)
    
    # Define classifiers to test
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=2),
        'SVM (Linear)': SVC(kernel='linear', gamma=1, C=0.1),
        'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=35)
    }

    # Test each classifier using cross-validation
    results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, features_scaled, y, cv=5)
        results[name] = (scores.mean(), scores.std())
        print(f'{name} Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')
    
    return results

# Test feature sets
print("\n--- Testing Euclidean Distance Features ---")
euclidean_features = compute_euclidean_features(X)
euclidean_results = test_classifiers(euclidean_features, y)

print("\n--- Testing Manhattan Distance Features ---")
manhattan_features = compute_manhattan_features(X)
manhattan_results = test_classifiers(manhattan_features, y)

# Grid Search for Parameter Tuning - SVM with RBF Kernel
def tune_svm_rbf(features, y):
    features_scaled = scaler.fit_transform(features)
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    }
    svm_rbf = SVC(kernel='rbf', class_weight='balanced')
    grid_search = GridSearchCV(svm_rbf, param_grid, cv=5)
    grid_search.fit(features_scaled, y)
    
    print("\nBest SVM RBF parameters:")
    print(grid_search.best_params_)
    print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")

# Perform parameter tuning for SVM with RBF kernel
print("\n--- Tuning SVM (RBF Kernel) ---")
tune_svm_rbf(euclidean_features, y)