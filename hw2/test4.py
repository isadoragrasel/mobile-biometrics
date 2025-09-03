import warnings
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Suppress warnings
warnings.filterwarnings("ignore")

# Normalize features
scaler = StandardScaler()

# Handle NaN values in the features
def handle_nan(features):
    return np.nan_to_num(features, nan=0.0)

# Feature extraction functions
# 1. Euclidean Distance
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
    return handle_nan(np.array(features))

# 2. Manhattan Distance
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
    return handle_nan(np.array(features))

# 3. Angles Between Landmarks
def compute_angles(X):
    features = []
    for k in range(X.shape[0]):
        person_k = X[k]
        features_k = []
        for i in range(1, person_k.shape[0] - 1):  # Skipping the first and last points
            p1 = person_k[i - 1, :]
            p2 = person_k[i, :]
            p3 = person_k[i + 1, :]
            v1 = p1 - p2
            v2 = p3 - p2
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clamp to avoid NaNs
            angle = np.arccos(cosine_angle)
            features_k.append(angle)
        features.append(features_k)
    return handle_nan(np.array(features))

# 4. Ratios Between Key Distances
def compute_distance_ratios(X):
    features = []
    for k in range(X.shape[0]):
        person_k = X[k]
        features_k = []
        distance_eye_to_eye = math.sqrt((person_k[0, 0] - person_k[1, 0])**2 + (person_k[0, 1] - person_k[1, 1])**2)
        distance_nose_to_mouth = math.sqrt((person_k[2, 0] - person_k[3, 0])**2 + (person_k[2, 1] - person_k[3, 1])**2)
        if distance_nose_to_mouth != 0:
            ratio = distance_eye_to_eye / distance_nose_to_mouth
        else:
            ratio = 0
        features_k.append(ratio)
        features.append(features_k)
    return handle_nan(np.array(features))

# 5. Landmark Displacement (from centroid)
def compute_landmark_displacement(X):
    features = []
    for k in range(X.shape[0]):
        person_k = X[k]
        features_k = []
        centroid = np.mean(person_k, axis=0)
        for landmark in person_k:
            displacement = np.sqrt((landmark[0] - centroid[0])**2 + (landmark[1] - centroid[1])**2)
            features_k.append(displacement)
        features.append(features_k)
    return handle_nan(np.array(features))

# 6. Landmark Density
def compute_landmark_density(X, radius=0.1):
    features = []
    for k in range(X.shape[0]):
        person_k = X[k]
        features_k = []
        nbrs = NearestNeighbors(radius=radius).fit(person_k)
        for landmark in person_k:
            density = nbrs.radius_neighbors([landmark], radius=radius, return_distance=False)
            features_k.append(len(density[0]))
        features.append(features_k)
    return handle_nan(np.array(features))

# 7. Landmark Histograms
def compute_landmark_histograms(X, bins=10):
    features = []
    for k in range(X.shape[0]):
        person_k = X[k]
        hist_x, _ = np.histogram(person_k[:, 0], bins=bins)
        hist_y, _ = np.histogram(person_k[:, 1], bins=bins)
        features_k = np.concatenate((hist_x, hist_y))
        features.append(features_k)
    return handle_nan(np.array(features))

# Classifier Testing
def test_classifiers(features, y, dataset_name):
    features_scaled = scaler.fit_transform(features)
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM (Linear)': SVC(kernel='linear', class_weight='balanced'),
        'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', gamma=0.1, C=1),
        'Decision Tree': DecisionTreeClassifier(max_depth=10)
    }
    
    # Test each classifier
    results = {}
    print(f"\n--- Results for {dataset_name} ---")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, features_scaled, y, cv=5)
        results[name] = (scores.mean(), scores.std())
        print(f'{name} Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')
    
    return results

# Test both datasets (SoF and Caltech)
def test_datasets(X, y, dataset_name):
    print(f"\n--- Testing {dataset_name} ---")
    print("\n--- Euclidean Distance Features ---")
    euclidean_features = compute_euclidean_features(X)
    test_classifiers(euclidean_features, y, dataset_name)

    print("\n--- Manhattan Distance Features ---")
    manhattan_features = compute_manhattan_features(X)
    test_classifiers(manhattan_features, y, dataset_name)

    print("\n--- Angles Between Landmarks ---")
    angle_features = compute_angles(X)
    test_classifiers(angle_features, y, dataset_name)

    print("\n--- Distance Ratios ---")
    ratio_features = compute_distance_ratios(X)
    test_classifiers(ratio_features, y, dataset_name)

    print("\n--- Landmark Displacement Features ---")
    displacement_features = compute_landmark_displacement(X)
    test_classifiers(displacement_features, y, dataset_name)

    print("\n--- Landmark Density Features ---")
    density_features = compute_landmark_density(X)
    test_classifiers(density_features, y, dataset_name)

    print("\n--- Landmark Histograms ---")
    histogram_features = compute_landmark_histograms(X)
    test_classifiers(histogram_features, y, dataset_name)

# Load both datasets and test
X_sof = np.load("X-68-SoF.npy")
y_sof = np.load("y-68-SoF.npy")

X_caltech = np.load("X-68-Caltech.npy")
y_caltech = np.load("y-68-Caltech.npy")

# Test SoF dataset
test_datasets(X_sof, y_sof, "SoF")

# Test Caltech dataset
test_datasets(X_caltech, y_caltech, "Caltech")