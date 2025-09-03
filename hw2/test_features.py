import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings for SVM convergence
warnings.simplefilter("ignore", category=ConvergenceWarning)

# Functions to compute various features
def relative_distances(landmarks):
    features = []
    for i in range(landmarks.shape[0]):
        for j in range(i + 1, landmarks.shape[0]):
            p1 = landmarks[i, :]
            p2 = landmarks[j, :]
            distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            features.append(distance)
    return np.array(features)

def angles(landmarks):
    features = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            for k in range(j + 1, len(landmarks)):
                v1 = landmarks[i] - landmarks[j]
                v2 = landmarks[k] - landmarks[j]
                # Avoid dividing by zero in norm calculation
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    continue
                cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                # Clip the cosine angle to avoid NaNs
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.arccos(cosine_angle)
                features.append(angle)
    return np.array(features)

def ratios(landmarks):
    features = []
    for i in range(landmarks.shape[0]):
        for j in range(i + 1, landmarks.shape[0]):
            for k in range(j + 1, landmarks.shape[0]):
                distance_ij = np.linalg.norm(landmarks[i] - landmarks[j])
                distance_jk = np.linalg.norm(landmarks[j] - landmarks[k])
                if distance_jk != 0:
                    features.append(distance_ij / distance_jk)
    return np.array(features)

# Define a function to pad feature arrays to ensure consistent shape
def pad_features(features_list, max_length):
    padded_features = []
    for features in features_list:
        padded = np.pad(features, (0, max_length - len(features)), 'constant', constant_values=0)
        padded_features.append(padded)
    return np.array(padded_features)

# Feature extraction process
def extract_features(X, feature_type):
    extracted_features = []
    for landmarks in X:
        if feature_type == 'relative_distances':
            extracted_features.append(relative_distances(landmarks))
        elif feature_type == 'angles':
            extracted_features.append(angles(landmarks))
        elif feature_type == 'ratios':
            extracted_features.append(ratios(landmarks))
        # Add other feature types here
    
    # Find the maximum feature length and pad features accordingly
    max_length = max(len(f) for f in extracted_features)
    return pad_features(extracted_features, max_length)

# Function to evaluate classifiers
def evaluate_classifier(clf, features, y):
    scores = cross_val_score(clf, features, y, cv=5)
    return scores.mean(), scores.std()

# Load data
X = np.load("X-68-SoF.npy")
y = np.load("y-68-SoF.npy")

# Impute missing values (if any)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Define classifier settings for each feature type
classifier_settings = {
    'relative_distances': {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=5)
    },
    'angles': {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=3)
    },
    'ratios': {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=4)
    }
}

# Compare performance
for feature_type in classifier_settings.keys():
    print(f"\nEvaluating feature: {feature_type}")
    
    # Extract features
    features = extract_features(X_imputed, feature_type)
    
    # Check for NaN values and replace with zero (or handle differently)
    features = np.nan_to_num(features, nan=0.0)
    
    # Scale features and apply PCA if needed
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=min(features_scaled.shape[0], features_scaled.shape[1], 25))
    features_pca = pca.fit_transform(features_scaled)
    
    # Get the classifiers for this feature type
    for clf_name, clf in classifier_settings[feature_type].items():
        mean_score, std_score = evaluate_classifier(clf, features_pca, y)
        print(f"{clf_name} Accuracy: {mean_score:.2f} (+/- {std_score:.2f})")