import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load facial landmarks (5 or 68 landmarks)
X = np.load("X-68-SoF.npy")
y = np.load("y-68-SoF.npy")
num_identities = y.shape[0]

# Transform landmarks into features (pairwise distances between points)
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

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 1. KNN Classifier with Grid Search for parameter tuning
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_scaled, y)

best_knn = grid_search_knn.best_estimator_
print("Best KNN parameters: ", grid_search_knn.best_params_)

# 2. SVM Classifier with Grid Search
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': [1, 0.1, 0.01]
}

svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_scaled, y)

best_svm = grid_search_svm.best_estimator_
print("Best SVM parameters: ", grid_search_svm.best_params_)

# 3. Random Forest Classifier with Grid Search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_scaled, y)

best_rf = grid_search_rf.best_estimator_
print("Best Random Forest parameters: ", grid_search_rf.best_params_)

# 4. Gradient Boosting Classifier with Grid Search
param_grid_gbc = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gbc = GradientBoostingClassifier()
grid_search_gbc = GridSearchCV(gbc, param_grid_gbc, cv=5, scoring='accuracy')
grid_search_gbc.fit(X_scaled, y)

best_gbc = grid_search_gbc.best_estimator_
print("Best GBC parameters: ", grid_search_gbc.best_params_)

# 5. Evaluate the models using cross-validation
knn_score = cross_val_score(best_knn, X_scaled, y, cv=5)
svm_score = cross_val_score(best_svm, X_scaled, y, cv=5)
rf_score = cross_val_score(best_rf, X_scaled, y, cv=5)
gbc_score = cross_val_score(best_gbc, X_scaled, y, cv=5)

# Print the accuracy for each classifier
print("\nModel Performance:")
print("KNN Accuracy: %.2f (+/- %.2f)" % (knn_score.mean(), knn_score.std()))
print("SVM Accuracy: %.2f (+/- %.2f)" % (svm_score.mean(), svm_score.std()))
print("Random Forest Accuracy: %.2f (+/- %.2f)" % (rf_score.mean(), rf_score.std()))
print("Gradient Boosting Accuracy: %.2f (+/- %.2f)" % (gbc_score.mean(), gbc_score.std()))