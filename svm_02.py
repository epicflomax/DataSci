import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Preparation import df

# Load the data
data = df

# Define the label and features
label = "FiErg"
features = ["KT", "Inst", "Adr", "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand",
            "SA", "PtageStatT", "AustStatT", "NeugStatT", "Ops", "Gebs", "CMIb", "CMIn",
            "pPatWAU", "pPatWAK", "pPatLKP", "pPatHOK", "PersA", "PersP", "PersMT", "PersT",
            "PersAFall", "PersPFall", "PersMTFall", "PersTFall", "AnzBelA", "AnzBelP (nur ab KZP2010)"]

# Filter the data to include only the specified features and label
filtered_data = data[features + [label]]

# Separate numerical and categorical features
numerical_features = filtered_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = filtered_data.select_dtypes(include=['object']).columns.tolist()

# Handle numerical features - replace inf values with NaN first
filtered_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Create a deep copy to avoid modifying the original data
processed_data = filtered_data.copy(deep=True)

# Set up imputer for numerical features
num_imputer = SimpleImputer(strategy='mean')
processed_data[numerical_features] = num_imputer.fit_transform(filtered_data[numerical_features])

# Convert the label to binary: 1 for FiErg > 0, 0 for FiErg <= 0
label_data = processed_data[label]
binary_label_data = (label_data > 0).astype(int)

# Split the data into training and testing sets FIRST
# Important: Split before any transformations to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(processed_data, binary_label_data, test_size=0.2, random_state=42)

# Initialize the StandardScaler
sc = StandardScaler()

# Fit the scaler on training data and transform both training and test data
X_train[numerical_features] = sc.fit_transform(X_train[numerical_features])
# Only transform test data using parameters learned from training data
X_test[numerical_features] = sc.transform(X_test[numerical_features])

# Make sure there are no NaN values in the test set as well
# These should already be handled by the imputer, but just to be safe
X_train = X_train.fillna(0)  # Use 0 or another appropriate value
X_test = X_test.fillna(0)  # Use the same fill value for consistency

# Display the shapes of the training and testing sets
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Feature selection: Select top k features using ANOVA F-value
k = 10  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=k)

# CORRECTION: Fit the selector only on training data
# Now we're sure there are no NaN values
selector.fit(X_train[numerical_features], y_train)

# Transform both training and test data with the fitted selector
X_train_selected = selector.transform(X_train[numerical_features])
X_test_selected = selector.transform(X_test[numerical_features])

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = np.array(numerical_features)[selected_feature_indices]

# Exclude 'FiErg', 'Ort_', and 'Adr_' features from the selected features
filtered_feature_names = [feature for feature in selected_feature_names if feature != 'FiErg' and not feature.startswith('Ort_') and not feature.startswith('Adr_')]

# Filter the selected features
filtered_feature_indices = [i for i, feature in enumerate(selected_feature_names) if feature in filtered_feature_names]
X_train_filtered = X_train_selected[:, filtered_feature_indices]
X_test_filtered = X_test_selected[:, filtered_feature_indices]

# Train an SVM classifier on the filtered features
svm_classifier_filtered = SVC(kernel='linear', random_state=42)
svm_classifier_filtered.fit(X_train_filtered, y_train)

# Predict on the testing set
y_pred_filtered = svm_classifier_filtered.predict(X_test_filtered)

# Map the labels to meaningful names
label_names = ["FiErg <= 0", "FiErg > 0"]
y_test_named = [label_names[label] for label in y_test]
y_pred_filtered_named = [label_names[label] for label in y_pred_filtered]

# Evaluate the performance
accuracy_filtered = accuracy_score(y_test, y_pred_filtered)
classification_rep_filtered = classification_report(y_test_named, y_pred_filtered_named)

# Print the results
print("Accuracy:", accuracy_filtered)
print("Classification Report:")
print(classification_rep_filtered)
print("Selected Features:", filtered_feature_names)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_filtered)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM Classifier')
plt.show()

# Uncomment below for GridSearch if needed
"""
# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train_filtered, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Predict on the testing set using the best estimator
y_pred_best = best_estimator.predict(X_test_filtered)

# Evaluate the performance
accuracy_best = accuracy_score(y_test, y_pred_best)
y_test_named = [label_names[label] for label in y_test]
y_pred_best_named = [label_names[label] for label in y_pred_best]
classification_rep_best = classification_report(y_test_named, y_pred_best_named)

# Print the results
print("Best Parameters:", best_params)
print("Accuracy with Best Estimator:", accuracy_best)
print("Classification Report with Best Estimator:")
print(classification_rep_best)
"""