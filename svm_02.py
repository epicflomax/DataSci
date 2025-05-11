import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Preparation import df


# Load the CSV file
# file_path = '../kzp-2008-2020-timeseries.csv'
# data = pd.read_csv(file_path, encoding='latin1')
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

# Handle numerical features
numerical_data = filtered_data[numerical_features]
numerical_data.replace([np.inf, -np.inf], np.nan, inplace=True)
numerical_data.fillna(numerical_data.mean(), inplace=True)
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)

# Handle categorical features
categorical_data = filtered_data[categorical_features]
categorical_data_encoded = pd.get_dummies(categorical_data)

# Combine the preprocessed features
preprocessed_features = pd.concat([pd.DataFrame(numerical_data_scaled, columns=numerical_data.columns),
                                    categorical_data_encoded], axis=1)

# Separate the label
label_data = filtered_data[label]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, label_data, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Convert the label to binary: 1 for FiErg > 0, 0 for FiErg <= 0
binary_label_data = (label_data > 0).astype(int)

# Feature selection: Select top k features using ANOVA F-value
k = 10  # You can choose the number of features you want to select
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(preprocessed_features, binary_label_data)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = preprocessed_features.columns[selected_feature_indices]

# Exclude 'FiErg', 'Ort_', and 'Adr_' features from the selected features
filtered_feature_names = [feature for feature in selected_feature_names if feature != 'FiErg' and not feature.startswith('Ort_') and not feature.startswith('Adr_')]

# Get the indices of the filtered features
filtered_feature_indices = [i for i, feature in enumerate(selected_feature_names) if feature in filtered_feature_names]

# Select the filtered features
X_new_filtered = X_new[:, filtered_feature_indices]

# Split the filtered features into training and testing sets
X_train_filtered, X_test_filtered, y_train_binary, y_test_binary = train_test_split(
    X_new_filtered, binary_label_data, test_size=0.2, random_state=42)

#----------------GRID SEARCH -------------------------------------------------------------------------------
# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear', 'poly']
# }

# # Create a GridSearchCV object
# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

# # Fit the grid search to the data
# grid_search.fit(X_train_filtered, y_train_binary)

# # Get the best parameters and best estimator
# best_params = grid_search.best_params_
# best_estimator = grid_search.best_estimator_

# # Predict on the testing set using the best estimator
# y_pred_best = best_estimator.predict(X_test_filtered)

# # Map the labels to meaningful names
# label_names = ["FiErg <= 0", "FiErg > 0"]
# y_test_binary_named = [label_names[label] for label in y_test_binary]
# y_pred_best_named = [label_names[label] for label in y_pred_best]

# # Evaluate the performance
# accuracy_best = accuracy_score(y_test_binary, y_pred_best)
# classification_rep_best = classification_report(y_test_binary_named, y_pred_best_named)

# # Print the results
# print("Best Parameters:", best_params)
# print("Accuracy with Best Estimator:", accuracy_best)
# print("Classification Report with Best Estimator:")
# print(classification_rep_best)
# print("Selected Features:", filtered_feature_names)

# # Generate the confusion matrix
# cm = confusion_matrix(y_test_binary, y_pred_best)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix for SVM Classifier')
# plt.show()

# # Plot feature importances
# if hasattr(best_estimator, 'coef_'):
#     importances = best_estimator.coef_[0]
#     indices = np.argsort(importances)[::-1]

#     plt.figure(figsize=(10, 6))
#     plt.title('Feature Importances')
#     plt.bar(range(X_new_filtered.shape[1]), importances[indices], align='center')
#     plt.xticks(range(X_new_filtered.shape[1]), [filtered_feature_names[i] for i in indices], rotation=90)
#     plt.xlabel('Features')
#     plt.ylabel('Importance Score')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Feature importances are not available for the chosen kernel.")

#---------------END OF GRID SEARCH--------------------------------------------------------------------------------

# Train an SVM classifier on the filtered features
svm_classifier_filtered = SVC(kernel='linear', random_state=42)
svm_classifier_filtered.fit(X_train_filtered, y_train_binary)

# Predict on the testing set
y_pred_filtered = svm_classifier_filtered.predict(X_test_filtered)

# Map the labels to meaningful names
label_names = ["FiErg <= 0", "FiErg > 0"]
y_test_binary_named = [label_names[label] for label in y_test_binary]
y_pred_filtered_named = [label_names[label] for label in y_pred_filtered]

# Evaluate the performance
accuracy_filtered = accuracy_score(y_test_binary, y_pred_filtered)
classification_rep_filtered = classification_report(y_test_binary_named, y_pred_filtered_named)

# Print the results
print("Accuracy:", accuracy_filtered)
print("Classification Report:")
print(classification_rep_filtered)
print("Selected Features:", filtered_feature_names)

# Generate the confusion matrix
cm = confusion_matrix(y_test_binary, y_pred_filtered)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM Classifier')
plt.show()
