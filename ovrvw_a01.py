
from Data_Preparation import data
#from Data_Preparation import missing_percentage_cat
from Data_Preparation import df
from Data_Preparation import num_features
from Data_Preparation import cat_features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Step 1: Data Preparation -----------

# Define categorical features
categorical_features = ['KT', 'Inst', 'Ort', 'Typ', 'RWStatus', 'Akt', 'SL', 'WB', 'SA']

# Make sure required columns exist
required_columns = categorical_features + ['FiErg']
missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing values
data_clean = data[required_columns].dropna()
data_clean['Target'] = (data_clean['FiErg'] > 0).astype(int)

# Optional: sample for smaller size
data_sample = data_clean.sample(frac=0.3, random_state=42).reset_index(drop=True)
X = data_sample[categorical_features]
y = data_sample['Target']

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# ----------- Step 2: SVM Model -----------

svm_model = GridSearchCV(
    SVC(), 
    param_grid={'C': [0.1, 1, 10], 'kernel': ['linear']}, 
    cv=5, scoring='accuracy'
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# ----------- Step 3: k-NN Model -----------

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# ----------- Step 4: Evaluation -----------

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(classification_report(y_true, y_pred))
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

results = []
results.append(evaluate_model("SVM", y_test, svm_pred))
results.append(evaluate_model("k-NN", y_test, knn_pred))

# ----------- Step 5: Confusion Matrices -----------

plt.figure(figsize=(10, 4))
for i, (pred, title) in enumerate(zip([svm_pred, knn_pred], ['SVM', 'k-NN'])):
    cm = confusion_matrix(y_test, pred)
    plt.subplot(1, 2, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ----------- Step 6: Metric Comparison -----------

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)
