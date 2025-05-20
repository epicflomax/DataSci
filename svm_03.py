import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Preparation import df

# --- Parameters --------
tt_size = 0.2
yr = 2019
model_type = "svc"  # Options: 'svc', 'hgb' (HistGradientBoosting)

# --- Load and prepare data --------
data = df.copy()
label = "FiErg"
features = ["KT", "Inst", "Adr", "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand",
            "SA", "PtageStatT", "AustStatT", "NeugStatT", "Ops", "Gebs", "CMIb", "CMIn",
            "pPatWAU", "pPatWAK", "pPatLKP", "pPatHOK", "PersA", "PersP", "PersMT", "PersT",
            "PersAFall", "PersPFall", "PersMTFall", "PersTFall", "AnzBelA", "AnzBelP (nur ab KZP2010)"]

# Drop rows with missing label
data = data.dropna(subset=[label])

# Drop features that are entirely NaN
columns_to_drop = [col for col in features if data[col].isna().sum() == len(data[col])]
data.drop(columns=columns_to_drop, inplace=True)
features = [col for col in features if col not in columns_to_drop]

# Rename column with parentheses
if "AnzBelP (nur ab KZP2010)" in data.columns:
    data.rename(columns={"AnzBelP (nur ab KZP2010)": "AnzBelP_KZP2010"}, inplace=True)
    features = ["AnzBelP_KZP2010" if col == "AnzBelP (nur ab KZP2010)" else col for col in features]

# Create binary label
data[label] = (data[label] > 0).astype(int)

# Split data
X = data[features]
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_size, random_state=42)

# Separate feature types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# --- Define pipelines and fit model ---
if model_type == "svc":
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=None)), #find explanation and alternatives for this choice- fehlende werden als 'missing values' klassifiziert.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel="linear", random_state=42))
    ])
    X_test_clean = X_test.dropna(subset=numerical_features + categorical_features)
    y_test_clean = y_test.loc[X_test_clean.index]
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test_clean)
    y_test_final = y_test_clean

    # --- Feature importance analysis ---
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)

    svc_model = model_pipeline.named_steps['classifier']
    coefs = svc_model.coef_[0]

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': np.abs(coefs)
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Important Features (SVC):")
    print(coef_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=coef_df.head(10))
    plt.title("Top 10 Feature Importances (SVC - Coefficients)")
    plt.tight_layout()
    plt.show()

elif model_type == "hgb":
    categorical_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_features)
    ], remainder='passthrough')  # this keeps numerical features

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    y_test_final = y_test

    # --- Feature importance analysis ---
    hgb_model = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Transform X_test using the preprocessor
    X_test_preprocessed = preprocessor.transform(X_test)

    # Build processed feature names
    def get_processed_feature_names(preprocessor, categorical_features, numerical_features):
        feature_names = []

        # OrdinalEncoder doesn't expand categories like OneHotEncoder does
        # So categorical feature names stay the same
        feature_names.extend(categorical_features)

        # 'remainder=passthrough' keeps numerical features as-is
        feature_names.extend(numerical_features)

        return feature_names

    processed_feature_names = get_processed_feature_names(preprocessor, categorical_features, numerical_features)

    # SHAP explainer
    explainer = shap.Explainer(hgb_model)
    shap_values = explainer(X_test_preprocessed)

    feature_importances = np.abs(shap_values.values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': processed_feature_names,
        'Importance': feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Optional: visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(20), x='Importance', y='Feature')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()


    print("\nTop Important Features:")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
    plt.title("Top 10 Feature Importances (HistGradientBoostingClassifier)")
    plt.tight_layout()
    plt.show()

else:
    raise ValueError("Invalid model_type. Choose 'svc' or 'hgb'.")

# --- Evaluation ---
label_names = ["FiErg <= 0", "FiErg > 0"]
y_test_named = [label_names[label] for label in y_test_final]
y_pred_named = [label_names[label] for label in y_pred]
accuracy = accuracy_score(y_test_final, y_pred)
print(f"\nModel: {model_type.upper()}")
print(f"Accuracy: {round(accuracy, 2)}")
print("Classification Report:")
print(classification_report(y_test_named, y_pred_named))

cm = confusion_matrix(y_test_final, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix ({model_type.upper()})')
plt.show()


#füge wissenschaftliche Referenzen hinzu

#ergänze mit Mathematischen Formeln aus den empfohlenen Büchern
#ergänz mit Kaggle Blog

#also understand the math for the HistGradientBooster as used in the model above (probably better results)