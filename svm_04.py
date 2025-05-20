import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.svm import LinearSVC
import seaborn as sns
from Data_Preparation import df

# --- Parameters --------
tt_size = 0.2
n_ftrs = 10
#yr = 2011
model_type = "svc"  # Options: 'svc', 'hgb'
data = df.copy()

md_choice = input('Which model do you prefer? (default is svc) ')

for i in md_choice.split(' '): 
    if i.startswith('20'):
        yer = int(i)
        print(yer)
        if yer > 2008 and yer < 2020:
            yr = yer
            data = df[df["JAHR"]==yr].copy()
            print('chosen_year', yr)
    elif i == 'svc':
        model_type = i
    elif i == 'hgb':
        model_type = i
    elif i.startswith('0.'): #testsplit
        tt_size = float(i)
        print('chosen test-split size:', tt_size)


# --- Load and prepare data --------
#data = df.copy()
#data = df[df["JAHR"]==yr].copy()


label = "FiErg"
# features = ["KT", "Inst", "Adr", "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand",
#             "SA", "PtageStatT", "AustStatT", "NeugStatT", "Ops", "Gebs", "CMIb", "CMIn",
#             "pPatWAU", "pPatWAK", "pPatLKP", "pPatHOK", "PersA", "PersP", "PersMT", "PersT",
#             "PersAFall", "PersPFall", "PersMTFall", "PersTFall", "AnzBelA", "AnzBelP (nur ab KZP2010)"]
features = features = ["KT", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]


# features = [ "Typ", "RWStatus", "Akt", "WB", "AnzStand",
#             "PtageStatT", "AustStatT", "NeugStatT", "Ops", "Gebs", "CMIb", "CMIn",
#             "pPatWAU", "pPatWAK", "pPatLKP", "pPatHOK", "PersA", "PersP", "PersMT", "PersT",
#             "PersAFall", "PersPFall", "PersMTFall", "PersTFall", "AnzBelA", "AnzBelP (nur ab KZP2010)"]

data = data.dropna(subset=[label])  # Drop missing labels

# Drop all-NaN columns
columns_to_drop = [col for col in features if data[col].isna().sum() == len(data[col])]
data.drop(columns=columns_to_drop, inplace=True)
features = [col for col in features if col not in columns_to_drop]

# Rename columns
if "AnzBelP (nur ab KZP2010)" in data.columns:
    data.rename(columns={"AnzBelP (nur ab KZP2010)": "AnzBelP"}, inplace=True)
    features = ["AnzBelP" if col == "AnzBelP (nur ab KZP2010)" else col for col in features]

# Binary target
data[label] = (data[label] > 0).astype(int)

# Split
X = data[features]
y = data[label]
yar = data["JAHR"]

stratify_key = pd.Series(list(zip(y, yar)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_size, random_state=42, stratify = stratify_key)

# Feature types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# --- Preprocessing and modeling ---

if model_type == "svc":
    # --- Numerical preprocessing ---
    num_imputer = SimpleImputer(strategy="mean") ## handling missing data, alternatives: mode
    X_train_num = num_imputer.fit_transform(X_train[numerical_features])
    X_test_num = num_imputer.transform(X_test[numerical_features])

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # --- Categorical preprocessing ---
    cat_imputer = SimpleImputer(strategy = "constant", fill_value = None) # handling missing data, alternative: most_frequent
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_features])
    X_test_cat = cat_imputer.transform(X_test[categorical_features])

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False) #handle_unknown : {'error', 'ignore', 'infrequent_if_exist', 'warn'}
    X_train_cat_encoded = onehot.fit_transform(X_train_cat)
    X_test_cat_encoded = onehot.transform(X_test_cat)




    # --- Combine numeric + categorical ---
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])

    # Drop NaNs that couldn't be handled
    non_nan_rows = ~np.isnan(X_test_processed).any(axis=1)
    X_test_final = X_test_processed[non_nan_rows]
    y_test_final = y_test.iloc[non_nan_rows]

    # --- Train SVC ---
    clf = SVC(kernel="linear", random_state=42, class_weight='balanced') #class weight -> balances, improves score
    #clf = LinearSVC(random_state=42, class_weight='balanced')
    #clf = SVC(kernel="rbf", gamma = 1.0) #also polynomial would be possible
    #clf = SVC(kernel = "poly", degree = 2, gamma = 0.8)

    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_final)

    # --- Feature importance ---
    cat_feature_names = onehot.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)
    coefs = clf.coef_[0] # ---
    #coefs = clf.coef0 # : float, default=0.0
        # Independent term in kernel function.
        # It is only significant in 'poly' and 'sigmoid'.

    #print("KOEFFS", coefs) # ABS auswählen statt genaue Zahl

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': np.abs(coefs) #np.abs(coefs) #generally checks for most important weights, positive (increasing) as negative (decreasing)
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Important Features (SVC):")
    print(coef_df.head(n_ftrs))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=coef_df.head(n_ftrs))
    plt.title(f"Top {n_ftrs} Feature Importances (SVC)")
    plt.tight_layout()
    plt.savefig(f'SVC_features_top-{n_ftrs}.png', dpi=300)
    #plt.show()

# --
    # # --- Plot decision boundary of top feature ---

    # Get the most important feature and its index
    top_feature = coef_df.iloc[0]["Feature"]
    top_index = all_feature_names.index(top_feature)

    # Get the corresponding 1D feature column
    X_plot = X_train_processed[:, top_index].reshape(-1, 1)
    y_plot = y_train

    # Fit a 1D SVC model on just this feature (for plotting)
    clf_1d = SVC(kernel="linear", class_weight="balanced")
    clf_1d.fit(X_plot, y_plot)

    # Create a range of values for this feature
    x_vals = np.linspace(X_plot.min(), X_plot.max(), 500).reshape(-1, 1)
    decision_function = clf_1d.decision_function(x_vals)

    # Plot decision boundary and margins
    plt.figure(figsize=(8, 5))
    plt.scatter(X_plot, y_plot, c=y_plot, cmap=plt.cm.Paired, edgecolors="k")
    plt.plot(x_vals, decision_function, label="Decision function")
    plt.axhline(y=0, color="k", linestyle="-", label="Decision boundary")
    plt.axhline(y=1, color="k", linestyle="--", label="Margin")
    plt.axhline(y=-1, color="k", linestyle="--")

    plt.title(f"SVC Decision Boundary (Top Feature: {top_feature})")
    plt.xlabel(top_feature)
    plt.ylabel("Decision Function Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"SVC_decision_boundary_top_feature_{top_feature}.png", dpi=300)
    #plt.show()

   
#---

elif model_type == "hgb":
    # --- Categorical preprocessing ---
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) #sets handle unknown to unknown_value -> -1, ordinal encoder uses positive only.
    X_train_cat = cat_encoder.fit_transform(X_train[categorical_features])
    X_test_cat = cat_encoder.transform(X_test[categorical_features])

    # Numerical features stay as-is
    X_train_num = X_train[numerical_features].to_numpy()
    X_test_num = X_test[numerical_features].to_numpy()

    # Combine - alternatively combine all numerical first and categorical second
    X_train_processed = np.hstack([X_train_cat, X_train_num])
    X_test_processed = np.hstack([X_test_cat, X_test_num])

    # X_train_processed = np.concatenate((X_train_cat, X_train_num), axis=1)
    # X_test_processed = np.concatenate((X_test_cat, X_test_num), axis=1)


    # Train
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_processed)
    y_test_final = y_test

    # Feature names
    feature_names = categorical_features + numerical_features

    # SHAP
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test_processed)
    #feature_importances = np.abs(shap_values.values).mean(axis=0) # ---- ABS auswählen
    feature_importances = (shap_values.values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Important Features (HGB):")
    print(feature_importance_df.head(n_ftrs))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(n_ftrs), x='Importance', y='Feature')
    plt.title(f'Top {n_ftrs} Feature Importances (HGB)')
    plt.tight_layout()
    plt.savefig(f'HGB_features_top-{n_ftrs}.png', dpi=300)
    #plt.show()

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
plt.savefig(f'confusion_matrix_{model_type}.png', dpi=300)
#plt.show()

# Ensure y_test_final and y_score are available
if model_type == "svc":
    # For SVC, use decision_function - explain use
    y_score = clf.decision_function(X_test_final)
elif model_type == "hgb":
    # For HGB, use predict_proba and take the score for the positive class (1)
    y_score = clf.predict_proba(X_test_processed)[:, 1]

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_final, y_score)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.3f}")

# Plot ROC Curve
# plt.figure()
# plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})') #, color='darkorange'
# plt.plot([0, 1], [0, 1], lw=2, linestyle='--') # color='navy'
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'Receiver Operating Characteristic ({model_type.upper()})')
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig(f'roc_curve_{model_type}.png', dpi=300)
#plt.show()

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic ({model_type.upper()})')
plt.legend(loc='lower right')
plt.savefig(f'roc_curve_{model_type}.png', dpi=300)



##1.  summarize the general built up of the analysis

##2. list the parameters to modify (how & why) for each important function.

#useful links to answer questions of handling modifications -> source code of scikit-learn and mathematical concept
# https://scikit-learn.org/stable/api/index.html
# 

#HistGradientBooster is inspired by https://github.com/Microsoft/LightGBM

# Citing scikit-learn in the paper and presentation
# @article{scikit-learn,
#   title={Scikit-learn: Machine Learning in {P}ython},
#   author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
#           and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
#           and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
#           Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
#   journal={Journal of Machine Learning Research},
#   volume={12},
#   pages={2825--2830},
#   year={2011}
# }


# more Information about the StandardScaler in scikit-learn
# https://papers.probabl.ai/the-standardscaler-is-not-standard

# Bishop Pattern recognition in Machine Learning
# A Tutorial on support vector regression