from Data_Preparation import data
from Data_Preparation import label
from Data_Preparation import features
from Data_Preparation import num_features
from Data_Preparation import cat_features
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

#SPLIT DATA
features_train, features_test, label_train, label_test = train_test_split(
    features, label, test_size=0.2, random_state=2023, stratify=label)

#SCALE NUMERIC FEATURES

#ONE HOT ENCODE CATEGORICAL FEATURES