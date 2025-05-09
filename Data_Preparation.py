import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


df = pd.read_csv("kzp-2008-2020-timeseries.csv",encoding="latin1")

data =df[df["JAHR"]==2011].copy()

label = data["FiErg"]
features = ["KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]

#print(data[features].isna().sum())
#missing_percentage = data[features].isnull().mean() * 100

#print(missing_percentage[missing_percentage > 50].sort_values(ascending=False))

#Dropping Adr
data = data.drop(columns=["Adr"])
features.remove("Adr")

num_features = data[features].select_dtypes("number").columns
cat_features = data[features].select_dtypes(exclude=["number"]).columns


#Percentage of missing values per categorical column
missing_percentage_cat = (data[cat_features].isnull().sum() / len(data[cat_features])) * 100
print("Missing percentage Categorical columns\n", missing_percentage_cat)

#Percentage of missing values per numeric column
missing_percentage_num = (data[num_features].isnull().sum() / len(data[num_features])) * 100
print("Missing percentage numeric columns\n", missing_percentage_num)

#Filling Categorical Columns
fill_in = 'NA'
data['SA'] = data['SA'].fillna(fill_in)
data['SL'] = data['SL'].fillna(fill_in)
data['WB'] = data['WB'].fillna(fill_in)

#Filling Numeric Columns
for i in data[num_features]:
    data[i] = data[i].fillna(data[i].mean())


#Transforming object col into category
print(data[cat_features].dtypes)
data[cat_features] = data[cat_features].astype('category')
print(data[cat_features].dtypes)

print(data[features])

#check

#Percentage of missing values per categorical column
missing_percentage_cat = (data[cat_features].isnull().sum() / len(data[cat_features])) * 100
print("Missing percentage Categorical columns\n", missing_percentage_cat)

#Percentage of missing values per numeric column
missing_percentage_num = (data[num_features].isnull().sum() / len(data[num_features])) * 100
print("Missing percentage numeric columns\n", missing_percentage_num)



