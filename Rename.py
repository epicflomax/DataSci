import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
#meow

df = pd.read_csv("kzp-2008-2020-timeseries.csv",encoding="latin1")

df_2011=df[df["JAHR"]==2011]

y = df_2011["FiErg"]
X = df_2011[["KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]]

print(X.isna().sum())
missing_percentage = X.isnull().mean() * 100

print(missing_percentage[missing_percentage > 50].sort_values(ascending=False))

df['SA'] = df['SA'].fillna(None)
df['SL'] = df['SL'].fillna(None)
df['WB'] = df['WB'].fillna(None)
