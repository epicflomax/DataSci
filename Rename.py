import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
#meow

df = pd.read_csv("kzp-2008-2020-timeseries.csv",encoding="latin1")

df_2011=df[df["JAHR"]==2011]

y = df_2011["FiErg"]
X = df_2011[["FiErg","KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP"]]
