import pandas as pd

df = pd.read_csv("data/predictions.csv")

print(df.groupby(['label', 'anomaly_label']).size())

print(df.groupby('label')['anomaly_score'].mean())

fp = df[(df["label"]!="attack") & (df["anomaly_label"]==-1)] #label is not attack while anomaly label says -1
fn = df[(df['label'] == 'attack') & (df['anomaly_label']==1)] #label is attack but anomaly label says not

import matplotlib.pyplot as plt

for lbl, grp in df.groupby("label"):
    plt.hist(grp["anomaly_score"], bins=30, alpha=0.5, label=lbl)
plt.legend()
plt.xlabel("Anomaly score")
plt.ylabel("Count")
plt.title("Score distribution by label")
plt.show()
