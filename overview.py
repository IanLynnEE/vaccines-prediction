import numpy as np
import pandas as pd

from preprocessing import read_and_encode, impute_features

df = read_and_encode('data/training_set_features.csv')
# x, _ = impute_features(df, df, strategy='mean')
df.info()

x = df.to_numpy()
y = pd.read_csv('data/training_set_labels.csv').to_numpy()
for i in range(3, 16):
    dist = 0
    missing = 0
    for j in range(x.shape[0]):
        if np.isnan(x[j, i]):
            missing += 1
        elif x[j, i] != y[j, 1]:
            dist += 1
    print(i, dist, missing)


both = 0
h1n1_only = 0
seasonal_only = 0
neither = 0
for row in y:
    if row[1] == 1:
        if row[2] == 1:
            both += 1
        elif row[2] == 0:
            h1n1_only += 1
    elif row[2] == 1:
        seasonal_only += 1
    else:
        neither += 1
print(y.shape)
print(h1n1_only, both, seasonal_only, neither)
