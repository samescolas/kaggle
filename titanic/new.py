#!/usr/bin/python

import pandas as pd

df = pd.read_csv("data/train.csv")

print("Training data size: {}".format(df.shape))

print("First 5 rows: {}".format(df.head(10)))

print("Survival info: {}".format(df.Survived.value_counts()))
