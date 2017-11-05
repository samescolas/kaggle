#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV into dataframe
df = pd.read_csv("../data/train.csv")
print(df.head(10))

# Number of records
entries = df.shape[0]

# Separate results from features
Y = df.pop('Survived')

print(Y.head())

#print("First 5 rows: {}".format(df.head(10)))

# Number of people that survived/died
survival_info = Y.value_counts()
sex_info = df.Sex.value_counts()

print("Survival rate: {}".format(float(survival_info[1])/entries))
print("Percent female: {}".format(float(sex_info[1])/entries))

# Ex: Filter dataset
#print(df[df.Sex=='male'])

#print(df.describe())

plt.plot(df.Sex, Y)
plt.show()
