#!/usr/bin/python

import pandas as pd

# Read CSV into dataframe
df = pd.read_csv("data/train.csv")

# Number of records
entries = df.shape[0]

#print("First 5 rows: {}".format(df.head(10)))

# Number of people that survived/died
survival_info = df.Survived.value_counts()
sex_info = df.Sex.value_counts()

print("Survival rate: {}".format(float(survival_info[1])/entries))
print("Percent female: {}".format(float(sex_info[1])/entries))

# Ex: Filter dataset
#print(df[df.Sex=='male'])

#print(df.describe())

df.Fare.hist()
