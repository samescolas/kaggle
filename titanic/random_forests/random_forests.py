#!/usr/bin/python

# Import RF model generator
from sklearn.ensemble import RandomForestRegressor

# Import error metric
from sklearn.metrics import roc_auc_score
import pandas as pd

# Read in data
df = pd.read_csv("../data/train.csv")

# Separate results
Y = df.pop('Survived')

# Replace null values with mean
df['Age'].fillna(df.Age.mean(), inplace=True)

# Clean up categorical values
df.fillna('None', inplace=True)

# Lists of variables
numeric_vars = list(df.dtypes[df.dtypes != 'object'].index)
categorical_vars = ['Sex', 'Cabin', 'Embarked']

#df[numeric_vars].describe()

# Drop extra information
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Strip numbers from cabin
df['Cabin'] = ['None' if c == 'None' else c[0] for c in df.Cabin]

# Convert categorical variables to numeric
for variable in categorical_vars:
  dummies = pd.get_dummies(df[variable])
  df = pd.concat([df, dummies], axis=1)

# Remove remaining categorical variables
df.drop(categorical_vars, axis=1, inplace=True)

# Create model with 1000 unique decision trees
# Need to do a hyperparameter test to see what these
# values really should be...
model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, oob_score=True, min_samples_leaf=5)
model.fit(df, Y)

print("Accuracy: {}".format(roc_auc_score(Y, model.oob_prediction_)))
