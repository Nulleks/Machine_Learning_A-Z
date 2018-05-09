# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # Todas menos la ultima, porque el limite superior esta excluido
y = dataset.iloc[:, 3].values

# Taking care of missing data
# df.fillna(df.mean()) <- mas simple xD
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])  # Limite superior esta excluido, toma las columnas 1 y 2
X[:, 1:3] = imputer.transform(X[:, 1:3])