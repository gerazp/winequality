import pandas as pd
import scipy
import csv
import numpy as np
import random

wine = pd.read_csv("winequality-red.csv")

X = wine.drop(['quality'], axis = 1)
Y = wine['quality']


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state = 42)

X_train['pH'] = X_train['pH'].sample(frac=0.67)

print(X_train['pH'])

X_train['pH'].fillna(X_train['pH'].mean(), inplace=True)

print(X_train['pH'])