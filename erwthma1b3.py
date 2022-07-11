import pandas as pd
import scipy
import csv
import numpy as np
import random

wine = pd.read_csv("winequality-red.csv")


X = wine.drop(['quality','pH'], axis = 1)
Y = wine['pH']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state = 42)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.67, random_state = 42)

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
Y_encoded = lab_enc.fit_transform(Y_train)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_encoded)

Y_pred=logreg.predict(X_test)

print(Y_pred)