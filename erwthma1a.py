import pandas as pd
import scipy
import csv
import numpy as np

wine = pd.read_csv("winequality-red.csv")

X = wine.drop(['quality'], axis = 1)
Y = wine['quality']


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state = 42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC

svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')

svc2.fit(X_train, Y_train)

Y_pred = svc2.predict(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score

F1 = f1_score(Y_test, Y_pred, average= 'micro')
prec = precision_score(Y_test, Y_pred, average='micro')
rec = recall_score(Y_test, Y_pred, average='micro')

print(F1)
print(prec)
print(rec)