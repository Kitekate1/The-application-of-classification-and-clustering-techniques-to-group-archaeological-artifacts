import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('../KERAMIKA.csv')#, names='')   

print(dataset.head())
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, -1].values
print(X)
print()
print()
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
scaler = StandardScaler()
print(X_test)
scaler.fit(X_train)
x2 = scaler.fit_transform(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_pred)
print("--------------", 10*'\n')
print(roc_auc_score(y_test, y_pred))
"""
pca = PCA(n_components=2)
princip = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
principalDf = pd.DataFrame(data = princip, columns = ['principal component 1', 'principal component 2'])
prTest = pd.DataFrame(data = X_test, columns = ['principal component 1', 'principal component 2'])

plt.figure(figsize=(24, 8), dpi=96)

print(principalDf.shape[0])
for i in range(principalDf.shape[0]):
        if y_train[i] == 1:
            plt.plot(principalDf.iloc[i]['principal component 1'], principalDf.iloc[i]['principal component 2'], marker='o',
            markerfacecolor='r',
            markeredgecolor='r',
            linestyle = 'None')
        elif y_train[i] == 2:
            plt.plot(principalDf.iloc[i]['principal component 1'], principalDf.iloc[i]['principal component 2'], marker='v',
            markerfacecolor='b',
            markeredgecolor='b',
            linestyle = 'None')
        elif y_train[i] == 3:
            plt.plot(principalDf.iloc[i]['principal component 1'], principalDf.iloc[i]['principal component 2'], marker='h',
            markerfacecolor='y',
            markeredgecolor='y',
            linestyle = 'None')
        elif y_train[i] == 4:
            plt.plot(principalDf.iloc[i]['principal component 1'], principalDf.iloc[i]['principal component 2'], marker='s',
            markerfacecolor='c',
            markeredgecolor='c',
            linestyle = 'None')
for i in range(prTest.shape[0]):
        if y_pred[i] == 1:
            plt.plot(prTest.iloc[i]['principal component 1'], prTest.iloc[i]['principal component 2'], marker='D',
            markerfacecolor='none',
            markeredgecolor='r',
            markersize = 10.55,
            linestyle = 'None')
        elif y_pred[i] == 2:
            plt.plot(prTest.iloc[i]['principal component 1'], prTest.iloc[i]['principal component 2'], marker='D',
            markerfacecolor='none',
            markeredgecolor='b',
            markersize = 10.55,
            linestyle = 'None')
        elif y_pred[i] == 3:
            plt.plot(prTest.iloc[i]['principal component 1'], prTest.iloc[i]['principal component 2'], marker='D',
            markerfacecolor='none',
            markeredgecolor='y',
            markersize = 10.55,
            linestyle = 'None')
        elif y_pred[i] == 4:
            plt.plot(prTest.iloc[i]['principal component 1'], prTest.iloc[i]['principal component 2'], marker='D',
            markerfacecolor='none',
            markeredgecolor='c',
            markersize = 10.55,
            linestyle = 'None')

        

plt.show()
"""