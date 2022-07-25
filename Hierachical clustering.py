import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

X = df.iloc[:, [3, 4]].values
print(X)

import scipy.cluster.hierarchy as sch
'''
# The ward method tries to minimise the variance in each cluster
dendrogram = sch.dendrogram(sch.linkage(X, method='ward')) 
plt.title('Dendrogram')
plt.xlabel('Flower')
plt.ylabel('no of clusters')
plt.show()
'''
# Here we get 3 clusters by dendrogram
# Height in dendrogram at which two clusters are merged represents
#distance between two clusters in data space

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
'''
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_hc, s=50, cmap='viridis')

plt.title('Clusters using HC')
plt.ylabel('Petal Width')
plt.xlabel('Petal Lenght')
plt.show()
'''

from sklearn.preprocessing import LabelEncoder

columns = ['Species']

def encoder(df):
    for col in columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])
    return df

df = encoder(df)

fig = plt.figure(figsize=(15,6))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_hc, s=50, cmap='viridis')
plt.title('Clusters using hierarchical method')
plt.ylabel('Petal Width')
plt.xlabel('Petal Lenght')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=df.iloc[:,5].values, s=50, cmap='prism')
plt.title('Original Data')
plt.ylabel('Petal Width')
plt.xlabel('Petal Lenght')
plt.show()
