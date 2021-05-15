import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataframe = pd.read_csv("Mall_customers.csv", index_col="CustomerID")
print(dataframe)

#setup variables to be input in k-means
X = dataframe.iloc[:,[1,3]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=30)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(x=range(1,11), y=wcss, marker='o', color='red')
plt.xlabel('number of cluster centers')
plt.ylabel('wcss')
plt.title('elbow method')
plt.show()

#after deducting that n_clusters=4 is the elbow value we use it to plot the scatterplot
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=30)
y_pred = kmeans.fit_predict(X)
plt.figure(figsize=(10,5))

for i in range(4):
    plt.scatter(X[y_pred==i,0], X[y_pred==i,1], label=f'cluster{i + 1}')
    plt.legend()

plt.title('Customer Clusters')
plt.xlabel('age')
plt.ylabel('spending score')
plt.show()