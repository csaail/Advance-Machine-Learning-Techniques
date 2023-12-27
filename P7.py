#P7: Implement the classification model using clustering for the following techniques with K means clustering with Prediction, Test Score and Confusion Matrix.
#DATASET: x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12], y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()
data = list(zip(x, y))

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
