
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(45,6,1000)
y2 = np.random.normal(33,5,1000)

x3 = np.random.normal(44,4,1000)
y3 = np.random.normal(10,2,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

single = {"x":x,"y":y}
data = pd.DataFrame(single)

plt.scatter(x1,y1,alpha= 0.3)
plt.scatter(x2,y2,alpha= 0.3)
plt.scatter(x3,y3,alpha= 0.3)
plt.show()

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("K")
plt.ylabel("wcss")
plt.show()

Kmeans2 = KMeans(n_clusters=3)
clusters = Kmeans2.fit_predict(data)

data["Clusters"] = clusters

plt.scatter(data.x[data.Clusters == 0 ],data.y[data.Clusters == 0],color = "black")
plt.scatter(data.x[data.Clusters == 1 ],data.y[data.Clusters == 1],color = "red")
plt.scatter(data.x[data.Clusters == 2 ],data.y[data.Clusters == 2],color = "yellow")
plt.scatter(Kmeans2.cluster_centers_[:,0],Kmeans2.cluster_centers_[:,1],color = "green")
plt.show()




































