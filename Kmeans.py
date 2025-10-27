#Kmeans
"""KMeans"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import time
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# Step 2: Select features
df = df.select_dtypes(include=[np.number])   # only numeric columns
print(df.head())

# Define n (samples) and m (features) from dataset
n, m = df.shape
print(f"\nNumber of samples: {n}, Number of features: {m}")

data = df.values

# Step 3: Elbow Method
sse = []
K_range = range(1, min(10, len(df)))

for k in K_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init=10) # Added n_init=10 to suppress warning
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.plot(K_range, sse, marker='*', linestyle='--', color='blue')
plt.xticks(K_range)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE")
plt.title("Elbow Method for K")
plt.grid(True)
plt.show()

print("\n Take a moment to observe the Elbow Plot above...")
time.sleep(5)

# Step 4: User inputs best K
k = int(input("Enter the best value of K (number of clusters): "))

# Step 5: Initialize centroids randomly
np.random.seed(0)
initial_centroids = data[np.random.choice(n, k, replace=False)]
print("\nInitial Centroids:")
print(initial_centroids)

# Step 6: Repeat until centroids converge
t = 0.000001
iteration = 1
while True:
    print(f"\nIteration {iteration}")
    iteration += 1

    # Compute distances
    distance = []
    for point in data:
        dist = [np.sqrt(np.sum((point - centroid) ** 2)) for centroid in initial_centroids]
        distance.append(dist)
    cluster = np.argmin(distance, axis=1)
    df['Cluster'] = cluster

    # Calculate new centroids
    new_centroid = []
    for l in range(k):
        cluster_points = df[df['Cluster'] == l].iloc[:, :-1].values
        if len(cluster_points) > 0:
            centroid_k = np.mean(cluster_points, axis=0)
        else:
            new_centroid.append(initial_centroids[l]) # keep old if empty cluster
            continue
        new_centroid.append(centroid_k)

    new_centroid = np.array(new_centroid)
    print("\nNew Centroids:")
    print(new_centroid)

    # Check convergence
    diff = np.abs(new_centroid - initial_centroids)
    if np.all(diff < t):
        break
    initial_centroids = new_centroid

# Final Output
print("\nFinal Clusters:")
print(df)