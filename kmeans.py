import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Customers.csv')

# Custom KMeans class to track changes across epochs
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Store history of cluster centers and labels
        self.cluster_centers_history = []
        self.labels_history = []

        # Initialize KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1, random_state=0)
        
        # Initial fit
        kmeans.fit(X)
        self.cluster_centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        self.cluster_centers_history.append(self.cluster_centers)
        self.labels_history.append(self.labels)

        print(f"Epoch 1:")
        print(f"Cluster Centers:\n{self.cluster_centers}")
        print(f"Labels: {self.labels}")
        print(f"Inertia: {kmeans.inertia_}\n")

        for epoch in range(1, self.max_iter):
            # Assign labels based on current cluster centers
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Update cluster centers
            self.cluster_centers = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Store results
            self.cluster_centers_history.append(self.cluster_centers)
            self.labels_history.append(self.labels)

            # Calculate inertia as the sum of squared distances to the closest cluster center
            inertia = np.sum((X - self.cluster_centers[self.labels])**2)

            # Only print results for the last epoch
            if epoch == self.max_iter - 1:
                print(f"Epoch {epoch + 1}:")
                print(f"Cluster Centers:\n{self.cluster_centers}")
                print(f"Labels: {self.labels}")
                print(f"Inertia: {inertia}\n")

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)

# KMeans Clustering Implementation
def kmeans_clustering(data, n_clusters):
    # Preprocessing
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create an instance of CustomKMeans
    custom_kmeans = CustomKMeans(n_clusters=n_clusters, max_iter=5)

    # Initial Clustering
    print("Initial Clustering Results:")
    custom_kmeans.fit(X_scaled)

    # Initial Clusters
    y_initial = custom_kmeans.labels_history[0]

    # Plot Initial Clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_initial, s=50, cmap='viridis')
    plt.title('KMeans Clusters - Epoch 1')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.savefig('initial_kmeans_clusters.png')
    plt.show()

    # Final Clustering after Epochs
    y_final = custom_kmeans.predict(X_scaled)

    # Plot Final Clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_final, s=50, cmap='viridis')
    plt.title('KMeans Clusters - Epoch 5')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.savefig('final_kmeans_clusters.png')
    plt.show()

    # Calculate Silhouette Score (Error Rate)
    error_rate = silhouette_score(X_scaled, y_final)
    print(f'Final clusters (KMeans): {y_final}')
    print(f'Silhouette Score (Error Rate): {error_rate:.4f}')

# Execute KMeans clustering
kmeans_clustering(data, n_clusters=3)
