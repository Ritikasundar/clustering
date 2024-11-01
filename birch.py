import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('Customers.csv')

# BIRCH Clustering Implementation
def birch_clustering(data, n_clusters):
    # Preprocessing
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create an instance of BIRCH
    birch_model = Birch(n_clusters=n_clusters)

    # Fit the model
    y_birch = birch_model.fit_predict(X_scaled)

    # Plot BIRCH Clusters in 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_birch, s=50, cmap='viridis')
    plt.title('BIRCH Clusters (2D View)')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.savefig('birch_clusters_2d.png')
    plt.show()

    # Calculate Silhouette Score
    error_rate = silhouette_score(X_scaled, y_birch)
    print(f'Final clusters (BIRCH): {y_birch}')
    print(f'Silhouette Score (Error Rate): {error_rate:.4f}')

    # Create a 3D scatter plot of the clusters
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adding a third dimension: customer index (for example purposes)
    customer_index = np.arange(X_scaled.shape[0])

    # Plotting the clusters in 3D
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], customer_index, c=y_birch, s=50, cmap='viridis', alpha=0.6)
    ax.set_title('BIRCH Clusters (3D View)')
    ax.set_xlabel('Annual Income (scaled)')
    ax.set_ylabel('Spending Score (scaled)')
    ax.set_zlabel('Customer Index')
    plt.savefig('birch_clusters_3d.png')
    plt.show()

# Execute BIRCH clustering
birch_clustering(data, n_clusters=3)
