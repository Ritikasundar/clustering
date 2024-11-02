# Customer Segmentation with KMeans and BIRCH

This project uses KMeans and BIRCH clustering to segment customers based on `Annual Income` and `Spending Score`. Clustering helps identify distinct groups, useful for targeted marketing and analysis.

## Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/Ritikasundar/clustering.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Ensure `Customers.csv` is in the project root with `Annual Income (k$)` and `Spending Score (1-100)` columns.

## Usage

### Run KMeans Clustering
```bash
python kmeans_clustering.py
```

### Run BIRCH Clustering
```bash
python birch_clustering.py
```

## Results

**Outputs include:**

* Cluster centers, labels, and inertia across epochs.
* 2D and 3D cluster visualizations.
* Silhouette scores for cluster quality.

## Dependencies

**Dependencies:**

* `pandas`
* `numpy`
* `matplotlib`
* `sklearn`

## Visualizations

**Visualizations:**

Images saved as:

* `initial_kmeans_clusters.png` and `final_kmeans_clusters.png` (KMeans)
* `birch_clusters_2d.png` and `birch_clusters_3d.png` (BIRCH)
