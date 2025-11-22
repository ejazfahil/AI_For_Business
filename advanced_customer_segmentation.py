"""
Advanced Customer Segmentation
Using multiple clustering algorithms and evaluation metrics

Based on latest research:
- K-Means clustering
- DBSCAN for density-based clustering
- Gaussian Mixture Models (GMM)
- Hierarchical clustering
- Comprehensive evaluation with multiple metrics
- UMAP for dimensionality reduction (alternative to PCA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# Load data
data = pd.read_csv('Wholesale customers data.csv')

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nDataset statistics:")
print(data.describe())

# Remove Channel and Region for clustering
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
X = data[features]

# Log transformation to handle skewness
X_log = np.log(X + 1)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Dimensionality Reduction
print("\n" + "="*50)
print("Dimensionality Reduction")
print("="*50)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# UMAP (if available)
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    print("\nUMAP reduction completed")

# Function to evaluate clustering
def evaluate_clustering(X, labels, algorithm_name):
    """Calculate clustering evaluation metrics"""
    # Remove noise points for DBSCAN
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    if len(np.unique(labels_filtered)) < 2:
        print(f"{algorithm_name}: Not enough clusters for evaluation")
        return None
    
    silhouette = silhouette_score(X_filtered, labels_filtered)
    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
    
    return {
        'algorithm': algorithm_name,
        'n_clusters': len(np.unique(labels_filtered)),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz
    }

# Dictionary to store results
results = []
clustering_models = {}

print("\n" + "="*50)
print("Clustering Algorithms")
print("="*50)

# 1. K-Means Clustering
print("\n1. K-Means Clustering")
print("-" * 30)

# Find optimal number of clusters using elbow method and silhouette score
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, labels))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('K-Means Elbow Method')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('K-Means Silhouette Score')
ax2.grid(True)

plt.tight_layout()
plt.savefig('kmeans_optimization.png', dpi=300, bbox_inches='tight')
print("K-Means optimization plots saved as 'kmeans_optimization.png'")

# Use optimal k (based on silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca)
clustering_models['kmeans'] = kmeans

result = evaluate_clustering(X_pca, kmeans_labels, 'K-Means')
if result:
    results.append(result)
    print(f"Silhouette Score: {result['silhouette_score']:.4f}")

# 2. DBSCAN
print("\n2. DBSCAN Clustering")
print("-" * 30)

# Try different eps values
eps_values = [0.3, 0.5, 0.7, 1.0]
best_dbscan_score = -1
best_dbscan_eps = None

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_pca)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters >= 2:
        result = evaluate_clustering(X_pca, labels, f'DBSCAN (eps={eps})')
        if result and result['silhouette_score'] > best_dbscan_score:
            best_dbscan_score = result['silhouette_score']
            best_dbscan_eps = eps
            best_dbscan_labels = labels

print(f"Best DBSCAN eps: {best_dbscan_eps}")
dbscan = DBSCAN(eps=best_dbscan_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)
clustering_models['dbscan'] = dbscan

result = evaluate_clustering(X_pca, dbscan_labels, 'DBSCAN')
if result:
    results.append(result)
    print(f"Silhouette Score: {result['silhouette_score']:.4f}")
    print(f"Number of noise points: {list(dbscan_labels).count(-1)}")

# 3. Gaussian Mixture Model
print("\n3. Gaussian Mixture Model (GMM)")
print("-" * 30)

# Find optimal number of components
bic_scores = []
aic_scores = []
n_components_range = range(2, 11)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_pca)
    bic_scores.append(gmm.bic(X_pca))
    aic_scores.append(gmm.aic(X_pca))

# Plot BIC and AIC
plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bic_scores, 'bo-', label='BIC')
plt.plot(n_components_range, aic_scores, 'ro-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('GMM Model Selection')
plt.legend()
plt.grid(True)
plt.savefig('gmm_optimization.png', dpi=300, bbox_inches='tight')
print("GMM optimization plot saved as 'gmm_optimization.png'")

# Use optimal number of components (lowest BIC)
optimal_components = n_components_range[np.argmin(bic_scores)]
print(f"Optimal number of components: {optimal_components}")

gmm = GaussianMixture(n_components=optimal_components, random_state=42)
gmm_labels = gmm.fit_predict(X_pca)
clustering_models['gmm'] = gmm

result = evaluate_clustering(X_pca, gmm_labels, 'GMM')
if result:
    results.append(result)
    print(f"Silhouette Score: {result['silhouette_score']:.4f}")

# 4. Hierarchical Clustering
print("\n4. Hierarchical Clustering")
print("-" * 30)

hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_pca)
clustering_models['hierarchical'] = hierarchical

result = evaluate_clustering(X_pca, hierarchical_labels, 'Hierarchical')
if result:
    results.append(result)
    print(f"Silhouette Score: {result['silhouette_score']:.4f}")

# Compare all algorithms
print("\n" + "="*50)
print("Clustering Comparison")
print("="*50)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
algorithms = ['kmeans', 'dbscan', 'gmm', 'hierarchical']
labels_dict = {
    'kmeans': kmeans_labels,
    'dbscan': dbscan_labels,
    'gmm': gmm_labels,
    'hierarchical': hierarchical_labels
}

for idx, (ax, algo) in enumerate(zip(axes.flat, algorithms)):
    labels = labels_dict[algo]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title(f'{algo.upper()} Clustering')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
print("\nClustering comparison plot saved as 'clustering_comparison.png'")

# Analyze cluster characteristics for best model
best_model_name = results_df.loc[results_df['silhouette_score'].idxmax(), 'algorithm']
print(f"\n" + "="*50)
print(f"Best Model: {best_model_name}")
print("="*50)

best_labels = labels_dict[best_model_name.lower().split()[0]]
data['Cluster'] = best_labels

# Cluster statistics
print("\nCluster Statistics:")
cluster_stats = data.groupby('Cluster')[features].mean()
print(cluster_stats)

# Save cluster statistics
cluster_stats.to_csv('cluster_statistics.csv')
print("\nCluster statistics saved as 'cluster_statistics.csv'")

print("\n" + "="*50)
print("Advanced Customer Segmentation Complete!")
print("="*50)
