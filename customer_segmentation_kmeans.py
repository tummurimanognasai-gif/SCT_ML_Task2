import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

np.random.seed(42)
n_customers = 200

customer_data = pd.DataFrame({
    'Customer_ID': range(1, n_customers + 1),
    'Annual_Spending': np.concatenate([
        np.random.normal(2000, 500, 60),
        np.random.normal(8000, 1500, 70),
        np.random.normal(15000, 2000, 70)
    ]),
    'Purchase_Frequency': np.concatenate([
        np.random.normal(5, 2, 60),
        np.random.normal(15, 3, 70),
        np.random.normal(30, 5, 70)
    ]),
    'Average_Order_Value': np.concatenate([
        np.random.normal(50, 15, 60),
        np.random.normal(200, 40, 70),
        np.random.normal(500, 100, 70)
    ])
})

scaler = StandardScaler()
features = customer_data[['Annual_Spending', 'Purchase_Frequency', 'Average_Order_Value']]
features_scaled = scaler.fit_transform(features)

inertias = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(features_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, kmeans_temp.labels_))

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(features_scaled)

print("=" * 60)
print("K-MEANS CLUSTERING - CUSTOMER SEGMENTATION")
print("=" * 60)

cluster_names = {0: 'Budget Conscious', 1: 'Standard', 2: 'Premium'}

for cluster in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    print(f"\nCluster {cluster} - {cluster_names[cluster]}")
    print(f"  Customer Count: {len(cluster_data)}")
    print(f"  Avg Annual Spending: ${cluster_data['Annual_Spending'].mean():,.2f}")
    print(f"  Avg Purchase Frequency: {cluster_data['Purchase_Frequency'].mean():.1f} times/year")
    print(f"  Avg Order Value: ${cluster_data['Average_Order_Value'].mean():.2f}")

print(f"\nModel Performance:")
print(f"  Optimal Clusters: {optimal_k}")
print(f"  Silhouette Score: {silhouette_score(features_scaled, customer_data['Cluster']):.4f}")
print(f"  Inertia: {kmeans.inertia_:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method for Optimal k')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score Analysis')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

scatter = axes[1, 0].scatter(customer_data['Annual_Spending'], 
                             customer_data['Purchase_Frequency'],
                             c=customer_data['Cluster'], cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Annual Spending ($)')
axes[1, 0].set_ylabel('Purchase Frequency')
axes[1, 0].set_title('Clusters: Spending vs Frequency')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')

scatter2 = axes[1, 1].scatter(customer_data['Purchase_Frequency'], 
                              customer_data['Average_Order_Value'],
                              c=customer_data['Cluster'], cmap='viridis', 
                              s=100, alpha=0.6, edgecolors='black')
axes[1, 1].set_xlabel('Purchase Frequency')
axes[1, 1].set_ylabel('Average Order Value ($)')
axes[1, 1].set_title('Clusters: Frequency vs Order Value')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1, 1], label='Cluster')

plt.tight_layout()
plt.savefig('customer_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

customer_data.to_csv('customer_clusters.csv', index=False)
print("\n✓ Visualization saved as 'customer_clustering_analysis.png'")
print("✓ Customer data with clusters saved as 'customer_clusters.csv'")