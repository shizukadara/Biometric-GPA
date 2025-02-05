import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset 
df = pd.read_csv('finalfinal.csv')

# Step 2: Remove irrelevant features 
irrelevant_features = [
    'TotalTimeOutsideNonOvernight_Weekend', 
    'TotalTimeOutsideOvernight_Weekend', 
    'TotalTimeOutsideOvernight_Weekday', 
    'GymnasiumVisitCount'
]
df = df.drop(columns=irrelevant_features)

# Step 3: Select features for clustering (exclude target variable CGPA and any previous cluster labels)
X = df.drop(columns=['CGPA'])

# Ensure only numeric columns are used
X = X.select_dtypes(include=[np.number])

# Handle missing values by filling them with the mean
X = X.fillna(X.mean())

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply KMeans clustering with 3 clusters (since we confirmed k=3 was optimal)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Add the new cluster labels to the dataframe
df['Cluster_KMeans_Final'] = kmeans_clusters

# Step 6: Calculate Silhouette Score for the clustering
silhouette_avg = silhouette_score(X_scaled, kmeans_clusters)
print(f'Silhouette Score for KMeans Clustering: {silhouette_avg:.3f}')

# Step 7: Generate updated cluster profiles for KMeans Clusters
kmeans_cluster_profile = df.groupby(df['Cluster_KMeans_Final']).mean(numeric_only=True)

# Rename the cluster index for better readability
kmeans_cluster_profile.index = [f'KMeans_Final_Cluster_{i}' for i in kmeans_cluster_profile.index]

# Display the updated cluster profiles
print("\n--- KMeans Cluster Profiles ---\n")
print(kmeans_cluster_profile)

# Step 8: Feature Importance using RandomForestClassifier
# Drop cluster and CGPA columns for the feature importance calculation
X = df.drop(columns=['CGPA', 'Cluster_KMeans_Final'])

# Ensure only numeric columns are used
X = X.select_dtypes(include=[np.number])

# Handle missing values by filling them with the mean
X = X.fillna(X.mean())

# Train a Random Forest Classifier to predict the cluster labels
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, df['Cluster_KMeans_Final'])

# Get feature importances
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("\n--- Feature Importance for Cluster Prediction ---\n")
print(importance_df)

# Step 9: Visualize Cluster Differences Using Boxplots
# List of key features to visualize
features_to_visualize = [
    'CGPA', 
    'TotalLibraryTime', 
    'TotalTrips_Weekday', 
    'TotalTimeOutsideNonOvernight_Weekday', 
    'BreakfastCount', 
    'LunchCount', 
    'DinnerCount'
]

# Generate boxplots one-by-one for the full dataset
for feature in features_to_visualize:
    plt.figure(figsize=(8, 6))
    sns.boxplot(hue=df['Cluster_KMeans_Final'], y=df[feature], palette='viridis')
    plt.title(f'Boxplot of {feature} Across Clusters (All Students)')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()

# Step 10: Visualize Clusters using PCA1 and PCA2
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
pca1_variance = explained_variance_ratio[0] * 100
pca2_variance = explained_variance_ratio[1] * 100

print(f"Explained Variance by PCA1: {pca1_variance:.2f}%")
print(f"Explained Variance by PCA2: {pca2_variance:.2f}%")

# Create a DataFrame for the PCA components
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster_KMeans_Final']

# Plot the clusters using PCA1 and PCA2
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_df['PCA1'], y=pca_df['PCA2'], hue=pca_df['Cluster'], palette='viridis', s=60, alpha=0.7)
plt.title(f'Clusters Visualized Using PCA1 and PCA2\n(PCA1: {pca1_variance:.2f}% variance, PCA2: {pca2_variance:.2f}% variance)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
