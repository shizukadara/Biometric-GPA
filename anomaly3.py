# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'finaltable.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Standardizing the features
X = data.drop(columns=["CGPA"])  # Drop the target column for anomaly detection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Contamination sets the proportion of anomalies
iso_forest.fit(X_scaled)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
anomaly_scores = iso_forest.decision_function(X_scaled)  # Higher scores indicate normal instances
anomaly_labels = iso_forest.predict(X_scaled)

# Convert labels (-1 to anomaly, 1 to normal) into boolean for easier interpretation
anomalies = anomaly_labels == -1

# Results
anomaly_indices = np.where(anomalies)[0]  # Indices of detected anomalies
print(f"Number of anomalies detected: {len(anomaly_indices)}")
print(f"Anomaly indices: {anomaly_indices}")

# Scatter plot for visualizing anomalies vs. normal data
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], label='Normal', c='blue', s=20)
plt.scatter(X_scaled[anomaly_indices, 0], X_scaled[anomaly_indices, 1], label='Anomalies', c='red', s=50)
plt.title("Anomalies vs. Normal Data")
plt.legend()
plt.show()

# Extract anomalies and normal data
normal_data = data.iloc[~data.index.isin(anomaly_indices)]
anomalies_data = data.iloc[anomaly_indices]

# Save anomalies and normal data to separate files
anomalies_data_file = 'anomalies_data.csv'
normal_data_file = 'normal_data.csv'

anomalies_data.to_csv(anomalies_data_file, index=False)
normal_data.to_csv(normal_data_file, index=False)

print(f"Anomalies saved to: {anomalies_data_file}")
print(f"Normal data saved to: {normal_data_file}")

# Feature comparison between normal data and anomalies
feature_comparison = pd.concat([
    normal_data.describe().loc[['mean', 'std', 'min', 'max']],
    anomalies_data.describe().loc[['mean', 'std', 'min', 'max']]
], axis=1, keys=['Normal Data', 'Anomalies'])

print("Feature Comparison:")
print(feature_comparison)

# Visualize feature distributions with box plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, orient="h", palette="Set2")
plt.title("Feature Distributions with Potential Anomalies")
plt.show()

# Highlight anomalies on pair plots
sns.pairplot(data, corner=True, diag_kind="kde")
plt.suptitle("Pair Plot with Feature Relationships", y=1.02)
plt.show()
