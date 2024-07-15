import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns


df = pd.read_csv("Retail_Sales_Data.csv")

# Display the first few rows of the dataframe to verify the data
print(df.head())

# Select Features: Using 'Age', 'Quantity', 'Price per Unit', and 'Total Amount' for clustering
X = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Cluster labels
labels = kmeans.labels_
df['Cluster'] = labels

# Plotting the clusters
# Since we have more than 2 dimensions, we'll use a pair plot for visualization

sns.pairplot(df, hue='Cluster', diag_kind='kde', vars=['Age', 'Quantity', 'Price per Unit', 'Total Amount'])
plt.show()

# Optional: Print the cluster centers (in standardized form)
print("Cluster centers (standardized):\n", kmeans.cluster_centers_)

# To interpret the cluster centers in the original scale:
original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster centers (original scale):\n", original_centers)
