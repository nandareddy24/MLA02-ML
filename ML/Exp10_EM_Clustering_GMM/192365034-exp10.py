import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Create dataset
X = np.array([
    [1], [2], [3], [4], [5],
    [10], [11], [12], [13], [14],
    [20], [21], [22], [23], [24]
])

# Create Gaussian Mixture Model
model = GaussianMixture(
    n_components=3,
    random_state=42
)

# Apply EM algorithm
model.fit(X)

# Predict clusters
labels = model.predict(X)

# Display results
print("Data:")
print(X.flatten())

print("\nCluster Labels:")
print(labels)

print("\nCluster Means:")
print(model.means_)

print("\nModel Converged:", model.converged_)

# Plot graph
plt.scatter(
    X.flatten(),
    np.zeros(len(X)),
    c=labels,
    cmap='viridis',
    s=100
)

plt.title("Expectation Maximization Clustering")
plt.xlabel("Data Values")
plt.ylabel("Clusters")
plt.show()
