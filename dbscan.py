from sklearn.cluster import DBSCAN

# Data points
X = [
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 8],
    [8, 9],
    [25, 25]
]

# Create DBSCAN model
model = DBSCAN(eps=2, min_samples=2)

# Train and predict clusters
labels = model.fit_predict(X)

print("Cluster Labels:", labels)
