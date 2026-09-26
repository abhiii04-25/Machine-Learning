from sklearn.linear_model import Perceptron

# Input data
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# AND gate output
y = [0, 0, 0, 1]

# Create Perceptron model
model = Perceptron()

# Train the model
model.fit(X, y)

# Predict output
predictions = model.predict(X)

print("Predictions:", predictions)

# Test new input
result = model.predict([[1, 1]])

print("Prediction for [1, 1]:", result[0])
