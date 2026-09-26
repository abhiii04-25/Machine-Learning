from sklearn.metrics import confusion_matrix, accuracy_score

# Actual values
y_actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

# Predicted values
y_predicted = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]

# Create confusion matrix
cm = confusion_matrix(y_actual, y_predicted)

print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_actual, y_predicted)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
