from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 2: Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Step 3: Define 5-Fold Cross Validation
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Step 4: Perform Cross Validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold
)

# Step 5: Display results
print("Accuracy of each fold:")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2f}")

print("Average Accuracy:", round(scores.mean(), 2))
