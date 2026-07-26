# Data Classification Using AI
# Author: Your Name

# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
iris = load_iris()

X = iris.data        # Features
y = iris.target      # Labels

print("Dataset Loaded Successfully!")
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("Total Samples:", len(X))

# -----------------------------
# Step 2: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------
# Step 3: Train Model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# -----------------------------
# Step 4: Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Step 6: Predict New Data
# -----------------------------
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nSample Data:", sample)
print("Predicted Flower:", iris.target_names[prediction[0]])
